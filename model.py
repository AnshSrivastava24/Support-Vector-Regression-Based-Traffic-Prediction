import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import joblib
import random


file_path = "<path to the SVR dataset file>"
print("Loading data...")
data = pd.read_csv(file_path)
print(f'Data loaded. Shape: {data.shape[0]} rows, {data.shape[1]} columns.')

#Data Cleaning
print("converting time to datetime format")
data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce')


print("cleaning invalid entries in time")
data.dropna(subset=['Time'], inplace=True)


print("extracting hour and minute from time")
data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute


print("dropping original time column")
data.drop(columns=['Time'], inplace=True)

#Handle Missing Values
print("imputing missing values")
imputer = SimpleImputer(strategy='mean')
numeric_columns = data.select_dtypes(include=[np.number]).columns

data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

#Feature Engineering
#One-hot encode 'Day of the week'
print("one-hot encoding day of the week")
data = pd.get_dummies(data, columns=['Day of the week'], drop_first=True)

#Ensure Junction is treated as categorical
print("treating junction as categorical")
data['Junction'] = data['Junction'].astype(int).astype(str)
data = pd.get_dummies(data, columns=['Junction'], prefix='Junction', drop_first=True)

#Create Lane-wise Weighted Vehicle Count
x = 1  # Base weight for light vehicles (LV)

for lane in range(1, 5):
    data[f'Lane {lane} Weighted'] = (
        data[f'Lane {lane}(LV)'] * x + data[f'Lane {lane}(HV)'] * 2 * x
    )
print("weighted vehicle counts created")
#Create Target Variable - Synthetic Signal Duration
for lane in range(1, 5):
    data[f'Lane {lane} Signal Duration'] = (
        data[f'Lane {lane} Weighted'] * 0.7 + np.random.normal(scale=5, size=len(data))
    )
print("target variable created")

#Prepare the Dataset for Modeling
lane_data_list = []

for lane in range(1, 5):
    lane_data = data.copy()
    lane_data['Lane Number'] = lane
    lane_data['Lane Weighted'] = lane_data[f'Lane {lane} Weighted']
    lane_data['Signal Duration'] = lane_data[f'Lane {lane} Signal Duration']
    
    
    for other_lane in range(1, 5):
        lane_data.drop(columns=[f'Lane {other_lane} Weighted', f'Lane {other_lane}(LV)', f'Lane {other_lane}(HV)', f'Lane {other_lane} Signal Duration'], inplace=True)
    
    lane_data_list.append(lane_data)
print("data prepared for modeling")

final_data = pd.concat(lane_data_list, ignore_index=True)

#Define Features and Target Variable
#Features: 'Lane Weighted', 'Hour', 'Minute', 'Day of the week' (encoded), 'Junction' (encoded), 'Lane Number' (encoded)
#Target: 'Signal Duration'
#Encode 'Lane Number' as a categorical variable
final_data['Lane Number'] = final_data['Lane Number'].astype(str)
final_data = pd.get_dummies(final_data, columns=['Lane Number'], prefix='Lane', drop_first=True)
print("features and target variable defined")

feature_columns = ['Lane Weighted', 'Hour', 'Minute'] + \
                  [col for col in final_data.columns if 'Day of the w_' in col or 'Junction_' in col or 'Lane_' in col]
X = final_data[feature_columns]
print("features defined")




# Save the encoded values of the first 10 rows to a CSV file
encoded_data_first_10 = final_data.head(10)
encoded_data_first_10.to_csv('encoded_data_first_10.csv', index=False)
print("Encoded values of the first 10 rows saved to 'encoded_data_first_10.csv'")




# Target
y = final_data['Signal Duration']
print("target variable defined")
#Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("data split into train and test sets")

joblib.dump((X_train, X_test, y_train, y_test), 'split_data.pkl')
print("Preprocessed data saved to 'split_data.pkl'")

#Train the SVR Model
svr_model = SVR(kernel='rbf', C=150, gamma=0.0005, epsilon=0.02)
svr_model.fit(X_train, y_train)
print("model trained")

#Save the trained model
joblib.dump(svr_model, 'trained_svr_model.pkl')

#Evaluate the Model
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {MAE}")

#Function to Take User Input and Predict Signal Duration

def predict_signal_duration():
    # Load the trained model
    svr_model = joblib.load('trained_svr_model.pkl')

    # Take user inputs
    day_of_week = input("Enter day of the week (e.g., Tuesday): ").strip()
    time_input = input("Enter time of day (HH:MM in 24-hour format, e.g., 14:30): ").strip()
    junction_number = input("Enter junction number (1 to 4): ").strip()
    lane_number = input("Enter lane number (1 to 4): ").strip()

    # Parse time input
    try:
        time_obj = datetime.datetime.strptime(time_input, '%H:%M')
        hour = time_obj.hour
        minute = time_obj.minute
    except ValueError:
        print("Invalid time format. Please enter time in HH:MM format.")
        return

    # Prepare the input data
    input_data = pd.DataFrame({
        'Lane Weighted': [final_data['Lane Weighted'].mean()],  # Using average weighted count
        'Hour': [hour],
        'Minute': [minute]
    })

    # Encode 'Day of the week'
    for col in [col for col in final_data.columns if 'Day of the w_' in col]:
        input_data[col] = 0

    # Handle case sensitivity and input issues with day of the week
    day_column = f'Day of the week_{day_of_week}'
    if day_column in input_data.columns:
        input_data[day_column] = 1
    #else:
        print(f"Day '{day_of_week}' not found in training data. Using default encoding.")

    # Encode 'Junction'
    for col in [col for col in final_data.columns if 'Junction_' in col]:
        input_data[col] = 0
    junction_column = f'Junction_{junction_number}'
    if junction_column in input_data.columns:
        input_data[junction_column] = 1
    #else:
        print(f"Junction '{junction_number}' not found in training data. Using default encoding.")

    # Encode 'Lane Number'
    for col in [col for col in final_data.columns if 'Lane_' in col]:
        input_data[col] = 0
    lane_column = f'Lane_{lane_number}'
    if lane_column in input_data.columns:
        input_data[lane_column] = 1
    #else:
        print(f"Lane '{lane_number}' not found in training data. Using default encoding.")

    # Ensure all required columns are present
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[X.columns]

    # Make prediction
    predicted_duration = svr_model.predict(input_data)[0]

    print(f"\nPredicted green signal duration for Junction {junction_number}, Lane {lane_number}, on {day_of_week} at {time_input}: {predicted_duration:.2f} seconds.")

# Call the prediction function
while True:
    predict_signal_duration()
