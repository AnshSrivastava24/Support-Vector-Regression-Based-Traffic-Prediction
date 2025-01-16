import joblib
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


#Load the preprocessed data
X_train, X_test, y_train, y_test = joblib.load('split_data.pkl')
print("Preprocessed data loaded successfully")

# Step 1: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid and perform grid search (same as the previous grid search code)
param_grid = {
    'C': [50, 100, 150],  # A slight range around the best value
    'gamma': [0.0005, 0.001, 0.002],  # Keep a bit of flexibility
    'epsilon': [0.005, 0.01, 0.02],  # Not too tightly bound
}


grid_search = GridSearchCV(SVR(), param_grid, refit=True, verbose=2, cv=10)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found by grid search:", grid_search.best_params_)
best_svr_model = grid_search.best_estimator_
best_svr_model.fit(X_train_scaled, y_train)

# Save the best model
joblib.dump(best_svr_model, 'best_trained_svr_model.pkl')

# Evaluate the model
y_pred = best_svr_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error after tuning: {mse}")
print(f"R-squared after tuning: {r2}")
print(f"Mean Absolute Error: {mae}")
