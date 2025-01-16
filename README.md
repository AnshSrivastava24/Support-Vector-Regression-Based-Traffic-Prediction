# Support-Vector-Regression-Based-Traffic-Prediction

This repository contains the implementation of paper titled [**Support Vector Regression Based Traffic Prediction Machine Learning Model**](https://ieeexplore.ieee.org/abstract/document/10816969), presented at [2024 8th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS)](https://ieeexplore.ieee.org/xpl/conhome/10816706/proceeding)

Designed to forecast and optimize green signal durations at traffic intersections. The system utilizes machine learning, specifically Support Vector Regression (SVR), and is trained on real-world traffic volume data to provide intelligent traffic control solutions.

**Authors : [Ansh Srivastava](https://www.linkedin.com/in/ansh-srivastava-ab908524b/) , [Mrigaannkaa Singh](https://www.linkedin.com/in/mrigaannkaa-singh-39b1a9191/) , [Somesh Nandi](https://www.linkedin.com/in/dr-somesh-nandi-647469a8/)**
---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

Urban traffic congestion poses significant challenges in modern cities, leading to delays, environmental pollution, and resource inefficiencies. This project addresses these challenges by introducing a predictive model that optimizes green signal durations based on traffic load. By taking into account both light and heavy vehicle counts, the system adapts to varying traffic conditions, ensuring smoother traffic flow.

The model is built using Support Vector Regression (SVR) and achieves high accuracy through hyperparameter tuning and performance optimization techniques.

---

## ✨ Features

1. **Weighted Traffic Analysis**:
   - Differentiates between light and heavy vehicles.

2. **Dynamic Signal Duration Calculation**:
   - Predicts green signal duration using a weighted count formula:
   
     Signal Duration = (Weighted Count x 0.7) + \epsilon
     
   - Includes Gaussian noise (\(\epsilon\)) for realistic predictions, simulating real-world traffic fluctuations.

3. **Efficient Data Processing**:
   - Handles missing data with mean imputation.
   - Transforms categorical features (e.g., day of the week, junction) using one-hot encoding.

4. **Machine Learning Model**:
   - Employs Support Vector Regression (SVR) with an RBF kernel.
   - Optimized through GridSearchCV for parameters like \(C\), \(\gamma\), and \(\epsilon\).

5. **Robust Evaluation**:
   - Evaluated using Mean Squared Error (MSE), R-squared (\(R^2\)), and Mean Absolute Error (MAE).

---

## 📊 Dataset

The dataset used in this project contains traffic volume data for a period of 10 months. Key details:
- **Size**: 13 columns and 29,713 rows.
- **Key Columns**: `Time`, `Date`, `Day of the Week`, `Junction`, `Traffic Volume (Light Vehicles and Heavy Vehicles)`.

### Data Preprocessing
- Extracted `hour` and `minute` from the `Time` column.
- Cleaned invalid time entries and handled missing data with imputation.
- Categorical variables were transformed using one-hot encoding.

---

## ⚙️ Methodology

1. **Weighted Vehicle Count**:
   
   Weighted Count = Light Vehicle Count + 2 x Heavy Vehicle Count
   

2. **Signal Duration Formula**:
   Signal Duration = (Weighted Count x 0.7) + \epsilon

3. **Model Training**:
   - The SVR model minimizes an epsilon-insensitive loss function to balance error tolerance and generalization.
   - Hyperparameters C, gamma, epsilon were tuned for optimal performance.

---

## 🔧 Hyperparameter Tuning

Hyperparameters were optimized using **GridSearchCV** with cross-validation. Key parameters:
- C (Regularization): Balances error tolerance and margin size.
- gamma (Kernel Coefficient): Controls the influence of individual data points.
- epsilon (Insensitivity Margin): Defines the error margin within which predictions are not penalized.

Best hyperparameters:
- \(C = 150\)
- \(\gamma = 0.0005\)
- \(\epsilon = 0.02\)

---

## 📈 Results

Performance metrics for the optimized model:
- **Mean Squared Error (MSE)**: 24.63
- **R-squared (\(R^2\))**: 0.97
- **Mean Absolute Error (MAE)**: 3.95

The model demonstrated high accuracy and robustness, making it suitable for real-world deployment in traffic management systems.

---

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AnshSrivastava24/Support-Vector-Regression-Based-Traffic-Prediction.git
   cd Support-Vector-Regression-Based-Traffic-Prediction
