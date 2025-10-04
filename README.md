# Regression on California Housing Dataset

## Objective
This project implements and compares different regression algorithms on the **California Housing dataset** using `scikit-learn`.  
The aim is to evaluate performance across models and identify the most suitable regressor for predicting median house values.


## Dataset
- **Source**: `fetch_california_housing` from `sklearn.datasets`  
- **Target Variable**: `MedHouseVal` (Median house value)  
- **Features**: 8 numerical attributes, such as:
  - `MedInc`: Median income
  - `HouseAge`: Median house age
  - `AveRooms`: Average number of rooms
  - `Population`, `AveOccup`, etc.


## Workflow

### 1. Data Loading & Preprocessing
- Dataset loaded into a Pandas DataFrame.
- Split into **independent features (X)** and **dependent variable (y)**.
- Standardized features using `StandardScaler` (important for Linear Regression and SVR).
- Train-test split (67/33).

### 2. Models Implemented
- **Linear Regression** → baseline model.  
- **Decision Tree Regressor** → handles non-linear splits.  
- **Random Forest Regressor** → ensemble of decision trees.  
- **Gradient Boosting Regressor** → sequential boosting, high accuracy.  
- **Support Vector Regressor (SVR)** → kernel-based, sensitive to scaling.  

### 3. Model Evaluation
Each model evaluated on:
- **Mean Squared Error (MSE)**  
- **Mean Absolute Error (MAE)**  
- **R-squared Score (R²)**  


## Results Summary
- **Best Model**: Gradient Boosting Regressor (highest R², lowest error).  
- **Worst Model**: SVR (did not perform well due to dataset size and scale).  

