import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor

# LOAD DATA
df = pd.read_csv("Imputed_Data_Case_Study_1.csv")

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# DROP IDENTIFIER
if "Property ID" in df.columns:
    df = df.drop(columns=["Property ID"])

# DATE FEATURE ENGINEERING
df["Date Sold"] = pd.to_datetime(df["Date Sold"])

df["Sold_Year"] = df["Date Sold"].dt.year
df["Sold_Month"] = df["Date Sold"].dt.month
df["Sold_Quarter"] = df["Date Sold"].dt.quarter
df["Property_Age_At_Sale"] = df["Sold_Year"] - df["Year Built"]

df = df.drop(columns=["Date Sold"])

# ENCODE CATEGORICAL FEATURES
cat_cols = df.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df[cat_cols] = encoder.fit_transform(df[cat_cols])

# FEATURE / TARGET SPLIT
X = df.drop(columns=["Price"])
y = df["Price"]

print("Final features:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BASELINE MODEL - LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nBASELINE MODEL (LINEAR REGRESSION)")
print(f"RMSE : {rmse_lr:.2f}")
print(f"R2   : {r2_lr:.4f}")

# XGBOOST MODEL WITH HYPERPARAMETER TUNING
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [4, 6, 8],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="r2",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

print("\nBEST XGBOOST CONFIG")
print(grid.best_params_)

y_pred_xgb = best_xgb.predict(X_test)

rmse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBOOST MODEL")
print(f"RMSE : {rmse_xgb:.2f}")
print(f"R2   : {r2_xgb:.4f}")

# SAVE ARTIFACTS
joblib.dump(best_xgb, "price_model.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("\nModel, encoder, and feature list saved successfully.")
