# House Price Prediction (XGBoost)

This project builds an end-to-end **house price prediction system** using structured real estate data, covering EDA, feature engineering, model training, and deployment via a Flask API.

### TEST THE API AT https://verdant-khapse-2373d6.netlify.app/
## Files Overview

- **eda.py**  
  Performs exploratory data analysis, KNN-based missing value imputation, date-based feature engineering, and generates insights and visualizations.

- **training_prediction_model.py**  
  Trains a baseline Linear Regression model and an advanced XGBoost model with hyperparameter tuning.  
  Final model achieves strong performance (R² ≈ 0.91 on test data).

- **app.py**  
  Flask API for real-time price prediction with inference-safe feature engineering and consistent categorical encoding.

- **price_model.pkl**  
  Trained XGBoost regression model.

- **price_model.pkl**  
  Fitted categorical encoder used during training and inference.

- **model_features.pkl**  
  Ordered list of features used by the model.

## Key Techniques Used
- KNN Imputation
- Date feature engineering
- Baseline vs advanced model comparison
- Hyperparameter tuning
- Production-ready inference pipeline
