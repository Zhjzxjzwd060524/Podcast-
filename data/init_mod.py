import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the processed dataset
df = pd.read_csv("processed_features.csv")

# Define features and target
# Drop any non-numerical columns (e.g., Episode_Sentiment) before splitting
non_numerical_cols = df.select_dtypes(include=['object']).columns
if len(non_numerical_cols) > 0:
    print(f"Dropping non-numerical columns: {list(non_numerical_cols)}")
    df = df.drop(columns=non_numerical_cols)

X = df.drop(columns=['Listening_Time_minutes'])
y = df['Listening_Time_minutes']

# Split into training and validation sets (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression with Lasso
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_val)
rmse_lasso = np.sqrt(mean_squared_error(y_val, y_pred_lasso))
print(f"Lasso RMSE: {rmse_lasso}")

# Print selected features (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0].tolist()
print(f"Selected features by Lasso: {selected_features}")

# 2. XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# Save the models (optional, for later use on test set)
import joblib
joblib.dump(lasso, 'lasso_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
print("Models saved as 'lasso_model.pkl' and 'xgb_model.pkl'.")