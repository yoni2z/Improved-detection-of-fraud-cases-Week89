import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import numpy as np

# Define paths
DATA_PATH = 'data/processed/'
MODEL_PATH = 'models/'

# Load train-test splits
try:
    X_fraud_train = pd.read_csv(os.path.join(DATA_PATH, 'X_fraud_train_resampled.csv'))
    y_fraud_train = pd.read_csv(os.path.join(DATA_PATH, 'y_fraud_train_resampled.csv'))
    X_fraud_test = pd.read_csv(os.path.join(DATA_PATH, 'X_fraud_test.csv'))
    y_fraud_test = pd.read_csv(os.path.join(DATA_PATH, 'y_fraud_test.csv'))
    X_credit_train = pd.read_csv(os.path.join(DATA_PATH, 'X_credit_train_resampled.csv'))
    y_credit_train = pd.read_csv(os.path.join(DATA_PATH, 'y_credit_train_resampled.csv'))
    X_credit_test = pd.read_csv(os.path.join(DATA_PATH, 'X_credit_test.csv'))
    y_credit_test = pd.read_csv(os.path.join(DATA_PATH, 'y_credit_test.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure train-test split CSVs are in 'data/processed/'")
    raise

# Verify scaling
def check_scaling(X, dataset_name):
    means = X.mean()
    stds = X.std()
    print(f"{dataset_name} scaling check - Mean (should be ~0):\n{means}\nStd (should be ~1):\n{stds}")
    if (means.abs() > 0.1).any() or (stds < 0.5).any() or (stds > 1.5).any():
        print(f"Warning: {dataset_name} may not be properly scaled.")

check_scaling(X_fraud_train, "Fraud_Data train")
check_scaling(X_credit_train, "creditcard train")

# Train Logistic Regression with increased max_iter and saga solver
lr_fraud = LogisticRegression(random_state=42, max_iter=5000, solver='saga')
lr_fraud.fit(X_fraud_train, y_fraud_train.values.ravel())
lr_credit = LogisticRegression(random_state=42, max_iter=5000, solver='saga')
lr_credit.fit(X_credit_train, y_credit_train.values.ravel())

# Train XGBoost
xgb_fraud = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_fraud.fit(X_fraud_train, y_fraud_train.values.ravel())
xgb_credit = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_credit.fit(X_credit_train, y_credit_train.values.ravel())

# Save models
os.makedirs(MODEL_PATH, exist_ok=True)
joblib.dump(lr_fraud, os.path.join(MODEL_PATH, 'fraud_lr_model.pkl'))
joblib.dump(xgb_fraud, os.path.join(MODEL_PATH, 'fraud_xgb_model.pkl'))
joblib.dump(lr_credit, os.path.join(MODEL_PATH, 'credit_lr_model.pkl'))
joblib.dump(xgb_credit, os.path.join(MODEL_PATH, 'credit_xgb_model.pkl'))

print("Models trained and saved successfully.")