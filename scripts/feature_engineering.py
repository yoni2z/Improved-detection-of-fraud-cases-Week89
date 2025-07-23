import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# Define paths
DATA_PATH = 'data/processed/'
OUTPUT_PATH = 'data/processed/'

# Load datasets
try:
    df_fraud = pd.read_csv(os.path.join(DATA_PATH, 'Fraud_Data_with_country.csv'))
    df_credit = pd.read_csv(os.path.join(DATA_PATH, 'creditcard_cleaned.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure 'Fraud_Data_with_country.csv' and 'creditcard_cleaned.csv' are in 'data/processed/'")
    raise

# Ensure datetime columns for Fraud_Data.csv
df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'], errors='coerce')
df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'], errors='coerce')

# Check for invalid datetime values
if df_fraud['signup_time'].isna().any() or df_fraud['purchase_time'].isna().any():
    print("Warning: Invalid datetime values found. Dropping rows with NaT.")
    df_fraud = df_fraud.dropna(subset=['signup_time', 'purchase_time'])

# Fraud_Data.csv: Transaction Frequency
df_fraud['user_transaction_count'] = df_fraud.groupby('user_id')['user_id'].transform('count')
df_fraud['device_transaction_count'] = df_fraud.groupby('device_id')['device_id'].transform('count')

# Fraud_Data.csv: Transaction Velocity
df_fraud = df_fraud.sort_values(['user_id', 'purchase_time'])
df_fraud['time_diff'] = df_fraud.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0)

# Fraud_Data.csv: Time-Based Features
df_fraud['hour_of_day'] = df_fraud['purchase_time'].dt.hour
df_fraud['day_of_week'] = df_fraud['purchase_time'].dt.dayofweek
df_fraud['time_since_signup'] = (df_fraud['purchase_time'] - df_fraud['signup_time']).dt.total_seconds() / 3600

# creditcard.csv: Time-Based Features
df_credit['hour_of_day'] = (df_credit['Time'] // 3600) % 24
df_credit['log_amount'] = np.log1p(df_credit['Amount'])

# Normalization and Scaling: Fraud_Data.csv
scaler = StandardScaler()
numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'user_transaction_count', 'device_transaction_count', 'time_diff']
df_fraud[numerical_cols] = scaler.fit_transform(df_fraud[numerical_cols])

# Normalization and Scaling: creditcard.csv
df_credit[['Amount', 'log_amount']] = scaler.fit_transform(df_credit[['Amount', 'log_amount']])

# Encode Categorical Features: Fraud_Data.csv
categorical_cols = ['source', 'browser', 'sex', 'country']
df_fraud = pd.get_dummies(df_fraud, columns=categorical_cols, drop_first=True)

# Train-Test Split
X_fraud = df_fraud.drop(['class', 'user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address'], axis=1)
y_fraud = df_fraud['class']
X_credit = df_credit.drop(['Class'], axis=1)
y_credit = df_credit['Class']

X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42)
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, stratify=y_credit, random_state=42)

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_fraud_train_resampled, y_fraud_train_resampled = smote.fit_resample(X_fraud_train, y_fraud_train)
X_credit_train_resampled, y_credit_train_resampled = smote.fit_resample(X_credit_train, y_credit_train)

# Save train-test splits
os.makedirs(OUTPUT_PATH, exist_ok=True)
pd.DataFrame(X_fraud_train_resampled).to_csv(os.path.join(OUTPUT_PATH, 'X_fraud_train_resampled.csv'), index=False)
pd.DataFrame(y_fraud_train_resampled).to_csv(os.path.join(OUTPUT_PATH, 'y_fraud_train_resampled.csv'), index=False)
pd.DataFrame(X_fraud_test).to_csv(os.path.join(OUTPUT_PATH, 'X_fraud_test.csv'), index=False)
pd.DataFrame(y_fraud_test).to_csv(os.path.join(OUTPUT_PATH, 'y_fraud_test.csv'), index=False)
pd.DataFrame(X_credit_train_resampled).to_csv(os.path.join(OUTPUT_PATH, 'X_credit_train_resampled.csv'), index=False)
pd.DataFrame(y_credit_train_resampled).to_csv(os.path.join(OUTPUT_PATH, 'y_credit_train_resampled.csv'), index=False)
pd.DataFrame(X_credit_test).to_csv(os.path.join(OUTPUT_PATH, 'X_credit_test.csv'), index=False)
pd.DataFrame(y_credit_test).to_csv(os.path.join(OUTPUT_PATH, 'y_credit_test.csv'), index=False)

# Save engineered datasets
df_fraud.to_csv(os.path.join(OUTPUT_PATH, 'Fraud_Data_engineered.csv'), index=False)
df_credit.to_csv(os.path.join(OUTPUT_PATH, 'creditcard_engineered.csv'), index=False)

# Verify new features
print("Fraud_Data new features:\n", df_fraud[['user_transaction_count', 'device_transaction_count', 'time_diff', 'hour_of_day', 'day_of_week', 'time_since_signup']].head())
print("creditcard new features:\n", df_credit[['hour_of_day', 'log_amount']].head())
print("Fraud_Data class distribution after SMOTE:\n", y_fraud_train_resampled.value_counts(normalize=True))
print("creditcard class distribution after SMOTE:\n", y_credit_train_resampled.value_counts(normalize=True))