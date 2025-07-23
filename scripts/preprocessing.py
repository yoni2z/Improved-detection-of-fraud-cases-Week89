import pandas as pd
import os

# Define paths
DATA_PATH = 'data/raw/'
PROCESSED_PATH = 'data/processed/'

# Load datasets
try:
    df_fraud = pd.read_csv(os.path.join(DATA_PATH, 'Fraud_Data.csv'))
    df_credit = pd.read_csv(os.path.join(DATA_PATH, 'creditcard.csv'))
    df_ip = pd.read_csv(os.path.join(DATA_PATH, 'IpAddress_to_Country.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure datasets are in 'data/raw/'")
    raise

# Handle missing values for Fraud_Data.csv
print("Missing values in Fraud_Data:\n", df_fraud.isnull().sum())
df_fraud['purchase_value'].fillna(df_fraud['purchase_value'].median(), inplace=True)
df_fraud['source'].fillna(df_fraud['source'].mode()[0], inplace=True)
df_fraud['browser'].fillna(df_fraud['browser'].mode()[0], inplace=True)
df_fraud['sex'].fillna(df_fraud['sex'].mode()[0], inplace=True)
df_fraud['age'].fillna(df_fraud['age'].median(), inplace=True)

# Handle missing values for creditcard.csv
print("Missing values in creditcard:\n", df_credit.isnull().sum())
df_credit['Amount'].fillna(df_credit['Amount'].median(), inplace=True)
df_credit.dropna(subset=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class'], inplace=True)

# Remove duplicates
df_fraud.drop_duplicates(inplace=True)
df_credit.drop_duplicates(inplace=True)

# Correct data types for Fraud_Data.csv
df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'], errors='coerce')
df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'], errors='coerce')
df_fraud['user_id'] = df_fraud['user_id'].astype(str)
df_fraud['device_id'] = df_fraud['device_id'].astype(str)
df_fraud['class'] = df_fraud['class'].astype(int)

# Correct data types for creditcard.csv
df_credit['Time'] = df_credit['Time'].astype(float)
df_credit['Amount'] = df_credit['Amount'].astype(float)
df_credit['Class'] = df_credit['Class'].astype(int)

# Validate ranges
df_fraud = df_fraud[(df_fraud['purchase_value'] >= 0) & (df_fraud['age'] >= 0) & (df_fraud['age'] <= 100)]
df_credit = df_credit[df_credit['Amount'] >= 0]

# Convert IP addresses to integer
df_fraud['ip_address'] = df_fraud['ip_address'].astype(int)

# Function to map IP to country
def map_ip_to_country(ip, ip_df):
    match = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
    return match['country'].iloc[0] if not match.empty else 'Unknown'

# Apply mapping
df_fraud['country'] = df_fraud['ip_address'].apply(lambda x: map_ip_to_country(x, df_ip))

# Save datasets
os.makedirs(PROCESSED_PATH, exist_ok=True)
df_fraud.to_csv(os.path.join(PROCESSED_PATH, 'Fraud_Data_cleaned.csv'), index=False)
df_fraud.to_csv(os.path.join(PROCESSED_PATH, 'Fraud_Data_with_country.csv'), index=False)
df_credit.to_csv(os.path.join(PROCESSED_PATH, 'creditcard_cleaned.csv'), index=False)

# Verify dtypes
print("Fraud_Data dtypes:\n", df_fraud.dtypes)
print("creditcard dtypes:\n", df_credit.dtypes)