import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import joblib
import os

# Load test data and models
X_fraud_test = pd.read_csv('data/processed/X_fraud_test.csv')
y_fraud_test = pd.read_csv('data/processed/y_fraud_test.csv')
X_credit_test = pd.read_csv('data/processed/X_credit_test.csv')
y_credit_test = pd.read_csv('data/processed/y_credit_test.csv')
lr_fraud = joblib.load('models/fraud_lr_model.pkl')
xgb_fraud = joblib.load('models/fraud_xgb_model.pkl')
lr_credit = joblib.load('models/credit_lr_model.pkl')
xgb_credit = joblib.load('models/credit_xgb_model.pkl')

# Evaluation function
def evaluate_model(model, X_test, y_test, dataset_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{dataset_name} - AUC-PR: {auc_pr:.4f}, F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {dataset_name}')
    plt.savefig(f'reports/eda_plots/{dataset_name}_cm.png')
    plt.close()

# Evaluate models
evaluate_model(lr_fraud, X_fraud_test, y_fraud_test, 'Fraud_Data_Logistic_Regression')
evaluate_model(xgb_fraud, X_fraud_test, y_fraud_test, 'Fraud_Data_XGBoost')
evaluate_model(lr_credit, X_credit_test, y_credit_test, 'creditcard_Logistic_Regression')
evaluate_model(xgb_credit, X_credit_test, y_credit_test, 'creditcard_XGBoost')