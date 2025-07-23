import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os

# Set Matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Load test data and models
X_fraud_test = pd.read_csv('data/processed/X_fraud_test.csv')
y_fraud_test = pd.read_csv('data/processed/y_fraud_test.csv')
X_credit_test = pd.read_csv('data/processed/X_credit_test.csv')
xgb_fraud = joblib.load('models/fraud_xgb_model.pkl')
xgb_credit = joblib.load('models/credit_xgb_model.pkl')

# Initialize SHAP explainer
explainer_fraud = shap.TreeExplainer(xgb_fraud)
explainer_credit = shap.TreeExplainer(xgb_credit)

# Compute SHAP values
shap_values_fraud = explainer_fraud.shap_values(X_fraud_test)
shap_values_credit = explainer_credit.shap_values(X_credit_test)

# Create directory for plots
os.makedirs('reports/shap_plots', exist_ok=True)

# Summary Plots
shap.summary_plot(shap_values_fraud, X_fraud_test, show=False)
plt.savefig('reports/shap_plots/fraud_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values_credit, X_credit_test, show=False)
plt.savefig('reports/shap_plots/credit_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Force Plots with Matplotlib mode and error handling
try:
    shap.force_plot(explainer_fraud.expected_value, shap_values_fraud[0], X_fraud_test.iloc[0], matplotlib=True, show=False)
    plt.savefig('reports/shap_plots/fraud_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error generating fraud force plot: {e}")

try:
    shap.force_plot(explainer_credit.expected_value, shap_values_credit[0], X_credit_test.iloc[0], matplotlib=True, show=False)
    plt.savefig('reports/shap_plots/credit_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error generating credit force plot: {e}")