import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, log_loss, matthews_corrcoef, mean_absolute_error
)
import xgboost as xgb
import optuna

# Load dataset
df = pd.read_csv('combined.csv', low_memory=False)

# Convert Timestamp to total seconds
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(float, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan

df['Timestamp'] = df['Timestamp'].apply(convert_time_to_seconds)

# Drop unnecessary columns
df.drop(['Source IP', 'Destination IP', 'Reason', 'Header Info'], axis=1, inplace=True)

# Encode categorical target variable
label_encoder = LabelEncoder()
df['Classification'] = label_encoder.fit_transform(df['Classification'])

# Separate features and target
y = df['Classification']
X = df.drop('Classification', axis=1)

# Handle non-numeric values
def handle_non_numeric(X):
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                X[col] = pd.factorize(X[col])[0]
    return X

X = handle_non_numeric(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective function for hyperparameter tuning
def objective(trial):
    # Define hyperparameters
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 0.2)
    }
    
    # Initialize XGBoost model with trial parameters
    model = xgb.XGBClassifier(**param, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate performance on validation set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Create Optuna study and perform optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

# Print the best parameters found by Optuna
print(f"Best Hyperparameters: {study.best_params}")

# Train model with best parameters
best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Predict on test data
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability scores for positive class

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute other performance metrics
log_loss_value = log_loss(y_test, best_model.predict_proba(X_test))
mcc = matthews_corrcoef(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Evaluation Metric Plots
# Log Loss plot
plt.figure(figsize=(8, 5))
plt.bar(['Log Loss'], [log_loss_value], color='salmon')
plt.ylabel('Log Loss')
plt.title('Log Loss Evaluation')
plt.show()

# Mean Absolute Error plot
plt.figure(figsize=(8, 5))
plt.bar(['MAE'], [mae], color='skyblue')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error Evaluation')
plt.show()

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'Log Loss: {log_loss_value:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
print(f'AUC Score: {auc_score:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', cm)
