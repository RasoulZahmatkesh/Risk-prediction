import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
data = pd.read_csv('credit_risk_data.csv')

# Preprocess data (e.g., drop missing values)
data = data.dropna()

# Split data into features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build initial model (Logistic Regression) and evaluate
model = LogisticRegression()

# Train the Logistic Regression model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Logistic Regression model
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Logistic Regression Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Logistic Regression Classification Report:\n{classification_report(y_test, y_pred)}')

# Step 3: Add other models (Random Forest, XGBoost)
# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_y_pred)}')
print(f'Random Forest Confusion Matrix:\n{confusion_matrix(y_test, rf_y_pred)}')
print(f'Random Forest Classification Report:\n{classification_report(y_test, rf_y_pred)}')

# XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

# Evaluate XGBoost model
print(f'XGBoost Accuracy: {accuracy_score(y_test, xgb_y_pred)}')
print(f'XGBoost Confusion Matrix:\n{confusion_matrix(y_test, xgb_y_pred)}')
print(f'XGBoost Classification Report:\n{classification_report(y_test, xgb_y_pred)}')

# Step 4: Optimize models using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10], 
    'solver': ['liblinear', 'saga']
}

# Create GridSearchCV model for Logistic Regression
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=1)

# Train the model using Grid Search
grid_search.fit(X_train, y_train)

# Best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# Evaluate optimized model
grid_y_pred = grid_search.best_estimator_.predict(X_test)
print(f'Optimized Logistic Regression Accuracy: {accuracy_score(y_test, grid_y_pred)}')
print(f'Optimized Logistic Regression Confusion Matrix:\n{confusion_matrix(y_test, grid_y_pred)}')
print(f'Optimized Logistic Regression Classification Report:\n{classification_report(y_test, grid_y_pred)}')

# Step 5: Evaluate models with Cross-Validation
cv_scores = cross_val_score(LogisticRegression(), X, y, cv=10)

# Display results
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean()}')

# Step 6: Plot model evaluation metrics (Confusion Matrix, ROC Curve)
# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Step 7: Save predictions to a CSV file
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('predictions.csv', index=False)
