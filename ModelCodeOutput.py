import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =================================================================
# 1. Data Loading (Constraint: Must load the actual creditcard.csv)
# =================================================================
try:
    df = pd.read_csv('creditcard.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# =================================================================
# 2. Preprocessing: Scaling 'Time' and 'Amount'
# =================================================================
scaler = StandardScaler()
X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
print("Time and Amount features standardized.")

# =================================================================
# 3. Stratified Data Splitting
# =================================================================
# Stratification ensures fair distribution of rare fraud cases
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

N_positive_train = y_train.sum()
N_negative_train = len(y_train) - N_positive_train
N_positive_test = y_test.sum()

print(f"\nTrain Fraud Cases: {N_positive_train} | Test Fraud Cases: {N_positive_test}")

# =================================================================
# 4. XGBoost Configuration and Training
# =================================================================

# Calculate the optimal scale_pos_weight
# Weight = Negative Cases / Positive Cases
scale_pos_weight_value = N_negative_train / N_positive_train
SCALE_POS_WEIGHT = round(scale_pos_weight_value)

print(f"Calculated scale_pos_weight for training: {SCALE_POS_WEIGHT} (1:{SCALE_POS_WEIGHT})")

# Define XGBoost Parameters for High-Imbalance
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',         # Focus on Precision-Recall AUC
    'eta': 0.05,                    # Learning Rate
    'max_depth': 4,                 # Shallow depth to generalize
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'scale_pos_weight': SCALE_POS_WEIGHT, # CRITICAL: Adjusts the loss function
    'early_stopping_rounds':15
}

model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False)

print("\nStarting XGBoost training with weighted optimization...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("XGBoost training complete.")

# =================================================================
# 5. Prediction and Evaluation
# =================================================================

# Predict probabilities on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- 5a. PR-AUC Score ---
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"\n--- XGBoost Model Performance Summary ---")
print(f"1. Area Under Precision-Recall Curve (PR-AUC): {pr_auc:.4f}")

# --- 5b. Optimal Threshold Calibration (Maximizing F1 Score) ---
# We use the PR curve to find the best threshold to balance Precision and Recall
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find the threshold that maximizes the F1 Score
fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10) # Add epsilon to avoid division by zero
idx = np.nanargmax(fscores)
optimal_threshold = thresholds[idx]

# Apply the optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate final metrics
final_recall = recall_score(y_test, y_pred_optimal)
final_precision = precision_score(y_test, y_pred_optimal)
final_f1 = f1_score(y_test, y_pred_optimal)
cm = confusion_matrix(y_test, y_pred_optimal)

print(f"\n2. Results using Optimal Threshold (Max F1 = {final_f1:.4f}):")
print(f"   Optimal Decision Threshold: {optimal_threshold:.4f}")
print(f"   Recall (Fraud Catch Rate): {final_recall:.4f}")
print(f"   Precision (False Alarm Rate): {final_precision:.4f}")

print("\n   Confusion Matrix:")
print("       Predicted Legit | Predicted Fraud")
print(f"Actual Legit | {cm[0, 0]:>13} | {cm[0, 1]:>13}")
print(f"Actual Fraud | {cm[1, 0]:>13} | {cm[1, 1]:>13}")