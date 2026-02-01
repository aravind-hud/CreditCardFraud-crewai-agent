Credit Card Fraud Detection with CrewAI & XGBoost
This project implements a robust fraud detection pipeline designed to identify fraudulent credit card transactions with high precision and recall.

⚙️ The Build Process
The initial codebase for this project was generated using CrewAI and the Gemini 3 Flash LLM, then manually refined to implement advanced evaluation strategies and model calibration.

📊 Dataset
The model utilizes the Kaggle Credit Card Fraud Detection Dataset.

Requirements: The local directory must contain creditcard.csv.

Nature of Data: Highly imbalanced (0.17% fraud rate), requiring specialized handling via scale_pos_weight and PR-AUC optimization.

🧪 Model Performance
The final model was evaluated using 5-Fold Stratified Cross-Validation to ensure stability across the entire dataset.

Key Metrics:

PR-AUC: 0.7298

Optimal F1-Score: 0.8200 (at threshold 0.6229)

Recall (Fraud Catch Rate): 83.67% — Caught 82 out of 98 fraud cases in the test set.

Precision: 80.39% — Only 20 false alarms out of nearly 57,000 transactions.

🛠️ Key Technical Features
XGBoost Classifier: Tuned for imbalanced classes.

Automated Thresholding: Dynamically finds the "sweet spot" to balance catching criminals vs. annoying customers.

Feature Scaling: Standardized Time and Amount for model stability.

Visualization: Built-in Seaborn Heatmaps for performance analysis.