# fraud_detection.py

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

print("🚀 Starting Fraud Detection Pipeline...")

# Step 1: Load dataset
try:
    df = pd.read_csv('creditcard.csv')
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ ERROR while loading CSV: {e}")
    exit()

# Step 2: Preprocess - scale 'Amount' and 'Time'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
print("🔧 Features 'Time' and 'Amount' scaled.")

# Step 3: Prepare features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("📊 Dataset split into train and test sets.")

# Step 5: Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("🔄 SMOTE applied to balance the dataset.")

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
print("✅ Model training completed.")

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

print("\n📌 Evaluation Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Step 8: Save the model
joblib.dump(model, 'fraud_detector_model.pkl')
print("\n💾 Model saved as 'fraud_detector_model.pkl'")
print("✅ Fraud Detection Pipeline Completed.")


