import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load trained model
try:
    model = joblib.load("fraud_detector_model.pkl")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# Feature list in order (same as used in training)
feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

def predict():
    try:
        # Collect inputs
        values = []
        for entry in entries:
            val = float(entry.get())
            values.append(val)

        input_data = np.array([values])  # Shape (1, 30)
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            messagebox.showwarning("Prediction Result", "⚠️ Transaction is FRAUDULENT!")
        else:
            messagebox.showinfo("Prediction Result", "✅ Transaction is NOT fraud.")

    except Exception as e:
        messagebox.showerror("Error", f"Please enter valid numbers.\n{e}")

# Build GUI
root = tk.Tk()
root.title("Credit Card Fraud Detector")

tk.Label(root, text="Enter transaction features:", font=('Arial', 14)).grid(row=0, column=0, columnspan=2, pady=10)

entries = []
for i, feature in enumerate(feature_names):
    tk.Label(root, text=feature, font=('Arial', 10)).grid(row=i+1, column=0, sticky="e", padx=5, pady=2)
    entry = tk.Entry(root, width=30)
    entry.grid(row=i+1, column=1, padx=5)
    entries.append(entry)

tk.Button(root, text="Predict", command=predict, bg="green", fg="white", width=20).grid(row=len(feature_names)+1, column=0, columnspan=2, pady=15)

root.mainloop()
