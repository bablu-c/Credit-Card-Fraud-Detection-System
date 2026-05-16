import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

sample_data = pd.DataFrame([{
    "Time": 10000,
    "V1": -1.2,
    "V2": 2.3,
    "V3": -0.5,
    "V4": 1.2,
    "V5": 0.1,
    "V6": -0.3,
    "V7": 1.5,
    "V8": 0.2,
    "V9": -1.1,
    "V10": 0.7,
    "V11": -0.4,
    "V12": 0.6,
    "V13": -0.2,
    "V14": 1.3,
    "V15": -0.7,
    "V16": 0.8,
    "V17": -0.1,
    "V18": 0.2,
    "V19": 0.4,
    "V20": -0.5,
    "V21": 0.1,
    "V22": -0.2,
    "V23": 0.3,
    "V24": -0.1,
    "V25": 0.5,
    "V26": -0.3,
    "V27": 0.2,
    "V28": -0.1,
    "Amount": 5000
}])

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("🚨 Fraud Transaction Detected")
else:
    print("✅ Normal Transaction")