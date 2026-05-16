from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    probability = model.predict_proba(df)[0][1]

    result = "Fraud" if prediction[0] == 1 else "Normal"

    return {
        "prediction": result,
        "fraud_probability": float(probability)
    }