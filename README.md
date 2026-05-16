# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using XGBoost, SMOTE, FastAPI, and real-time fraud prediction APIs.

---

# 📌 Project Overview

Credit card fraud is one of the major challenges in digital banking and online payments. This project uses Machine Learning techniques to identify fraudulent transactions from highly imbalanced transaction data.

The system:
- trains an ML model on transaction data
- handles class imbalance using SMOTE
- predicts fraud probability
- exposes a FastAPI prediction API
- supports real-time fraud scoring

---

# 🚀 Features

✅ Fraud Detection using Machine Learning  
✅ XGBoost Classifier  
✅ SMOTE for Imbalanced Dataset Handling  
✅ FastAPI Backend  
✅ Real-Time Fraud Prediction API  
✅ Confusion Matrix Visualization  
✅ Precision-Recall Curve  
✅ Swagger UI API Testing  
✅ Model Saving using Joblib  

---

# 🛠 Tech Stack

## Languages & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- Joblib

---

# 📂 Folder Structure

```bash
Credit-Card-Fraud-Detection/
│
├── data/
├── notebooks/
├── src/
│   ├── train.py
│   ├── predict.py
│
├── serving/
│   └── app.py
│
├── models/
├── outputs/
├── images/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

📊 Dataset

Dataset used:

Kaggle Credit Card Fraud Dataset

Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚠️ Dataset is not uploaded to this repository because of GitHub file size limits.

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/bablu-c/Credit-Card-Fraud-Detection-System.git

cd Credit-Card-Fraud-Detection-System
2️⃣ Create Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
Mac/Linux
python3 -m venv venv

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run Project
Train Model
python main.py
Run Prediction
python src/predict.py
Run FastAPI Server
uvicorn serving.app:app --reload
🌐 API Documentation

Open in browser:

http://127.0.0.1:8000/docs

Swagger UI will open for API testing.

📈 Model Workflow
Transaction Data
       ↓
Data Cleaning
       ↓
SMOTE Balancing
       ↓
Feature Processing
       ↓
XGBoost Model
       ↓
Fraud Prediction
       ↓
FastAPI Response
📊 Evaluation Metrics

The project evaluates:

Precision
Recall
F1 Score
Confusion Matrix
Precision-Recall Curve
📸 Screenshots
Recommended Screenshots
Dataset Preview
Model Training Output
Confusion Matrix
PR Curve
Swagger API UI
Prediction Output
GitHub Repository
🧠 Sample API Response
{
  "prediction": "Normal",
  "fraud_probability": 0.01
}
💡 Future Improvements
Streamlit Dashboard
Next.js Frontend
SHAP Explainability
Docker Deployment
Kafka Streaming
AWS/GCP Deployment
Real-Time Fraud Alerts
🎯 Learning Outcomes

This project helped in understanding:

Fraud Analytics
Imbalanced Classification
XGBoost
API Development
FastAPI
Model Deployment
Real-Time Prediction Systems
📌 Resume Project Description

Built an end-to-end Credit Card Fraud Detection System using XGBoost, SMOTE, and FastAPI with real-time fraud prediction and API deployment.

👨‍💻 Author

Bablu kumar

⭐ If You Like This Project

Give this repository a star ⭐ on GitHub.