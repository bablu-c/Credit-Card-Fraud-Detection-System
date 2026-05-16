import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

def train_model():

    # Load Dataset
    df = pd.read_csv("data/creditcard.csv")

    print(df.head())

    # Features and Target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Handle Imbalance
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(y_resampled.value_counts())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = XGBClassifier()

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Report
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt='d')

    plt.title("Fraud Detection Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")

    plt.show()

    # Precision Recall Curve
    y_scores = model.predict_proba(X_test)[:,1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_scores
    )

    plt.figure(figsize=(6,5))

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.savefig("outputs/pr_curve.png")

    plt.show()

    # Save Model
    joblib.dump(model, "models/fraud_model.pkl")

    print("Model Saved Successfully")