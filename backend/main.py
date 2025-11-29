"""
Simple Fraud Detection API
"""

import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model on startup
model = None
scaler = None
threshold = None

@app.on_event("startup")
async def load_model():
    global model, scaler, threshold
    
    with open('../models/artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('../models/artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('../models/artifacts/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    
    print(f"✅ Model loaded! Threshold: {threshold:.4f}")


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: Transaction):
    # Convert to array
    features = [
        transaction.Time, transaction.V1, transaction.V2, transaction.V3,
        transaction.V4, transaction.V5, transaction.V6, transaction.V7,
        transaction.V8, transaction.V9, transaction.V10, transaction.V11,
        transaction.V12, transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18, transaction.V19,
        transaction.V20, transaction.V21, transaction.V22, transaction.V23,
        transaction.V24, transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Amount
    ]
    
    # Scale and predict
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    probability = model.predict_proba(features_scaled)[0, 1]
    is_fraud = probability >= threshold
    
    return {
        "is_fraud": bool(is_fraud),
        "probability": float(probability),
        "threshold": float(threshold)
    }