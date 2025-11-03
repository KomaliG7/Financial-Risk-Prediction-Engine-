# src/api.py
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI()
model = joblib.load('models/xgb_baseline.pkl')
preproc_obj = joblib.load('models/preprocessor.joblib')['preprocessor']

@app.post("/predict")
def predict(payload: dict):
    # payload: { "features": { col1: val1, ... } }
    features = payload.get('features', {})
    X = pd.DataFrame([features])
    X_trans = preproc_obj.transform(X)
    proba = model.predict_proba(X_trans)[:,1][0]
    pred = int(proba >= 0.5)
    return {"probability": float(proba), "prediction": pred}
