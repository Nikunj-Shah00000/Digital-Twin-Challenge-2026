import joblib
import numpy as np
import pandas as pd
from .config import MODEL_DIR, FEATURES

def load_model():
    return joblib.load(MODEL_DIR/"cardiotwin_model.joblib")

def risk_band(p):
    if p < .35:
        return "Low"
    if p <= .65:
        return "Moderate"
    return "High"

def predict_dataframe(model, df):
    X = df[FEATURES].astype(float)
    p = model.predict_proba(X)[:,1]
    out = df[["patient_id"]].copy()
    out["risk_probability"] = p
    out["risk_percent"] = (100*p).round(1)
    out["risk_band"] = [risk_band(x) for x in p]
    return out
