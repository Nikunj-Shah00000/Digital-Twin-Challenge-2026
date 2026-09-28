import numpy as np
import pandas as pd
from .config import STATIC_FEATURES, DYNAMIC_FEATURES

def build_features(ehr, wearable, window=7):
    recent = wearable.sort_values(["patient_id","day"]).groupby("patient_id").tail(window).copy()

    def slope(s):
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s))
        return float(np.polyfit(x, s.to_numpy(), 1)[0])

    dyn = recent.groupby("patient_id").agg(
        hr_mean=("heart_rate","mean"),
        hr_resting=("heart_rate","min"),
        hrv_mean=("hrv","mean"),
        sleep_hours=("sleep_hours","mean"),
        steps=("steps","mean"),
        activity_min=("activity_min","mean"),
    ).reset_index()

    slopes = recent.groupby("patient_id")["hrv"].apply(slope).reset_index(name="hrv_slope")
    changes = recent.groupby("patient_id")["heart_rate"].apply(
        lambda s: float(s.iloc[-1]-s.iloc[0]) if len(s)>1 else 0.0
    ).reset_index(name="hr_change")

    dyn = dyn.merge(slopes, on="patient_id").merge(changes, on="patient_id")
    out = ehr.merge(dyn, on="patient_id")
    return out

def feature_matrix(df):
    return df[STATIC_FEATURES + DYNAMIC_FEATURES].astype(float)
