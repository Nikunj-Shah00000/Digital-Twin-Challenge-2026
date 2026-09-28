import numpy as np
import pandas as pd
from .config import FEATURES

def contributions(model, row):
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    x = row[FEATURES].astype(float).to_numpy().reshape(1,-1)
    z = scaler.transform(x)[0] * clf.coef_[0]
    result = pd.DataFrame({"feature": FEATURES, "contribution": z})
    result["direction"] = np.where(result.contribution >= 0, "increases risk", "reduces risk")
    return result.sort_values("contribution", ascending=False)
