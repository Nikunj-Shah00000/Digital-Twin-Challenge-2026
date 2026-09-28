import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from .config import DATA_DIR, MODEL_DIR, FEATURES, RANDOM_STATE
from .features import build_features, feature_matrix

def train():
    ehr = pd.read_csv(DATA_DIR/"synthetic_ehr.csv")
    wearable = pd.read_csv(DATA_DIR/"synthetic_wearable.csv")
    df = build_features(ehr, wearable)

    X = feature_matrix(df)
    y = df["adverse_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:,1]
    pred = (prob >= .5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean())
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR/"cardiotwin_model.joblib")
    (MODEL_DIR/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
