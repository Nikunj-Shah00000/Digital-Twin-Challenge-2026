from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
STATIC_FEATURES = [
    "age", "bmi", "hypertension", "diabetes", "smoker",
    "sbp", "ldl", "prior_cvd"
]
DYNAMIC_FEATURES = [
    "hr_mean", "hr_resting", "hrv_mean", "hrv_slope",
    "sleep_hours", "steps", "activity_min", "hr_change"
]
FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES
