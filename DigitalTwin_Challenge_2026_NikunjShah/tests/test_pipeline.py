from src.generate_data import generate
from src.features import build_features

def test_generation_and_features(tmp_path):
    ehr, wearable = generate(n_patients=20, days=10, seed=7)
    df = build_features(ehr, wearable)
    assert len(ehr) == 20
    assert len(df) == 20
    assert "hrv_mean" in df.columns
    assert "adverse_event" in df.columns
