import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.features import build_features
from src.explain import contributions
from src.predict import risk_band
from src.config import DATA_DIR, MODEL_DIR, FEATURES

st.set_page_config(page_title="CardioTwin India", page_icon="🫀", layout="wide")

st.title("🫀 CardioTwin India")
st.caption("Digital Twin Challenge 2026 • Synthetic EHR + wearable fusion PoC")

if not (DATA_DIR/"synthetic_ehr.csv").exists() or not (MODEL_DIR/"cardiotwin_model.joblib").exists():
    st.warning("Run `python run_pipeline.py` first.")
    st.stop()

ehr = pd.read_csv(DATA_DIR/"synthetic_ehr.csv")
wear = pd.read_csv(DATA_DIR/"synthetic_wearable.csv")
df = build_features(ehr, wear)
model = joblib.load(MODEL_DIR/"cardiotwin_model.joblib")

pid = st.sidebar.selectbox("Virtual patient", df.patient_id.tolist())
row = df[df.patient_id == pid].iloc[0]
prob = float(model.predict_proba(row[FEATURES].to_frame().T)[0,1])

st.sidebar.markdown("### Twin status")
st.sidebar.success("LIVE SYNTHETIC STREAM")
st.sidebar.write("Last wearable window: 7 days")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Risk probability", f"{prob*100:.1f}%")
c2.metric("Risk band", risk_band(prob))
c3.metric("Age", int(row.age))
c4.metric("BMI", f"{row.bmi:.1f}")

st.divider()
left,right = st.columns([1.3,1])

with left:
    st.subheader("Physiological trends")
    w = wear[wear.patient_id == pid].copy()
    w["day"] = w["day"].astype(int)
    fig = px.line(w, x="day", y="heart_rate", markers=True,
                  title="Heart Rate")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(w, x="day", y="hrv", markers=True, title="Heart Rate Variability")
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Clinical context")
    st.write({
        "Hypertension": "Yes" if row.hypertension else "No",
        "Diabetes": "Yes" if row.diabetes else "No",
        "Smoking": "Yes" if row.smoker else "No",
        "Systolic BP": f"{row.sbp:.0f} mmHg",
        "LDL": f"{row.ldl:.0f} mg/dL",
        "Prior CVD": "Yes" if row.prior_cvd else "No",
        "Avg sleep": f"{row.sleep_hours:.1f} h",
        "Avg steps": f"{row.steps:,.0f}",
    })

    st.subheader("Risk drivers")
    contrib = contributions(model, row).head(8)
    fig3 = px.bar(contrib.sort_values("contribution"),
                  x="contribution", y="feature", orientation="h",
                  title="Model feature contributions")
    st.plotly_chart(fig3, use_container_width=True)

st.info("Prototype decision-support view only. Risk estimates are generated from synthetic data and are not clinical advice.")
