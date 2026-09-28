# 20-Minute Demo Script

## 0:00–2:00 — Opening
- Introduce CardioTwin India.
- Explain why a patient snapshot is different from a continuously refreshed patient state.
- State the localized outcome: cardiovascular-risk state.

## 2:00–5:00 — Challenge mapping
Show the README and point out:
- synthetic EHR stream;
- wearable time-series stream;
- fusion layer;
- predictive model;
- doctor dashboard.

## 5:00–8:00 — Data
Run `python run_pipeline.py`.
Explain:
- no real patient data;
- reproducible synthetic generation;
- static + dynamic data.

## 8:00–12:00 — Model
Open `src/train.py`.
Explain:
- feature engineering;
- standardization;
- Logistic Regression;
- probability output;
- evaluation metrics;
- limitations of synthetic validation.

## 12:00–16:00 — Digital Twin dashboard
Run:
`streamlit run dashboard/app.py`

Select two different virtual patients.
Show:
- risk probability;
- risk band;
- heart-rate trend;
- HRV trend;
- clinical context;
- feature contributions.

## 16:00–18:00 — Architecture
Show `docs/architecture_diagram.pdf`.
Explain the flow from EHR + wearable data to twin state and model output.

## 18:00–20:00 — Impact + limitations
Explain:
- proactive monitoring concept;
- clinician-in-the-loop workflow;
- privacy-first sandbox design;
- external validation required before clinical use;
- future temporal models and real-world validation.

End with:
“CardioTwin is a proof of concept for continuously updating a patient representation—not a replacement for clinical judgment.”
