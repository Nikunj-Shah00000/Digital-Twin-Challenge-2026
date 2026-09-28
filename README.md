# 🫀 CardioTwin India — Digital Twin Challenge 2026

> **A proof-of-concept Digital Twin for proactive cardiovascular-risk monitoring using synthetic EHR + wearable time-series data.**

CardioTwin India demonstrates how a patient-specific digital twin can combine **static/historical clinical context** with **dynamic wearable signals** to estimate short-horizon cardiovascular risk and explain the factors driving the alert.

## 1. Team Details

| Field | Details |
|---|---|
| Team Name | **CardioTwin India** |
| Team Leader | **Nikunj Shah** |
| College / Incubator | **Amity University Noida** |
| Team Members | 0 |
| Challenge | Happiest Health Digital Twin Challenge 2026 |
| Domain | AI + Digital Health + Data Science |

**Video Link** - https://www.youtube.com/watch?v=SSKee%5Crwhzu

## 2. Project Title

### CardioTwin India — Personalized Cardiovascular Risk Digital Twin

## 3. Problem Statement

Cardiovascular risk is influenced by a combination of relatively stable patient characteristics and rapidly changing physiological/behavioral signals. A conventional snapshot of a patient's medical record can miss short-term changes in risk.

This PoC models a **localized cardiovascular-risk digital twin** that fuses:

1. **Static EHR-like information:** age, sex, BMI, hypertension/diabetes history, smoking status, cholesterol and prior cardiovascular history.
2. **Dynamic wearable information:** heart rate, HRV, sleep duration, steps and activity intensity.

The system produces a continuously refreshed **risk score**, risk band and contributing factors. It is designed as a prototype decision-support layer—not a clinical diagnostic device.

## 4. Healthcare Use Case

**Target outcome:** elevated cardiovascular-risk state over the next monitoring window.

A doctor can open a patient's virtual twin and inspect:

- current risk score and risk band;
- recent heart-rate / HRV / sleep trends;
- historical clinical context;
- top factors contributing to the alert;
- recent change in risk;
- recommended next step for **clinical review**, not autonomous treatment.

### Why this is a Digital Twin PoC

The patient representation is not only a static profile. It is refreshed by a stream of simulated wearable observations and combines those observations with historical EHR features to update the model's risk estimate.

## 5. Architecture

```text
                 ┌──────────────────────────┐
                 │ Synthetic EHR Generator  │
                 │ demographics + labs +   │
                 │ diagnoses + history      │
                 └────────────┬─────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Patient Profile  │
                    │ Static Features  │
                    └────────┬─────────┘
                             │
                             │
                             ▼
┌──────────────────┐  ┌──────────────────────┐
│ Wearable / IoT   │─▶│ Time-Series Pipeline │
│ HR, HRV, sleep,  │  │ validation + rolling │
│ steps, activity  │  │ feature extraction  │
└──────────────────┘  └──────────┬───────────┘
                                 │
                                 ▼
                      ┌────────────────────┐
                      │ Digital Twin State │
                      │ EHR + latest       │
                      │ physiological      │
                      │ context            │
                      └─────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Risk Model          │
                     │ Logistic Regression │
                     │ + calibration       │
                     └─────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          Risk score      Risk band      Explainability
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌────────────────────┐
                    │ Doctor Dashboard   │
                    │ trends + alert +   │
                    │ patient context    │
                    └────────────────────┘
```

## 6. Technical Stack

- **Python 3.10+**
- Pandas / NumPy — data processing
- scikit-learn — baseline ML model
- Joblib — model persistence
- Streamlit — conceptual clinician dashboard
- Plotly — interactive trend visualization
- Matplotlib — offline architecture/figure generation
- Synthetic data generation — reproducible sandbox dataset
- GitHub — source/version control

## 7. AI/ML Model

### Baseline

A **Logistic Regression** model is used because this is a PoC intended to demonstrate a transparent and auditable risk pipeline.

The model estimates:

\[
P(Y=1|X)=\sigma(\beta_0+\beta^TX)
\]

where:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The feature vector combines static EHR context with rolling wearable features.

### Static features

- age
- BMI
- hypertension
- diabetes
- smoking
- systolic BP
- LDL cholesterol
- prior cardiovascular history

### Dynamic features

- mean heart rate
- resting heart rate
- mean HRV
- HRV trend
- sleep duration
- step count
- activity minutes
- heart-rate variability
- recent change from personal baseline

### Risk bands

For the prototype UI:

- **Low:** probability < 0.35
- **Moderate:** 0.35–0.65
- **High:** > 0.65

These thresholds are **prototype visualization thresholds**, not validated clinical cutoffs.

## 8. Data Strategy

No real patient-identifiable data is included.

The repository generates synthetic EHR and wearable data locally. This allows the complete pipeline to be demonstrated without exposing patient information.

### Synthetic EHR

The generator creates realistic-looking but artificial patient profiles with:

- demographics;
- clinical history;
- laboratory measurements;
- cardiovascular history.

### Synthetic wearable stream

The generator creates timestamped measurements representing:

- heart rate;
- HRV;
- sleep;
- steps;
- activity.

The signal contains patient-specific baselines plus controlled deviations, allowing the model to learn a relationship between risk factors and short-horizon adverse-state labels.

## 9. Project Structure

```text
CardioTwin_India/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_pipeline.py
├── data/
│   ├── README.md
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── models/
├── dashboard/
│   └── app.py
├── docs/
│   ├── architecture.md
│   ├── architecture_diagram.pdf
│   ├── presentation.md
│   ├── presentation.pdf
│   └── demo_script.md
├── notebooks/
├── tests/
│   └── test_pipeline.py
└── requirements.txt
```

## 10. Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

python run_pipeline.py

streamlit run dashboard/app.py
```

The pipeline generates synthetic data, trains the model, evaluates it, and saves model artifacts.

## 11. Example API-like Prediction

```python
from src.predict import load_artifacts, predict_patient

model, scaler, features = load_artifacts()
result = predict_patient(model, scaler, features, patient_row)

print(result)
```

## 12. Evaluation

The PoC reports:

- ROC-AUC
- accuracy
- precision
- recall
- F1
- confusion matrix

Because the dataset is synthetic, these metrics demonstrate **pipeline behavior**, not clinical validity or real-world performance.

## 13. Explainability

The dashboard displays feature contributions using the model's coefficients and the patient's standardized feature values.

A high-risk alert can therefore be accompanied by statements such as:

- elevated systolic blood pressure contributed to risk;
- reduced HRV compared with baseline contributed to risk;
- short sleep duration contributed to risk.

This is intended to make the model easier for a clinician to inspect.

## 14. Safety & Clinical Scope

This project is a **research/prototype demonstration**.

It is not:

- a medical device;
- a diagnosis engine;
- a replacement for a clinician;
- a treatment recommendation system.

A real clinical deployment would require prospective validation, clinical governance, bias analysis, cybersecurity controls, privacy controls, calibration assessment, external validation and appropriate regulatory review.

## 15. Reproducibility

Synthetic data generation uses a fixed random seed by default. The complete pipeline can therefore be regenerated locally.

## 16. 20-Minute Demo Video

**YouTube (Unlisted):** `PASTE_FINAL_UNLISTED_YOUTUBE_LINK_HERE`

Suggested demonstration flow is provided in `docs/demo_script.md`.

## 17. Presentation

- PDF: `docs/presentation.pdf`
- Editable source: `docs/presentation.md`

## 18. Architecture Diagram

- PDF: `docs/architecture_diagram.pdf`
- Source: `docs/architecture.md`

## 19. Open-Source License

MIT License — see `LICENSE`.

## 20. Submission Checklist

- [x] Team details
- [x] College/incubator information
- [x] Project title
- [x] Problem statement
- [x] Healthcare use case
- [x] Technical stack
- [x] AI/ML model/framework details
- [X] Final 15–20 minute unlisted demo video link
- [x] Open-source license
- [x] Architecture diagram PDF
- [x] Presentation PDF
- [x] Source code
- [x] Synthetic data generation
- [x] Public-repository-ready structure

### Final submission actions

1. Replace the placeholder YouTube URL.
2. Add verified team members.
3. Create the public GitHub repository.
4. Push all files.
5. Open every PDF/link in an incognito/private browser window to verify public accessibility.
6. Submit only the team leader's requested details + public repository URL on the challenge platform.
