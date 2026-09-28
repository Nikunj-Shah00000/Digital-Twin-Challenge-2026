# CardioTwin India — Architecture Diagram

## Data flow

1. Synthetic EHR generator creates patient demographics, diagnoses, labs and history.
2. Wearable generator creates time-series heart rate, HRV, sleep, steps and activity.
3. Feature engineering computes a rolling seven-day physiological state.
4. The Digital Twin state combines historical EHR + current dynamic features.
5. Logistic Regression estimates short-horizon adverse cardiovascular-risk probability.
6. Explainability uses model coefficients and standardized feature values.
7. Streamlit presents a clinician-oriented conceptual dashboard.

## Design principle

The PoC deliberately separates:
- **patient history** from
- **dynamic physiological state** from
- **model inference** from
- **clinical visualization**.

This makes the architecture replaceable: a future version can swap synthetic streams for consented/open datasets, and the baseline model for a temporal model, without redesigning the whole application.
