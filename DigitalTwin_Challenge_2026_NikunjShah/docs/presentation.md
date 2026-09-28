# CardioTwin India — Presentation

## Slide 1 — Title
**CardioTwin India**
Personalized Cardiovascular Risk Digital Twin  
Digital Twin Challenge 2026  
Amity University Noida

## Slide 2 — The problem
Healthcare often observes a patient through periodic snapshots. Physiological state changes continuously.

**Challenge:** How can we combine historical clinical context with dynamic wearable signals to identify an emerging risk state?

## Slide 3 — Our use case
**Localized outcome:** elevated cardiovascular-risk state over the next monitoring window.

Inputs:
- EHR-like history
- wearable physiology
- personal baseline deviations

Output:
- risk probability
- risk band
- explainable contributing factors

## Slide 4 — What is the Digital Twin?
A continuously refreshed computational representation of an individual.

Our PoC state:

**Twin(t) = Historical EHR + Dynamic physiological window(t)**

The twin changes as new wearable observations arrive.

## Slide 5 — Two-stream data fusion
**Static stream**
- age
- BMI
- hypertension
- diabetes
- smoking
- SBP
- LDL
- prior CVD

**Dynamic stream**
- heart rate
- HRV
- sleep
- steps
- activity
- recent change from baseline

## Slide 6 — Architecture
EHR → feature layer  
Wearables → time-series feature layer  
Both → Digital Twin State  
Twin → risk model → explainability → dashboard

## Slide 7 — ML model
Logistic Regression baseline.

\[
P(Y=1|X)=rac{1}{1+e^{-(eta_0+eta^TX)}}
\]

Why:
- transparent;
- fast;
- reproducible;
- appropriate as a PoC baseline.

## Slide 8 — Synthetic-data sandbox
No real patient-identifiable data.

The pipeline generates:
- synthetic EHR;
- synthetic wearable observations;
- controlled physiological deviations;
- synthetic event labels.

## Slide 9 — Dashboard
Show:
- risk probability;
- risk band;
- clinical context;
- HR trend;
- HRV trend;
- model contributions.

## Slide 10 — Explainability
The dashboard surfaces which features move the model output.

Examples:
- BP;
- HRV;
- sleep;
- heart-rate change;
- chronic risk factors.

## Slide 11 — Evaluation
Report:
- ROC-AUC
- accuracy
- precision
- recall
- F1
- confusion matrix

**Important:** Synthetic performance is not evidence of clinical efficacy.

## Slide 12 — Clinical workflow
Wearable stream → Twin update → risk alert → clinician review → decision

The system does not autonomously diagnose or prescribe.

## Slide 13 — Privacy and safety
- synthetic data in PoC;
- no patient identifiers;
- least-privilege architecture for future deployment;
- auditability;
- human-in-the-loop;
- clinical validation required.

## Slide 14 — Future roadmap
1. Temporal models
2. Personal baseline learning
3. Uncertainty estimation
4. External validation
5. Federated/privacy-preserving learning
6. EHR interoperability
7. prospective clinical study

## Slide 15 — Conclusion
CardioTwin India demonstrates the core Digital Twin loop:

**Observe → Update → Predict → Explain → Review**

The PoC establishes a foundation for future clinically validated digital-twin research.
