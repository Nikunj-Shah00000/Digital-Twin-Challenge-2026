import numpy as np
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, RANDOM_STATE

def generate(n_patients=800, days=14, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    patients = []
    wearable = []

    for i in range(n_patients):
        pid = f"P{i+1:04d}"
        age = int(rng.integers(30, 81))
        sex = rng.choice(["F", "M"])
        bmi = float(np.clip(rng.normal(25.5, 4.2), 17, 42))
        hypertension = int(rng.random() < (0.12 + max(age-40,0)*0.012))
        diabetes = int(rng.random() < (0.07 + max(age-45,0)*0.008))
        smoker = int(rng.random() < 0.18)
        sbp = float(np.clip(rng.normal(124 + 14*hypertension + 5*smoker, 13), 90, 200))
        ldl = float(np.clip(rng.normal(118 + 18*diabetes, 30), 50, 240))
        prior_cvd = int(rng.random() < (0.015 + max(age-55,0)*0.004))

        # latent patient-specific physiology
        base_hr = rng.normal(72, 5) + 2*smoker + 3*hypertension
        base_hrv = max(18, rng.normal(55, 10) - 7*smoker - 6*hypertension)
        base_sleep = np.clip(rng.normal(7.2, 0.7), 5, 9)
        base_steps = np.clip(rng.normal(7200, 1800), 1800, 12000)

        stress = 0.0
        for d in range(days):
            day = d + 1
            # controlled physiological deviation in final days for a subset
            event_pressure = 0
            if d >= days - 3 and rng.random() < 0.18:
                event_pressure = 1
            hr = base_hr + rng.normal(0, 4) + event_pressure*rng.normal(8, 2)
            hrv = max(10, base_hrv + rng.normal(0, 7) - event_pressure*rng.normal(12, 3))
            sleep = np.clip(base_sleep + rng.normal(0, .55) - event_pressure*rng.uniform(0.8, 1.8), 3.5, 9.5)
            steps = np.clip(base_steps + rng.normal(0, 1200) - event_pressure*rng.uniform(500, 1800), 500, 14000)
            activity = np.clip(steps / 85 + rng.normal(0, 8), 5, 180)
            wearable.append([pid, day, hr, hrv, sleep, steps, activity])

        # event label is generated from combined chronic + dynamic risk in final 3 days
        final = pd.DataFrame([x[2:] for x in wearable if x[0] == pid and x[1] > days-4],
                             columns=["hr","hrv","sleep","steps","activity"])
        hr_mean = final.hr.mean()
        hrv_mean = final.hrv.mean()
        sleep_mean = final.sleep.mean()
        steps_mean = final.steps.mean()
        hrv_slope = np.polyfit(np.arange(len(final)), final.hrv.values, 1)[0]
        hr_change = final.hr.iloc[-1] - final.hr.iloc[0]

        z = (
            -5.0
            + 0.035*age + 0.55*hypertension + 0.45*diabetes + 0.40*smoker
            + 0.018*(sbp-120) + 0.007*(ldl-110) + 0.65*prior_cvd
            + 0.05*(hr_mean-72) - 0.025*(hrv_mean-50)
            - 0.32*(sleep_mean-7) - 0.00008*(steps_mean-7000)
            - 0.12*hrv_slope + 0.08*hr_change
        )
        p = 1/(1+np.exp(-z))
        label = int(rng.random() < p)
        patients.append([pid, age, sex, bmi, hypertension, diabetes, smoker, sbp, ldl, prior_cvd, label])

    ehr = pd.DataFrame(patients, columns=[
        "patient_id","age","sex","bmi","hypertension","diabetes","smoker",
        "sbp","ldl","prior_cvd","adverse_event"
    ])
    wear = pd.DataFrame(wearable, columns=[
        "patient_id","day","heart_rate","hrv","sleep_hours","steps","activity_min"
    ])

    DATA_DIR.mkdir(exist_ok=True)
    ehr.to_csv(DATA_DIR/"synthetic_ehr.csv", index=False)
    wear.to_csv(DATA_DIR/"synthetic_wearable.csv", index=False)
    return ehr, wear

if __name__ == "__main__":
    e,w = generate()
    print(f"Generated {len(e)} synthetic patients and {len(w)} wearable observations.")
