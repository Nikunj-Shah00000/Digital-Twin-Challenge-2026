import json
from src.generate_data import generate
from src.train import train

if __name__ == "__main__":
    print("1/2 Generating synthetic EHR + wearable streams...")
    generate()
    print("2/2 Training CardioTwin model...")
    metrics = train()
    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print("\nLaunch dashboard with: streamlit run dashboard/app.py")
