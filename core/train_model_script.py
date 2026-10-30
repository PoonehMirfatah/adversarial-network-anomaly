import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.student_detector import StudentAnomalyDetector, generate_normal_data

rng = np.random.default_rng(42)
dataset = [generate_normal_data(rng) for _ in range(1000)]

(PROJECT_ROOT / "dataset").mkdir(exist_ok=True)
(PROJECT_ROOT / "models").mkdir(exist_ok=True)

with open(PROJECT_ROOT / "dataset" / "training_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

df = pd.DataFrame(dataset)
detector = StudentAnomalyDetector(contamination=0.03, random_state=42)
detector.fit(df)

joblib.dump(detector, PROJECT_ROOT / "models" / "anomaly_model.joblib")
print("models/anomaly_model.joblib saved.")
