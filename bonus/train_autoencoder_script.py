
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.student_detector import generate_normal_data
from shared.autoencoder_detector import StudentAutoencoderDetector

rng = np.random.default_rng(42)

normal_dataset = [generate_normal_data(rng) for _ in range(2000)]

(PROJECT_ROOT / "dataset").mkdir(exist_ok=True)
(PROJECT_ROOT / "models").mkdir(exist_ok=True)

with open(PROJECT_ROOT / "dataset" / "autoencoder_training_data.json", "w", encoding="utf-8") as f:
    json.dump(normal_dataset, f, indent=2)

df = pd.DataFrame(normal_dataset)

detector = StudentAutoencoderDetector(
    hidden_layer_sizes=(8, 3, 8),
    random_state=42,
    threshold_percentile=99,
)
detector.fit(df)

joblib.dump(detector, PROJECT_ROOT / "models" / "autoencoder_model.joblib")
print("models/autoencoder_model.joblib saved.")
print(f"Reconstruction-error threshold: {detector.threshold_:.6f}")
