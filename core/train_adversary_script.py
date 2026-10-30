import random
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)
np.random.seed(42)

model_path = PROJECT_ROOT / "models" / "anomaly_model.joblib"
try:
    defender_model = joblib.load(model_path)
except FileNotFoundError:
    print("Error: models/anomaly_model.joblib not found. Run core/train_model_script.py first.")
    raise SystemExit(1)


def query_defender_api(packet_size, duration_ms, src_port=443, dst_port=12345, protocol="TCP"):
    packet = {
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_size": int(packet_size),
        "duration_ms": int(duration_ms),
        "protocol": protocol,
    }
    return int(defender_model.predict(packet)[0])


probe_data = []
for _ in range(600):
    probe_data.append({
        "packet_size": random.randint(50, 4000),
        "duration_ms": random.randint(10, 2000),
    })

df_probes = pd.DataFrame(probe_data)
df_probes["label"] = [
    query_defender_api(row["packet_size"], row["duration_ms"])
    for _, row in df_probes.iterrows()
]

X_surrogate = df_probes[["packet_size", "duration_ms"]]
y_surrogate = df_probes["label"]
surrogate_model = DecisionTreeClassifier(max_depth=6, random_state=42)
surrogate_model.fit(X_surrogate, y_surrogate)

accuracy = accuracy_score(y_surrogate, surrogate_model.predict(X_surrogate))
print(f"Surrogate model approximation accuracy: {accuracy * 100:.2f}%")

(PROJECT_ROOT / "models").mkdir(exist_ok=True)
joblib.dump(surrogate_model, PROJECT_ROOT / "models" / "surrogate_model.joblib")
print("models/surrogate_model.joblib saved.")
