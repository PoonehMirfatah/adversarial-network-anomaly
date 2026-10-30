
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.student_detector import generate_normal_data, preprocess_data

rng = np.random.default_rng(42)
random.seed(42)
SUSPICIOUS_PORTS = [1337, 9999, 6666]


def generate_anomaly_data():
    anomaly_type = random.choice(["port", "packet", "duration", "protocol"])
    if anomaly_type == "port":
        return {
            "src_port": random.choice(SUSPICIOUS_PORTS),
            "dst_port": random.randint(60000, 65535),
            "packet_size": random.randint(100, 1500),
            "duration_ms": random.randint(50, 500),
            "protocol": "TCP",
        }
    if anomaly_type == "packet":
        return {
            "src_port": 443,
            "dst_port": random.randint(1024, 65535),
            "packet_size": random.randint(2000, 10000),
            "duration_ms": random.randint(50, 500),
            "protocol": "TCP",
        }
    if anomaly_type == "duration":
        return {
            "src_port": 80,
            "dst_port": random.randint(1024, 65535),
            "packet_size": random.randint(100, 1500),
            "duration_ms": random.randint(2000, 5000),
            "protocol": "TCP",
        }
    return {
        "src_port": 443,
        "dst_port": random.randint(1024, 65535),
        "packet_size": random.randint(100, 1500),
        "duration_ms": random.randint(50, 500),
        "protocol": "UNKNOWN",
    }


normal_data = [generate_normal_data(rng) for _ in range(400)]
anomaly_data = [generate_anomaly_data() for _ in range(120)]

all_data = normal_data + anomaly_data
labels = ["Normal"] * len(normal_data) + ["Anomaly"] * len(anomaly_data)

x = preprocess_data(all_data).values
x_scaled = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, random_state=42)
x_2d = pca.fit_transform(x_scaled)

plot_df = pd.DataFrame({
    "PC1": x_2d[:, 0],
    "PC2": x_2d[:, 1],
    "Label": labels,
})

(PROJECT_ROOT / "outputs").mkdir(exist_ok=True)

plt.figure(figsize=(8, 6))
for label in ["Normal", "Anomaly"]:
    subset = plot_df[plot_df["Label"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.75)

plt.title("PCA Visualization of Normal and Anomalous Traffic")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "pca_visualization.png", dpi=150)
print("PCA visualization saved to outputs/pca_visualization.png")
print("Explained variance ratio:", pca.explained_variance_ratio_)
