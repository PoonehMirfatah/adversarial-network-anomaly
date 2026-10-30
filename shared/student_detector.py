"""A small student-style anomaly detector for the course project.

The model part is IsolationForest. A few simple sanity checks are also used so
obvious impossible packets, such as UNKNOWN protocol or very large packets, are
reported clearly to the client and to the simulated black-box defender API.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

COMMON_PORTS = [80, 443, 22, 8080]
NORMAL_PROTOCOLS = ["TCP", "UDP"]
FEATURE_COLUMNS = [
    "src_port",
    "dst_port",
    "packet_size",
    "duration_ms",
    "protocol_TCP",
    "protocol_UDP",
]


def generate_normal_data(rng):
    """Generate one normal traffic record similar to server.py."""
    return {
        "src_port": int(rng.choice(COMMON_PORTS)),
        "dst_port": int(rng.integers(1024, 65536)),
        "packet_size": int(rng.integers(100, 1501)),
        "duration_ms": int(rng.integers(50, 501)),
        "protocol": str(rng.choice(NORMAL_PROTOCOLS)),
    }


def to_dataframe(data):
    """Accept dict/list/DataFrame and return a clean DataFrame."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)

    for col in ["src_port", "dst_port", "packet_size", "duration_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["protocol"] = df["protocol"].fillna("UNKNOWN").astype(str)
    return df


def preprocess_data(data):
    """Convert traffic records to the numerical format used by IsolationForest."""
    df = to_dataframe(data)
    df = pd.get_dummies(df, columns=["protocol"])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLUMNS].astype(float)


def rule_based_anomaly(row):
    """Simple readable checks for anomalies that are obvious in this dataset."""
    if row["protocol"] not in NORMAL_PROTOCOLS:
        return True
    if int(row["src_port"]) not in COMMON_PORTS:
        return True
    if int(row["packet_size"]) < 50 or int(row["packet_size"]) > 1500:
        return True
    if int(row["duration_ms"]) < 10 or int(row["duration_ms"]) > 500:
        return True
    return False


class StudentAnomalyDetector:
    """IsolationForest plus small project-specific checks."""

    def __init__(self, contamination=0.03, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, data):
        x_train = preprocess_data(data)
        self.model.fit(x_train)
        return self

    def predict(self, data):
        df = to_dataframe(data)
        x = preprocess_data(df)
        model_pred = self.model.predict(x)

        # Final decision of the defender: either the model or the simple checks can alert.
        rule_pred = np.array([-1 if rule_based_anomaly(row) else 1 for _, row in df.iterrows()])
        final_pred = np.where((model_pred == -1) | (rule_pred == -1), -1, 1)
        return final_pred

    def decision_function(self, data):
        x = preprocess_data(data)
        return self.model.decision_function(x)


def describe_anomaly(data):
    """Create a local readable alert if no LLM API key is available."""
    protocol = str(data.get("protocol", "UNKNOWN"))
    src_port = int(data.get("src_port", 0))
    packet_size = int(data.get("packet_size", 0))
    duration_ms = int(data.get("duration_ms", 0))

    if protocol not in NORMAL_PROTOCOLS:
        return "Unknown Protocol", f"Protocol '{protocol}' is not normal. Expected TCP or UDP."
    if src_port not in COMMON_PORTS:
        return "Suspicious Source Port", f"Source port {src_port} is outside the normal service ports."
    if packet_size > 1500:
        return "Large Packet", f"Packet size {packet_size} is bigger than the normal maximum of 1500 bytes."
    if packet_size < 50:
        return "Very Small Packet", f"Packet size {packet_size} is unusually small."
    if duration_ms > 500:
        return "Long Duration", f"Duration {duration_ms} ms is bigger than the normal maximum of 500 ms."
    if duration_ms < 10:
        return "Very Short Duration", f"Duration {duration_ms} ms is unusually short."
    return "Statistical Anomaly", "IsolationForest classified this traffic as unusual compared with normal training traffic."
