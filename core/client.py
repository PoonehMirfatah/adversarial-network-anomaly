import json
import os
import socket
import time

import joblib
import requests

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.student_detector import describe_anomaly

HOST = "localhost"
PORT = 9999
MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.joblib"


def load_detector():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: models/anomaly_model.joblib not found. Run core/train_model_script.py first.")
        raise SystemExit(1)


def connect_with_retry():
    """Allows running the client before running server.py, as requested in the PDF."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((HOST, PORT))
            return s
        except ConnectionRefusedError:
            print("Waiting for server to start...")
            time.sleep(1)


def ask_llm_for_alert(data):
    """Optional OpenRouter call. If no key exists, a local explanation is used."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        label, reason = describe_anomaly(data)
        return label, reason

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You label sensor/network anomalies. Return a short label and reason.",
                    },
                    {
                        "role": "user",
                        "content": f"Sensor reading: {data}. Describe the anomaly briefly.",
                    },
                ],
            },
            timeout=15,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"]
        return "LLM Anomaly Label", message
    except Exception as exc:
        label, reason = describe_anomaly(data)
        return label, reason + f" (LLM fallback used: {exc})"


def main():
    detector = load_detector()

    with connect_with_retry() as s:
        buffer = ""
        print("Core IsolationForest client connected to server.\n")

        while True:
            chunk = s.recv(1024).decode()
            if not chunk:
                print("Connection closed by server.")
                break
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print("Error decoding JSON.")
                    continue

                print(f"Data Received: {data}")
                prediction = detector.predict(data)[0]

                if prediction == -1:
                    label, reason = ask_llm_for_alert(data)
                    print("\nAnomaly Detected!")
                    print(f"Label: {label}")
                    print(f"Reason: {reason}\n")
                else:
                    print("Normal traffic.\n")


if __name__ == "__main__":
    main()
