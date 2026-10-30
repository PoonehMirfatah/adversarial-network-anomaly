import json
import math
import socket
import time
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HOST = "localhost"
PORT = 9999

TOTAL_PAYLOAD_SIZE = 10000
TOTAL_DURATION_MS = 4000


def load_surrogate_model():
    model_path = PROJECT_ROOT / "models" / "surrogate_model.joblib"
    try:
        print("Successfully loaded the surrogate model.")
        return joblib.load(model_path)
    except FileNotFoundError:
        print("Error: models/surrogate_model.joblib not found. Run core/train_adversary_script.py first.")
        raise SystemExit(1)


surrogate_model = load_surrogate_model()


def surrogate_predict(packet_size, duration_ms):
    x = pd.DataFrame([[packet_size, duration_ms]], columns=["packet_size", "duration_ms"])
    return int(surrogate_model.predict(x)[0])


def find_optimal_packet_parameters():
    """Search offline for the largest packet parameters predicted as normal."""
    best_size = 100
    best_duration = 50
    best_score = -1

    for packet_size in range(100, 2201, 25):
        for duration_ms in range(50, 1001, 25):
            if surrogate_predict(packet_size, duration_ms) == 1:
                score = packet_size * 2 + duration_ms
                if score > best_score:
                    best_score = score
                    best_size = packet_size
                    best_duration = duration_ms

    best_size = max(100, int(best_size * 0.95))
    best_duration = max(50, int(best_duration * 0.95))
    return best_size, best_duration


def split_total(total, number_of_packets):
    base = total // number_of_packets
    remainder = total % number_of_packets
    return [base + 1 if i < remainder else base for i in range(number_of_packets)]


def evade_and_transmit(conn, max_safe_size, max_safe_duration):
    print(f"Optimal parameters from surrogate -> Size: {max_safe_size}, Duration: {max_safe_duration}")
    print("Beginning optimized evasion attack transmission...")

    number_of_packets = max(
        math.ceil(TOTAL_PAYLOAD_SIZE / max_safe_size),
        math.ceil(TOTAL_DURATION_MS / max_safe_duration),
    )

    payload_parts = split_total(TOTAL_PAYLOAD_SIZE, number_of_packets)
    duration_parts = split_total(TOTAL_DURATION_MS, number_of_packets)

    sent_payload = 0
    sent_duration = 0

    for packet_size, duration_ms in zip(payload_parts, duration_parts):
        packet = {
            "src_port": 443,
            "dst_port": 12345,
            "packet_size": packet_size,
            "duration_ms": duration_ms,
            "protocol": "TCP",
        }
        conn.sendall((json.dumps(packet) + "\n").encode())
        sent_payload += packet_size
        sent_duration += duration_ms
        print(f"Sent safe packet: {packet}")
        time.sleep(0.4)

    print(f"Total payload sent: {sent_payload}/{TOTAL_PAYLOAD_SIZE}")
    print(f"Total duration sent: {sent_duration}/{TOTAL_DURATION_MS}")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Core adversary server running and waiting for client to connect...")
    conn, addr = server.accept()
    print(f"Connected by defender client at {addr}")

    try:
        max_size, max_duration = find_optimal_packet_parameters()
        evade_and_transmit(conn, max_size, max_duration)
        print("Attack transmission completed successfully!")
    except KeyboardInterrupt:
        print("Adversary server stopped.")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
