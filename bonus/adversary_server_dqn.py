
import json
import socket
import time
from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HOST = "localhost"
PORT = 9999
TOTAL_PAYLOAD_SIZE = 10000
TOTAL_DURATION_MS = 4000


def load_agent():
    try:
        return joblib.load(PROJECT_ROOT / "models" / "dqn_agent.joblib")
    except FileNotFoundError:
        print("Error: models/dqn_agent.joblib not found. Run bonus/train_dqn_script.py first.")
        raise SystemExit(1)


def load_defender_for_safety_check():
    try:
        return joblib.load(PROJECT_ROOT / "models" / "autoencoder_model.joblib")
    except FileNotFoundError:
        print("Warning: models/autoencoder_model.joblib not found. Running without local safety check.")
        return None


def defender_predict(defender_model, packet_size, duration_ms):
    if defender_model is None:
        return 1
    packet = {
        "src_port": 443,
        "dst_port": 12345,
        "packet_size": int(packet_size),
        "duration_ms": int(duration_ms),
        "protocol": "TCP",
    }
    return int(defender_model.predict(packet)[0])


def choose_safe_action(agent, defender_model, remaining_payload, remaining_duration):
    packet_size, duration_ms = agent.best_action(remaining_payload, remaining_duration)

    packet_size = min(packet_size, remaining_payload)
    duration_ms = min(duration_ms, remaining_duration)
    if packet_size <= 0:
        packet_size = 1
    if duration_ms <= 0:
        duration_ms = 1

    if defender_predict(defender_model, packet_size, duration_ms) == 1:
        return packet_size, duration_ms

    for safe_size, safe_duration in sorted(agent.actions, key=lambda x: x[0]):
        safe_size = min(safe_size, remaining_payload)
        safe_duration = min(safe_duration, remaining_duration)
        if safe_size <= 0:
            safe_size = 1
        if safe_duration <= 0:
            safe_duration = 1
        if defender_predict(defender_model, safe_size, safe_duration) == 1:
            return safe_size, safe_duration

    return min(100, remaining_payload), min(50, remaining_duration)


def dqn_evade_and_transmit(conn, agent, defender_model):
    remaining_payload = TOTAL_PAYLOAD_SIZE
    remaining_duration = TOTAL_DURATION_MS
    packet_counter = 0

    print("Beginning DQN-based evasion attack transmission...")

    while remaining_payload > 0:
        packet_size, duration_ms = choose_safe_action(agent, defender_model, remaining_payload, remaining_duration)

        packet = {
            "src_port": 443,
            "dst_port": 12345,
            "packet_size": int(packet_size),
            "duration_ms": int(duration_ms),
            "protocol": "TCP",
        }

        conn.sendall((json.dumps(packet) + "\n").encode())
        packet_counter += 1
        remaining_payload = max(0, remaining_payload - packet_size)
        remaining_duration = max(0, remaining_duration - duration_ms)

        print(f"Sent DQN packet #{packet_counter}: {packet}")
        time.sleep(0.4)

    print("DQN attack transmission completed successfully!")
    print(f"Total payload sent: {TOTAL_PAYLOAD_SIZE - remaining_payload}/{TOTAL_PAYLOAD_SIZE}")
    print(f"Total packets sent: {packet_counter}")


def main():
    agent = load_agent()
    defender_model = load_defender_for_safety_check()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("DQN adversary server running and waiting for client to connect...")
    conn, addr = server.accept()
    print(f"Connected by defender client at {addr}")

    try:
        dqn_evade_and_transmit(conn, agent, defender_model)
    except KeyboardInterrupt:
        print("DQN adversary server stopped.")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
