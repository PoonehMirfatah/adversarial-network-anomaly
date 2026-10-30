
from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.dqn_agent import StudentDQNAgent, DQNConfig

try:
    defender_model = joblib.load(PROJECT_ROOT / "models" / "autoencoder_model.joblib")
except FileNotFoundError:
    print("Error: models/autoencoder_model.joblib not found. Run bonus/train_autoencoder_script.py first.")
    raise SystemExit(1)


def query_autoencoder_defender(packet_size, duration_ms, src_port=443, dst_port=12345, protocol="TCP"):
    packet = {
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_size": int(packet_size),
        "duration_ms": int(duration_ms),
        "protocol": protocol,
    }
    return int(defender_model.predict(packet)[0])


actions = [
    (250, 100),
    (500, 150),
    (750, 200),
    (1000, 250),
    (1250, 300),
    (1450, 450),
]

config = DQNConfig(
    total_payload_size=10000,
    total_duration_ms=4000,
    episodes=120,
    max_steps_per_episode=35,
    random_state=42,
)

agent = StudentDQNAgent(actions=actions, config=config)
agent.train(query_autoencoder_defender)

(PROJECT_ROOT / "models").mkdir(exist_ok=True)
joblib.dump(agent, PROJECT_ROOT / "models" / "dqn_agent.joblib")
print("models/dqn_agent.joblib saved.")
print("Learned action list:")
for i, action in enumerate(actions):
    print(f"  action {i}: packet_size={action[0]}, duration_ms={action[1]}")
