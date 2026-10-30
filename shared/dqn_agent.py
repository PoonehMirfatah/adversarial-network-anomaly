"""A small DQN-style attacker for the bonus part of the project.

For simplicity and easy installation, the Q-network is implemented with
sklearn's MLPRegressor instead of PyTorch/TensorFlow.  It is still a neural
network that approximates Q(s, a), which is the key idea of DQN.
"""

import random
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class DQNConfig:
    total_payload_size: int = 10000
    total_duration_ms: int = 4000
    gamma: float = 0.90
    epsilon_start: float = 0.90
    epsilon_end: float = 0.05
    episodes: int = 120
    max_steps_per_episode: int = 35
    random_state: int = 42


class StudentDQNAgent:
    """DQN-like agent that selects packet_size and duration actions."""

    def __init__(self, actions, config=None):
        self.actions = list(actions)
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        random.seed(self.config.random_state)

        # Q-network: state -> Q value for each action.
        self.q_network = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.003,
            max_iter=1,
            warm_start=True,
            random_state=self.config.random_state,
        )

        # Small initial training so predict() works before real training.
        dummy_states = np.array([
            [1.0, 1.0],
            [0.75, 0.75],
            [0.50, 0.50],
            [0.25, 0.25],
            [0.0, 0.0],
        ])
        dummy_targets = np.zeros((len(dummy_states), len(self.actions)))
        self.q_network.fit(dummy_states, dummy_targets)

    def state_from_remaining(self, remaining_payload, remaining_duration):
        return np.array([
            max(0.0, remaining_payload / self.config.total_payload_size),
            max(0.0, remaining_duration / self.config.total_duration_ms),
        ], dtype=float)

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(len(self.actions))
        q_values = self.q_network.predict([state])[0]
        return int(np.argmax(q_values))

    def train(self, defender_predict_func):
        """Train against a black-box defender predictor.

        defender_predict_func(packet_size, duration_ms) must return 1 for normal
        and -1 for anomaly.
        """
        memory = []
        batch_size = 32

        for episode in range(self.config.episodes):
            epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start - episode / self.config.episodes,
            )
            remaining_payload = self.config.total_payload_size
            remaining_duration = self.config.total_duration_ms

            for _ in range(self.config.max_steps_per_episode):
                state = self.state_from_remaining(remaining_payload, remaining_duration)
                action_index = self.select_action(state, epsilon=epsilon)
                packet_size, duration_ms = self.actions[action_index]

                label = defender_predict_func(packet_size, duration_ms)
                done = False

                if label == 1:
                    # The defender accepted the packet, so the attacker gets reward.
                    remaining_payload = max(0, remaining_payload - packet_size)
                    remaining_duration = max(0, remaining_duration - duration_ms)
                    reward = packet_size / 200.0 + duration_ms / 500.0
                    if remaining_payload == 0 and remaining_duration == 0:
                        reward += 30.0
                        done = True
                else:
                    # The defender detected the packet.
                    reward = -15.0

                next_state = self.state_from_remaining(remaining_payload, remaining_duration)
                memory.append((state, action_index, reward, next_state, done))

                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    states = np.array([item[0] for item in batch])
                    targets = self.q_network.predict(states)

                    for i, (s, a, r, ns, d) in enumerate(batch):
                        if d:
                            targets[i, a] = r
                        else:
                            next_q = self.q_network.predict([ns])[0]
                            targets[i, a] = r + self.config.gamma * float(np.max(next_q))

                    self.q_network.fit(states, targets)

                if done:
                    break

        return self

    def best_action(self, remaining_payload, remaining_duration):
        state = self.state_from_remaining(remaining_payload, remaining_duration)
        action_index = self.select_action(state, epsilon=0.0)
        return self.actions[action_index]
