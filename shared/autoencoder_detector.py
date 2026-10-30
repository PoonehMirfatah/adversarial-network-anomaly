"""Bonus defender: a simple Autoencoder based anomaly detector.

This file is intentionally written in a clear student-project style.  It keeps
exactly the same traffic features as the normal project, but instead of
IsolationForest it trains a neural network to reconstruct normal packets.
Packets with high reconstruction error are considered anomalous.
"""

import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from shared.student_detector import preprocess_data, to_dataframe, rule_based_anomaly


class StudentAutoencoderDetector:
    """Autoencoder-like detector using sklearn MLPRegressor.

    The model is trained only on normal traffic:
        input features  -> neural network -> reconstructed input features
    If reconstruction error is higher than a learned threshold, the packet is
    marked as anomaly (-1). Otherwise it is normal (1).
    """

    def __init__(self, hidden_layer_sizes=(8, 3, 8), random_state=42, threshold_percentile=99):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=700,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=25,
        )
        self.threshold_ = None

    def fit(self, data):
        x = preprocess_data(data).values
        x_scaled = self.scaler.fit_transform(x)

        # Autoencoder training: target output is the same as input.
        self.model.fit(x_scaled, x_scaled)

        train_errors = self.reconstruction_error_from_scaled(x_scaled)
        self.threshold_ = float(np.percentile(train_errors, self.threshold_percentile))
        return self

    def reconstruction_error_from_scaled(self, x_scaled):
        reconstructed = self.model.predict(x_scaled)
        errors = np.mean((x_scaled - reconstructed) ** 2, axis=1)
        return errors

    def reconstruction_error(self, data):
        x = preprocess_data(data).values
        x_scaled = self.scaler.transform(x)
        return self.reconstruction_error_from_scaled(x_scaled)

    def predict(self, data):
        if self.threshold_ is None:
            raise RuntimeError("Autoencoder detector is not trained yet.")

        df = to_dataframe(data)
        errors = self.reconstruction_error(df)
        autoencoder_pred = np.where(errors > self.threshold_, -1, 1)

        # These small checks make the alert easier to explain and catch clearly
        # impossible values such as UNKNOWN protocol.
        rule_pred = np.array([-1 if rule_based_anomaly(row) else 1 for _, row in df.iterrows()])
        final_pred = np.where((autoencoder_pred == -1) | (rule_pred == -1), -1, 1)
        return final_pred

    def decision_function(self, data):
        """Positive means more normal; negative means suspicious."""
        errors = self.reconstruction_error(data)
        return self.threshold_ - errors


def save_autoencoder(detector, path="../models/autoencoder_model.joblib"):
    joblib.dump(detector, path)


def load_autoencoder(path="../models/autoencoder_model.joblib"):
    return joblib.load(path)
