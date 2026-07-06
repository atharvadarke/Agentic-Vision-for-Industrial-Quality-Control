"""
AnomalyDetector — CPU-Bound Unsupervised Anomaly Detection via Isolation Forest.

This module wraps Scikit-Learn's IsolationForest to classify 512-dimensional
feature vectors as "Normal" (inlier) or "Anomaly" (outlier). The model is
trained exclusively on normal/good samples and flags deviations at inference.

Design Rationale:
    - IsolationForest has virtually zero RAM overhead compared to deep models.
    - Contamination=0.05 assumes ~5% of training data may contain subtle defects.
    - Serialization via joblib enables fast model loading without re-training.
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Lightweight anomaly detector for manufacturing quality control.
    Operates on 512-D feature embeddings from the VisionEncoder.
    """

    def __init__(self):
        """
        Initialize the IsolationForest with conservative contamination.

        contamination=0.05: Expects at most 5% of the training data to be
        anomalous. This is a standard heuristic for industrial QC where
        the vast majority of parts are conforming.

        random_state=42: Ensures reproducible anomaly boundaries across
        training runs for audit compliance.
        """
        self.model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100,
            n_jobs=1  # Single-threaded to minimize RAM on edge device
        )
        self._is_trained = False

    def train(self, embeddings: np.ndarray) -> None:
        """
        Fit the IsolationForest on a matrix of "normal" embeddings.

        Args:
            embeddings: np.ndarray of shape (n_samples, 512) containing
                        feature vectors extracted from known-good parts.

        Post-condition:
            self._is_trained is set to True, enabling predict() calls.
        """
        self.model.fit(embeddings)
        self._is_trained = True

    def save_model(self, model_path: str) -> None:
        """
        Serialize the trained model to disk via joblib.

        Args:
            model_path: File path for the serialized model (e.g., 'models/anomaly_detector.joblib').
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model. Call train() first.")
        joblib.dump(self.model, model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a previously trained model from disk.

        Args:
            model_path: File path to the serialized joblib model.
        """
        self.model = joblib.load(model_path)
        self._is_trained = True

    def predict(self, embedding: np.ndarray) -> dict:
        """
        Classify a single 512-D embedding as Normal or Anomaly.

        Args:
            embedding: np.ndarray of shape (512,) — a single feature vector.

        Returns:
            dict: {"status": "Normal", "anomaly_score": float} or
                  {"status": "Anomaly", "anomaly_score": float}

        IsolationForest scoring:
            - predict() returns  1 for inliers (Normal)
            - predict() returns -1 for outliers (Anomaly)
            - decision_function() returns the raw anomaly score
              (more negative = more anomalous)
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        # Reshape to (1, 512) for single-sample prediction
        sample = embedding.reshape(1, -1)

        # Get binary prediction and continuous anomaly score
        prediction = self.model.predict(sample)[0]
        anomaly_score = self.model.decision_function(sample)[0]

        status = "Normal" if prediction == 1 else "Anomaly"

        return {
            "status": status,
            "anomaly_score": float(anomaly_score)
        }
