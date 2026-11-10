"""
personalized_anomaly_model.py — Optional machine-learning anomaly detection using Isolation Forests.

Simple explanation
------------------
While robust statistical Z-scores (in anomaly_detector.py) evaluate individual
features independently, an Isolation Forest can identify complex multivariate outliers
(for example: when pressure variability and stance duration are subtly abnormal in combination,
even if each value is marginally within standard deviation limits).

MEDICAL GUARDRAIL:
This model identifies statistical multidimensional outliers in sensor data.
It does NOT predict osteoarthritis progression or diagnose disease.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config import config
from ml.model_utils import MODEL_FEATURE_NAMES, extract_feature_vector

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES_ISOLATION_FOREST = 10


class PersonalizedIsolationForest:
    """
    Fits a patient-specific Isolation Forest on historical baseline gait sessions.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.fitted_patient_id: Optional[str] = None
        self.n_samples_trained: int = 0

    def fit(self, gait_sessions: List[Dict[str, Any]], patient_id: str) -> bool:
        """
        Trains the isolation forest on a patient's historical gait sessions.
        Requires at least 10 baseline trials.
        """
        if len(gait_sessions) < MIN_TRAINING_SAMPLES_ISOLATION_FOREST:
            logger.info(
                "Insufficient observations (%d < %d) to train personalized Isolation Forest.",
                len(gait_sessions), MIN_TRAINING_SAMPLES_ISOLATION_FOREST
            )
            return False

        # Build feature matrix
        matrix = []
        for s in gait_sessions:
            vec = extract_feature_vector(s, feature_names=MODEL_FEATURE_NAMES)
            matrix.append(vec.flatten())

        X = np.array(matrix)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.fitted_patient_id = patient_id
        self.n_samples_trained = len(X)
        logger.info("Fitted personalized Isolation Forest for %s (%d trials).", patient_id, len(X))
        return True

    def score_session(self, session_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a new gait trial.
        Returns outlier status (-1 for outlier, 1 for normal) and raw anomaly score.
        """
        if self.model is None:
            return {
                "status": "not_trained",
                "is_outlier": False,
                "anomaly_score": 0.0,
                "message": "Isolation Forest not trained (insufficient baseline trials).",
            }

        X = extract_feature_vector(session_features, feature_names=MODEL_FEATURE_NAMES)
        pred = int(self.model.predict(X)[0])  # 1 = inlier, -1 = outlier
        score = float(self.model.decision_function(X)[0])  # lower score = more anomalous

        is_outlier = (pred == -1)
        return {
            "status": "success",
            "is_outlier": is_outlier,
            "anomaly_score": round(score, 4),
            "interpretation": "Multivariate outlier detected" if is_outlier else "Consistent with multivariate baseline",
        }

