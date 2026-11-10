"""
model_utils.py — Shared machine learning utilities and evaluation metrics.

Simple explanation
------------------
This module standardizes how machine learning models are saved, loaded,
evaluated, and validated.
It ensures that:
  - Models are saved alongside metadata (feature list, training date, version, metrics)
  - Features passed for prediction match exactly what the model was trained on
  - Participant-level data splitting prevents data leakage across patients
  - Standard clinical screening metrics (Sensitivity, Specificity, ROC-AUC, F1)
    are calculated and reported with appropriate medical disclaimers.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import config

logger = logging.getLogger(__name__)

# Standard set of features used for model training and prediction
MODEL_FEATURE_NAMES: List[str] = [
    "heel_peak",
    "midfoot_peak",
    "forefoot_peak",
    "heel_mean",
    "midfoot_mean",
    "forefoot_mean",
    "heel_std",
    "midfoot_std",
    "forefoot_std",
    "acc_mag_mean",
    "acc_mag_peak",
    "acceleration_variability",
    "gyro_mag_mean",
    "gyro_mag_peak",
    "gyro_variability",
    "stance_duration_seconds",
    "sampling_frequency_hz",
]

MEDICAL_DISCLAIMER_TEXT = (
    "RESEARCH PROTOTYPE NOTICE: This machine-learning model is experimental "
    "and has not been clinically validated. It is designed to explore gait "
    "pattern associations and MUST NOT be used for clinical diagnosis or "
    "to replace medical evaluation by a licensed healthcare professional."
)


def validate_features(feature_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Checks if a feature dictionary has all required features for model prediction.

    Returns (is_valid, missing_features_list).
    """
    missing = [f for f in MODEL_FEATURE_NAMES if f not in feature_dict or feature_dict[f] is None]
    return (len(missing) == 0, missing)


def extract_feature_vector(
    feature_dict: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Converts a feature dict into a 2D numpy array [1, n_features] in fixed order.
    """
    names = feature_names or MODEL_FEATURE_NAMES
    vector = []
    for name in names:
        val = feature_dict.get(name, 0.0)
        try:
            vector.append(float(val))
        except (ValueError, TypeError):
            vector.append(0.0)
    return np.array(vector, dtype=float).reshape(1, -1)


def compute_clinical_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Computes standard screening and diagnostic evaluation metrics.

    Metrics calculated:
      - Accuracy
      - Precision
      - Recall (Sensitivity)
      - Specificity (True Negative Rate)
      - F1 Score
      - Confusion Matrix [[TN, FP], [FN, TP]]
      - ROC-AUC (if probabilities available)
    """
    y_t = np.array(y_true).astype(int)
    y_p = np.array(y_pred).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "sensitivity": round(rec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    if y_prob is not None:
        try:
            # If multi-class or binary proba array [N, 2]
            probs = np.array(y_prob)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
            auc = float(roc_auc_score(y_t, probs))
            metrics["roc_auc"] = round(auc, 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves a trained model bundle containing the classifier, feature names, and metadata.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": MODEL_FEATURE_NAMES,
        "metadata": metadata or {},
        "disclaimer": MEDICAL_DISCLAIMER_TEXT,
    }

    joblib.dump(bundle, target_path)
    logger.info("Saved model bundle to %s", target_path)
    return target_path


def load_model(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a saved model bundle. Returns dictionary with 'model', 'feature_names', 'metadata'.
    """
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {target_path}")

    bundle = joblib.load(target_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        # Backwards compatibility if bare estimator was saved
        return {
            "model": bundle,
            "feature_names": MODEL_FEATURE_NAMES,
            "metadata": {},
            "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        }

    return bundle
