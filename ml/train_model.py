"""
train_model.py — Training pipeline for experimental Random Forest gait classifier.

Simple explanation
------------------
This script trains a Random Forest model on gait features to examine whether
foot pressure and kinematic patterns correlate with confirmed clinical osteoarthritis.

CRITICAL MEDICAL & METHODOLOGICAL RULES:
1. Participant-level validation: We use Group-aware splitting (GroupKFold / GroupShuffleSplit)
   by participant_id. Multiple walking trials from the same person NEVER appear in
   both train and test sets simultaneously (preventing data leakage).
2. Synthetic data safeguard: Synthetic data is strictly for verifying software pipeline
   execution. A synthetic model is explicitly stamped as 'synthetic_testing_only'
   and must never be used for clinical findings.
3. Full evaluation: We report Sensitivity, Specificity, F1, Accuracy, and Confusion Matrix,
   not just accuracy.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure project root is importable when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from app.config import config
from ml.model_utils import (
    MEDICAL_DISCLAIMER_TEXT,
    MODEL_FEATURE_NAMES,
    compute_clinical_metrics,
    save_model,
)

logger = logging.getLogger(__name__)


def generate_synthetic_training_dataset(
    n_participants: int = 20,
    trials_per_participant: int = 4,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic feature dataset solely for unit testing and verifying
    the training pipeline without physical hardware or real patient records.
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for p in range(1, n_participants + 1):
        pid = f"test_pt_{p:03d}"
        # Binary clinical label: 0 = Control / Asymptomatic, 1 = OA Pattern
        label = int(p % 2 == 0)

        # Baseline shift for OA vs Control in simulated gait
        heel_bias = -120 if label == 1 else 0
        fore_bias = 80 if label == 1 else 0
        var_bias = 0.05 if label == 1 else 0.0

        for trial in range(1, trials_per_participant + 1):
            sid = f"sess_{p:03d}_{trial:02d}"
            rec: Dict[str, Any] = {
                "participant_id": pid,
                "session_id": sid,
                "data_source": "synthetic_demo",
                "clinical_label": label,
                "heel_peak": float(rng.normal(820 + heel_bias, 40)),
                "midfoot_peak": float(rng.normal(650, 30)),
                "forefoot_peak": float(rng.normal(900 + fore_bias, 45)),
                "heel_mean": float(rng.normal(340 + heel_bias * 0.4, 20)),
                "midfoot_mean": float(rng.normal(280, 15)),
                "forefoot_mean": float(rng.normal(310 + fore_bias * 0.4, 20)),
                "heel_std": float(rng.normal(120, 10)),
                "midfoot_std": float(rng.normal(90, 8)),
                "forefoot_std": float(rng.normal(130, 10)),
                "acc_mag_mean": float(rng.normal(1.02, 0.03)),
                "acc_mag_peak": float(rng.normal(1.45, 0.08)),
                "acceleration_variability": float(rng.normal(0.25 + var_bias, 0.03)),
                "gyro_mag_mean": float(rng.normal(2.5, 0.4)),
                "gyro_mag_peak": float(rng.normal(5.8, 0.6)),
                "gyro_variability": float(rng.normal(1.1, 0.15)),
                "stance_duration_seconds": float(rng.normal(0.72, 0.05)),
                "sampling_frequency_hz": 50.0,
            }
            records.append(rec)

    return pd.DataFrame(records)


def train_pipeline(
    data_source: Union[str, Path, pd.DataFrame],
    model_version: str = "rf_v1.0",
    test_size: float = 0.25,
    random_state: int = 42,
    output_model_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Executes the training and participant-level evaluation pipeline.
    """
    if isinstance(data_source, (str, Path)):
        df = pd.read_csv(Path(data_source))
    else:
        df = data_source.copy()

    if "clinical_label" not in df.columns:
        raise ValueError("Dataset is missing required 'clinical_label' column.")
    if "participant_id" not in df.columns:
        raise ValueError("Dataset is missing 'participant_id' column needed for participant-level splitting.")

    # Drop incomplete rows
    df = df.dropna(subset=MODEL_FEATURE_NAMES + ["clinical_label", "participant_id"]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"Insufficient training samples ({len(df)} rows).")

    is_synthetic = "synthetic" in str(df["data_source"].iloc[0]).lower()

    X = df[MODEL_FEATURE_NAMES].values
    y = df["clinical_label"].values.astype(int)
    groups = df["participant_id"].values

    # Participant-level split to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_participants = set(groups[train_idx])
    test_participants = set(groups[test_idx])

    # Assert zero overlap between participant sets
    assert len(train_participants.intersection(test_participants)) == 0, "Data leakage: Participant in both train and test!"

    logger.info(
        "Train set: %d samples (%d participants) | Test set: %d samples (%d participants)",
        len(X_train), len(train_participants), len(X_test), len(test_participants)
    )

    # Train Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        random_state=random_state,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test) if hasattr(rf, "predict_proba") else None
    metrics = compute_clinical_metrics(y_test, y_pred, y_prob)

    # Compute Feature Importances
    importances = dict(zip(MODEL_FEATURE_NAMES, [round(float(x), 4) for x in rf.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    # Compile model metadata
    metadata = {
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic_training_data": is_synthetic,
        "clinical_validation_status": "not_validated",
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_train_participants": int(len(train_participants)),
        "n_test_participants": int(len(test_participants)),
        "metrics": metrics,
        "features": MODEL_FEATURE_NAMES,
        "feature_importances": sorted_importances,
    }

    # Save model
    if output_model_path:
        save_path = Path(output_model_path)
    else:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = config.MODELS_DIR / f"{model_version}.joblib"

    save_model(rf, save_path, metadata=metadata)

    logger.info(
        "Model training complete. Saved to %s | Accuracy: %.3f, Sens: %.3f, Spec: %.3f",
        save_path, metrics["accuracy"], metrics["sensitivity"], metrics["specificity"]
    )
    return save_path, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OA Gait Screening Random Forest Model")
    parser.add_argument("--data", default=None, help="Path to clinical labelled CSV")
    parser.add_argument("--synthetic_demo", action="store_true", help="Train pipeline test model on synthetic data")
    parser.add_argument("--version", default="rf_v1.0", help="Model version name")
    args = parser.parse_args()

    if args.synthetic_demo:
        print("[INFO] Generating synthetic testing dataset...")
        synth_df = generate_synthetic_training_dataset()
        saved_file, meta = train_pipeline(synth_df, model_version=f"{args.version}_synthetic_demo")
        print(f"[SUCCESS] Synthetic test model saved to: {saved_file}")
        print(f"[METRICS] {json.dumps(meta['metrics'], indent=2)}")
        print(f"[TOP FEATURES] {json.dumps(list(meta['feature_importances'].items())[:5], indent=2)}")
    elif args.data:
        saved_file, meta = train_pipeline(args.data, model_version=args.version)
        print(f"[SUCCESS] Clinically trained model saved to: {saved_file}")
    else:
        print("Please provide a --data path to clinical data or use --synthetic_demo for pipeline testing.")
