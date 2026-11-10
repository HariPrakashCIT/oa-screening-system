"""
predict.py — Inference module for gait pattern evaluation.

Simple explanation
------------------
Takes a dictionary or file of extracted gait features, verifies them against
the trained Random Forest model, and produces an experimental pattern-matching
assessment score.

MEDICAL SAFETY:
If a validated model does not exist or if features are invalid,
this module returns an explicit status:
    "AI assessment unavailable – model not clinically validated."
It NEVER fabricates predictions or diagnoses.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure project root is importable when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.firebase_service import get_firebase_service
from ml.model_utils import (
    MEDICAL_DISCLAIMER_TEXT,
    extract_feature_vector,
    load_model,
    validate_features,
)

logger = logging.getLogger(__name__)


def find_latest_model(models_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Finds the most recent model file (.joblib or .pkl) in the models/ directory.
    """
    m_dir = models_dir or config.MODELS_DIR
    if not m_dir.exists():
        return None

    model_files = list(m_dir.glob("*.joblib")) + list(m_dir.glob("*.pkl"))
    if not model_files:
        return None

    # Return the file with the newest modification time
    return max(model_files, key=lambda f: f.stat().st_mtime)


def predict(
    features: Union[Dict[str, Any], str, Path],
    model_path: Optional[Union[str, Path]] = None,
    upload_to_firebase: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates extracted gait features using an eligible trained model.

    Returns a standardized assessment dictionary.
    """
    # 1. Resolve features
    if isinstance(features, (str, Path)):
        f_path = Path(features)
        if not f_path.exists():
            return {
                "status": "error",
                "prediction": "not_evaluated",
                "message": f"Feature file not found: {f_path}",
                "disclaimer": MEDICAL_DISCLAIMER_TEXT,
            }
        with open(f_path, "r", encoding="utf-8") as f:
            feat_dict = json.load(f)
    else:
        feat_dict = dict(features)

    # 2. Check for an eligible model
    resolved_model_path = Path(model_path) if model_path else find_latest_model()
    if not resolved_model_path or not resolved_model_path.exists():
        logger.warning("No eligible trained model found in %s", config.MODELS_DIR)
        return {
            "status": "unavailable",
            "prediction": "not_evaluated",
            "model_version": None,
            "model_score": None,
            "message": "AI assessment unavailable – model not clinically validated.",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        }

    # 3. Load model bundle
    try:
        bundle = load_model(resolved_model_path)
        clf = bundle["model"]
        meta = bundle.get("metadata", {})
        version = meta.get("model_version", resolved_model_path.stem)
        feature_names = bundle.get("feature_names")
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", resolved_model_path, exc)
        return {
            "status": "error",
            "prediction": "not_evaluated",
            "message": f"Error loading model: {exc}",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        }

    # 4. Validate input features
    is_valid, missing = validate_features(feat_dict)
    if not is_valid:
        logger.warning("Features missing for prediction: %s", missing)
        return {
            "status": "invalid_features",
            "prediction": "not_evaluated",
            "missing_features": missing,
            "message": f"Missing required features: {missing}",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        }

    # 5. Extract feature vector and run inference
    X = extract_feature_vector(feat_dict, feature_names=feature_names)
    pred_raw = int(clf.predict(X)[0])

    # Probability estimation if available (note: not calibrated confidence)
    score = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        # Probability of class 1
        score = round(float(probs[1]), 4) if len(probs) > 1 else round(float(probs[0]), 4)

    # Human-readable experimental label
    label_str = "OA_Associated_Pattern" if pred_raw == 1 else "Normal_Gait_Pattern"

    result = {
        "status": "success",
        "prediction": label_str,
        "prediction_code": pred_raw,
        "model_version": version,
        "model_score": score,
        "model_file": resolved_model_path.name,
        "clinical_validation_status": meta.get("clinical_validation_status", "not_validated"),
        "is_synthetic_model": meta.get("is_synthetic_training_data", False),
        "disclaimer": MEDICAL_DISCLAIMER_TEXT,
    }

    # 6. Optionally upload to Firebase if session_id is available
    if upload_to_firebase:
        session_id = feat_dict.get("session_id")
        if session_id and session_id != "unknown":
            svc = get_firebase_service()
            if svc.is_connected:
                svc.save_assessment_result(
                    session_id=session_id,
                    prediction=label_str,
                    model_version=version,
                    model_score=score,
                    validated=False,
                    clinical_validation_status=result["clinical_validation_status"],
                )
                logger.info("Uploaded assessment for session %s to Firebase.", session_id)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gait Features with Trained Model")
    parser.add_argument("--features", required=True, help="Path to extracted features JSON")
    parser.add_argument("--model", default=None, help="Path to specific model file")
    args = parser.parse_args()

    res = predict(args.features, model_path=args.model)
    print(json.dumps(res, indent=2))
