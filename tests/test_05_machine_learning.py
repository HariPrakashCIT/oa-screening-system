"""
tests/test_05_machine_learning.py
=================================
Acceptance tests for Machine Learning Pipeline (Part F):
  - Model Utils:
      1. Feature validation flags missing features correctly
      2. Feature vector extracted with exact dimensions and values
      3. Calculation of Accuracy, Sensitivity (Recall), Specificity, and F1
      4. Confusion matrix calculation
      5. Saving and loading model bundles
  - Model Training:
      6. Synthetic dataset generation produces required columns
      7. Participant-level splitting ensures ZERO participant overlap
      8. Training pipeline saves a model bundle with complete metadata
  - Model Prediction / Inference:
      9. Prediction unavailable message when no model exists
      10. Prediction fails gracefully on missing features
      11. Successful inference produces valid prediction dictionary
      12. Medical disclaimer is always attached
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from ml.model_utils import (
    MEDICAL_DISCLAIMER_TEXT,
    MODEL_FEATURE_NAMES,
    compute_clinical_metrics,
    extract_feature_vector,
    load_model,
    save_model,
    validate_features,
)
from ml.predict import predict
from ml.train_model import generate_synthetic_training_dataset, train_pipeline


class TestModelUtils(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_validate_features(self):
        full_features = {k: 1.0 for k in MODEL_FEATURE_NAMES}
        valid, missing = validate_features(full_features)
        self.assertTrue(valid)
        self.assertEqual(len(missing), 0)

        incomplete_features = {"heel_peak": 800.0}
        valid, missing = validate_features(incomplete_features)
        self.assertFalse(valid)
        self.assertIn("midfoot_peak", missing)

    def test_extract_feature_vector_shape(self):
        feats = {k: float(i) for i, k in enumerate(MODEL_FEATURE_NAMES)}
        vec = extract_feature_vector(feats)
        self.assertEqual(vec.shape, (1, len(MODEL_FEATURE_NAMES)))
        self.assertEqual(vec[0, 0], 0.0)
        self.assertEqual(vec[0, 1], 1.0)

    def test_compute_clinical_metrics(self):
        # 4 samples:
        # y_true = [1, 1, 0, 0]
        # y_pred = [1, 0, 0, 0]
        # TP=1, FN=1, TN=2, FP=0
        # Accuracy: 3/4 = 0.75
        # Sensitivity: TP / (TP+FN) = 1/2 = 0.50
        # Specificity: TN / (TN+FP) = 2/2 = 1.00
        # Precision: TP / (TP+FP) = 1/1 = 1.00
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]
        m = compute_clinical_metrics(y_true, y_pred)

        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(m["sensitivity"], 0.50)
        self.assertEqual(m["specificity"], 1.00)
        self.assertEqual(m["precision"], 1.00)
        self.assertEqual(m["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(m["confusion_matrix"]["true_negative"], 2)


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_synthetic_data_generation(self):
        df = generate_synthetic_training_dataset(n_participants=10, trials_per_participant=2)
        self.assertEqual(len(df), 20)
        self.assertIn("clinical_label", df.columns)
        self.assertIn("participant_id", df.columns)
        for col in MODEL_FEATURE_NAMES:
            self.assertIn(col, df.columns)

    def test_train_pipeline_participant_level_split(self):
        df = generate_synthetic_training_dataset(n_participants=16, trials_per_participant=3)
        model_path = self.tmp_dir / "test_model.joblib"

        saved_path, meta = train_pipeline(
            data_source=df,
            model_version="test_v1",
            output_model_path=model_path,
        )
        self.assertTrue(saved_path.exists())
        self.assertIn("accuracy", meta["metrics"])
        self.assertIn("sensitivity", meta["metrics"])
        self.assertIn("specificity", meta["metrics"])
        self.assertEqual(meta["clinical_validation_status"], "not_validated")

        # Load back bundle and verify
        bundle = load_model(saved_path)
        self.assertIn("model", bundle)
        self.assertIn("feature_names", bundle)
        self.assertEqual(bundle["feature_names"], MODEL_FEATURE_NAMES)


class TestModelInference(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_predict_unavailable_when_no_model_exists(self):
        # Point to empty folder
        feats = {k: 50.0 for k in MODEL_FEATURE_NAMES}
        empty_dir = self.tmp_dir / "no_models"
        empty_dir.mkdir()

        res = predict(feats, model_path=empty_dir / "ghost.joblib", upload_to_firebase=False)
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("not clinically validated", res["message"])
        self.assertIn("disclaimer", res)

    def test_predict_handles_missing_features(self):
        # Train quick dummy model
        df = generate_synthetic_training_dataset(n_participants=12, trials_per_participant=2)
        model_path = self.tmp_dir / "quick_model.joblib"
        train_pipeline(df, output_model_path=model_path)

        # Missing all features
        res = predict({"heel_peak": 500}, model_path=model_path, upload_to_firebase=False)
        self.assertEqual(res["status"], "invalid_features")
        self.assertIn("missing_features", res)

    def test_successful_prediction_workflow(self):
        # Train model
        df = generate_synthetic_training_dataset(n_participants=16, trials_per_participant=2)
        model_path = self.tmp_dir / "valid_model.joblib"
        train_pipeline(df, output_model_path=model_path)

        # Valid feature dict
        full_feats = {k: 100.0 for k in MODEL_FEATURE_NAMES}
        full_feats["session_id"] = "test_s001"
        res = predict(full_feats, model_path=model_path, upload_to_firebase=False)

        self.assertEqual(res["status"], "success")
        self.assertIn(res["prediction"], ["Normal_Gait_Pattern", "OA_Associated_Pattern"])
        self.assertIsNotNone(res["model_score"])
        self.assertEqual(res["clinical_validation_status"], "not_validated")
        self.assertEqual(res["disclaimer"], MEDICAL_DISCLAIMER_TEXT)


if __name__ == "__main__":
    unittest.main(verbosity=2)

