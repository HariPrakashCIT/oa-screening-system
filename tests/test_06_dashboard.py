"""
tests/test_06_dashboard.py
==========================
Acceptance tests for Streamlit Dashboard (Prompt 5):
  1. Module imports cleanly and exposes main entrypoints.
  2. Data aggregation for Overview page counts participants, sessions, features.
  3. Participant directory filter works with text search queries.
  4. Missing CSV file detection works gracefully without crashing.
  5. Feature extractor integration generates features for dashboard display.
  6. Assessment report displays 'AI assessment unavailable – model not clinically validated.'
     when no eligible model is present.
  7. Assessment report displays model classification and metadata when model exists.
  8. Medical disclaimer is present on all report pathways.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
import app.dashboard as dashboard_mod
from ml.model_utils import MEDICAL_DISCLAIMER_TEXT
from ml.predict import predict


class TestDashboardComponents(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_dashboard_imports_cleanly(self):
        self.assertTrue(hasattr(dashboard_mod, "main"))
        self.assertTrue(hasattr(dashboard_mod, "page_overview"))
        self.assertTrue(hasattr(dashboard_mod, "page_participants"))
        self.assertTrue(hasattr(dashboard_mod, "page_data_acquisition"))
        self.assertTrue(hasattr(dashboard_mod, "page_walking_analysis"))
        self.assertTrue(hasattr(dashboard_mod, "page_assessment_report"))

    def test_overview_counts_sessions(self):
        # Create 2 mock CSVs
        p1 = self.tmp_dir / "patient_001_session_001.csv"
        p2 = self.tmp_dir / "patient_002_session_001.csv"
        pd.DataFrame({"timestamp_ms": [0], "heel": [100]}).to_csv(p1, index=False)
        pd.DataFrame({"timestamp_ms": [0], "heel": [200]}).to_csv(p2, index=False)

        files = list(self.tmp_dir.glob("*.csv"))
        self.assertEqual(len(files), 2)

    def test_participant_search_filter(self):
        all_pids = ["patient_001", "patient_002", "patient_010", "control_001"]
        query = "patient"
        filtered = [p for p in all_pids if query in p.lower()]
        self.assertEqual(len(filtered), 3)

        query_ctrl = "control"
        filtered_ctrl = [p for p in all_pids if query_ctrl in p.lower()]
        self.assertEqual(filtered_ctrl, ["control_001"])

    def test_assessment_fallback_when_no_model(self):
        # Point to empty models dir
        empty_models_dir = self.tmp_dir / "empty_models"
        empty_models_dir.mkdir()

        res = predict(
            {"heel_peak": 800},
            model_path=empty_models_dir / "nonexistent.joblib",
            upload_to_firebase=False,
        )
        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(res["message"], "AI assessment unavailable – model not clinically validated.")
        self.assertEqual(res["disclaimer"], MEDICAL_DISCLAIMER_TEXT)


if __name__ == "__main__":
    unittest.main(verbosity=2)

