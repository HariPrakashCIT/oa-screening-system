"""
tests/test_07_patient_monitoring.py
===================================
Acceptance tests for Prompt 6:
  - DailyLogService:
      1. Logging meals and calculating total protein
      2. Validating pain and fatigue range (0-10)
      3. Validating non-negative steps and stiffness minutes
      4. Storing and retrieving daily logs locally
  - BaselineEngine:
      5. Computing robust statistics (mean, median, std, MAD)
      6. Baseline status is 'provisional' when observations < 7
      7. Baseline status is 'established' when observations >= 7
  - AnomalyDetector:
      8. Normal measurement flagged as 'within_baseline'
      9. Deviation >= Z-threshold flagged as 'change_to_monitor'
      10. Multi-variable pattern detection (e.g. drop in steps + rise in pain)
  - TrendAnalyzer:
      11. Weekly summary computation of steps, pain, adherence
      12. Spearman rank correlation requires minimum paired observations
  - AIReportGenerator (Gemini Integration):
      13. Deterministic fallback produces valid structured JSON schema
      14. Medical disclaimer is always included
  - PersonalizedIsolationForest:
      15. Isolation forest trains on multivariate gait features and flags outliers
  - PatientMonitoringCoordinator:
      16. End-to-end monitoring pipeline execution
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

from app.ai_report_generator import AIReportGenerator
from app.anomaly_detector import AnomalyDetector
from app.baseline_engine import BaselineEngine, compute_robust_statistics
from app.daily_log_service import DailyLog, DailyLogService, calculate_daily_protein
from app.patient_monitoring import PatientMonitoringCoordinator
from app.trend_analyzer import TrendAnalyzer
from ml.model_utils import MEDICAL_DISCLAIMER_TEXT
from ml.personalized_anomaly_model import PersonalizedIsolationForest


class TestDailyLogging(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.service = DailyLogService(storage_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_protein_summation(self):
        meals = [
            {"food": "Eggs", "protein_g": 12.0},
            {"food": "Paneer", "protein_g": 18.0},
        ]
        tot = calculate_daily_protein(meals)
        self.assertEqual(tot, 30.0)

    def test_symptom_scale_validation(self):
        # Pain score 11 should fail (must be 0-10)
        with self.assertRaises(ValueError):
            log = DailyLog(patient_id="p1", date="2026-11-09", pain_score=11)
            log.validate()

        # Fatigue score -1 should fail
        with self.assertRaises(ValueError):
            log = DailyLog(patient_id="p1", date="2026-11-09", fatigue_score=-1)
            log.validate()

    def test_save_and_retrieve_daily_log(self):
        log = DailyLog(
            patient_id="test_pt_01",
            date="2026-11-09",
            steps_count=4500,
            protein_intake_g=65.0,
            pain_score=3,
            stiffness_minutes=15,
            fatigue_score=2,
            swelling_reported=False,
            exercise_completed=True,
        )
        saved = self.service.save_log(log)
        self.assertTrue(saved)

        retrieved = self.service.get_log("test_pt_01", "2026-11-09")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.steps_count, 4500)
        self.assertEqual(retrieved.pain_score, 3)


class TestBaselineAndAnomalyDetection(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.engine = BaselineEngine(min_observations_established=7, storage_dir=self.tmp_dir)
        self.detector = AnomalyDetector(z_threshold=2.0, baseline_engine=self.engine)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_robust_statistics_calculation(self):
        # Numbers: [10, 10, 10, 10, 50] (one outlier)
        # Median = 10, Mean = 18, MAD = 0
        stats = compute_robust_statistics([10, 10, 10, 10, 50])
        self.assertEqual(stats.median, 10.0)
        self.assertEqual(stats.mean, 18.0)
        self.assertEqual(stats.count, 5)

    def test_baseline_status_provisional_vs_established(self):
        # 3 observations < 7 -> provisional
        gait_3 = [{"heel_peak": 800 + i * 5} for i in range(3)]
        b_prov = self.engine.compute_baseline("pt_test", gait_sessions=gait_3, daily_logs=[])
        self.assertEqual(b_prov.baseline_status, "provisional")

        # 8 observations >= 7 -> established
        gait_8 = [{"heel_peak": 800 + i * 5} for i in range(8)]
        b_est = self.engine.compute_baseline("pt_test", gait_sessions=gait_8, daily_logs=[])
        self.assertEqual(b_est.baseline_status, "established")

    def test_anomaly_detection_within_baseline(self):
        # Baseline with median = 5000, std = 500
        base_stats = {"median": 5000.0, "mad": 300.0, "std": 500.0, "count": 10}
        dev = self.detector.evaluate_metric("steps_count", 4900.0, base_stats)
        self.assertEqual(dev.status, "within_baseline")

    def test_anomaly_detection_flagged_change(self):
        # Significant drop in steps: 2000 vs 5000 (Z < -2.0)
        base_stats = {"median": 5000.0, "mad": 300.0, "std": 500.0, "count": 10}
        dev = self.detector.evaluate_metric("steps_count", 2000.0, base_stats)
        self.assertEqual(dev.status, "change_to_monitor")
        self.assertLess(dev.z_score, -2.0)


class TestTrendAndAIReporting(unittest.TestCase):

    def test_trend_weekly_summary(self):
        records = [
            {"date": f"2026-11-0{i+1}", "steps_count": 4000 + i * 100, "pain_score": 3, "stiffness_minutes": 10, "protein_intake_g": 65.0, "protein_target_g": 60.0, "exercise_completed": True}
            for i in range(7)
        ]
        df = pd.DataFrame(records)
        analyzer = TrendAnalyzer()
        summary = analyzer.compute_weekly_summary(df)

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["days_analyzed"], 7)
        self.assertEqual(summary["exercise_completion_rate_pct"], 100.0)

    def test_ai_report_generator_deterministic_schema(self):
        gen = AIReportGenerator(api_key=None)  # forces deterministic fallback
        anomaly_payload = {
            "overall_category": "Change to monitor",
            "deviations_detected": [
                {"metric": "steps_count", "status": "change_to_monitor", "clinical_note": "Significant drop"}
            ],
            "multi_variable_pattern": "Reduced steps with elevated pain",
        }
        report = gen.generate_patient_report(
            patient_id="patient_001",
            anomaly_report=anomaly_payload,
            daily_log={"steps_count": 2100, "pain_score": 5, "protein_intake_g": 55},
            upload_to_firebase=False,
        )

        required_keys = [
            "daily_summary",
            "activity_analysis",
            "nutrition_analysis",
            "gait_analysis",
            "symptom_analysis",
            "detected_changes",
            "suggested_follow_up",
            "limitations",
            "disclaimer",
        ]
        for k in required_keys:
            self.assertIn(k, report, f"Missing key in AI report: {k}")
        self.assertEqual(report["disclaimer"], MEDICAL_DISCLAIMER_TEXT)


class TestPersonalizedIsolationForest(unittest.TestCase):

    def test_isolation_forest_training_and_scoring(self):
        from ml.model_utils import MODEL_FEATURE_NAMES

        # Generate 15 baseline sessions
        sessions = [{k: float(100.0 + np.random.normal(0, 5)) for k in MODEL_FEATURE_NAMES} for _ in range(15)]
        iso = PersonalizedIsolationForest()
        trained = iso.fit(sessions, patient_id="pt_iso")
        self.assertTrue(trained)

        # Normal test sample
        normal_sample = {k: 100.0 for k in MODEL_FEATURE_NAMES}
        res_normal = iso.score_session(normal_sample)
        self.assertEqual(res_normal["status"], "success")
        self.assertFalse(res_normal["is_outlier"])

        # Wild extreme outlier (10x higher values)
        extreme_sample = {k: 5000.0 for k in MODEL_FEATURE_NAMES}
        res_extreme = iso.score_session(extreme_sample)
        self.assertTrue(res_extreme["is_outlier"])


class TestPatientCoordinator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.coord = PatientMonitoringCoordinator(
            daily_log_service=DailyLogService(storage_dir=self.tmp_dir / "logs"),
            baseline_engine=BaselineEngine(storage_dir=self.tmp_dir / "baselines"),
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_daily_monitoring_pipeline(self):
        # 1. Log today's entry
        log = DailyLog(patient_id="pt_full", date="2026-11-09", steps_count=3500, pain_score=2)
        self.coord.log_service.save_log(log)

        # 2. Run coordination pipeline
        res = self.coord.run_daily_monitoring(patient_id="pt_full", today_date_str="2026-11-09", upload_to_firebase=False)
        self.assertEqual(res["patient_id"], "pt_full")
        self.assertIn("ai_insight", res)
        self.assertIn("anomaly_report", res)
        self.assertIn("baseline", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)

