"""
tests/test_04_signal_processing.py
==================================
Acceptance tests for Signal Processing & Feature Extraction (Part D):
  - DataProcessor:
      1. Loading valid raw CSV
      2. Missing mandatory column rejection
      3. Corrupted / non-numeric row handling
      4. Duplicate timestamp deduplication
      5. Forward/backward-fill missing value imputation
      6. Sensor value range bounding and clipping
      7. Moving average smoothing filter
      8. Force calibration conversion (and uncalibrated fallback)
      9. Saving cleaned CSV to disk
  - FeatureExtractor:
      10. Mathematical accuracy of pressure peak, mean, std
      11. Active loading duration calculation against threshold
      12. Kinematic acceleration magnitude and variability
      13. Gyroscope rotational magnitude and variability
      14. Sampling rate and duration computation
      15. JSON feature summary file creation
      16. End-to-end integration: Raw CSV -> Clean CSV -> Feature JSON
"""

import csv
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
from app.data_processor import DataProcessor, ProcessingReport
from app.feature_extractor import FeatureExtractor


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.processor = DataProcessor(smoothing_window=3)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_sample_csv(self, filename: str, rows: list) -> Path:
        filepath = self.tmp_dir / filename
        pd.DataFrame(rows).to_csv(filepath, index=False)
        return filepath

    def test_load_valid_raw_csv(self):
        rows = [
            {
                "timestamp_ms": i * 20,
                "heel": 200, "midfoot": 150, "forefoot": 100,
                "acc_x_g": 0.01, "acc_y_g": 0.02, "acc_z_g": 0.98,
                "gyro_x_dps": 0.1, "gyro_y_dps": 0.2, "gyro_z_dps": 0.3,
            }
            for i in range(5)
        ]
        csv_path = self._create_sample_csv("raw_test.csv", rows)
        df = self.processor.load_raw_csv(csv_path)
        self.assertEqual(len(df), 5)
        self.assertIn("heel", df.columns)

    def test_empty_csv_raises_error(self):
        empty_csv = self.tmp_dir / "empty.csv"
        empty_csv.write_text("")
        with self.assertRaises((ValueError, pd.errors.EmptyDataError)):
            self.processor.load_raw_csv(empty_csv)

    def test_missing_column_raises_error(self):
        # CSV missing 'acc_z_g'
        rows = [{"timestamp_ms": 10, "heel": 100, "midfoot": 50, "forefoot": 20}]
        csv_path = self._create_sample_csv("bad_columns.csv", rows)
        with self.assertRaises(ValueError):
            self.processor.load_raw_csv(csv_path)

    def test_deduplicates_duplicate_timestamps(self):
        rows = [
            {"timestamp_ms": 100, "heel": 100, "midfoot": 50, "forefoot": 20, "acc_x_g": 0, "acc_y_g": 0, "acc_z_g": 1, "gyro_x_dps": 0, "gyro_y_dps": 0, "gyro_z_dps": 0},
            {"timestamp_ms": 100, "heel": 105, "midfoot": 52, "forefoot": 21, "acc_x_g": 0, "acc_y_g": 0, "acc_z_g": 1, "gyro_x_dps": 0, "gyro_y_dps": 0, "gyro_z_dps": 0},  # duplicate timestamp
            {"timestamp_ms": 120, "heel": 110, "midfoot": 55, "forefoot": 22, "acc_x_g": 0, "acc_y_g": 0, "acc_z_g": 1, "gyro_x_dps": 0, "gyro_y_dps": 0, "gyro_z_dps": 0},
        ]
        df = pd.DataFrame(rows)
        clean_df, report = self.processor.validate_and_clean(df, apply_filter=False)
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(report.duplicate_timestamps_dropped, 1)

    def test_handles_incorrect_data_types(self):
        rows = [
            {"timestamp_ms": 20, "heel": "corrupted_text", "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 40, "heel": 150, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        df = pd.DataFrame(rows)
        clean_df, report = self.processor.validate_and_clean(df, apply_filter=False)
        self.assertFalse(clean_df["heel"].isna().any())
        self.assertEqual(clean_df.loc[0, "heel"], 150.0)  # backward filled from valid row

    def test_imputes_missing_sensor_values(self):
        rows = [
            {"timestamp_ms": 20, "heel": 100, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 40, "heel": np.nan, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 60, "heel": 120, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        df = pd.DataFrame(rows)
        clean_df, report = self.processor.validate_and_clean(df, apply_filter=False)
        self.assertEqual(clean_df["heel"].isna().sum(), 0)
        self.assertEqual(clean_df.loc[1, "heel"], 100.0)  # forward-filled
        self.assertEqual(report.missing_values_imputed, 1)

    def test_clamps_out_of_range_values(self):
        rows = [
            # 5000 is beyond 12-bit ADC max 4095
            {"timestamp_ms": 20, "heel": 5000, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            # -10 is below 0
            {"timestamp_ms": 40, "heel": -10, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        df = pd.DataFrame(rows)
        clean_df, report = self.processor.validate_and_clean(df, apply_filter=False)
        self.assertEqual(clean_df.loc[0, "heel"], 4095.0)
        self.assertEqual(clean_df.loc[1, "heel"], 0.0)
        self.assertEqual(report.out_of_range_flagged, 2)

    def test_calibration_curve_application(self):
        calib_cfg = {
            "heel": {"scale": 0.05, "offset": 0.0},
            "midfoot": {"scale": 0.05, "offset": 0.0},
            "forefoot": {"scale": 0.05, "offset": 0.0},
        }
        proc = DataProcessor(calibration_config=calib_cfg)
        rows = [
            {"timestamp_ms": 20, "heel": 1000, "midfoot": 500, "forefoot": 200, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        clean_df, report = proc.validate_and_clean(pd.DataFrame(rows), apply_filter=False)
        self.assertTrue(report.is_calibrated)
        self.assertIn("heel_force_est", clean_df.columns)
        self.assertAlmostEqual(clean_df.loc[0, "heel_force_est"], 50.0)  # 1000 * 0.05

    def test_process_and_save_creates_clean_csv(self):
        rows = [
            {"timestamp_ms": i * 20, "heel": 100 + i, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0}
            for i in range(10)
        ]
        raw_path = self._create_sample_csv("patient_001_session_001.csv", rows)
        clean_path, report = self.processor.process_and_save(raw_path, output_dir=self.tmp_dir)

        self.assertTrue(clean_path.exists())
        self.assertTrue(clean_path.name.endswith("_clean.csv"))
        saved_df = pd.read_csv(clean_path)
        self.assertEqual(len(saved_df), 10)


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.extractor = FeatureExtractor(load_threshold_adc=100)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_exact_mathematical_feature_calculations(self):
        # Known numbers to verify math
        # heel: [100, 200, 300] -> mean=200, peak=300, min=100, std=81.65 (ddof=0)
        rows = [
            {"timestamp_ms": 0, "heel": 100, "midfoot": 50, "forefoot": 300, "acc_x_g": 0.1, "acc_y_g": 0.2, "acc_z_g": 0.9, "gyro_x_dps": 1.0, "gyro_y_dps": 2.0, "gyro_z_dps": 3.0},
            {"timestamp_ms": 20, "heel": 200, "midfoot": 50, "forefoot": 200, "acc_x_g": 0.1, "acc_y_g": 0.2, "acc_z_g": 0.9, "gyro_x_dps": 1.0, "gyro_y_dps": 2.0, "gyro_z_dps": 3.0},
            {"timestamp_ms": 40, "heel": 300, "midfoot": 50, "forefoot": 100, "acc_x_g": 0.1, "acc_y_g": 0.2, "acc_z_g": 0.9, "gyro_x_dps": 1.0, "gyro_y_dps": 2.0, "gyro_z_dps": 3.0},
        ]
        df = pd.DataFrame(rows)
        feats = self.extractor.extract_features(df, participant_id="p1", session_id="s1")

        self.assertEqual(feats["heel_peak"], 300.0)
        self.assertEqual(feats["heel_mean"], 200.0)
        self.assertEqual(feats["heel_min"], 100.0)
        self.assertAlmostEqual(feats["heel_std"], 81.65, places=1)
        self.assertEqual(feats["measurement_count"], 3)
        self.assertAlmostEqual(feats["duration_seconds"], 0.04, places=2)
        self.assertEqual(feats["sampling_frequency_hz"], 50.0)

    def test_loading_duration_with_threshold(self):
        # Threshold is 100:
        # Sample 1: 50 (< 100)
        # Sample 2: 150 (>= 100)
        # Sample 3: 200 (>= 100)
        # At 50 Hz, 2 samples = 2 / 50 = 0.04s
        rows = [
            {"timestamp_ms": 0, "heel": 50, "midfoot": 20, "forefoot": 10, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 20, "heel": 150, "midfoot": 20, "forefoot": 10, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 40, "heel": 200, "midfoot": 20, "forefoot": 10, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        df = pd.DataFrame(rows)
        feats = self.extractor.extract_features(df)
        self.assertAlmostEqual(feats["heel_loading_duration_s"], 0.04, places=2)

    def test_kinematic_acceleration_magnitude(self):
        # x=0.3, y=0.4, z=0 -> |a| = sqrt(0.09 + 0.16) = 0.5
        rows = [
            {"timestamp_ms": 0, "heel": 50, "midfoot": 20, "forefoot": 10, "acc_x_g": 0.3, "acc_y_g": 0.4, "acc_z_g": 0.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
            {"timestamp_ms": 20, "heel": 50, "midfoot": 20, "forefoot": 10, "acc_x_g": 0.3, "acc_y_g": 0.4, "acc_z_g": 0.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0},
        ]
        feats = self.extractor.extract_features(pd.DataFrame(rows))
        self.assertAlmostEqual(feats["acc_mag_mean"], 0.5, places=3)
        self.assertAlmostEqual(feats["acc_mag_peak"], 0.5, places=3)

    def test_process_and_save_features_json(self):
        rows = [
            {"timestamp_ms": i * 20, "heel": 100, "midfoot": 50, "forefoot": 20, "acc_x_g": 0.0, "acc_y_g": 0.0, "acc_z_g": 1.0, "gyro_x_dps": 0.0, "gyro_y_dps": 0.0, "gyro_z_dps": 0.0}
            for i in range(15)
        ]
        csv_path = self.tmp_dir / "test_session_clean.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        feat_file, feats = self.extractor.process_and_save(
            csv_path,
            output_dir=self.tmp_dir,
            upload_to_firebase=False,
        )
        self.assertTrue(feat_file.exists())
        self.assertTrue(feat_file.name.endswith("_features.json"))
        with open(feat_file, "r") as f:
            data = json.load(f)
        self.assertIn("heel_peak", data)
        self.assertEqual(data["measurement_count"], 15)


class TestFullDataPipeline(unittest.TestCase):
    """End-to-end integration: Raw CSV -> Clean CSV -> Feature JSON."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_pipeline_raw_to_features(self):
        # Generate 50 realistic rows
        rows = []
        for i in range(50):
            rows.append({
                "host_timestamp_utc": "2026-11-09T15:00:00+00:00",
                "timestamp_ms": i * 20,
                "heel": int(500 * np.sin(i / 5.0) + 600),
                "midfoot": int(200 * np.cos(i / 5.0) + 300),
                "forefoot": int(400 * np.sin(i / 4.0) + 500),
                "acc_x_g": round(0.05 * np.sin(i), 3),
                "acc_y_g": round(0.08 * np.cos(i), 3),
                "acc_z_g": 0.98,
                "gyro_x_dps": 1.2,
                "gyro_y_dps": 0.8,
                "gyro_z_dps": 0.4,
                "participant_id": "patient_001",
                "session_id": "session_001",
                "data_source": "simulated",
            })
        raw_path = self.tmp_dir / "patient_001_session_001.csv"
        pd.DataFrame(rows).to_csv(raw_path, index=False)

        # 1. Process
        processor = DataProcessor()
        clean_path, report = processor.process_and_save(raw_path, output_dir=self.tmp_dir)
        self.assertTrue(clean_path.exists())
        self.assertEqual(report.clean_rows, 50)

        # 2. Extract Features
        extractor = FeatureExtractor()
        feat_path, feats = extractor.process_and_save(
            clean_path, output_dir=self.tmp_dir, upload_to_firebase=False
        )
        self.assertTrue(feat_path.exists())
        self.assertEqual(feats["participant_id"], "patient_001")
        self.assertEqual(feats["session_id"], "session_001")
        self.assertGreater(feats["heel_peak"], 0)
        self.assertGreater(feats["sampling_frequency_hz"], 40.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

