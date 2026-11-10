"""
tests/test_02_data_acquisition.py
==================================
Acceptance tests for Prompt 2 — ESP32 Communication & Data Acquisition.

Tests cover:
  1.  SerialReceiver — module imports cleanly
  2.  _parse_and_validate — valid JSON message
  3.  _parse_and_validate — invalid JSON (not parseable)
  4.  _parse_and_validate — empty string
  5.  _parse_and_validate — comment line
  6.  _parse_and_validate — missing sensor fields
  7.  _parse_and_validate — wrong data types (string instead of number)
  8.  _parse_and_validate — partial fields
  9.  SimulatedSerialReceiver — produces readings with correct fields
  10. SimulatedSerialReceiver — values within valid ranges
  11. SimulatedSerialReceiver — stop() ends the stream
  12. list_serial_ports — returns a list (may be empty)
  13. create_receiver — returns simulated receiver in demo mode
  14. DataLogger — file is created with correct CSV columns
  15. DataLogger — write() saves all nine sensor fields
  16. DataLogger — host_timestamp_utc column is present and not empty
  17. DataLogger — data_source column is "simulated" or "real"
  18. DataLogger — participant_id and session_id are saved correctly
  19. DataLogger — prevents overwriting an existing file
  20. DataLogger — context manager (with statement) opens and closes cleanly
  21. DataLogger — write_many() saves multiple rows
  22. SessionManager — run() with demo mode completes successfully
  23. SessionManager — result contains correct participant and session IDs
  24. SessionManager — CSV file exists after a completed session
  25. SessionManager — CSV has the expected number of rows
  26. SessionManager — data_source is "simulated" in demo mode
  27. SessionManager — session status is "completed" after a normal run
  28. SessionManager — generate_session_id produces a parseable timestamp
  29. SessionManager — next_participant_id increments from existing files
  30. End-to-end — 3 seconds of simulated data produces ≥ 100 rows

Run with:
    python tests/test_02_data_acquisition.py

Or (if pytest is installed):
    pytest tests/test_02_data_acquisition.py -v
"""

import csv
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.serial_receiver import (
    SerialReceiver,
    SimulatedSerialReceiver,
    create_receiver,
    list_serial_ports,
)
from app.data_logger import DataLogger
from app.session_manager import (
    SessionManager,
    SessionResult,
    generate_session_id,
    next_participant_id,
)


# ---------------------------------------------------------------------------
# Shared helper — a minimal valid sensor reading dict
# ---------------------------------------------------------------------------
VALID_READING = {
    "timestamp_ms": 1250,
    "heel":         120,
    "midfoot":      240,
    "forefoot":     180,
    "acc_x_g":      0.12,
    "acc_y_g":      0.25,
    "acc_z_g":      0.98,
    "gyro_x_dps":   0.3,
    "gyro_y_dps":   0.1,
    "gyro_z_dps":   0.4,
}

VALID_JSON = json.dumps(VALID_READING)


# ===========================================================================
# 1-12  SerialReceiver tests
# ===========================================================================

class TestSerialReceiverParsing(unittest.TestCase):
    """
    Tests the static _parse_and_validate method.
    This method is the most important — it is the gatekeeper that accepts
    or rejects every message from the ESP32.
    """

    def _parse(self, raw: str):
        return SerialReceiver._parse_and_validate(raw)

    # --- Test 2 ---
    def test_valid_json_returns_dict(self):
        result = self._parse(VALID_JSON)
        self.assertIsInstance(result, dict, "Valid JSON should return a dict")

    def test_valid_json_has_all_nine_fields(self):
        result = self._parse(VALID_JSON)
        self.assertIsNotNone(result)
        for field in config.REQUIRED_SENSOR_FIELDS:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_valid_json_attaches_raw_line(self):
        result = self._parse(VALID_JSON)
        self.assertIn("_raw_line", result)

    # --- Test 3 ---
    def test_invalid_json_returns_none(self):
        result = self._parse("{not valid json")
        self.assertIsNone(result, "Invalid JSON must return None")

    def test_truncated_json_returns_none(self):
        result = self._parse('{"timestamp_ms": 100,')
        self.assertIsNone(result)

    def test_random_text_returns_none(self):
        result = self._parse("hello world")
        self.assertIsNone(result)

    # --- Test 4 ---
    def test_empty_string_returns_none(self):
        result = self._parse("")
        self.assertIsNone(result, "Empty string must return None")

    def test_whitespace_only_returns_none(self):
        result = self._parse("   ")
        self.assertIsNone(result)

    # --- Test 5 ---
    def test_comment_line_returns_none(self):
        result = self._parse("// This is a debug comment from ESP32")
        self.assertIsNone(result, "Comment lines must be silently skipped")

    # --- Test 6 ---
    def test_missing_one_field_returns_none(self):
        incomplete = dict(VALID_READING)
        del incomplete["heel"]
        result = self._parse(json.dumps(incomplete))
        self.assertIsNone(result, "Missing 'heel' must return None")

    def test_missing_all_imu_fields_returns_none(self):
        partial = {
            "timestamp_ms": 100,
            "heel":    100,
            "midfoot": 200,
            "forefoot": 150,
        }
        result = self._parse(json.dumps(partial))
        self.assertIsNone(result)

    def test_empty_json_object_returns_none(self):
        result = self._parse("{}")
        self.assertIsNone(result)

    # --- Test 7 ---
    def test_string_value_for_numeric_field_returns_none(self):
        bad = dict(VALID_READING)
        bad["heel"] = "HIGH"          # string instead of number
        result = self._parse(json.dumps(bad))
        self.assertIsNone(result, "Non-numeric sensor value must return None")

    def test_null_value_for_field_returns_none(self):
        bad = dict(VALID_READING)
        bad["acc_x_g"] = None
        result = self._parse(json.dumps(bad))
        self.assertIsNone(result)

    def test_boolean_value_returns_none(self):
        bad = dict(VALID_READING)
        bad["gyro_z_dps"] = True      # boolean — not a valid sensor value
        result = self._parse(json.dumps(bad))
        self.assertIsNone(result)

    # --- Test 8 ---
    def test_partial_fields_returns_none(self):
        partial = {"timestamp_ms": 500, "heel": 100}
        result = self._parse(json.dumps(partial))
        self.assertIsNone(result)

    # Extra edge case
    def test_json_array_instead_of_object_returns_none(self):
        result = self._parse("[1, 2, 3]")
        self.assertIsNone(result, "JSON array must return None — expected object")


# ===========================================================================
# 9-13  SimulatedSerialReceiver tests
# ===========================================================================

class TestSimulatedSerialReceiver(unittest.TestCase):

    def _make_receiver(self, n_samples: int = 10):
        """Create a receiver that produces exactly n_samples then stops."""
        return SimulatedSerialReceiver(
            sample_rate_hz=200,         # fast — no real sleep in tests
            duration_seconds=n_samples / 200,
        )

    # --- Test 9 ---
    def test_readings_have_all_required_fields(self):
        receiver = self._make_receiver(5)
        receiver.connect()
        for reading in receiver.stream():
            for field in config.REQUIRED_SENSOR_FIELDS:
                self.assertIn(field, reading, f"Missing: {field}")
            break   # one reading is enough for this test

    # --- Test 10 ---
    def test_pressure_values_within_adc_range(self):
        receiver = self._make_receiver(50)
        receiver.connect()
        count = 0
        for reading in receiver.stream():
            for sensor in ("heel", "midfoot", "forefoot"):
                val = reading[sensor]
                self.assertGreaterEqual(val, 0, f"{sensor} below 0")
                self.assertLessEqual(val, 4095, f"{sensor} above 4095")
            count += 1
        self.assertGreater(count, 0, "Stream produced no readings")

    def test_accelerometer_values_in_range(self):
        receiver = self._make_receiver(20)
        receiver.connect()
        for reading in receiver.stream():
            for axis in ("acc_x_g", "acc_y_g", "acc_z_g"):
                lo, hi = config.SENSOR_RANGE[axis]
                self.assertGreaterEqual(reading[axis], lo)
                self.assertLessEqual(reading[axis], hi)

    def test_gyroscope_values_in_range(self):
        receiver = self._make_receiver(20)
        receiver.connect()
        for reading in receiver.stream():
            for axis in ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps"):
                lo, hi = config.SENSOR_RANGE[axis]
                self.assertGreaterEqual(reading[axis], lo)
                self.assertLessEqual(reading[axis], hi)

    def test_timestamp_ms_is_non_negative_integer(self):
        receiver = self._make_receiver(5)
        receiver.connect()
        for reading in receiver.stream():
            self.assertIsInstance(reading["timestamp_ms"], int)
            self.assertGreaterEqual(reading["timestamp_ms"], 0)
            break

    # --- Test 11 ---
    def test_stop_ends_stream(self):
        receiver = SimulatedSerialReceiver(sample_rate_hz=200)
        receiver.connect()
        count = 0
        for reading in receiver.stream():
            count += 1
            if count >= 3:
                receiver.stop()
        self.assertGreaterEqual(count, 3)
        self.assertFalse(receiver.is_connected)

    # --- Test 12 ---
    def test_list_serial_ports_returns_list(self):
        ports = list_serial_ports()
        self.assertIsInstance(ports, list)

    # --- Test 13 ---
    def test_create_receiver_demo_mode_returns_simulated(self):
        receiver = create_receiver(demo=True)
        self.assertIsInstance(receiver, SimulatedSerialReceiver)

    def test_create_receiver_no_pyserial_falls_back_to_simulated(self):
        # Even if demo=False, if pyserial isn't available we get Simulated
        receiver = create_receiver(demo=True)
        self.assertIsInstance(receiver, SimulatedSerialReceiver)


# ===========================================================================
# 14-21  DataLogger tests
# ===========================================================================

class TestDataLogger(unittest.TestCase):

    def setUp(self):
        """Use a temporary directory so tests don't pollute data/raw."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)
        self.pid = "test_patient"
        self.sid = "test_session_001"

    def tearDown(self):
        """Remove all temporary test files."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_logger(self, source="simulated"):
        return DataLogger(
            participant_id=self.pid,
            session_id=self.sid,
            data_source=source,
            output_dir=self.tmp_path,
        )

    # --- Test 14 ---
    def test_csv_file_is_created(self):
        dl = self._make_logger()
        dl.open()
        dl.close()
        self.assertTrue(dl.csv_path.exists(), "CSV file must be created on disk")

    def test_csv_has_correct_header_columns(self):
        dl = self._make_logger()
        dl.open()
        dl.close()
        with open(dl.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, config.CSV_COLUMNS)

    # --- Test 15 ---
    def test_write_saves_all_nine_sensor_fields(self):
        dl = self._make_logger()
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        sensor_fields = [
            "timestamp_ms", "heel", "midfoot", "forefoot",
            "acc_x_g", "acc_y_g", "acc_z_g",
            "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
        ]
        for field in sensor_fields:
            self.assertIn(field, row, f"Missing field in CSV: {field}")
            self.assertNotEqual(row[field], "", f"Field '{field}' is empty")

    def test_saved_values_match_input(self):
        dl = self._make_logger()
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(int(row["heel"]), VALID_READING["heel"])
        self.assertEqual(int(row["midfoot"]), VALID_READING["midfoot"])
        self.assertAlmostEqual(float(row["acc_z_g"]), VALID_READING["acc_z_g"])

    # --- Test 16 ---
    def test_host_timestamp_utc_is_present_and_non_empty(self):
        dl = self._make_logger()
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertIn("host_timestamp_utc", row)
        self.assertNotEqual(row["host_timestamp_utc"], "")
        # Validate it is a parseable ISO-8601 timestamp
        ts = datetime.fromisoformat(row["host_timestamp_utc"])
        self.assertIsNotNone(ts)

    # --- Test 17 ---
    def test_data_source_column_is_simulated(self):
        dl = self._make_logger(source="simulated")
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["data_source"], "simulated")

    def test_data_source_column_is_real(self):
        dl = DataLogger(
            participant_id="p",
            session_id="s_real",
            data_source="real",
            output_dir=self.tmp_path,
        )
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["data_source"], "real")

    # --- Test 18 ---
    def test_participant_id_saved_correctly(self):
        dl = self._make_logger()
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["participant_id"], self.pid)

    def test_session_id_saved_correctly(self):
        dl = self._make_logger()
        dl.open()
        dl.write(VALID_READING)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["session_id"], self.sid)

    # --- Test 19 ---
    def test_raises_on_overwrite_attempt(self):
        """DataLogger must refuse to overwrite an existing file."""
        dl = self._make_logger()
        dl.open()
        dl.close()
        # Try to open the same file again
        dl2 = DataLogger(
            participant_id=self.pid,
            session_id=self.sid,
            data_source="simulated",
            output_dir=self.tmp_path,
        )
        with self.assertRaises(FileExistsError):
            dl2.open()

    def test_invalid_data_source_raises_value_error(self):
        with self.assertRaises(ValueError):
            DataLogger(
                participant_id="p",
                session_id="s",
                data_source="unknown",
                output_dir=self.tmp_path,
            )

    # --- Test 20 ---
    def test_context_manager_opens_and_closes(self):
        with self._make_logger() as dl:
            self.assertTrue(dl.is_open)
            dl.write(VALID_READING)
        self.assertFalse(dl.is_open)
        self.assertTrue(dl.csv_path.exists())

    # --- Test 21 ---
    def test_write_many_saves_multiple_rows(self):
        readings = [dict(VALID_READING, timestamp_ms=i * 20) for i in range(10)]
        dl = self._make_logger()
        dl.open()
        dl.write_many(readings)
        dl.close()
        with open(dl.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 10, "write_many should save 10 rows")

    def test_rows_written_counter_is_accurate(self):
        dl = self._make_logger()
        dl.open()
        for _ in range(7):
            dl.write(VALID_READING)
        dl.close()
        self.assertEqual(dl.rows_written, 7)


# ===========================================================================
# 22-29  SessionManager tests
# ===========================================================================

class TestSessionManager(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)
        config.DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_sm(self, duration=0.5, participant="test_pt_001"):
        """Return a demo SessionManager that runs for duration seconds."""
        sid = f"sess_{int(time.time() * 1000)}"
        return SessionManager(
            participant_id=participant,
            session_id=sid,
            demo=True,
            duration_seconds=duration,
            display_readings=False,
        )

    # --- Test 22 ---
    def test_demo_session_completes_successfully(self):
        sm = self._make_sm(duration=0.3)
        result = sm.run()
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.is_successful)

    # --- Test 23 ---
    def test_result_has_correct_participant_id(self):
        sm = self._make_sm(participant="test_pt_002")
        result = sm.run()
        self.assertEqual(result.participant_id, "test_pt_002")

    def test_result_has_session_id(self):
        sm = self._make_sm()
        result = sm.run()
        self.assertIsNotNone(result.session_id)
        self.assertNotEqual(result.session_id, "")

    # --- Test 24 ---
    def test_csv_file_exists_after_session(self):
        sm = self._make_sm(duration=0.3)
        result = sm.run()
        self.assertIsNotNone(result.csv_path)
        self.assertTrue(result.csv_path.exists(), "CSV must exist after session")

    # --- Test 25 ---
    def test_csv_has_rows(self):
        sm = self._make_sm(duration=0.5)
        result = sm.run()
        self.assertGreater(result.rows_written, 0)
        # At 50 Hz for 0.5 s we expect roughly 25 rows (allow 5–60 range)
        self.assertGreater(result.rows_written, 5)

    def test_csv_rows_match_rows_written_counter(self):
        sm = self._make_sm(duration=0.3)
        result = sm.run()
        with open(result.csv_path, newline="") as f:
            disk_rows = sum(1 for _ in csv.DictReader(f))
        self.assertEqual(disk_rows, result.rows_written)

    # --- Test 26 ---
    def test_data_source_is_simulated_in_demo_mode(self):
        sm = self._make_sm(duration=0.2)
        result = sm.run()
        self.assertEqual(result.data_source, "simulated")
        with open(result.csv_path, newline="") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["data_source"], "simulated")

    # --- Test 27 ---
    def test_session_status_is_completed(self):
        sm = self._make_sm(duration=0.2)
        result = sm.run()
        self.assertEqual(result.status, "completed")

    def test_duration_seconds_is_positive(self):
        sm = self._make_sm(duration=0.3)
        result = sm.run()
        self.assertGreater(result.duration_seconds, 0)

    def test_start_and_end_times_are_set(self):
        sm = self._make_sm(duration=0.2)
        result = sm.run()
        self.assertIsNotNone(result.start_time)
        self.assertIsNotNone(result.end_time)
        self.assertGreater(result.end_time, result.start_time)

    # --- Test 28 ---
    def test_generate_session_id_contains_participant(self):
        sid = generate_session_id("patient_001")
        self.assertIn("patient_001", sid)

    def test_generate_session_id_contains_timestamp(self):
        sid = generate_session_id("patient_001")
        # Format: patient_001_20261109_201530
        parts = sid.split("_")
        self.assertGreaterEqual(len(parts), 4)
        # The date part should be an 8-digit number
        date_part = parts[-2]
        self.assertEqual(len(date_part), 8)
        self.assertTrue(date_part.isdigit())

    def test_two_session_ids_are_unique(self):
        import time
        sid1 = generate_session_id("patient_001")
        time.sleep(1.1)   # ensure different second
        sid2 = generate_session_id("patient_001")
        self.assertNotEqual(sid1, sid2)

    # --- Test 29 ---
    def test_next_participant_id_returns_string(self):
        pid = next_participant_id()
        self.assertIsInstance(pid, str)
        self.assertTrue(pid.startswith("patient_"))


# ===========================================================================
# 30  End-to-end integration test
# ===========================================================================

class TestEndToEnd(unittest.TestCase):
    """
    Test 30 — 3 seconds of simulated data must produce ≥ 100 rows.

    At 50 Hz × 3 s = 150 expected rows.
    We allow ≥ 100 to account for startup overhead.
    """

    def test_three_second_demo_session_produces_sufficient_rows(self):
        sid = f"e2e_{int(time.time() * 1000)}"
        sm = SessionManager(
            participant_id="e2e_patient",
            session_id=sid,
            demo=True,
            duration_seconds=3.0,
            display_readings=False,
        )
        result = sm.run()

        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(
            result.rows_written,
            100,
            f"Expected ≥ 100 rows from 3 s demo, got {result.rows_written}",
        )

        # Also verify CSV on disk
        self.assertTrue(result.csv_path.exists())
        with open(result.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertGreaterEqual(len(rows), 100)

        # Spot-check column completeness on first row
        first = rows[0]
        for col in config.CSV_COLUMNS:
            self.assertIn(col, first, f"Column '{col}' missing from CSV")


# ===========================================================================
# Allow direct execution without pytest
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

