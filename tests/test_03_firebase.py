"""
tests/test_03_firebase.py
==========================
Tests for Prompt 3 — Firebase Firestore Integration.

Test groups
-----------
  Group A — Module structure (always runs)
     1.  firebase_service.py imports cleanly
     2.  FirebaseService can be instantiated
     3.  All 12 required functions exist
     4.  initialize_firebase() returns a FirebaseService
     5.  test_firebase_connection() function exists at module level
     6.  get_firebase_service() function exists at module level

  Group B — Offline / no-credentials tests (always runs)
     7.  connect() returns False when credentials file is missing
     8.  connect() returns False when firebase-admin is not installed
     9.  is_connected is False before connecting
     10. create_patient() returns False when not connected
     11. get_patient() returns None when not connected
     12. list_patients() returns [] when not connected
     13. create_session() returns False when not connected
     14. update_session() returns False when not connected
     15. get_session() returns None when not connected
     16. get_patient_sessions() returns [] when not connected
     17. list_recent_sessions() returns [] when not connected
     18. save_session_features() returns False when not connected
     19. save_assessment_result() returns False when not connected
     20. test_firebase_connection() returns False when not connected

  Group C — Live Firestore integration (SKIPPED if no credentials)
     21. initialize_firebase() connects successfully
     22. test_firebase_connection() round-trip passes
     23. create_patient() writes a document
     24. get_patient() reads the document back
     25. patient data matches what was written
     26. create_patient() is idempotent (calling twice is safe)
     27. list_patients() includes the test patient
     28. create_session() writes a session document
     29. get_session() reads the session back
     30. session data matches what was written
     31. update_session() changes a field
     32. updated field is reflected in get_session()
     33. save_session_features() writes features
     34. get_session() shows saved features
     35. save_assessment_result() writes assessment
     36. get_session() shows saved assessment
     37. get_patient_sessions() returns the test session
     38. list_recent_sessions() includes the test session
     39. Cleanup: test documents removed from Firestore

Run offline tests only:
    python tests/test_03_firebase.py

Run all tests including live Firestore (requires credentials):
    FIREBASE_CREDENTIALS_PATH=secrets/firebase-key.json \\
    python tests/test_03_firebase.py --live
"""

import sys
import os
import time
import unittest
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
import app.firebase_service as _fb_module
from app.firebase_service import (
    FirebaseService,
    initialize_firebase,
    test_firebase_connection,
    get_firebase_service,
)

# ---------------------------------------------------------------------------
# Determine whether live tests should run
# ---------------------------------------------------------------------------
_CREDENTIALS_EXIST = config.FIREBASE_CREDENTIALS_PATH.exists()
_RUN_LIVE = "--live" in sys.argv and _CREDENTIALS_EXIST

# Test-specific patient and session IDs (deleted after the test run)
_TEST_PATIENT_ID = "test_patient_firebase_pytest"
_TEST_SESSION_ID = f"test_session_firebase_{int(time.time())}"
_TEST_FEATURES = {
    "heel_peak":               820,
    "midfoot_peak":            650,
    "forefoot_peak":           900,
    "heel_mean":               340,
    "midfoot_mean":            280,
    "forefoot_mean":           310,
    "acceleration_variability": 0.35,
    "gyro_variability":        0.24,
    "measurement_count":       1500,
    "sampling_frequency_hz":   50,
}
_TEST_ASSESSMENT = {
    "prediction":               "pattern_A",
    "model_version":            "rf_v0.1_test",
    "model_score":              0.72,
    "validated":                False,
    "clinical_validation_status": "not_validated",
}


# ===========================================================================
# GROUP A — Module structure (no network needed)
# ===========================================================================

class TestModuleStructure(unittest.TestCase):
    """Verify that all 12 required functions exist on the class."""

    REQUIRED_METHODS = [
        # Names from the prompt
        "connect",                # maps to initialize_firebase()
        "test_firebase_connection",
        "create_patient",
        "get_patient",
        "list_patients",
        "create_session",
        "update_session",
        "save_session_features",
        "save_assessment_result",
        "get_session",
        "get_patient_sessions",
        "list_recent_sessions",
    ]

    # Test 1
    def test_module_imports_cleanly(self):
        mod = importlib.import_module("app.firebase_service")
        self.assertIsNotNone(mod)

    # Test 2
    def test_firebase_service_instantiates(self):
        svc = FirebaseService()
        self.assertIsInstance(svc, FirebaseService)

    # Test 3
    def test_all_12_required_methods_exist(self):
        svc = FirebaseService()
        missing = [m for m in self.REQUIRED_METHODS if not hasattr(svc, m)]
        self.assertFalse(
            missing,
            f"FirebaseService is missing required methods: {missing}",
        )

    # Test 4
    def test_initialize_firebase_returns_firebase_service(self):
        result = initialize_firebase()
        self.assertIsInstance(result, FirebaseService)

    # Test 5
    def test_module_level_test_firebase_connection_exists(self):
        self.assertTrue(callable(test_firebase_connection))

    # Test 6
    def test_module_level_get_firebase_service_exists(self):
        self.assertTrue(callable(get_firebase_service))


# ===========================================================================
# GROUP B — Offline / no-credentials tests
# ===========================================================================

class TestOfflineBehaviour(unittest.TestCase):
    """
    These tests verify the FirebaseService handles missing credentials,
    missing packages and disconnected states gracefully.

    They run entirely offline — no network required.
    """

    # Test 7
    def test_connect_returns_false_when_credentials_missing(self):
        svc = FirebaseService()
        # Point credentials to a non-existent file
        with patch.object(config, "FIREBASE_CREDENTIALS_PATH",
                          Path("/tmp/does_not_exist_credentials.json")):
            result = svc.connect()
        self.assertFalse(result)

    # Test 8
    def test_connect_returns_false_when_firebase_not_installed(self):
        svc = FirebaseService()
        with patch.dict(sys.modules, {"firebase_admin": None}):
            with patch.object(
                _fb_module, "_FIREBASE_AVAILABLE", False
            ):
                result = svc.connect()
        self.assertFalse(result)

    # Test 9
    def test_is_connected_false_before_connect(self):
        svc = FirebaseService()
        self.assertFalse(svc.is_connected)

    def _disconnected_svc(self):
        """Return a FirebaseService that cannot connect."""
        svc = FirebaseService()
        # Patch so connect() always returns False
        svc.connect = lambda: False
        svc._connected = False
        svc._db = None
        return svc

    # Test 10
    def test_create_patient_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.create_patient("patient_offline_test", age=40)
        self.assertFalse(result)

    # Test 11
    def test_get_patient_returns_none_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.get_patient("patient_offline_test")
        self.assertIsNone(result)

    # Test 12
    def test_list_patients_returns_empty_list_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.list_patients()
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    # Test 13
    def test_create_session_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.create_session("s_offline", "p_offline")
        self.assertFalse(result)

    # Test 14
    def test_update_session_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.update_session("s_offline", {"status": "completed"})
        self.assertFalse(result)

    # Test 15
    def test_get_session_returns_none_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.get_session("s_offline")
        self.assertIsNone(result)

    # Test 16
    def test_get_patient_sessions_returns_empty_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.get_patient_sessions("p_offline")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    # Test 17
    def test_list_recent_sessions_returns_empty_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.list_recent_sessions()
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    # Test 18
    def test_save_session_features_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.save_session_features("s_offline", {"heel_peak": 500})
        self.assertFalse(result)

    # Test 19
    def test_save_assessment_result_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.save_assessment_result("s_offline", "not_evaluated")
        self.assertFalse(result)

    # Test 20
    def test_method_test_firebase_connection_returns_false_when_disconnected(self):
        svc = self._disconnected_svc()
        result = svc.test_firebase_connection()
        self.assertFalse(result)

    def test_save_session_features_rejects_empty_dict(self):
        svc = self._disconnected_svc()
        result = svc.save_session_features("s", {})
        self.assertFalse(result)

    def test_repr_shows_connected_false(self):
        svc = FirebaseService()
        self.assertIn("False", repr(svc))


# ===========================================================================
# GROUP C — Live Firestore integration
# ===========================================================================

@unittest.skipUnless(_RUN_LIVE, (
    "Live Firebase tests are SKIPPED (no credentials found or --live not passed).\n"
    "To run live tests:\n"
    "  1. Place your service account JSON at: "
    + str(config.FIREBASE_CREDENTIALS_PATH) + "\n"
    "  2. Run:  python tests/test_03_firebase.py --live"
))
class TestFirebaseLiveIntegration(unittest.TestCase):
    """
    Live round-trip tests against your real Firestore database.

    These create a test patient and session, then delete them afterwards.
    Use a development Firebase project — not production data.
    """

    @classmethod
    def setUpClass(cls):
        """Connect to Firebase once for all live tests in this class."""
        cls.svc = initialize_firebase()
        if not cls.svc.is_connected:
            raise unittest.SkipTest(
                "Could not connect to Firebase — skipping live tests."
            )
        # Clean up any leftover test data from previous runs
        cls._cleanup()

    @classmethod
    def tearDownClass(cls):
        """Remove all test documents from Firestore."""
        cls._cleanup()

    @classmethod
    def _cleanup(cls):
        """Delete test patient and session from Firestore."""
        try:
            if hasattr(cls, "svc") and cls.svc.is_connected:
                db = cls.svc._db
                db.collection(config.PATIENTS_COLLECTION).document(_TEST_PATIENT_ID).delete()
                db.collection(config.SESSIONS_COLLECTION).document(_TEST_SESSION_ID).delete()
        except Exception:
            pass

    # Test 21
    def test_21_initialize_firebase_connects(self):
        self.assertTrue(self.svc.is_connected)

    # Test 22
    def test_22_connection_roundtrip_passes(self):
        ok = self.svc.test_firebase_connection()
        self.assertTrue(ok, "Firestore connection round-trip must succeed")

    # Test 23
    def test_23_create_patient_succeeds(self):
        ok = self.svc.create_patient(_TEST_PATIENT_ID, age=52)
        self.assertTrue(ok)

    # Test 24
    def test_24_get_patient_reads_document(self):
        data = self.svc.get_patient(_TEST_PATIENT_ID)
        self.assertIsNotNone(data)

    # Test 25
    def test_25_patient_data_matches(self):
        data = self.svc.get_patient(_TEST_PATIENT_ID)
        self.assertIsNotNone(data)
        self.assertEqual(data["patient_id"], _TEST_PATIENT_ID)
        self.assertEqual(data["age"], 52)

    # Test 26
    def test_26_create_patient_is_idempotent(self):
        ok1 = self.svc.create_patient(_TEST_PATIENT_ID, age=52)
        ok2 = self.svc.create_patient(_TEST_PATIENT_ID, age=52)
        self.assertTrue(ok1)
        self.assertTrue(ok2)

    # Test 27
    def test_27_list_patients_includes_test_patient(self):
        patients = self.svc.list_patients()
        ids = [p.get("patient_id") for p in patients]
        self.assertIn(_TEST_PATIENT_ID, ids)

    # Test 28
    def test_28_create_session_succeeds(self):
        ok = self.svc.create_session(
            session_id=_TEST_SESSION_ID,
            patient_id=_TEST_PATIENT_ID,
            test_type="normal_walking",
            duration_seconds=30.0,
            data_source="simulated",
            status="completed",
            csv_filename=f"{_TEST_PATIENT_ID}_{_TEST_SESSION_ID}.csv",
        )
        self.assertTrue(ok)

    # Test 29
    def test_29_get_session_reads_document(self):
        data = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNotNone(data)

    # Test 30
    def test_30_session_data_matches(self):
        data = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNotNone(data)
        self.assertEqual(data["session_id"], _TEST_SESSION_ID)
        self.assertEqual(data["patient_id"], _TEST_PATIENT_ID)
        self.assertEqual(data["test_type"], "normal_walking")
        self.assertEqual(data["data_source"], "simulated")
        self.assertEqual(data["status"], "completed")

    # Test 31
    def test_31_update_session_changes_a_field(self):
        ok = self.svc.update_session(
            _TEST_SESSION_ID,
            {"duration_seconds": 45.0, "status": "completed"},
        )
        self.assertTrue(ok)

    # Test 32
    def test_32_updated_field_is_reflected(self):
        data = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNotNone(data)
        self.assertAlmostEqual(data.get("duration_seconds", 0), 45.0, places=1)

    # Test 33
    def test_33_save_session_features_succeeds(self):
        ok = self.svc.save_session_features(_TEST_SESSION_ID, _TEST_FEATURES)
        self.assertTrue(ok)

    # Test 34
    def test_34_features_appear_in_get_session(self):
        data = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNotNone(data)
        features = data.get("features")
        self.assertIsNotNone(features)
        self.assertEqual(features.get("heel_peak"), _TEST_FEATURES["heel_peak"])
        self.assertEqual(features.get("forefoot_peak"), _TEST_FEATURES["forefoot_peak"])

    # Test 35
    def test_35_save_assessment_result_succeeds(self):
        ok = self.svc.save_assessment_result(
            session_id=_TEST_SESSION_ID,
            prediction=_TEST_ASSESSMENT["prediction"],
            model_version=_TEST_ASSESSMENT["model_version"],
            model_score=_TEST_ASSESSMENT["model_score"],
            validated=_TEST_ASSESSMENT["validated"],
        )
        self.assertTrue(ok)

    # Test 36
    def test_36_assessment_appears_in_get_session(self):
        data = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNotNone(data)
        assessment = data.get("assessment")
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.get("prediction"), "pattern_A")
        self.assertFalse(assessment.get("validated"))
        self.assertEqual(
            assessment.get("clinical_validation_status"), "not_validated"
        )

    # Test 37
    def test_37_get_patient_sessions_returns_test_session(self):
        sessions = self.svc.get_patient_sessions(_TEST_PATIENT_ID)
        ids = [s.get("session_id") for s in sessions]
        self.assertIn(_TEST_SESSION_ID, ids)

    # Test 38
    def test_38_list_recent_sessions_includes_test_session(self):
        sessions = self.svc.list_recent_sessions(limit=50)
        ids = [s.get("session_id") for s in sessions]
        self.assertIn(_TEST_SESSION_ID, ids)

    # Test 39
    def test_39_cleanup_test_documents(self):
        """Remove test documents. Verifies delete works."""
        db = self.svc._db
        db.collection(config.PATIENTS_COLLECTION).document(_TEST_PATIENT_ID).delete()
        db.collection(config.SESSIONS_COLLECTION).document(_TEST_SESSION_ID).delete()

        patient = self.svc.get_patient(_TEST_PATIENT_ID)
        session = self.svc.get_session(_TEST_SESSION_ID)
        self.assertIsNone(patient, "Test patient must be deleted")
        self.assertIsNone(session, "Test session must be deleted")


# ===========================================================================
# Allow direct execution
# ===========================================================================

if __name__ == "__main__":
    # Remove --live from argv so unittest doesn't complain about it
    argv = [a for a in sys.argv if a != "--live"]
    unittest.main(verbosity=2, argv=argv)

