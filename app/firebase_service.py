"""
firebase_service.py — Firebase Firestore integration for the OA Screening System.

Simple explanation
------------------
Firebase Firestore is Google's cloud database.
Data is stored in "documents" grouped into "collections" —
think of a collection as a drawer, and each document as a labelled folder.

This module is the ONLY place in the project that talks to Firebase.
Everything stored in the cloud passes through here.

Data flow (as required):
    ESP32 → USB → Python → THIS MODULE → Firebase Firestore

The ESP32 never connects to Firebase directly.

Firestore structure
-------------------
  patients/
    patient_001/  → { patient_id, age, created_at }
    patient_002/  → { patient_id, age, created_at }

  sessions/
    session_id/   → {
        session_id, patient_id, test_type, duration_seconds,
        data_source, status, created_at,
        features:   { heel_peak, midfoot_peak, ... },
        assessment: { prediction, model_version, model_score, validated }
    }

Usage (once credentials are configured)
-----------------------------------------
    from app.firebase_service import FirebaseService

    svc = FirebaseService()
    initialize_firebase(svc)                 # connect
    svc.create_patient("patient_001", age=52)
    svc.create_session("session_001", "patient_001", duration_seconds=30)
    svc.save_session_features("session_001", features_dict)
    data = svc.get_session("session_001")

Required functions (12)
-----------------------
  1.  initialize_firebase()
  2.  test_firebase_connection()
  3.  create_patient()
  4.  get_patient()
  5.  list_patients()
  6.  create_session()
  7.  update_session()
  8.  save_session_features()
  9.  save_assessment_result()
  10. get_session()
  11. get_patient_sessions()
  12. list_recent_sessions()
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import firebase_admin — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as _fs
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    logger.warning(
        "firebase-admin is not installed — Firebase is disabled.\n"
        "Install it with:  pip install firebase-admin"
    )

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 3          # maximum attempts per Firestore operation
_RETRY_DELAY_SECONDS: float = 1.5   # wait between retries


# ===========================================================================
# Internal retry helper
# ===========================================================================

def _with_retry(operation_name: str, fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) up to _MAX_RETRIES times.

    If every attempt raises an exception, the last exception is logged
    and None / False is returned (depending on what the caller expects).

    Simple explanation: if the cloud database is briefly unavailable,
    the code tries again a few times before giving up.
    """
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s — retrying in %.1fs…",
                    operation_name, attempt, _MAX_RETRIES, exc, _RETRY_DELAY_SECONDS,
                )
                time.sleep(_RETRY_DELAY_SECONDS)
            else:
                logger.error(
                    "[%s] All %d attempts failed. Last error: %s",
                    operation_name, _MAX_RETRIES, exc,
                )
    return None   # signals failure to the caller


# ===========================================================================
# FirebaseService class
# ===========================================================================

class FirebaseService:
    """
    All Firebase Firestore operations for the OA Screening System.

    Instantiation does NOT connect immediately.
    Call initialize_firebase(this_instance) or svc.connect() first.

    Every public method returns a safe default (False, None, [])
    when Firebase is unavailable — the application keeps running.
    """

    def __init__(self):
        self._db = None           # Firestore client, set by connect()
        self._connected: bool = False
        self._app = None          # firebase_admin App handle

    # ======================================================================
    # 1. initialize_firebase  — connects this instance to Firestore
    # ======================================================================

    def connect(self) -> bool:
        """
        Initialise the Firebase Admin SDK and open a Firestore connection.

        Reads the service account JSON path from config.FIREBASE_CREDENTIALS_PATH
        (which is loaded from the .env file — credentials are NEVER hardcoded).

        Returns True on success, False on any failure.
        """
        if self._connected:
            return True

        if not _FIREBASE_AVAILABLE:
            logger.error(
                "firebase-admin is not installed.\n"
                "Run:  pip install firebase-admin"
            )
            return False

        cred_path: Path = config.FIREBASE_CREDENTIALS_PATH

        if not cred_path.exists():
            logger.error(
                "Firebase credentials file not found at: %s\n\n"
                "To fix this:\n"
                "  1. Open https://console.firebase.google.com\n"
                "  2. Select your project  →  Project Settings\n"
                "  3. Service Accounts  →  'Generate new private key'\n"
                "  4. Save the downloaded JSON as:  %s\n"
                "  5. The file is listed in .gitignore — it will NOT be "
                "committed to GitHub.",
                cred_path, cred_path,
            )
            return False

        try:
            # Only initialise once per Python process
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(cred_path))
                self._app = firebase_admin.initialize_app(cred)
                logger.debug("Firebase app initialised from credentials.")
            else:
                self._app = firebase_admin.get_app()
                logger.debug("Re-using existing Firebase app.")

            self._db = _fs.client()
            self._connected = True
            logger.info(
                "Firebase Firestore connected. Project: %s",
                config.FIREBASE_PROJECT_ID or "(from credentials file)",
            )
            return True

        except Exception as exc:
            logger.error("Firebase initialisation failed: %s", exc)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """True if Firestore is ready to use."""
        return self._connected and self._db is not None

    def _require_connection(self) -> bool:
        """Auto-connect on first use. Returns True if connected."""
        if self.is_connected:
            return True
        return self.connect()

    # ======================================================================
    # 2. test_firebase_connection  — round-trip verification
    # ======================================================================

    def test_firebase_connection(self) -> bool:
        """
        Write a test document, read it back, then delete it.

        Use this during setup to confirm Firestore is reachable.
        Returns True if the round-trip succeeds.

        Simple explanation: it is like sending a letter to yourself
        to confirm the postal service is working.
        """
        if not self._require_connection():
            return False

        test_collection = "_connection_tests"
        test_doc_id = f"ping_{int(time.time())}"

        def _do_test():
            ref = self._db.collection(test_collection).document(test_doc_id)
            ref.set({"ping": True, "ts": _fs.SERVER_TIMESTAMP})
            doc = ref.get()
            ok = doc.exists and doc.to_dict().get("ping") is True
            ref.delete()
            return ok

        result = _with_retry("test_firebase_connection", _do_test)
        if result:
            logger.info("Firebase connection test PASSED ✓")
        else:
            logger.error("Firebase connection test FAILED.")
        return bool(result)

    # ======================================================================
    # 3. create_patient
    # ======================================================================

    def create_patient(
        self,
        patient_id: str,
        age: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Create a patient document in Firestore.

        Only pseudonymous research data is stored (no names or addresses).
        If the patient already exists, the existing record is kept unchanged.

        Parameters
        ----------
        patient_id : Pseudonymous ID, e.g. "patient_001".
        age        : Age in years (optional).
        notes      : Free-text research notes (optional).

        Returns True on success (including if the patient already existed).
        """
        if not self._require_connection():
            return False

        def _do():
            ref = self._db.collection(config.PATIENTS_COLLECTION).document(patient_id)
            existing = ref.get()

            if existing.exists:
                logger.info("Patient '%s' already exists — no changes made.", patient_id)
                return True

            data: Dict[str, Any] = {
                "patient_id": patient_id,
                "created_at": _fs.SERVER_TIMESTAMP,
            }
            if age is not None:
                data["age"] = age
            if notes is not None:
                data["notes"] = notes

            ref.set(data)
            logger.info("Patient '%s' created in Firestore.", patient_id)
            return True

        result = _with_retry("create_patient", _do)
        return bool(result)

    # ======================================================================
    # 4. get_patient
    # ======================================================================

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a patient document by ID.

        Returns the data as a Python dict, or None if not found.
        """
        if not self._require_connection():
            return None

        def _do():
            doc = (
                self._db.collection(config.PATIENTS_COLLECTION)
                .document(patient_id)
                .get()
            )
            if doc.exists:
                return doc.to_dict()
            logger.debug("Patient '%s' not found in Firestore.", patient_id)
            return None

        return _with_retry("get_patient", _do)

    # ======================================================================
    # 5. list_patients
    # ======================================================================

    def list_patients(self) -> List[Dict[str, Any]]:
        """
        Return all patient documents from Firestore as a list of dicts.
        Returns an empty list if Firebase is unavailable.
        """
        if not self._require_connection():
            return []

        def _do():
            docs = self._db.collection(config.PATIENTS_COLLECTION).stream()
            return [d.to_dict() for d in docs]

        result = _with_retry("list_patients", _do)
        return result if isinstance(result, list) else []

    # ======================================================================
    # 6. create_session
    # ======================================================================

    def create_session(
        self,
        session_id: str,
        patient_id: str,
        test_type: str = "normal_walking",
        duration_seconds: float = 0.0,
        data_source: str = "real",
        status: str = "in_progress",
        csv_filename: Optional[str] = None,
    ) -> bool:
        """
        Create a session document in Firestore.

        Call this at the START of recording so a record exists even if
        the session fails partway through.

        Parameters
        ----------
        session_id       : Unique ID, e.g. "patient_001_20261109_201530".
        patient_id       : Links this session to a patient document.
        test_type        : Description of the walking task.
        duration_seconds : Will be updated when the session ends.
        data_source      : "real" (hardware) or "simulated" (demo).
        status           : "in_progress" at start; updated to "completed"
                           or "failed" afterwards.
        csv_filename     : Name of the local raw CSV file.

        Returns True on success.
        """
        if not self._require_connection():
            return False

        def _do():
            ref = (
                self._db.collection(config.SESSIONS_COLLECTION)
                .document(session_id)
            )
            # Prevent accidental duplicate creation
            if ref.get().exists:
                logger.warning(
                    "Session '%s' already exists in Firestore — skipping create.",
                    session_id,
                )
                return True

            data: Dict[str, Any] = {
                "session_id":       session_id,
                "patient_id":       patient_id,
                "test_type":        test_type,
                "duration_seconds": duration_seconds,
                "data_source":      data_source,
                "status":           status,
                "created_at":       _fs.SERVER_TIMESTAMP,
                "features":         None,
                "assessment": {
                    "prediction":               "not_evaluated",
                    "model_version":            None,
                    "model_score":              None,
                    "validated":                False,
                    "clinical_validation_status": "not_validated",
                },
            }
            if csv_filename:
                data["csv_filename"] = csv_filename

            ref.set(data)
            logger.info("Session '%s' created in Firestore.", session_id)
            return True

        result = _with_retry("create_session", _do)
        return bool(result)

    # ======================================================================
    # 7. update_session
    # ======================================================================

    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update specific fields in an existing session document.

        Example:
            svc.update_session("session_001", {
                "status": "completed",
                "duration_seconds": 32.5,
            })

        Parameters
        ----------
        session_id : Session to update.
        updates    : Dict of field names → new values.

        Returns True on success.
        """
        if not self._require_connection():
            return False

        def _do():
            (
                self._db.collection(config.SESSIONS_COLLECTION)
                .document(session_id)
                .update(updates)
            )
            logger.info(
                "Session '%s' updated: %s", session_id, list(updates.keys())
            )
            return True

        result = _with_retry("update_session", _do)
        return bool(result)

    # ======================================================================
    # 8. save_session_features
    # ======================================================================

    def save_session_features(
        self,
        session_id: str,
        features: Dict[str, Any],
    ) -> bool:
        """
        Store the extracted gait feature dictionary in a session document.

        Features are stored as a nested dict inside the session document —
        NOT as a separate collection or as individual sensor readings.

        Example features dict:
            {
                "heel_peak": 820,
                "midfoot_peak": 650,
                "forefoot_peak": 900,
                "acceleration_variability": 0.35,
                "gyro_variability": 0.24,
                ...
            }

        These values are computed by feature_extractor.py from the raw CSV.

        Parameters
        ----------
        session_id : Session to attach features to.
        features   : Feature dictionary from FeatureExtractor.

        Returns True on success.
        """
        if not self._require_connection():
            return False

        if not features:
            logger.warning("save_session_features: empty features dict for '%s'.", session_id)
            return False

        def _do():
            (
                self._db.collection(config.SESSIONS_COLLECTION)
                .document(session_id)
                .update({
                    "features":    features,
                    "features_at": _fs.SERVER_TIMESTAMP,
                })
            )
            logger.info(
                "Features saved for session '%s' (%d features).",
                session_id, len(features),
            )
            return True

        result = _with_retry("save_session_features", _do)
        return bool(result)

    # ======================================================================
    # 9. save_assessment_result
    # ======================================================================

    def save_assessment_result(
        self,
        session_id: str,
        prediction: str,
        model_version: Optional[str] = None,
        model_score: Optional[float] = None,
        validated: bool = False,
        clinical_validation_status: str = "not_validated",
    ) -> bool:
        """
        Store the ML model assessment result in a session document.

        IMPORTANT — Medical safety:
        This stores an EXPERIMENTAL gait-pattern classification only.
        It is NOT a clinical diagnosis of osteoarthritis.
        The 'validated' flag must remain False unless a qualified
        clinician has formally reviewed and approved the model outputs.

        Parameters
        ----------
        session_id                 : Session to attach the result to.
        prediction                 : Model output label, e.g. "pattern_A"
                                     or "not_evaluated".
        model_version              : Version string of the saved model file.
        model_score                : Raw model output score / probability.
                                     Do not describe this as a confidence
                                     percentage unless calibration has been
                                     performed and documented.
        validated                  : Always False in this prototype.
        clinical_validation_status : Always "not_validated" in this prototype.

        Returns True on success.
        """
        if not self._require_connection():
            return False

        def _do():
            (
                self._db.collection(config.SESSIONS_COLLECTION)
                .document(session_id)
                .update({
                    "assessment": {
                        "prediction":                prediction,
                        "model_version":             model_version,
                        "model_score":               model_score,
                        "validated":                 validated,
                        "clinical_validation_status": clinical_validation_status,
                    },
                    "assessment_at": _fs.SERVER_TIMESTAMP,
                })
            )
            logger.info("Assessment result saved for session '%s'.", session_id)
            return True

        result = _with_retry("save_assessment_result", _do)
        return bool(result)

    # ======================================================================
    # 10. get_session
    # ======================================================================

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session document by ID.

        Returns the session as a Python dict, or None if not found.
        """
        if not self._require_connection():
            return None

        def _do():
            doc = (
                self._db.collection(config.SESSIONS_COLLECTION)
                .document(session_id)
                .get()
            )
            return doc.to_dict() if doc.exists else None

        return _with_retry("get_session", _do)

    # ======================================================================
    # 11. get_patient_sessions
    # ======================================================================

    def get_patient_sessions(
        self,
        patient_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Return all sessions for one participant, newest first.

        Parameters
        ----------
        patient_id : The patient whose sessions to retrieve.
        limit      : Maximum number of sessions to return.

        Returns a list of session dicts (may be empty).
        """
        if not self._require_connection():
            return []

        def _do():
            docs = (
                self._db.collection(config.SESSIONS_COLLECTION)
                .where("patient_id", "==", patient_id)
                .limit(limit)
                .stream()
            )
            sessions = [d.to_dict() for d in docs]
            # Sort by created_at descending (newest first) where possible
            sessions.sort(
                key=lambda s: s.get("created_at") or datetime.min,
                reverse=True,
            )
            return sessions

        result = _with_retry("get_patient_sessions", _do)
        return result if isinstance(result, list) else []

    # ======================================================================
    # 12. list_recent_sessions
    # ======================================================================

    def list_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Return the most recent sessions across all participants.

        Parameters
        ----------
        limit : Maximum number of sessions to return (default 20).

        Returns a list of session dicts ordered by creation time (newest first).
        """
        if not self._require_connection():
            return []

        def _do():
            docs = (
                self._db.collection(config.SESSIONS_COLLECTION)
                .order_by("created_at", direction=_fs.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            return [d.to_dict() for d in docs]

        result = _with_retry("list_recent_sessions", _do)
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Aliases — make the spec function names work as well
    # ------------------------------------------------------------------

    def upload_features(
        self, session_id: str, features: Dict[str, Any]
    ) -> bool:
        """Alias for save_session_features (backward compatibility)."""
        return self.save_session_features(session_id, features)

    def upload_assessment(
        self,
        session_id: str,
        prediction: str,
        model_version: Optional[str] = None,
        model_score: Optional[float] = None,
        clinical_validation_status: str = "not_validated",
    ) -> bool:
        """Alias for save_assessment_result (backward compatibility)."""
        return self.save_assessment_result(
            session_id=session_id,
            prediction=prediction,
            model_version=model_version,
            model_score=model_score,
            clinical_validation_status=clinical_validation_status,
        )

    def list_sessions(
        self,
        patient_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Alias that routes to get_patient_sessions or list_recent_sessions."""
        if patient_id:
            return self.get_patient_sessions(patient_id, limit=limit)
        return self.list_recent_sessions(limit=limit)

    # ======================================================================
    # Daily Logs, Baselines, and AI Insights
    # ======================================================================

    def save_daily_log(self, log_dict: Dict[str, Any]) -> bool:
        """Stores a daily patient log (activity, nutrition, symptoms) in daily_logs."""
        if not self._require_connection():
            return False
        patient_id = log_dict.get("patient_id")
        log_date = log_dict.get("date")
        if not patient_id or not log_date:
            return False

        def _do():
            doc_id = f"{patient_id}_{log_date}"
            self._db.collection(config.DAILY_LOGS_COLLECTION).document(doc_id).set(log_dict)
            logger.info("Saved daily log to Firestore: %s", doc_id)
            return True

        result = _with_retry("save_daily_log", _do)
        return bool(result)

    def get_daily_log(self, patient_id: str, log_date: str) -> Optional[Dict[str, Any]]:
        """Retrieves a daily log document from daily_logs."""
        if not self._require_connection():
            return None

        def _do():
            doc_id = f"{patient_id}_{log_date}"
            doc = self._db.collection(config.DAILY_LOGS_COLLECTION).document(doc_id).get()
            return doc.to_dict() if doc.exists else None

        return _with_retry("get_daily_log", _do)

    def save_patient_baseline(self, patient_id: str, baseline_dict: Dict[str, Any]) -> bool:
        """Stores or updates a patient baseline document in patient_baselines."""
        if not self._require_connection():
            return False

        def _do():
            self._db.collection(config.BASELINES_COLLECTION).document(patient_id).set(baseline_dict)
            logger.info("Saved baseline to Firestore for: %s", patient_id)
            return True

        result = _with_retry("save_patient_baseline", _do)
        return bool(result)

    def get_patient_baseline(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a patient baseline from patient_baselines."""
        if not self._require_connection():
            return None

        def _do():
            doc = self._db.collection(config.BASELINES_COLLECTION).document(patient_id).get()
            return doc.to_dict() if doc.exists else None

        return _with_retry("get_patient_baseline", _do)

    def save_ai_insight(self, insight_dict: Dict[str, Any]) -> bool:
        """Stores an AI-generated monitoring insight in ai_insights."""
        if not self._require_connection():
            return False
        patient_id = insight_dict.get("patient_id")
        date_str = datetime.now().strftime("%Y-%m-%d")

        def _do():
            doc_id = f"{patient_id}_{date_str}"
            self._db.collection(config.INSIGHTS_COLLECTION).document(doc_id).set(insight_dict)
            logger.info("Saved AI Insight to Firestore: %s", doc_id)
            return True

        result = _with_retry("save_ai_insight", _do)
        return bool(result)

    def get_ai_insights(self, patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves historical AI insights for a patient."""
        if not self._require_connection():
            return []

        def _do():
            docs = (
                self._db.collection(config.INSIGHTS_COLLECTION)
                .where("patient_id", "==", patient_id)
                .limit(limit)
                .stream()
            )
            return [d.to_dict() for d in docs]

        result = _with_retry("get_ai_insights", _do)
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"FirebaseService(connected={self.is_connected})"


# ===========================================================================
# Module-level function 1 — initialize_firebase
# ===========================================================================

def initialize_firebase(svc: Optional[FirebaseService] = None) -> FirebaseService:
    """
    Initialise and return a connected FirebaseService instance.

    This is the **primary entry point** for all modules that need Firebase.

    Parameters
    ----------
    svc : FirebaseService | None
        Pass an existing instance to reconnect it.
        If None, a new instance is created.

    Returns
    -------
    FirebaseService
        A connected instance. Check .is_connected before using it.

    Example
    -------
        from app.firebase_service import initialize_firebase

        svc = initialize_firebase()
        if svc.is_connected:
            svc.create_patient("patient_001", age=52)
    """
    if svc is None:
        svc = FirebaseService()
    svc.connect()
    return svc


# ===========================================================================
# Module-level function 2 — test_firebase_connection (standalone)
# ===========================================================================

def test_firebase_connection() -> bool:
    """
    Quick standalone connection test.

    Creates, reads and deletes a test document in Firestore.
    Returns True if the round-trip succeeds.

    Use this to verify your credentials are correct before collecting data.

    Example:
        python -c "from app.firebase_service import test_firebase_connection; print(test_firebase_connection())"
    """
    svc = initialize_firebase()
    return svc.test_firebase_connection()


# ===========================================================================
# Convenience singleton — the dashboard and session manager import this
# ===========================================================================

def get_firebase_service() -> FirebaseService:
    """
    Return a connected FirebaseService.
    Callers must check .is_connected before performing operations.
    """
    return initialize_firebase()
