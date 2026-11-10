"""
daily_log_service.py — Service for logging and retrieving daily patient lifestyle and symptom records.

Tracks four essential daily pillars for patients diagnosed with knee osteoarthritis:
  1. Nutrition: Food entries, total daily protein intake (g), personal protein targets.
  2. Physical Activity: Step counts, walking duration, tracking source (manual, insole, wearable).
  3. Symptoms: Validated Pain score (0-10), morning stiffness (minutes), fatigue (0-10), swelling.
  4. Lifestyle: Daily therapeutic exercise completion, notes.

Storage:
  - Saves local structured JSON copies under data/daily_logs/ for offline reliability.
  - Synchronizes to Firebase Firestore collection 'daily_logs' when online.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.config import config
from app.firebase_service import get_firebase_service

logger = logging.getLogger(__name__)


@dataclass
class MealEntry:
    food: str
    protein_g: float
    quantity: str = "1 serving"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "food": self.food,
            "protein_g": round(float(self.protein_g), 1),
            "quantity": self.quantity,
        }


@dataclass
class DailyLog:
    patient_id: str
    date: str  # YYYY-MM-DD
    steps_count: int = 0
    steps_source: str = "manual"  # manual | insole_estimated | wearable
    walking_duration_minutes: float = 0.0
    protein_intake_g: float = 0.0
    protein_target_g: float = 60.0
    pain_score: int = 0  # 0 to 10
    stiffness_minutes: int = 0
    fatigue_score: int = 0  # 0 to 10
    swelling_reported: bool = False
    exercise_completed: bool = False
    meals: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        """Enforces clinical scale limits and data integrity."""
        if not (0 <= self.pain_score <= 10):
            raise ValueError(f"Pain score must be between 0 and 10 (got {self.pain_score}).")
        if not (0 <= self.fatigue_score <= 10):
            raise ValueError(f"Fatigue score must be between 0 and 10 (got {self.fatigue_score}).")
        if self.stiffness_minutes < 0:
            raise ValueError("Stiffness minutes cannot be negative.")
        if self.steps_count < 0:
            raise ValueError("Step count cannot be negative.")
        if self.protein_intake_g < 0:
            raise ValueError("Protein intake cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


class DailyLogService:
    """
    Manages daily lifestyle, nutrition, and symptom logs for OA patients.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or config.DATA_DAILY_LOGS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_local_path(self, patient_id: str, log_date: str) -> Path:
        return self.storage_dir / f"{patient_id}_{log_date}.json"

    def save_log(self, log: DailyLog) -> bool:
        """
        Validates, saves to local disk, and attempts cloud sync to Firestore.
        """
        log.validate()
        # Automatically sum meals if provided and not explicitly set
        if log.meals and log.protein_intake_g == 0.0:
            log.protein_intake_g = sum(m.get("protein_g", 0.0) for m in log.meals)

        data = log.to_dict()

        # 1. Local Persistence (guarantees zero data loss)
        local_file = self._get_local_path(log.patient_id, log.date)
        try:
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved daily log locally: %s", local_file)
        except Exception as exc:
            logger.error("Failed saving daily log to %s: %s", local_file, exc)
            return False

        # 2. Cloud Firestore Persistence
        fb = get_firebase_service()
        if fb.is_connected:
            try:
                doc_id = f"{log.patient_id}_{log.date}"
                fb._db.collection(config.DAILY_LOGS_COLLECTION).document(doc_id).set(data)
                logger.info("Synced daily log %s to Firestore.", doc_id)
            except Exception as exc:
                logger.warning("Could not sync daily log to Firestore: %s", exc)

        return True

    def get_log(self, patient_id: str, log_date: str) -> Optional[DailyLog]:
        """
        Retrieves a log for a specific date, checking local storage first then cloud.
        """
        # 1. Check local
        local_file = self._get_local_path(patient_id, log_date)
        if local_file.exists():
            try:
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DailyLog(**data)
            except Exception as exc:
                logger.warning("Error reading local log %s: %s", local_file, exc)

        # 2. Check cloud Firestore
        fb = get_firebase_service()
        if fb.is_connected:
            try:
                doc_id = f"{patient_id}_{log_date}"
                doc = fb._db.collection(config.DAILY_LOGS_COLLECTION).document(doc_id).get()
                if doc.exists:
                    return DailyLog(**doc.to_dict())
            except Exception as exc:
                logger.warning("Error reading Firestore log %s: %s", doc_id, exc)

        return None

    def get_patient_history(
        self,
        patient_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit_days: int = 30,
    ) -> List[DailyLog]:
        """
        Returns chronological list of daily logs for a patient.
        """
        logs_map: Dict[str, DailyLog] = {}

        # 1. Load local logs
        pattern = f"{patient_id}_*.json"
        for p in self.storage_dir.glob(pattern):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                dt_str = d.get("date")
                if dt_str:
                    logs_map[dt_str] = DailyLog(**d)
            except Exception:
                continue

        # 2. Augment from Firestore if connected
        fb = get_firebase_service()
        if fb.is_connected:
            try:
                query = (
                    fb._db.collection(config.DAILY_LOGS_COLLECTION)
                    .where("patient_id", "==", patient_id)
                    .limit(limit_days)
                    .stream()
                )
                for doc in query:
                    d = doc.to_dict()
                    dt_str = d.get("date")
                    if dt_str and dt_str not in logs_map:
                        logs_map[dt_str] = DailyLog(**d)
            except Exception as exc:
                logger.warning("Error fetching cloud history for %s: %s", patient_id, exc)

        # Filter by date window if specified
        sorted_logs = sorted(logs_map.values(), key=lambda x: x.date)

        if start_date:
            sorted_logs = [l for l in sorted_logs if l.date >= start_date]
        if end_date:
            sorted_logs = [l for l in sorted_logs if l.date <= end_date]

        return sorted_logs[-limit_days:]

    def get_history_dataframe(self, patient_id: str, limit_days: int = 30) -> pd.DataFrame:
        """
        Returns patient daily history as a Pandas DataFrame for analytics and graphing.
        """
        logs = self.get_patient_history(patient_id, limit_days=limit_days)
        if not logs:
            return pd.DataFrame()
        records = [l.to_dict() for l in logs]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)
        return df


def calculate_daily_protein(meals: List[Dict[str, Any]]) -> float:
    """Helper to sum protein across entered meal items."""
    return round(float(sum(float(m.get("protein_g", 0.0)) for m in meals)), 1)

