"""
baseline_engine.py — Personalized baseline computation engine for OA patients.

Simple explanation
------------------
Osteoarthritis affects every person differently:
  - An elderly patient may comfortably walk 2,500 steps/day with a baseline pain score of 3.
  - An active younger patient may average 6,500 steps/day with a baseline pain score of 1.
Comparing both against a universal static threshold produces misleading alerts.

This engine computes an individual profile for each patient using:
  - Medians and Means
  - Standard Deviations (SD) and Median Absolute Deviations (MAD)
  - Baseline Status: 'provisional' (< 7 valid sessions) vs. 'established' (>= 7 valid sessions)
  - Configurable rolling observation window (default: 14 days)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.config import config
from app.daily_log_service import DailyLogService
from app.firebase_service import get_firebase_service

logger = logging.getLogger(__name__)

# Essential gait metrics tracked for personal baseline
BASELINE_GAIT_METRICS = [
    "heel_peak",
    "midfoot_peak",
    "forefoot_peak",
    "heel_mean",
    "midfoot_mean",
    "forefoot_mean",
    "pressure_variability",
    "acceleration_variability",
    "gyro_variability",
    "stance_duration_seconds",
]

# Essential lifestyle metrics tracked for personal baseline
BASELINE_LIFESTYLE_METRICS = [
    "steps_count",
    "walking_duration_minutes",
    "protein_intake_g",
    "pain_score",
    "stiffness_minutes",
    "fatigue_score",
]


@dataclass
class MetricBaseline:
    mean: float
    std: float
    median: float
    mad: float  # Median Absolute Deviation (robust measure of dispersion)
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": round(float(self.mean), 2),
            "std": round(float(self.std), 2),
            "median": round(float(self.median), 2),
            "mad": round(float(self.mad), 2),
            "count": int(self.count),
        }


@dataclass
class PatientBaseline:
    patient_id: str
    baseline_status: str  # 'provisional' | 'established' | 'insufficient_data'
    valid_observation_count: int
    baseline_start_date: Optional[str] = None
    baseline_end_date: Optional[str] = None
    gait_baseline: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    activity_baseline: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    symptom_baseline: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nutrition_baseline: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_updated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_robust_statistics(values: Union[List[float], np.ndarray]) -> MetricBaseline:
    """Computes mean, std, median, and median absolute deviation (MAD)."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return MetricBaseline(mean=0.0, std=0.0, median=0.0, mad=0.0, count=0)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    med_val = float(np.median(arr))
    # MAD = median(|x_i - median(x)|)
    mad_val = float(np.median(np.abs(arr - med_val)))

    return MetricBaseline(
        mean=mean_val,
        std=std_val,
        median=med_val,
        mad=mad_val,
        count=len(arr),
    )


class BaselineEngine:
    """
    Computes, caches, and maintains individual patient baselines.
    """

    def __init__(
        self,
        min_observations_established: int = 7,
        baseline_window_days: int = 14,
        storage_dir: Optional[Path] = None,
    ):
        self.min_observations_established = min_observations_established
        self.baseline_window_days = baseline_window_days
        self.storage_dir = storage_dir or (config.DATA_PROCESSED_DIR / "baselines")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_local_file(self, patient_id: str) -> Path:
        return self.storage_dir / f"{patient_id}_baseline.json"

    def compute_baseline(
        self,
        patient_id: str,
        gait_sessions: Optional[List[Dict[str, Any]]] = None,
        daily_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> PatientBaseline:
        """
        Calculates a personal baseline from historical gait sessions and daily logs.
        """
        # 1. Fetch gait feature records if not provided
        if gait_sessions is None:
            gait_sessions = self._load_patient_gait_features(patient_id)

        # 2. Fetch daily logs if not provided
        if daily_logs is None:
            log_svc = DailyLogService()
            history = log_svc.get_patient_history(patient_id, limit_days=self.baseline_window_days)
            daily_logs = [h.to_dict() for h in history]

        total_obs = max(len(gait_sessions), len(daily_logs))

        if total_obs == 0:
            status = "insufficient_data"
        elif total_obs < self.min_observations_established:
            status = "provisional"
        else:
            status = "established"

        # --- Compute Gait Baselines ---
        gait_base: Dict[str, Dict[str, Any]] = {}
        if gait_sessions:
            df_gait = pd.DataFrame(gait_sessions)
            for m in BASELINE_GAIT_METRICS:
                if m in df_gait.columns:
                    stat = compute_robust_statistics(df_gait[m].dropna().values)
                    gait_base[m] = stat.to_dict()

        # --- Compute Activity & Symptom & Nutrition Baselines ---
        act_base: Dict[str, Dict[str, Any]] = {}
        sym_base: Dict[str, Dict[str, Any]] = {}
        nut_base: Dict[str, Dict[str, Any]] = {}

        if daily_logs:
            df_logs = pd.DataFrame(daily_logs)

            for act_col in ("steps_count", "walking_duration_minutes"):
                if act_col in df_logs.columns:
                    act_base[act_col] = compute_robust_statistics(df_logs[act_col].dropna().values).to_dict()

            for sym_col in ("pain_score", "stiffness_minutes", "fatigue_score"):
                if sym_col in df_logs.columns:
                    sym_base[sym_col] = compute_robust_statistics(df_logs[sym_col].dropna().values).to_dict()

            if "protein_intake_g" in df_logs.columns:
                nut_base["protein_intake_g"] = compute_robust_statistics(df_logs["protein_intake_g"].dropna().values).to_dict()

        baseline = PatientBaseline(
            patient_id=patient_id,
            baseline_status=status,
            valid_observation_count=total_obs,
            gait_baseline=gait_base,
            activity_baseline=act_base,
            symptom_baseline=sym_base,
            nutrition_baseline=nut_base,
        )

        # Save to disk
        self._save_baseline(baseline)
        return baseline

    def _load_patient_gait_features(self, patient_id: str) -> List[Dict[str, Any]]:
        """Collects saved JSON feature summaries matching patient_id from data/processed/."""
        records = []
        for f in config.DATA_PROCESSED_DIR.glob(f"{patient_id}_*_features.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    records.append(json.load(fp))
            except Exception:
                continue
        return records

    def _save_baseline(self, baseline: PatientBaseline) -> None:
        """Saves baseline to local file and syncs to Firestore."""
        data = baseline.to_dict()
        local_file = self._get_local_file(baseline.patient_id)
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Sync with Firestore
        fb = get_firebase_service()
        if fb.is_connected:
            try:
                fb._db.collection(config.BASELINES_COLLECTION).document(baseline.patient_id).set(data)
                logger.info("Synced patient baseline for %s to Firestore.", baseline.patient_id)
            except Exception as exc:
                logger.warning("Could not sync baseline to Firestore: %s", exc)

    def get_baseline(self, patient_id: str) -> Optional[PatientBaseline]:
        """Loads cached baseline if available, or calculates it."""
        local_file = self._get_local_file(patient_id)
        if local_file.exists():
            try:
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PatientBaseline(**data)
            except Exception:
                pass

        # If not cached, compute now
        return self.compute_baseline(patient_id)

