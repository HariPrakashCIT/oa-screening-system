"""
patient_profile_service.py — Patient Profile, Onboarding, and Medical Records Service.

Manages persistent clinical and lifestyle profiles for individuals monitored
for knee osteoarthritis. Stores profiles locally as JSON and synchronises with
Google Firebase Firestore when online.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import config
from app.firebase_service import get_firebase_service

logger = logging.getLogger(__name__)

# Directory where patient profiles and medical records are persisted
PROFILES_DIR: Path = config.PROJECT_ROOT / "data" / "patient_profiles"
REPORTS_DIR: Path = config.PROJECT_ROOT / "data" / "medical_reports"

PROFILES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_PROFILE_TEMPLATE: Dict[str, Any] = {
    "patient_id": "patient_001",
    "display_name": "Patient 001",
    "age": 58,
    "gender": "Female",
    "height_cm": 165.0,
    "weight_kg": 72.0,
    "bmi": 26.4,
    "oa_joint": "Knee",
    "affected_side": "Right",
    "kl_grade": "Grade 2 (Mild)",
    "diagnosis_year": 2023,
    "symptoms_duration_months": 18,
    "mobility_limitations": "Pain on stair descent, stiffness after sitting > 30 min",
    "current_medications": "Acetaminophen PRN, Topical NSAID gel",
    "daily_step_goal": 5000,
    "daily_protein_goal_g": 65.0,
    "exercise_regimen": "Quadriceps strengthening, low-impact stationary cycling 3x/week",
    "clinical_notes": "Mild medial joint space narrowing. No prior surgery.",
    "onboarding_completed": True,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate BMI from height in cm and weight in kg."""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m * h_m), 1)


class PatientProfileService:
    """
    Handles local and cloud persistence for patient profiles and uploaded medical records.
    """

    def __init__(self, profiles_dir: Optional[Path] = None, reports_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self.reports_dir = reports_dir or REPORTS_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_default_profile()

    def _ensure_default_profile(self) -> None:
        """Seed the first default profile if none exists."""
        default_file = self.profiles_dir / "patient_001.json"
        if not default_file.exists():
            try:
                with open(default_file, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_PROFILE_TEMPLATE, f, indent=2)
            except Exception as e:
                logger.warning("Could not write default profile: %s", e)

    def get_profile(self, patient_id: str) -> Dict[str, Any]:
        """
        Load a patient profile. Returns default template filled with ID if not found.
        """
        path = self.profiles_dir / f"{patient_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Auto-update BMI if height and weight exist
                    if data.get("height_cm") and data.get("weight_kg"):
                        data["bmi"] = calculate_bmi(data["height_cm"], data["weight_kg"])
                    return data
            except Exception as e:
                logger.error("Failed to read profile for %s: %s", patient_id, e)

        # Fallback template
        fallback = dict(DEFAULT_PROFILE_TEMPLATE)
        fallback["patient_id"] = patient_id
        fallback["display_name"] = f"Patient {patient_id.replace('patient_', '')}"
        fallback["created_at"] = datetime.now(timezone.utc).isoformat()
        return fallback

    def save_profile(self, profile: Dict[str, Any], sync_cloud: bool = True) -> bool:
        """
        Save profile to local JSON and sync with Firestore.
        """
        patient_id = profile.get("patient_id")
        if not patient_id:
            logger.error("Cannot save profile without patient_id.")
            return False

        # Recalculate BMI
        h = float(profile.get("height_cm", 0.0) or 0.0)
        w = float(profile.get("weight_kg", 0.0) or 0.0)
        profile["bmi"] = calculate_bmi(h, w)
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save locally
        path = self.profiles_dir / f"{patient_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
            logger.info("Saved local profile: %s", path)
        except Exception as e:
            logger.error("Failed to save local profile for %s: %s", patient_id, e)
            return False

        # Sync to Firebase if connected
        if sync_cloud:
            try:
                fb = get_firebase_service()
                if fb.is_connected:
                    notes_summary = (
                        f"Limb: {profile.get('affected_side', '')} {profile.get('oa_joint', '')}. "
                        f"KL Grade: {profile.get('kl_grade', '')}. "
                        f"Notes: {profile.get('clinical_notes', '')}"
                    ).strip()
                    fb.create_patient(
                        patient_id=patient_id,
                        age=int(profile.get("age", 55)),
                        notes=notes_summary,
                    )
            except Exception as exc:
                logger.warning("Cloud profile sync warning for %s: %s", patient_id, exc)

        return True

    def list_all_patients(self) -> List[str]:
        """
        Return sorted list of all unique patient IDs known from profiles, CSVs, or cloud.
        """
        pids = set()

        # From local profile JSONs
        for f in self.profiles_dir.glob("*.json"):
            pids.add(f.stem)

        # From raw and demo CSVs
        for d in [config.DATA_RAW_DIR, config.DATA_DEMO_DIR]:
            if d.exists():
                for f in d.glob("patient_*.csv"):
                    parts = f.stem.split("_")
                    if len(parts) >= 2:
                        pids.add(f"patient_{parts[1]}")

        # From daily logs
        if config.DATA_DAILY_LOGS_DIR.exists():
            for f in config.DATA_DAILY_LOGS_DIR.glob("patient_*_*.json"):
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    pids.add(f"patient_{parts[1]}")

        # From Firestore if available
        try:
            fb = get_firebase_service()
            if fb.is_connected:
                for cp in fb.list_patients():
                    cid = cp.get("patient_id")
                    if cid:
                        pids.add(cid)
        except Exception:
            pass

        if not pids:
            pids.add("patient_001")

        return sorted(list(pids))

    # ------------------------------------------------------------------
    # Medical Reports & Documents
    # ------------------------------------------------------------------

    def save_report(
        self,
        patient_id: str,
        uploaded_file,
        report_type: str = "Imaging / X-Ray",
        clinical_note: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Save an uploaded medical document to patient's report repository.
        """
        pt_dir = self.reports_dir / patient_id
        pt_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{timestamp_str}_{uploaded_file.name.replace(' ', '_')}"
        dest_path = pt_dir / safe_name

        try:
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            meta = {
                "file_name": safe_name,
                "original_name": uploaded_file.name,
                "report_type": report_type,
                "clinical_note": clinical_note,
                "size_bytes": dest_path.stat().st_size,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }

            # Update index
            index_path = pt_dir / "index.json"
            items = []
            if index_path.exists():
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                except Exception:
                    items = []
            items.insert(0, meta)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)

            return meta
        except Exception as e:
            logger.error("Failed to save report for %s: %s", patient_id, e)
            return None

    def list_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        """List all medical reports associated with a patient."""
        pt_dir = self.reports_dir / patient_id
        index_path = pt_dir / "index.json"
        if not index_path.exists():
            return []
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read report index for %s: %s", patient_id, e)
            return []

