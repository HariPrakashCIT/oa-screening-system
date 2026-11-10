"""
feature_extractor.py — Calculates biomechanical and kinematic gait features.

Simple explanation
------------------
This module takes cleaned time-series sensor data and summarizes it into
compact mathematical descriptors (features) such as:
  - Peak and mean pressures at the heel, midfoot, and forefoot
  - Variability and stance/contact duration
  - Acceleration and rotational dynamics from the MPU6050
  - Temporal sampling quality metrics

These extracted features form the input for both the dashboard analytics
and the machine learning pattern-classification models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from app.config import config
from app.firebase_service import get_firebase_service

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts clinical research features from walking sensor time-series data.

    Parameters
    ----------
    load_threshold_adc : int
        Threshold above which the foot sensor is considered 'loaded' (weight-bearing).
    """

    def __init__(self, load_threshold_adc: int = config.PRESSURE_LOAD_THRESHOLD_ADC):
        self.load_threshold_adc = load_threshold_adc

    def extract_features(
        self,
        data: Union[pd.DataFrame, str, Path],
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes all pressure, motion, and temporal gait features.

        Parameters
        ----------
        data : pd.DataFrame | str | Path
            Cleaned sensor dataframe or path to cleaned CSV.
        participant_id : str | None
            Optional identifier override.
        session_id : str | None
            Optional session identifier override.

        Returns
        -------
        Dict[str, Any]
            Extracted feature dictionary ready for dashboard visualization,
            Firebase upload, and ML processing.
        """
        if isinstance(data, (str, Path)):
            df = pd.read_csv(Path(data))
        else:
            df = data.copy()

        if df.empty:
            raise ValueError("Input dataset is empty — cannot extract features.")

        # Resolve participant and session IDs
        pid = participant_id or (df["participant_id"].iloc[0] if "participant_id" in df.columns else "unknown")
        sid = session_id or (df["session_id"].iloc[0] if "session_id" in df.columns else "unknown")
        data_source = df["data_source"].iloc[0] if "data_source" in df.columns else "unknown"

        features: Dict[str, Any] = {
            "participant_id": str(pid),
            "session_id": str(sid),
            "data_source": str(data_source),
        }

        # ------------------------------------------------------------------
        # 1. Temporal & Quality Features
        # ------------------------------------------------------------------
        measurement_count = len(df)
        features["measurement_count"] = int(measurement_count)

        if "timestamp_ms" in df.columns and measurement_count > 1:
            timestamps = df["timestamp_ms"].values
            duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
            diffs_ms = np.diff(timestamps)
            avg_interval_ms = float(np.mean(diffs_ms)) if len(diffs_ms) > 0 else 0.0
            sampling_hz = (1000.0 / avg_interval_ms) if avg_interval_ms > 0 else 0.0
        else:
            duration_s = 0.0
            avg_interval_ms = 0.0
            sampling_hz = 0.0

        features["duration_seconds"] = round(float(duration_s), 2)
        features["avg_sampling_interval_ms"] = round(float(avg_interval_ms), 2)
        features["sampling_frequency_hz"] = round(float(sampling_hz), 1)
        features["missing_sample_ratio"] = 0.0  # Cleaned input has imputed small gaps

        # ------------------------------------------------------------------
        # 2. Pressure / Foot Loading Features
        # ------------------------------------------------------------------
        for sensor in ("heel", "midfoot", "forefoot"):
            if sensor in df.columns:
                series = df[sensor].astype(float)
                mean_val = float(series.mean())
                std_val = float(series.std(ddof=0))
                min_val = float(series.min())
                max_val = float(series.max())

                features[f"{sensor}_mean"] = round(mean_val, 2)
                features[f"{sensor}_std"] = round(std_val, 2)
                features[f"{sensor}_min"] = round(min_val, 2)
                features[f"{sensor}_peak"] = round(max_val, 2)
                features[f"{sensor}_max"] = round(max_val, 2)

                # Active loading duration (seconds foot is bearing weight above threshold)
                loaded_samples = (series >= self.load_threshold_adc).sum()
                if sampling_hz > 0:
                    sensor_load_time_s = loaded_samples / sampling_hz
                else:
                    sensor_load_time_s = 0.0
                features[f"{sensor}_loading_duration_s"] = round(float(sensor_load_time_s), 2)

        # Composite pressure metrics
        if all(s in df.columns for s in ("heel", "midfoot", "forefoot")):
            total_pressure = df["heel"] + df["midfoot"] + df["forefoot"]
            features["total_pressure_mean"] = round(float(total_pressure.mean()), 2)
            features["total_pressure_peak"] = round(float(total_pressure.max()), 2)
            features["pressure_variability"] = round(float(total_pressure.std(ddof=0)), 2)

            # Estimated weight-bearing stance time (any sensor loaded)
            stance_mask = (
                (df["heel"] >= self.load_threshold_adc)
                | (df["midfoot"] >= self.load_threshold_adc)
                | (df["forefoot"] >= self.load_threshold_adc)
            )
            stance_samples = stance_mask.sum()
            features["stance_duration_seconds"] = (
                round(float(stance_samples / sampling_hz), 2) if sampling_hz > 0 else 0.0
            )

        # ------------------------------------------------------------------
        # 3. Kinematic Features — Accelerometer
        # ------------------------------------------------------------------
        acc_axes = [c for c in ("acc_x_g", "acc_y_g", "acc_z_g") if c in df.columns]
        if acc_axes:
            for axis in acc_axes:
                clean_name = axis.replace("_g", "")
                series = df[axis].astype(float)
                features[f"{clean_name}_mean"] = round(float(series.mean()), 4)
                features[f"{clean_name}_std"] = round(float(series.std(ddof=0)), 4)
                features[f"{clean_name}_max"] = round(float(series.max()), 4)
                features[f"{clean_name}_min"] = round(float(series.min()), 4)

            # Acceleration magnitude vector |a| = sqrt(x^2 + y^2 + z^2)
            if len(acc_axes) == 3:
                acc_mag = np.sqrt(df["acc_x_g"] ** 2 + df["acc_y_g"] ** 2 + df["acc_z_g"] ** 2)
                features["acc_mag_mean"] = round(float(acc_mag.mean()), 4)
                features["acc_mag_peak"] = round(float(acc_mag.max()), 4)
                features["acceleration_variability"] = round(float(acc_mag.std(ddof=0)), 4)

        # ------------------------------------------------------------------
        # 4. Kinematic Features — Gyroscope
        # ------------------------------------------------------------------
        gyro_axes = [c for c in ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps") if c in df.columns]
        if gyro_axes:
            for axis in gyro_axes:
                clean_name = axis.replace("_dps", "")
                series = df[axis].astype(float)
                features[f"{clean_name}_mean"] = round(float(series.mean()), 4)
                features[f"{clean_name}_std"] = round(float(series.std(ddof=0)), 4)
                features[f"{clean_name}_max"] = round(float(series.max()), 4)
                features[f"{clean_name}_min"] = round(float(series.min()), 4)

            # Gyroscope magnitude |w|
            if len(gyro_axes) == 3:
                gyro_mag = np.sqrt(df["gyro_x_dps"] ** 2 + df["gyro_y_dps"] ** 2 + df["gyro_z_dps"] ** 2)
                features["gyro_mag_mean"] = round(float(gyro_mag.mean()), 4)
                features["gyro_mag_peak"] = round(float(gyro_mag.max()), 4)
                features["gyro_variability"] = round(float(gyro_mag.std(ddof=0)), 4)

        # Overall composite movement variability
        acc_var = features.get("acceleration_variability", 0.0)
        gyro_var = features.get("gyro_variability", 0.0)
        features["movement_variability"] = round(float(acc_var + (gyro_var / 100.0)), 4)

        return features

    def process_and_save(
        self,
        cleaned_csv_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        upload_to_firebase: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Extracts features from cleaned CSV, saves JSON summary to data/processed/,
        and optionally syncs features into the session document in Firestore.
        """
        path = Path(cleaned_csv_path)
        features = self.extract_features(path)

        out_dir = output_dir or config.DATA_PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_file = out_dir / (path.stem + "_features.json")
        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        logger.info("Saved extracted features to %s", feature_file)

        # Sync to Firebase if connected & session_id exists
        if upload_to_firebase:
            session_id = features.get("session_id")
            if session_id and session_id != "unknown":
                svc = get_firebase_service()
                if svc.is_connected:
                    svc.save_session_features(session_id, features)
                    logger.info("Uploaded features for session %s to Firebase.", session_id)

        return feature_file, features
