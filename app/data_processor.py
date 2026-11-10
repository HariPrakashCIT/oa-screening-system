"""
data_processor.py — Cleans, validates, and pre-processes raw sensor CSV recordings.

Simple explanation
------------------
Raw recordings collected from the hardware may contain imperfections:
  - Jittery readings or electrical noise
  - Corrupted rows from early USB disconnects
  - Occasional duplicate timestamps or missing values
  - Out-of-bounds readings from hardware glitches

This module performs systematic quality checks on the raw CSV,
applies optional smoothing filters, performs uncalibrated vs. calibrated
force conversions if calibration parameters are provided, and outputs
a cleaned CSV saved into data/processed/ without ever altering the original file.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class ProcessingReport:
    """
    Summary of the data quality checks and cleaning actions performed.
    """
    total_raw_rows: int = 0
    clean_rows: int = 0
    missing_values_imputed: int = 0
    duplicate_timestamps_dropped: int = 0
    out_of_range_flagged: int = 0
    estimated_sampling_hz: float = 0.0
    duration_seconds: float = 0.0
    is_calibrated: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_raw_rows": self.total_raw_rows,
            "clean_rows": self.clean_rows,
            "missing_values_imputed": self.missing_values_imputed,
            "duplicate_timestamps_dropped": self.duplicate_timestamps_dropped,
            "out_of_range_flagged": self.out_of_range_flagged,
            "estimated_sampling_hz": self.estimated_sampling_hz,
            "duration_seconds": self.duration_seconds,
            "is_calibrated": self.is_calibrated,
            "notes": self.notes,
        }


class DataProcessor:
    """
    Loads, cleans, validates, and calibrates raw walking sensor CSV recordings.

    Parameters
    ----------
    calibration_config : dict | None
        Optional sensor calibration parameters. If None, sensors operate in
        'raw ADC count' mode and force values remain uncalibrated.
        Format example:
        {
            "heel": {"scale": 0.02, "offset": 0.0, "unit": "N_est"},
            "midfoot": {"scale": 0.02, "offset": 0.0, "unit": "N_est"},
            "forefoot": {"scale": 0.02, "offset": 0.0, "unit": "N_est"},
        }
    smoothing_window : int
        Window size for rolling average smoothing (set <= 1 to disable).
    """

    def __init__(
        self,
        calibration_config: Optional[Dict[str, Dict[str, float]]] = None,
        smoothing_window: int = 3,
    ):
        self.calibration_config = calibration_config
        self.smoothing_window = max(1, smoothing_window)

    def load_raw_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a raw sensor recording CSV into a Pandas DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw CSV file does not exist: {path}")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"CSV file is empty: {path}")

        # Verify minimal essential columns exist
        missing_columns = [
            col for col in config.REQUIRED_SENSOR_FIELDS if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"CSV {path.name} is missing mandatory sensor columns: {missing_columns}"
            )

        return df

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        apply_filter: bool = True,
    ) -> Tuple[pd.DataFrame, ProcessingReport]:
        """
        Executes cleaning pipeline:
          1. Drops corrupted/malformed non-numeric rows
          2. Deduplicates identical timestamp readings
          3. Interpolates minor missing gaps
          4. Checks valid ranges based on config.SENSOR_RANGE
          5. Applies moving average noise smoothing if enabled
          6. Computes sampling metrics and compiles ProcessingReport
        """
        report = ProcessingReport(total_raw_rows=len(df))
        clean_df = df.copy()

        # 1. Ensure numeric types for sensor columns
        numeric_cols = config.REQUIRED_SENSOR_FIELDS
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Drop rows where critical timestamp is NaN
        initial_count = len(clean_df)
        clean_df = clean_df.dropna(subset=["timestamp_ms"])
        if len(clean_df) < initial_count:
            report.notes.append(f"Dropped {initial_count - len(clean_df)} rows with invalid timestamps.")

        # 2. Check and remove duplicate timestamps
        dups = clean_df.duplicated(subset=["timestamp_ms"], keep="first")
        num_dups = dups.sum()
        if num_dups > 0:
            clean_df = clean_df[~dups].copy()
            report.duplicate_timestamps_dropped = int(num_dups)
            report.notes.append(f"Removed {num_dups} duplicate timestamp entries.")

        # Sort chronologically
        clean_df = clean_df.sort_values(by="timestamp_ms").reset_index(drop=True)

        # 3. Handle missing sensor values (forward-fill then backward-fill small gaps)
        missing_count = clean_df[numeric_cols].isna().sum().sum()
        if missing_count > 0:
            report.missing_values_imputed = int(missing_count)
            clean_df[numeric_cols] = clean_df[numeric_cols].ffill().bfill()
            report.notes.append(f"Imputed {missing_count} missing sensor data points.")

        # 4. Out-of-range audit (flagged, clamped to physical bounds)
        out_of_range_count = 0
        for sensor, (lo, hi) in config.SENSOR_RANGE.items():
            if sensor in clean_df.columns:
                mask = (clean_df[sensor] < lo) | (clean_df[sensor] > hi)
                invalid_n = mask.sum()
                if invalid_n > 0:
                    out_of_range_count += int(invalid_n)
                    clean_df[sensor] = clean_df[sensor].clip(lower=lo, upper=hi)

        report.out_of_range_flagged = out_of_range_count
        if out_of_range_count > 0:
            report.notes.append(f"Clamped {out_of_range_count} out-of-range sensor values.")

        # 5. Optional noise filtering (rolling mean smoothing)
        if apply_filter and self.smoothing_window > 1 and len(clean_df) >= self.smoothing_window:
            filtered_cols = [c for c in numeric_cols if c != "timestamp_ms"]
            clean_df[filtered_cols] = (
                clean_df[filtered_cols]
                .rolling(window=self.smoothing_window, min_periods=1, center=True)
                .mean()
            )
            report.notes.append(f"Applied centered rolling mean filter (window={self.smoothing_window}).")

        # 6. Apply calibration if configured
        if self.calibration_config:
            clean_df = self._apply_calibration(clean_df)
            report.is_calibrated = True
            report.notes.append("Applied physical sensor force calibration curves.")
        else:
            report.is_calibrated = False
            report.notes.append("No calibration curve provided; preserved raw ADC units.")

        # 7. Compute temporal summary
        if len(clean_df) > 1:
            time_diffs_ms = clean_df["timestamp_ms"].diff().dropna()
            avg_dt_ms = time_diffs_ms.mean()
            duration_s = (clean_df["timestamp_ms"].iloc[-1] - clean_df["timestamp_ms"].iloc[0]) / 1000.0
            sampling_hz = (1000.0 / avg_dt_ms) if avg_dt_ms > 0 else 0.0
        else:
            duration_s = 0.0
            sampling_hz = 0.0

        report.clean_rows = len(clean_df)
        report.duration_seconds = round(float(duration_s), 2)
        report.estimated_sampling_hz = round(float(sampling_hz), 1)

        return clean_df, report

    def _apply_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw ADC pressure values to estimated force using calibration parameters.
        """
        calibrated_df = df.copy()
        if not self.calibration_config:
            return calibrated_df

        for sensor in ("heel", "midfoot", "forefoot"):
            if sensor in self.calibration_config and sensor in calibrated_df.columns:
                params = self.calibration_config[sensor]
                scale = params.get("scale", 1.0)
                offset = params.get("offset", 0.0)
                # Output estimated force column alongside raw ADC
                calibrated_df[f"{sensor}_force_est"] = (calibrated_df[sensor] * scale) + offset
                calibrated_df[f"{sensor}_force_est"] = calibrated_df[f"{sensor}_force_est"].clip(lower=0.0)

        return calibrated_df

    def process_and_save(
        self,
        raw_csv_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        apply_filter: bool = True,
    ) -> Tuple[Path, ProcessingReport]:
        """
        Complete processing workflow: Loads raw CSV, cleans & validates it,
        and saves it to data/processed/ without modifying raw recording.
        """
        raw_path = Path(raw_csv_path)
        df = self.load_raw_csv(raw_path)
        clean_df, report = self.validate_and_clean(df, apply_filter=apply_filter)

        out_dir = output_dir or config.DATA_PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        clean_filename = raw_path.stem + "_clean.csv"
        clean_csv_path = out_dir / clean_filename

        clean_df.to_csv(clean_csv_path, index=False)
        logger.info(
            "Cleaned data saved to %s (%d rows, Hz=%.1f)",
            clean_csv_path,
            len(clean_df),
            report.estimated_sampling_hz,
        )
        return clean_csv_path, report
