"""
data_logger.py — Saves validated sensor readings to CSV files.

Simple explanation
------------------
Every time the ESP32 sends a measurement, this module:
  1. Adds a host-computer timestamp (in UTC).
  2. Adds participant and session labels.
  3. Marks whether the data came from real hardware or the simulator.
  4. Appends one row to a CSV file on disk.

The raw CSV is never modified after a row is written.
Processed versions are saved to data/processed/ separately.

Usage
-----
    from app.data_logger import DataLogger

    logger = DataLogger(
        participant_id="patient_001",
        session_id="session_001",
        data_source="simulated",
    )
    logger.open()
    logger.write(reading_dict)   # call once per sensor reading
    logger.close()
    print(logger.csv_path)
"""

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import config

logger = logging.getLogger(__name__)


class DataLogger:
    """
    Appends validated sensor readings to a CSV file row by row.

    Parameters
    ----------
    participant_id : str
        Pseudonymous identifier, e.g. "patient_001".
    session_id : str
        Unique session identifier, e.g. "session_001".
    data_source : str
        Either "real" or "simulated". This label is saved in every row
        so it is impossible to confuse real and synthetic data later.
    output_dir : Path | None
        Override the default data/raw directory (useful for testing).
    """

    def __init__(
        self,
        participant_id: str,
        session_id: str,
        data_source: str = "real",
        output_dir: Optional[Path] = None,
    ):
        if data_source not in ("real", "simulated"):
            raise ValueError(
                f"data_source must be 'real' or 'simulated', got '{data_source}'"
            )

        self.participant_id = participant_id
        self.session_id = session_id
        self.data_source = data_source

        # Decide where to save the file
        if output_dir is None:
            if data_source == "simulated":
                output_dir = config.DATA_DEMO_DIR
            else:
                output_dir = config.DATA_RAW_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = config.session_csv_filename(participant_id, session_id)
        self.csv_path: Path = output_dir / filename

        self._file = None
        self._writer = None
        self._is_open: bool = False
        self.rows_written: int = 0

        logger.info(
            "DataLogger created → %s  (source=%s)",
            self.csv_path,
            data_source,
        )

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the CSV file for writing.

        Raises FileExistsError if the file already exists to protect
        existing raw data from being overwritten.
        """
        if self._is_open:
            logger.warning("DataLogger.open() called but file is already open.")
            return

        if self.csv_path.exists():
            raise FileExistsError(
                f"Raw CSV already exists: {self.csv_path}\n"
                "Refusing to overwrite raw data. Use a different session_id "
                "or delete the existing file manually."
            )

        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=config.CSV_COLUMNS,
            extrasaction="ignore",   # silently drop _raw_line and other extras
        )
        self._writer.writeheader()
        self._file.flush()
        self._is_open = True

        logger.info("CSV opened: %s", self.csv_path)

    def close(self) -> None:
        """Flush and close the CSV file."""
        if not self._is_open:
            return

        if self._file:
            self._file.flush()
            self._file.close()

        self._is_open = False
        logger.info(
            "CSV closed: %s  (%d rows written)",
            self.csv_path,
            self.rows_written,
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False   # do not suppress exceptions

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, reading: dict) -> None:
        """
        Append one sensor reading to the CSV.

        Parameters
        ----------
        reading : dict
            A validated sensor dictionary from SerialReceiver or
            SimulatedSerialReceiver.  Must contain the nine ESP32 fields.
            Any extra keys (e.g. '_raw_line') are silently ignored.

        The method adds:
          - host_timestamp_utc : ISO-8601 UTC timestamp from the laptop clock.
          - participant_id     : from constructor.
          - session_id         : from constructor.
          - data_source        : "real" or "simulated".
        """
        if not self._is_open:
            raise RuntimeError(
                "DataLogger is not open. Call open() before write()."
            )

        row = {
            "host_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_ms":       reading.get("timestamp_ms", ""),
            "heel":               reading.get("heel", ""),
            "midfoot":            reading.get("midfoot", ""),
            "forefoot":           reading.get("forefoot", ""),
            "acc_x_g":            reading.get("acc_x_g", ""),
            "acc_y_g":            reading.get("acc_y_g", ""),
            "acc_z_g":            reading.get("acc_z_g", ""),
            "gyro_x_dps":         reading.get("gyro_x_dps", ""),
            "gyro_y_dps":         reading.get("gyro_y_dps", ""),
            "gyro_z_dps":         reading.get("gyro_z_dps", ""),
            "participant_id":     self.participant_id,
            "session_id":         self.session_id,
            "data_source":        self.data_source,
        }

        self._writer.writerow(row)
        self.rows_written += 1

        # Flush every 50 rows so data is not lost if the program crashes
        if self.rows_written % 50 == 0:
            self._file.flush()

    def write_many(self, readings: List[dict]) -> None:
        """Write a list of readings in one call."""
        for reading in readings:
            self.write(reading)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DataLogger("
            f"participant='{self.participant_id}', "
            f"session='{self.session_id}', "
            f"source='{self.data_source}', "
            f"rows={self.rows_written}, "
            f"open={self._is_open})"
        )
