"""
config.py — Centralised configuration for the OA Screening System.

All settings are loaded from environment variables (defined in .env).
Nothing sensitive is hardcoded here.

Usage:
    from app.config import Config
    cfg = Config()
    print(cfg.serial_baud_rate)
"""

import os
import logging
from pathlib import Path
# ---------------------------------------------------------------------------
# Load .env file from the project root (two levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    # Fallback parser if python-dotenv is not installed yet
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v


class Config:
    """
    Single source of truth for all runtime configuration.

    Reading from environment variables means you can change settings
    without touching any Python code — just edit .env and restart.
    """

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_RAW_DIR: Path = _PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    DATA_PROCESSED_DIR: Path = _PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    DATA_DEMO_DIR: Path = _PROJECT_ROOT / os.getenv("DATA_DEMO_DIR", "data/demo")
    DATA_DAILY_LOGS_DIR: Path = _PROJECT_ROOT / "data/daily_logs"
    MODELS_DIR: Path = _PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
    SECRETS_DIR: Path = _PROJECT_ROOT / "secrets"

    # ------------------------------------------------------------------
    # Firebase
    # ------------------------------------------------------------------
    FIREBASE_CREDENTIALS_PATH: Path = _PROJECT_ROOT / os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "secrets/firebase-key.json"
    )
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")

    # Firestore collection names
    PATIENTS_COLLECTION: str = "patients"
    SESSIONS_COLLECTION: str = "sessions"
    DAILY_LOGS_COLLECTION: str = "daily_logs"
    BASELINES_COLLECTION: str = "patient_baselines"
    INSIGHTS_COLLECTION: str = "ai_insights"

    # ------------------------------------------------------------------
    # Gemini AI Configuration
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    # ------------------------------------------------------------------
    # Serial communication
    # ------------------------------------------------------------------
    SERIAL_PORT: str = os.getenv("SERIAL_PORT", "")
    SERIAL_BAUD_RATE: int = int(os.getenv("SERIAL_BAUD_RATE", "115200"))
    SERIAL_TIMEOUT_SECONDS: float = 1.0
    SERIAL_READ_TIMEOUT_SECONDS: float = 2.0

    # ------------------------------------------------------------------
    # Demo / development mode
    # ------------------------------------------------------------------
    # Set DEMO_MODE=true in .env to run without physical hardware.
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    # Simulated sensor frequency when in demo mode
    DEMO_SAMPLE_RATE_HZ: int = 50

    # ------------------------------------------------------------------
    # Sensor field names (must match ESP32 firmware output exactly)
    # ------------------------------------------------------------------
    SENSOR_FIELDS = {
        "timestamp": "timestamp_ms",
        "heel":      "heel",
        "midfoot":   "midfoot",
        "forefoot":  "forefoot",
        "acc_x":     "acc_x_g",
        "acc_y":     "acc_y_g",
        "acc_z":     "acc_z_g",
        "gyro_x":    "gyro_x_dps",
        "gyro_y":    "gyro_y_dps",
        "gyro_z":    "gyro_z_dps",
    }

    # All fields that must be present in every incoming JSON message
    REQUIRED_SENSOR_FIELDS = list(SENSOR_FIELDS.values())

    # ------------------------------------------------------------------
    # Sensor value ranges — used for sanity checking incoming readings.
    # Readings outside these ranges are flagged as suspicious (not deleted).
    # Adjust these after reviewing your hardware's actual output range.
    # ------------------------------------------------------------------
    SENSOR_RANGE = {
        # FSR ADC counts (0–4095 for 12-bit ESP32 ADC)
        "heel":      (0, 4095),
        "midfoot":   (0, 4095),
        "forefoot":  (0, 4095),
        # Accelerometer in g-force (±4 g typical for MPU6050 default setting)
        "acc_x_g":   (-4.0, 4.0),
        "acc_y_g":   (-4.0, 4.0),
        "acc_z_g":   (-4.0, 4.0),
        # Gyroscope in degrees per second (±250 dps default)
        "gyro_x_dps": (-250.0, 250.0),
        "gyro_y_dps": (-250.0, 250.0),
        "gyro_z_dps": (-250.0, 250.0),
    }

    # ------------------------------------------------------------------
    # Pressure loading threshold
    # Readings above this ADC count are considered "loaded" (foot on ground).
    # This is a configurable starting point — tune based on your sensors.
    # ------------------------------------------------------------------
    PRESSURE_LOAD_THRESHOLD_ADC: int = 100

    # ------------------------------------------------------------------
    # CSV column order (defines the saved file structure)
    # ------------------------------------------------------------------
    CSV_COLUMNS = [
        "host_timestamp_utc",
        "timestamp_ms",
        "heel",
        "midfoot",
        "forefoot",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
        "participant_id",
        "session_id",
        "data_source",   # "real" or "simulated"
    ]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ------------------------------------------------------------------
    # Helper: ensure all data directories exist at startup
    # ------------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create required data directories if they do not already exist."""
        for directory in [
            self.DATA_RAW_DIR,
            self.DATA_PROCESSED_DIR,
            self.DATA_DEMO_DIR,
            self.DATA_DAILY_LOGS_DIR,
            self.MODELS_DIR,
            self.SECRETS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper: generate a CSV filename for a session
    # ------------------------------------------------------------------
    @staticmethod
    def session_csv_filename(participant_id: str, session_id: str) -> str:
        """Return a standardised CSV filename.

        Example: patient_001_session_001.csv
        """
        return f"{participant_id}_{session_id}.csv"

    # ------------------------------------------------------------------
    # Helper: return the full path to a raw CSV file
    # ------------------------------------------------------------------
    def raw_csv_path(self, participant_id: str, session_id: str) -> Path:
        filename = self.session_csv_filename(participant_id, session_id)
        return self.DATA_RAW_DIR / filename

    def processed_csv_path(self, participant_id: str, session_id: str) -> Path:
        filename = f"{participant_id}_{session_id}_clean.csv"
        return self.DATA_PROCESSED_DIR / filename

    def __repr__(self) -> str:
        return (
            f"Config(demo={self.DEMO_MODE}, "
            f"serial_port='{self.SERIAL_PORT}', "
            f"baud={self.SERIAL_BAUD_RATE}, "
            f"firebase_project='{self.FIREBASE_PROJECT_ID}')"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of re-creating it
# ---------------------------------------------------------------------------
config = Config()

# Configure root logger once when this module is first imported
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

