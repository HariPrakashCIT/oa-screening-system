"""
tests/test_01_project_setup.py
==============================
Acceptance tests for Prompt 1 — Project Setup.

Compatible with both:
  • pytest       →   pytest tests/test_01_project_setup.py -v
  • unittest     →   python run_tests.py

Tests verify:
  1. All required directories exist.
  2. All required files exist.
  3. All app module stubs are importable.
  4. All ml module stubs are importable.
  5. config.py exposes the correct attributes.
  6. Required third-party packages are installed.
  7. .env.example is present.
  8. .gitignore protects secrets/ and .env.
  9. README.md contains a medical disclaimer.
"""

import sys
import os
import importlib
import unittest
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of how the tests are run
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 1. Directory structure
# ===========================================================================
class TestDirectoryStructure(unittest.TestCase):

    REQUIRED_DIRS = [
        "app",
        "ml",
        "data/raw",
        "data/processed",
        "data/demo",
        "models",
        "secrets",
        "tests",
    ]

    def test_all_required_directories_exist(self):
        missing = [d for d in self.REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
        self.assertFalse(missing, f"Missing directories: {missing}")


# ===========================================================================
# 2. Required files
# ===========================================================================
class TestRequiredFiles(unittest.TestCase):

    REQUIRED_FILES = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        ".env",
        "app/__init__.py",
        "app/config.py",
        "app/serial_receiver.py",
        "app/data_logger.py",
        "app/data_processor.py",
        "app/feature_extractor.py",
        "app/firebase_service.py",
        "app/session_manager.py",
        "app/dashboard.py",
        "ml/__init__.py",
        "ml/train_model.py",
        "ml/predict.py",
        "ml/model_utils.py",
        "tests/__init__.py",
    ]

    def test_all_required_files_exist(self):
        missing = [f for f in self.REQUIRED_FILES if not (PROJECT_ROOT / f).is_file()]
        self.assertFalse(missing, f"Missing files: {missing}")


# ===========================================================================
# 3. App module imports
# ===========================================================================
class TestAppModuleImports(unittest.TestCase):

    APP_MODULES = [
        "app.config",
        "app.serial_receiver",
        "app.data_logger",
        "app.data_processor",
        "app.feature_extractor",
        "app.firebase_service",
        "app.session_manager",
    ]

    def test_all_app_modules_importable(self):
        """
        Stub modules must import without errors.
        NotImplementedError is only raised when the CLASS is instantiated —
        not on import. This is intentional.
        """
        failed = []
        for name in self.APP_MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed.append(f"  {name}: {exc}")
        self.assertFalse(failed, "Import failures:\n" + "\n".join(failed))


# ===========================================================================
# 4. ML module imports
# ===========================================================================
class TestMLModuleImports(unittest.TestCase):

    ML_MODULES = ["ml.train_model", "ml.predict", "ml.model_utils"]

    def test_all_ml_modules_importable(self):
        failed = []
        for name in self.ML_MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed.append(f"  {name}: {exc}")
        self.assertFalse(failed, "ML import failures:\n" + "\n".join(failed))


# ===========================================================================
# 5. Config module correctness
# ===========================================================================
class TestConfig(unittest.TestCase):

    def setUp(self):
        from app.config import Config, config
        self.Config = Config
        self.config = config

    def test_config_imports_cleanly(self):
        self.assertIsNotNone(self.Config)
        self.assertIsNotNone(self.config)

    def test_required_attributes_present(self):
        required = [
            "PROJECT_ROOT", "DATA_RAW_DIR", "DATA_PROCESSED_DIR",
            "DATA_DEMO_DIR", "MODELS_DIR", "FIREBASE_CREDENTIALS_PATH",
            "FIREBASE_PROJECT_ID", "SERIAL_BAUD_RATE", "DEMO_MODE",
            "SENSOR_FIELDS", "REQUIRED_SENSOR_FIELDS", "SENSOR_RANGE",
            "CSV_COLUMNS",
        ]
        missing = [a for a in required if not hasattr(self.config, a)]
        self.assertFalse(missing, f"Config missing: {missing}")

    def test_sensor_fields_complete(self):
        expected = {
            "timestamp", "heel", "midfoot", "forefoot",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
        }
        self.assertEqual(expected, set(self.config.SENSOR_FIELDS.keys()))

    def test_csv_columns_contain_mandatory_fields(self):
        mandatory = [
            "host_timestamp_utc", "timestamp_ms",
            "heel", "midfoot", "forefoot",
            "acc_x_g", "acc_y_g", "acc_z_g",
            "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
            "participant_id", "session_id", "data_source",
        ]
        missing = [c for c in mandatory if c not in self.config.CSV_COLUMNS]
        self.assertFalse(missing, f"CSV_COLUMNS missing: {missing}")

    def test_ensure_directories_creates_dirs(self):
        self.config.ensure_directories()
        self.assertTrue(self.config.DATA_RAW_DIR.is_dir())
        self.assertTrue(self.config.DATA_PROCESSED_DIR.is_dir())
        self.assertTrue(self.config.MODELS_DIR.is_dir())

    def test_session_csv_filename_format(self):
        result = self.Config.session_csv_filename("patient_001", "session_001")
        self.assertEqual(result, "patient_001_session_001.csv")

    def test_demo_mode_is_boolean(self):
        self.assertIsInstance(self.config.DEMO_MODE, bool)

    def test_baud_rate_is_positive_integer(self):
        self.assertIsInstance(self.config.SERIAL_BAUD_RATE, int)
        self.assertGreater(self.config.SERIAL_BAUD_RATE, 0)

    def test_sensor_range_covers_all_sensor_fields(self):
        """Every sensor that produces readings must have a range defined."""
        expected_keys = {
            "heel", "midfoot", "forefoot",
            "acc_x_g", "acc_y_g", "acc_z_g",
            "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
        }
        actual_keys = set(self.config.SENSOR_RANGE.keys())
        self.assertEqual(expected_keys, actual_keys)


# ===========================================================================
# 6. Required packages
# ===========================================================================
class TestPackagesInstalled(unittest.TestCase):

    CORE_PACKAGES = [
        ("pandas",  "pandas"),
        ("numpy",   "numpy"),
        ("sklearn", "scikit-learn"),
        ("dotenv",  "python-dotenv"),
        ("joblib",  "joblib"),
    ]

    NETWORK_PACKAGES = [
        ("serial",          "pyserial"),
        ("firebase_admin",  "firebase-admin"),
        ("streamlit",       "streamlit"),
        ("matplotlib",      "matplotlib"),
        ("colorama",        "colorama"),
    ]

    def test_core_packages_importable(self):
        """Core packages that ship with the system Python must be available."""
        failed = []
        for import_name, pip_name in self.CORE_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                failed.append(f"  pip install {pip_name}")
        self.assertFalse(
            failed,
            "Missing packages. Run in your terminal:\n" + "\n".join(failed)
        )

    def test_network_packages_present_or_warn(self):
        """
        Packages that require network to install may be absent in a sandbox.
        This test never fails — it only issues a warning with install commands.
        """
        missing = []
        for import_name, pip_name in self.NETWORK_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(pip_name)
        if missing:
            msg = (
                "\n[ACTION REQUIRED] Install these packages in your terminal "
                "(with network access):\n"
                "  pip install " + " ".join(missing)
            )
            warnings.warn(msg)
            # Not a test failure — the developer installs these manually


# ===========================================================================
# 7. Security — .gitignore
# ===========================================================================
class TestGitignore(unittest.TestCase):

    def _text(self):
        path = PROJECT_ROOT / ".gitignore"
        self.assertTrue(path.is_file(), ".gitignore not found")
        return path.read_text()

    def test_excludes_env_file(self):
        self.assertIn(".env", self._text())

    def test_excludes_secrets_directory(self):
        self.assertIn("secrets/", self._text())

    def test_excludes_pycache(self):
        self.assertIn("__pycache__/", self._text())

    def test_env_example_present(self):
        self.assertTrue(
            (PROJECT_ROOT / ".env.example").is_file(),
            ".env.example must exist so new developers know what to configure."
        )


# ===========================================================================
# 8. README sanity check
# ===========================================================================
class TestReadme(unittest.TestCase):

    def _text(self):
        path = PROJECT_ROOT / "README.md"
        self.assertTrue(path.is_file(), "README.md not found")
        return path.read_text().lower()

    def test_mentions_installation(self):
        self.assertIn("install", self._text())

    def test_mentions_virtual_environment(self):
        text = self._text()
        self.assertTrue("venv" in text or "virtual environment" in text)

    def test_mentions_streamlit(self):
        self.assertIn("streamlit", self._text())

    def test_has_medical_disclaimer(self):
        text = self._text()
        self.assertTrue(
            "not" in text and "diagnos" in text,
            "README must state the system does NOT diagnose osteoarthritis."
        )


# ===========================================================================
# Allow running directly:   python tests/test_01_project_setup.py
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
