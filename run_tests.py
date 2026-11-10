"""
run_tests.py — Master test runner for all OA Screening System modules.

Runs all unit and integration tests across tests/ using Python's
built-in unittest framework and prints a consolidated report.

Usage:
    python run_tests.py
"""

import sys
import unittest
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    # Discover all tests matching test_*.py in tests/ directory
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
