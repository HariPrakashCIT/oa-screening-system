"""
session_manager.py — Walking session lifecycle management.

Simple explanation
------------------
A "session" is one complete recording of someone walking.
This module manages the full lifecycle:

  1. You tell it who is walking (participant_id).
  2. It generates a unique session ID and records the start time.
  3. It opens a CSV file through DataLogger.
  4. It reads sensor data from SerialReceiver (real or simulated).
  5. It writes every reading to the CSV.
  6. When you say stop, it closes the file and records the end time.
  7. It returns a SessionResult with metadata about what was recorded.

Usage — demo mode (no hardware)
---------------------------------
    from app.session_manager import SessionManager

    sm = SessionManager(participant_id="patient_001", demo=True)
    result = sm.run(duration_seconds=5)
    print(result)

Usage — real hardware
---------------------
    sm = SessionManager(
        participant_id="patient_001",
        port="/dev/cu.usbserial-0001",
        demo=False,
    )
    result = sm.run()   # runs until you press Ctrl-C

Command-line
------------
    python app/session_manager.py --demo --duration 5 --participant patient_001
"""

import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.config import config
from app.data_logger import DataLogger
from app.serial_receiver import SimulatedSerialReceiver, create_receiver

logger = logging.getLogger(__name__)


# ===========================================================================
# Session result — returned after a session ends
# ===========================================================================

@dataclass
class SessionResult:
    """
    Everything that happened during one walking session.

    This object is returned by SessionManager.run().
    It contains enough information to upload to Firebase later.
    """
    participant_id: str
    session_id: str
    data_source: str                 # "real" or "simulated"
    status: str                      # "completed" | "failed" | "stopped"
    csv_path: Optional[Path]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: float
    rows_written: int
    error_message: Optional[str] = None

    # Fields that Firebase upload will fill in later
    firebase_uploaded: bool = False
    firebase_session_doc_id: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.status == "completed" and self.rows_written > 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SESSION SUMMARY",
            "=" * 60,
            f"  Participant   : {self.participant_id}",
            f"  Session ID    : {self.session_id}",
            f"  Data source   : {self.data_source.upper()}",
            f"  Status        : {self.status.upper()}",
            f"  Rows saved    : {self.rows_written}",
            f"  Duration      : {self.duration_seconds:.1f} s",
            f"  CSV file      : {self.csv_path}",
        ]
        if self.error_message:
            lines.append(f"  Error         : {self.error_message}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SessionResult(id='{self.session_id}', "
            f"status='{self.status}', rows={self.rows_written})"
        )


# ===========================================================================
# Session ID generator
# ===========================================================================

def generate_session_id(participant_id: str) -> str:
    """
    Create a unique session ID based on participant and timestamp.

    Example:  patient_001_20261109_201530
    This format makes it easy to sort sessions chronologically.
    """
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    return f"{participant_id}_{timestamp_str}"


# ===========================================================================
# Main session manager
# ===========================================================================

class SessionManager:
    """
    Coordinates sensor data collection for one walking session.

    Parameters
    ----------
    participant_id : str
        Pseudonymous ID for the person walking, e.g. "patient_001".
    session_id : str | None
        If None, an ID is generated automatically from the timestamp.
    port : str | None
        Serial port for real hardware. Ignored in demo mode.
    demo : bool | None
        If True, use the simulated receiver. If None, reads from config.
    duration_seconds : float | None
        Stop automatically after this many seconds.
        If None, runs until Ctrl-C or stop() is called.
    display_readings : bool
        Print each reading to the terminal as it arrives.
    display_interval : int
        Print every N-th reading (reduces terminal spam at 50 Hz).
    """

    def __init__(
        self,
        participant_id: str,
        session_id: Optional[str] = None,
        port: Optional[str] = None,
        demo: Optional[bool] = None,
        duration_seconds: Optional[float] = None,
        display_readings: bool = True,
        display_interval: int = 25,
    ):
        self.participant_id = participant_id
        self.session_id = session_id or generate_session_id(participant_id)
        self.port = port
        self.demo = demo if demo is not None else config.DEMO_MODE
        self.duration_seconds = duration_seconds
        self.display_readings = display_readings
        self.display_interval = display_interval

        self._stop_event = threading.Event()
        self._result: Optional[SessionResult] = None

        logger.info(
            "SessionManager ready: participant=%s  session=%s  demo=%s",
            self.participant_id,
            self.session_id,
            self.demo,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, duration_seconds: Optional[float] = None) -> SessionResult:
        """
        Start data collection and block until it ends.

        Returns a SessionResult with all metadata about the session.

        The session ends when any of the following happens:
          - duration_seconds elapses (if set)
          - stop() is called from another thread
          - The user presses Ctrl-C
          - The serial connection drops (real hardware mode)
        """
        # Allow per-call override of duration
        if duration_seconds is not None:
            self.duration_seconds = duration_seconds

        data_source = "simulated" if self.demo else "real"
        start_time = datetime.now(timezone.utc)
        status = "failed"
        error_message = None
        csv_path = None
        rows_written = 0

        # Create the receiver (real or simulated)
        receiver = create_receiver(
            port=self.port,
            demo=self.demo,
            duration_seconds=self.duration_seconds,
        )

        # Create the data logger
        dl = DataLogger(
            participant_id=self.participant_id,
            session_id=self.session_id,
            data_source=data_source,
        )
        csv_path = dl.csv_path

        # Register Ctrl-C handler so the session closes cleanly
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        print(f"\n{'='*60}")
        print(f"  SESSION STARTED")
        print(f"  Participant : {self.participant_id}")
        print(f"  Session ID  : {self.session_id}")
        print(f"  Data source : {data_source.upper()}")
        if self.duration_seconds:
            print(f"  Duration    : {self.duration_seconds} s")
        else:
            print(f"  Duration    : Until Ctrl-C")
        print(f"{'='*60}\n")

        try:
            connected = receiver.connect()
            if not connected:
                raise ConnectionError(
                    f"Could not connect to serial port: {self.port}"
                )

            dl.open()
            self._stop_event.clear()
            row_index = 0

            for reading in receiver.stream():
                if self._stop_event.is_set():
                    break

                # Write to CSV
                dl.write(reading)
                row_index += 1

                # Optional terminal display (every N readings)
                if self.display_readings and row_index % self.display_interval == 0:
                    self._display_reading(reading, row_index)

            status = "completed"

        except FileExistsError as exc:
            status = "failed"
            error_message = str(exc)
            logger.error("Session failed — file already exists: %s", exc)

        except ConnectionError as exc:
            status = "failed"
            error_message = str(exc)
            logger.error("Session failed — connection error: %s", exc)

        except KeyboardInterrupt:
            # Ctrl-C pressed — treat as a clean stop
            status = "stopped"
            logger.info("Session stopped by user (Ctrl-C).")

        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            logger.error("Session failed with unexpected error: %s", exc, exc_info=True)

        finally:
            # Always close the receiver and file
            try:
                receiver.disconnect()
            except Exception:
                pass

            try:
                dl.close()
                rows_written = dl.rows_written
            except Exception:
                pass

            # Restore original Ctrl-C handler
            signal.signal(signal.SIGINT, original_sigint)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        self._result = SessionResult(
            participant_id=self.participant_id,
            session_id=self.session_id,
            data_source=data_source,
            status=status,
            csv_path=csv_path if rows_written > 0 else None,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            rows_written=rows_written,
            error_message=error_message,
        )

        print(self._result.summary())
        return self._result

    def stop(self) -> None:
        """Signal the running session to stop after the current reading."""
        self._stop_event.set()

    @property
    def result(self) -> Optional[SessionResult]:
        """The SessionResult from the last run() call."""
        return self._result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_sigint(self, signum, frame) -> None:
        """Handle Ctrl-C by setting the stop event instead of crashing."""
        print("\n[Ctrl-C detected] Stopping session...")
        self._stop_event.set()

    @staticmethod
    def _display_reading(reading: dict, index: int) -> None:
        """Print a compact summary of one reading to the terminal."""
        print(
            f"  #{index:>5} | "
            f"t={reading.get('timestamp_ms','?'):>6} ms | "
            f"Heel={reading.get('heel','?'):>4}  "
            f"Mid={reading.get('midfoot','?'):>4}  "
            f"Fore={reading.get('forefoot','?'):>4} | "
            f"az={reading.get('acc_z_g','?'):.3f}g"
        )


# ===========================================================================
# Participant ID helpers
# ===========================================================================

def next_participant_id(data_dir: Optional[Path] = None) -> str:
    """
    Suggest the next unused pseudonymous participant ID.

    Scans existing CSVs in data/raw and data/demo, finds the highest
    patient_XXX number and returns the next one.

    Example: if patient_001 and patient_002 CSVs exist, returns "patient_003".
    """
    if data_dir is None:
        data_dir = config.DATA_RAW_DIR

    existing_dirs = [config.DATA_RAW_DIR, config.DATA_DEMO_DIR]
    highest = 0

    for d in existing_dirs:
        if not d.exists():
            continue
        for f in d.glob("patient_*.csv"):
            # Filename: patient_001_session_...
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    num = int(parts[1])
                    highest = max(highest, num)
                except ValueError:
                    pass

    return f"patient_{highest + 1:03d}"


def list_participants(data_dir: Optional[Path] = None) -> List[str]:
    """Return a sorted list of unique participant IDs from saved CSVs."""
    if data_dir is None:
        data_dir = config.DATA_RAW_DIR

    ids = set()
    for d in [config.DATA_RAW_DIR, config.DATA_DEMO_DIR]:
        if not d.exists():
            continue
        for f in d.glob("patient_*.csv"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                ids.add(f"patient_{parts[1]}")

    return sorted(ids)


# ===========================================================================
# Command-line entry point
# ===========================================================================

def _cli():
    """Run a session from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="OA Screening — Start a walking session"
    )
    parser.add_argument(
        "--participant", "-p",
        default=None,
        help="Participant ID (e.g. patient_001). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Session ID. Auto-generated from timestamp if omitted.",
    )
    parser.add_argument(
        "--port",
        default=config.SERIAL_PORT or None,
        help="Serial port (e.g. /dev/cu.usbserial-0001 or COM3).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=config.DEMO_MODE,
        help="Use simulated data (no ESP32 needed).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Stop after this many seconds.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Do not print individual readings.",
    )

    args = parser.parse_args()

    participant_id = args.participant or next_participant_id()
    print(f"Using participant ID: {participant_id}")

    sm = SessionManager(
        participant_id=participant_id,
        session_id=args.session,
        port=args.port,
        demo=args.demo,
        duration_seconds=args.duration,
        display_readings=not args.quiet,
    )
    result = sm.run()
    sys.exit(0 if result.is_successful else 1)


if __name__ == "__main__":
    _cli()
