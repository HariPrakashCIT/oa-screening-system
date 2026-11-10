"""
serial_receiver.py — USB serial communication with ESP32.

Responsibilities
----------------
1. List available serial ports on the host computer.
2. Connect to the chosen port at the configured baud rate.
3. Read incoming JSON lines from the ESP32.
4. Parse each JSON line into a Python dictionary.
5. Validate that all required sensor fields are present.
6. Reject malformed messages without crashing.
7. Handle serial disconnection gracefully.
8. Provide a generator interface so callers receive one clean reading at a time.

Simple explanation
------------------
Think of this module like a translator standing at the USB cable.
The ESP32 shouts raw numbers down the wire.
This module catches each message, checks it makes sense, and passes
a clean Python dictionary to whoever is listening.

Usage
-----
    from app.serial_receiver import SerialReceiver

    receiver = SerialReceiver(port="/dev/cu.usbserial-0001", baud=115200)
    receiver.connect()
    for reading in receiver.stream():
        print(reading)
    receiver.disconnect()
"""

import json
import logging
import threading
import time
from typing import Generator, Iterator, List, Optional

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pyserial; degrade gracefully if not installed yet
# ---------------------------------------------------------------------------
try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    logger.warning(
        "pyserial is not installed. "
        "Real hardware mode will not work. "
        "Run:  pip install pyserial"
    )


# ===========================================================================
# Helper — list available ports
# ===========================================================================

def list_serial_ports() -> List[str]:
    """
    Return a list of serial port names currently visible to the OS.

    On macOS these look like /dev/cu.usbserial-XXXX or /dev/cu.SLAB_USBtoUART.
    On Windows they look like COM3, COM4, etc.

    Returns an empty list if pyserial is not installed.
    """
    if not _SERIAL_AVAILABLE:
        logger.warning("pyserial not installed — cannot list serial ports.")
        return []

    ports = [p.device for p in serial.tools.list_ports.comports()]
    logger.debug("Available serial ports: %s", ports)
    return ports


# ===========================================================================
# Main class
# ===========================================================================

class SerialReceiver:
    """
    Reads JSON sensor messages from an ESP32 over USB serial.

    Parameters
    ----------
    port : str
        Serial port name, e.g. "/dev/cu.usbserial-0001" or "COM3".
    baud : int
        Baud rate.  Must match the ESP32 firmware setting (default 115200).
    timeout : float
        How many seconds to wait for a byte before giving up on that read.
    """

    def __init__(
        self,
        port: str,
        baud: int = config.SERIAL_BAUD_RATE,
        timeout: float = config.SERIAL_TIMEOUT_SECONDS,
    ):
        if not _SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial is required for real hardware mode. "
                "Install it with:  pip install pyserial"
            )

        self.port = port
        self.baud = baud
        self.timeout = timeout

        self._serial: Optional[serial.Serial] = None
        self._connected = False
        self._stop_event = threading.Event()

        # Counters for diagnostics
        self.messages_received: int = 0
        self.messages_rejected: int = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the serial connection.

        Returns True on success, False on failure (port busy or not found).
        """
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
            )
            # Allow the ESP32 time to reset after DTR toggle
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            self._connected = True
            logger.info("Connected to %s at %d baud.", self.port, self.baud)
            return True

        except serial.SerialException as exc:
            logger.error("Could not open %s: %s", self.port, exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the serial connection cleanly."""
        self._stop_event.set()
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False
        logger.info(
            "Disconnected from %s. "
            "Received: %d  Rejected: %d",
            self.port,
            self.messages_received,
            self.messages_rejected,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected and bool(self._serial and self._serial.is_open)

    # ------------------------------------------------------------------
    # Reading loop
    # ------------------------------------------------------------------

    def stream(self) -> Generator[dict, None, None]:
        """
        Yield one validated sensor reading dict per ESP32 message.

        This is a generator — the caller drives it with a for-loop.
        Call disconnect() or set the stop event to end the loop.

        Each yielded dict contains all REQUIRED_SENSOR_FIELDS plus:
            _raw_line: the original string (useful for debugging)
        """
        if not self.is_connected:
            raise RuntimeError("Call connect() before stream().")

        self._stop_event.clear()

        while not self._stop_event.is_set():
            try:
                raw_bytes = self._serial.readline()
            except serial.SerialException as exc:
                logger.error("Serial read error: %s — stopping stream.", exc)
                self._connected = False
                break

            if not raw_bytes:
                # Timeout — no data arrived within self.timeout seconds
                continue

            # Decode bytes → string
            try:
                raw_line = raw_bytes.decode("utf-8", errors="replace").strip()
            except Exception as exc:
                logger.debug("Decode error: %s", exc)
                self.messages_rejected += 1
                continue

            # Parse → validate → yield
            reading = self._parse_and_validate(raw_line)
            if reading is not None:
                self.messages_received += 1
                yield reading
            else:
                self.messages_rejected += 1

    def stop(self) -> None:
        """Signal the stream() loop to stop at the next iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_and_validate(raw_line: str) -> Optional[dict]:
        """
        Parse one raw text line from the ESP32.

        Returns a dict if the line is valid JSON with all required fields.
        Returns None if the line is malformed or missing fields.

        This method never raises — it always returns dict or None.
        """
        if not raw_line or raw_line.startswith("//"):
            # Empty line or comment — skip silently
            return None

        # Step 1: JSON parsing
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            logger.debug("JSON parse error: %s | raw='%s'", exc, raw_line[:120])
            return None

        if not isinstance(data, dict):
            logger.debug("Expected JSON object, got: %s", type(data).__name__)
            return None

        # Step 2: Check all required fields are present
        missing = [
            field
            for field in config.REQUIRED_SENSOR_FIELDS
            if field not in data
        ]
        if missing:
            logger.debug(
                "Message missing fields %s | raw='%s'", missing, raw_line[:120]
            )
            return None

        # Step 3: Check all required values are numeric (bool is NOT acceptable —
        #          Python's bool subclasses int, so we must exclude it explicitly)
        non_numeric = [
            field
            for field in config.REQUIRED_SENSOR_FIELDS
            if isinstance(data[field], bool) or not isinstance(data[field], (int, float))
        ]
        if non_numeric:
            logger.debug(
                "Non-numeric values for fields %s | raw='%s'",
                non_numeric,
                raw_line[:120],
            )
            return None

        # Step 4: Attach the raw line for debugging (not saved to CSV)
        data["_raw_line"] = raw_line
        return data

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SerialReceiver(port={self.port!r}, baud={self.baud}, "
            f"connected={self.is_connected})"
        )


# ===========================================================================
# Simulated receiver — produces synthetic data without hardware
# ===========================================================================

class SimulatedSerialReceiver:
    """
    Generates realistic synthetic sensor data for demo and testing.

    Uses the same interface as SerialReceiver so the rest of the code
    works identically regardless of whether hardware is attached.

    The simulated walk has a realistic gait pattern:
      - Heel loads first, then midfoot, then forefoot.
      - Accelerometer shows gentle body movement.
      - Gyroscope shows slight rotation.

    Data is clearly marked with data_source = "simulated".
    """

    def __init__(
        self,
        sample_rate_hz: int = config.DEMO_SAMPLE_RATE_HZ,
        duration_seconds: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        sample_rate_hz : int
            How many readings to generate per second.
        duration_seconds : float | None
            If set, the stream ends after this many seconds.
            If None, streams indefinitely until stop() is called.
        """
        import numpy as np  # only needed here

        self.sample_rate_hz = sample_rate_hz
        self.duration_seconds = duration_seconds
        self._stop_event = threading.Event()
        self._np = np

        self.messages_received: int = 0
        self.messages_rejected: int = 0

        logger.info(
            "SimulatedSerialReceiver ready at %d Hz "
            "(duration=%s s).",
            sample_rate_hz,
            duration_seconds,
        )

    # Satisfy the same interface
    def connect(self) -> bool:
        self._stop_event.clear()
        logger.info("[SIMULATED] Hardware connection skipped — demo mode.")
        return True

    def disconnect(self) -> None:
        self._stop_event.set()
        logger.info("[SIMULATED] Stream stopped.")

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_connected(self) -> bool:
        return not self._stop_event.is_set()

    def stream(self) -> Generator[dict, None, None]:
        """Yield synthetic sensor readings at the configured rate."""
        np = self._np
        interval = 1.0 / self.sample_rate_hz
        start_time = time.time()
        t = 0  # ms counter

        # Gait cycle period: ~1 second (50 samples at 50 Hz)
        cycle_len = self.sample_rate_hz

        while not self._stop_event.is_set():
            if self.duration_seconds is not None:
                if time.time() - start_time >= self.duration_seconds:
                    break

            # Position within the gait cycle (0 → 1)
            phase = (t % cycle_len) / cycle_len

            # -- Pressure pattern: heel → midfoot → forefoot bell curves --
            heel     = int(800 * np.exp(-((phase - 0.15) ** 2) / 0.01) + np.random.normal(0, 15))
            midfoot  = int(600 * np.exp(-((phase - 0.45) ** 2) / 0.02) + np.random.normal(0, 12))
            forefoot = int(850 * np.exp(-((phase - 0.75) ** 2) / 0.01) + np.random.normal(0, 15))

            # Clamp to valid ADC range
            heel     = max(0, min(4095, heel))
            midfoot  = max(0, min(4095, midfoot))
            forefoot = max(0, min(4095, forefoot))

            # -- Accelerometer: gravity dominates Z, small walk oscillation --
            acc_x = round(0.05 * np.sin(2 * np.pi * phase)      + np.random.normal(0, 0.02), 4)
            acc_y = round(0.08 * np.sin(2 * np.pi * phase + 0.5) + np.random.normal(0, 0.02), 4)
            acc_z = round(0.98                                    + np.random.normal(0, 0.02), 4)

            # -- Gyroscope: gentle rotation during stride --
            gyro_x = round(2.0 * np.sin(2 * np.pi * phase)      + np.random.normal(0, 0.3), 4)
            gyro_y = round(1.5 * np.cos(2 * np.pi * phase)      + np.random.normal(0, 0.2), 4)
            gyro_z = round(0.8 * np.sin(4 * np.pi * phase)      + np.random.normal(0, 0.15), 4)

            reading = {
                "timestamp_ms": t * (1000 // self.sample_rate_hz),
                "heel":         heel,
                "midfoot":      midfoot,
                "forefoot":     forefoot,
                "acc_x_g":      acc_x,
                "acc_y_g":      acc_y,
                "acc_z_g":      acc_z,
                "gyro_x_dps":   gyro_x,
                "gyro_y_dps":   gyro_y,
                "gyro_z_dps":   gyro_z,
                "_raw_line":    "[simulated]",
            }

            self.messages_received += 1
            yield reading

            t += 1
            time.sleep(interval)


# ===========================================================================
# Factory function — returns the right receiver based on config
# ===========================================================================

def create_receiver(
    port: Optional[str] = None,
    demo: Optional[bool] = None,
    duration_seconds: Optional[float] = None,
) -> "SerialReceiver | SimulatedSerialReceiver":
    """
    Return either a real SerialReceiver or a SimulatedSerialReceiver.

    Decision logic:
      1. If demo=True (or config.DEMO_MODE is True) → Simulated.
      2. If pyserial is not installed → Simulated.
      3. Otherwise → Real hardware receiver on `port`.

    Parameters
    ----------
    port : str | None
        Serial port for real hardware (ignored in demo mode).
    demo : bool | None
        Override config.DEMO_MODE.
    duration_seconds : float | None
        Only used for simulated mode (limits stream length).
    """
    use_demo = demo if demo is not None else config.DEMO_MODE

    if use_demo or not _SERIAL_AVAILABLE:
        reason = "DEMO_MODE=true" if use_demo else "pyserial not installed"
        logger.info("Using SimulatedSerialReceiver (%s).", reason)
        return SimulatedSerialReceiver(duration_seconds=duration_seconds)

    chosen_port = port or config.SERIAL_PORT
    if not chosen_port:
        available = list_serial_ports()
        if not available:
            logger.warning(
                "No serial port specified and none found. "
                "Falling back to simulated mode."
            )
            return SimulatedSerialReceiver(duration_seconds=duration_seconds)
        chosen_port = available[0]
        logger.info("No port specified — auto-selected: %s", chosen_port)

    return SerialReceiver(port=chosen_port)
