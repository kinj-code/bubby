#!/usr/bin/env python3
"""
UI State Listener — reactive, non-blocking connection to the Rust broadcast server.

Connects to the Rust BroadcastServer (Phase 4.2) via TCP and reads JSON-line
state events in real time. The UI should never poll — it should only listen.

Architecture:
    Rust core → BroadcastServer::broadcast() → TCP → StateListener → Qt signal → UI

The listener runs in a Qt worker thread, emitting Qt signals that the UI's
main thread slots handle. The UI remains fully responsive at 60fps.
"""

import json
import logging
import socket
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Try to import Qt; define a pure-Python fallback for testing/CLI use
try:
    from PySide6.QtCore import QObject, Signal, QThread
    HAS_QT = True
except ImportError:
    HAS_QT = False


class StateEvent:
    """A parsed state update from the Rust core."""

    __slots__ = ("event_type", "state", "detail", "tool", "progress", "timestamp", "raw")

    def __init__(self, data: dict):
        self.event_type = data.get("type", "unknown")
        self.state = data.get("state")
        self.detail = data.get("detail")
        self.tool = data.get("tool")
        self.progress = data.get("progress")
        self.timestamp = data.get("timestamp", 0.0)
        self.raw = data

    def __repr__(self) -> str:
        return (
            f"StateEvent(type={self.event_type}, state={self.state}, "
            f"detail={str(self.detail)[:40]})"
        )


class StateListener:
    """
    Non-blocking TCP listener for Rust broadcast events.

    Usage:
        listener = StateListener(port=9501)
        listener.on_state_change = lambda event: print(f"State is now: {event.state}")
        listener.connect()
        # ... UI runs independently ...
        listener.disconnect()

    The listener spawns a daemon thread that reads lines from the TCP
    socket and dispatches callbacks. The main thread never blocks.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 9501

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_state_change: Optional[Callable[[StateEvent], None]] = None
        self.on_tts_event: Optional[Callable[[StateEvent], None]] = None
        self.on_error: Optional[Callable[[StateEvent], None]] = None
        self.on_any: Optional[Callable[[StateEvent], None]] = None

        # Stats
        self._events_received = 0
        self._connected = False

    def connect(self) -> bool:
        """Open TCP connection and start the reader thread. Non-blocking."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self._host, self._port))
            self._sock.settimeout(None)  # switch to blocking for readline
            self._connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            logger.info(f"StateListener connected to {self._host}:{self._port}")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            logger.warning(f"StateListener connect failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the connection and stop the reader thread."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False
        logger.info("StateListener disconnected")

    def _read_loop(self) -> None:
        """Background thread: read JSON lines and dispatch callbacks."""
        buf = b""
        while self._running and self._sock:
            try:
                data = self._sock.recv(4096)
                if not data:
                    break  # connection closed

                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line.decode("utf-8"))
                        self._dispatch(StateEvent(parsed))
                        self._events_received += 1
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(f"StateListener: bad JSON line: {e}")

            except (ConnectionResetError, BrokenPipeError, OSError):
                break
            except Exception as e:
                logger.error(f"StateListener read error: {e}")
                break

        self._connected = False
        logger.debug("StateListener reader thread exited")

    def _dispatch(self, event: StateEvent) -> None:
        """Route the event to the appropriate callback."""
        if self.on_any:
            self.on_any(event)

        if event.event_type == "state_change" and self.on_state_change:
            self.on_state_change(event)
        elif event.event_type == "tts_event" and self.on_tts_event:
            self.on_tts_event(event)
        elif event.event_type == "error" and self.on_error:
            self.on_error(event)

    @property
    def events_received(self) -> int:
        return self._events_received

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Qt integration layer (optional) ──────────────────────────────

if HAS_QT:
    class QtStateListener(QObject):
        """
        Qt-aware StateListener that emits signals on the main thread.

        Usage:
            listener = QtStateListener()
            listener.state_changed.connect(ui.handle_state_change)
            listener.start(port=9501)
        """

        state_changed = Signal(dict)    # raw JSON dict for Signal compatibility
        tts_event = Signal(dict)
        error_occurred = Signal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._listener = StateListener()
            self._listener.on_state_change = lambda e: self.state_changed.emit(e.raw)
            self._listener.on_tts_event = lambda e: self.tts_event.emit(e.raw)
            self._listener.on_error = lambda e: self.error_occurred.emit(e.raw)

        def start(self, host: str = "127.0.0.1", port: int = 9501):
            self._listener._host = host
            self._listener._port = port
            self._listener.connect()

        def stop(self):
            self._listener.disconnect()


# ── Self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 60)
    logger.info("STATE LISTENER — Standalone Test")
    logger.info("=" * 60)

    listener = StateListener(port=0)  # port 0 won't connect — test local behavior

    # Test 1: connect fails gracefully
    ok = listener.connect()
    logger.info(f"  Connect (no server): {ok} (expected False)")
    assert ok is False
    logger.info("  ✓ Failed connect handled gracefully")

    # Test 2: callback dispatch
    events = []
    listener.on_any = lambda e: events.append(e)
    listener._dispatch(StateEvent({"type": "state_change", "state": "IDLE", "detail": "test"}))
    assert len(events) == 1
    assert events[0].state == "IDLE"
    logger.info("  ✓ Callback dispatch works")

    # Test 3: disconnect is safe
    listener.disconnect()
    assert not listener.is_connected
    logger.info("  ✓ Disconnect safe")

    # Test 4: StateEvent parsing
    e = StateEvent({
        "type": "state_change",
        "state": "EXECUTING",
        "tool": "bash",
        "detail": "Running command",
        "timestamp": 1234567890.0,
    })
    assert e.tool == "bash"
    assert e.timestamp == 1234567890.0
    logger.info("  ✓ StateEvent parsing correct")

    # Test 5: QtStateListener exists (import-only)
    if HAS_QT:
        logger.info(f"  QtStateListener available (Qt {HAS_QT})")
    else:
        logger.info("  QtStateListener skipped (no PySide6)")

    logger.info("\n" + "=" * 60)
    logger.info("STATE LISTENER TESTS PASSED")
    logger.info("=" * 60)