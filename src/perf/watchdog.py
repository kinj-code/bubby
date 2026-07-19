"""Process Watchdog — monitors and recovers stalled inference processes.

Pings the LLM inference engine periodically. If the process hangs
(common with local GGUF models on Linux), kills and restarts it
without losing application context.

RAM: ~1MB (single monitoring thread).
"""

import threading
import time
import logging
import os
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class WatchdogMonitor:
    """Lightweight background monitor for critical processes."""

    CHECK_INTERVAL = 5.0       # Seconds between health checks
    MAX_STALL_SECONDS = 30.0    # Max time before declaring process hung
    MAX_RESTARTS = 3            # Max auto-restarts before giving up

    def __init__(
        self,
        on_stall: Optional[Callable[[], None]] = None,
        check_interval: float = CHECK_INTERVAL,
        max_stall: float = MAX_STALL_SECONDS,
    ) -> None:
        self._check_interval = check_interval
        self._max_stall = max_stall
        self._on_stall = on_stall
        self._last_heartbeat = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stalls_detected = 0
        self._restarts = 0
        self._registered_services: dict = {}
        logger.info(f"WatchdogMonitor initialized (interval={check_interval}s, max_stall={max_stall}s)")

    def register_service(self, name: str, health_check: Callable[[], bool]) -> None:
        """Register a service for health monitoring."""
        self._registered_services[name] = health_check
        logger.debug(f"Watchdog registered: {name}")

    def heartbeat(self) -> None:
        """Signal that the main process is alive."""
        self._last_heartbeat = time.time()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("WatchdogMonitor started")

    def stop(self) -> None:
        self._running = False
        logger.info("WatchdogMonitor stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(self._check_interval)
            now = time.time()
            stall_time = now - self._last_heartbeat

            if stall_time > self._max_stall:
                self._stalls_detected += 1
                logger.warning(f"STALL DETECTED: {stall_time:.1f}s since last heartbeat (stall #{self._stalls_detected})")
                if self._on_stall and self._restarts < self.MAX_RESTARTS:
                    self._restarts += 1
                    logger.info(f"Auto-restart attempt {self._restarts}/{self.MAX_RESTARTS}")
                    try:
                        self._on_stall()
                    except Exception as e:
                        logger.error(f"Stall handler failed: {e}")
                elif self._restarts >= self.MAX_RESTARTS:
                    logger.error(f"MAX_RESTARTS ({self.MAX_RESTARTS}) reached — giving up on auto-recovery")

            # Check registered services
            for name, check in self._registered_services.items():
                try:
                    if not check():
                        logger.warning(f"Service health check failed: {name}")
                except Exception as e:
                    logger.error(f"Service check error for '{name}': {e}")

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "stalls_detected": self._stalls_detected,
            "restarts": self._restarts,
            "seconds_since_heartbeat": time.time() - self._last_heartbeat,
            "registered_services": list(self._registered_services.keys()),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger.info("=" * 60)
    logger.info("WATCHDOG TEST")

    restarts = []
    def on_stall():
        restarts.append(time.time())
        logger.info("Stall handler called")

    wd = WatchdogMonitor(on_stall=on_stall, check_interval=0.5, max_stall=1.0)
    wd.register_service("test_service", lambda: True)
    wd.start()

    # Wait for heartbeat timeout
    time.sleep(1.5)
    assert len(restarts) >= 1, "Should have triggered stall handler"
    logger.info(f"✓ Stalls detected: {wd.get_stats()['stalls_detected']}")

    # Heartbeat and verify recovery
    wd.heartbeat()
    time.sleep(0.7)
    stats = wd.get_stats()
    assert stats["seconds_since_heartbeat"] < 1.0, f"Should be near 0, got {stats['seconds_since_heartbeat']:.1f}"
    logger.info(f"✓ Heartbeat recovered: {stats}")

    wd.stop()
    logger.info("ALL WATCHDOG TESTS PASSED")