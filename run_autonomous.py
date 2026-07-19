#!/usr/bin/env python3
"""
Autonomous Mission Control — runs Bubby as a continuous, self-monitoring process.

Binds all Phase 1-13 subsystems into a single daemon loop. Designed for
nohup execution during extended unattended operation (days to weeks).

Usage:
    nohup python3 run_autonomous.py > logs/session_output.log 2>&1 &

Pre-Flight Checklist:
    1. Disable suspend/hibernate: systemctl mask sleep.target suspend.target
    2. Set performance mode: cpupower frequency-set -g performance
    3. Verify tests pass: python test_phase_12_unified.py
    4. Create initial backup: python -c "from src.data.persistence import CheckpointManager; CheckpointManager().snapshot('data/knowledge/feedback.db')"
"""

import logging
import logging.handlers
import signal
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.perf.watchdog import WatchdogMonitor
from src.data.persistence import CheckpointManager
from src.sensors.terminal import TerminalSensor
from src.integrations.calendar import CalendarSensor

# ── Logging Setup ──
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "mission_control.log",
            maxBytes=5_000_000,  # 5MB
            backupCount=5,
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mission_control")

# ── Global State ──
running = True
start_time = time.time()
restart_count = 0


def handle_shutdown(signum=None, frame=None):
    """Graceful shutdown handler."""
    global running
    logger.info(f"Shutdown signal received (signal={signum}). Wrapping up...")
    running = False


# ── Mission Initialization ──
def init_subsystems():
    """Initialize all monitoring and persistence subsystems."""
    logger.info("=" * 60)
    logger.info("MISSION CONTROL — Initializing Subsystems")
    logger.info("=" * 60)

    # Watchdog — monitors LLM/VLM process health
    watchdog = WatchdogMonitor(
        on_stall=lambda: logger.error("STALL DETECTED — Watchdog intervened"),
        check_interval=10.0,
        max_stall=60.0,
    )

    # Register critical services
    def llm_health_check():
        try:
            from src.llm.inference import LLMInference
            return True  # If import succeeds, engine is available
        except Exception:
            return False

    watchdog.register_service("llm_engine", llm_health_check)
    logger.info("Watchdog initialized with llm_engine health check")

    # Checkpoint Manager — periodic backups
    checkpoint = CheckpointManager()
    logger.info("CheckpointManager initialized")

    # Terminal Sensor — monitor for build errors
    terminal = TerminalSensor()
    logger.info("TerminalSensor initialized")

    # Calendar Sensor — monitor for imminent deadlines
    calendar = CalendarSensor()
    logger.info("CalendarSensor initialized")

    return watchdog, checkpoint, terminal, calendar


def periodic_maintenance(checkpoint, profiler_stats):
    """Run hourly maintenance tasks."""
    now = datetime.now()

    # Hourly backup
    if now.minute == 0:
        critical_paths = [
            "data/knowledge/feedback.db",
            "data/knowledge/graph.db",
            "data/memory/",
        ]
        for path in critical_paths:
            full_path = Path(path)
            if full_path.exists():
                checkpoint.snapshot(str(full_path), f"hourly_{now.strftime('%H%M')}")
        logger.info(f"Hourly backup completed at {now.strftime('%H:%M')}")

    # Daily summary
    if now.hour == 0 and now.minute == 0:
        uptime_hours = (time.time() - start_time) / 3600
        logger.info("=" * 40)
        logger.info(f"DAILY SUMMARY — Uptime: {uptime_hours:.1f} hours")
        logger.info(f"Profiler stats: {profiler_stats}")
        logger.info("=" * 40)


def run_mission():
    """Main autonomous mission loop."""
    global running, restart_count

    logger.info("")
    logger.info("=" * 60)
    logger.info("MISSION STARTED — Autonomous Stress Test")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)
    logger.info("")

    # Init
    watchdog, checkpoint, terminal, calendar = init_subsystems()
    watchdog.start()

    # Lazy-load profiler (doesn't need model loaded)
    from src.perf.profiler import PipelineProfiler
    profiler = PipelineProfiler()

    iteration = 0
    try:
        while running:
            iteration += 1
            watchdog.heartbeat()

            # ── Sensor Polling ──
            profiler.start("sensor_polling")

            # Terminal sensor — check for build errors
            try:
                term_state = terminal.poll()
                if term_state.event.value != "unknown":
                    logger.info(f"Terminal: {term_state.event.value} (urgency={term_state.urgency:.2f})")
            except Exception as e:
                logger.debug(f"Terminal sensor poll skipped: {e}")

            # Calendar sensor — check for imminent events
            try:
                cal_events = calendar.poll()
                if cal_events:
                    for evt in cal_events:
                        if evt.is_imminent:
                            logger.info(f"CALENDAR ALERT: '{evt.title}' in {evt.minutes_until:.0f} min (urgency={evt.urgency:.2f})")
            except Exception as e:
                logger.debug(f"Calendar sensor poll skipped: {e}")

            profiler.stop("sensor_polling")

            # ── Periodic Maintenance ──
            profiler_stats = profiler.get_stats()
            periodic_maintenance(checkpoint, profiler_stats)

            # ── Log health every 15 minutes ──
            if iteration % 15 == 0:
                wd_stats = watchdog.get_stats()
                cp_stats = checkpoint.get_stats()
                logger.info(
                    f"Heartbeat #{iteration}: watchdog_ok={wd_stats['stalls_detected'] == 0}, "
                    f"restarts={wd_stats['restarts']}, "
                    f"pipeline_ms={profiler_stats['total_last_pipeline_ms']:.1f}"
                )

            # ── Sleep — 60 second loop ──
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Mission interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Mission aborted due to error: {e}", exc_info=True)
    finally:
        # ── Mission Shutdown ──
        logger.info("")
        logger.info("=" * 60)
        logger.info("MISSION SHUTDOWN — Saving state...")
        logger.info("=" * 60)

        watchdog.stop()

        # Final backup
        for path in ["data/knowledge/feedback.db", "data/knowledge/graph.db"]:
            full_path = Path(path)
            if full_path.exists():
                checkpoint.snapshot(str(full_path), "shutdown")

        uptime_hours = (time.time() - start_time) / 3600
        logger.info(f"Total mission uptime: {uptime_hours:.1f} hours ({uptime_hours/24:.1f} days)")
        logger.info(f"Watchdog stalls detected: {watchdog.get_stats()['stalls_detected']}")
        logger.info(f"Watchdog restarts: {watchdog.get_stats()['restarts']}")
        logger.info(f"Final profiler heatmap: {profiler.get_stats()}")
        logger.info("Mission complete. Logs saved to logs/mission_control.log")
        logger.info("=" * 60)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    run_mission()