#!/usr/bin/env python3
"""
Autonomous Mission Control — runs Bubby as a continuous, self-monitoring process
with the FULL cognition stack (vision, reasoning, LLM, persona, autonomy loop).

Binds all Phase 1-13 subsystems into a single supervised daemon loop.
Designed for nohup execution during extended unattended operation (days to weeks).

Usage:
    nohup QT_QPA_PLATFORM=offscreen python3 run_autonomous.py > logs/session_output.log 2>&1 &

Pre-Flight Checklist:
    1. Disable suspend/hibernate: sudo systemctl mask sleep.target suspend.target hibernate.target
    2. Set performance mode: sudo cpupower frequency-set -g performance
    3. Verify tests pass: python test_phase_12_unified.py
    4. Create initial backup: python -c "from src.data.persistence import CheckpointManager; CheckpointManager().snapshot('data/knowledge/feedback.db')"
"""

import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from src.brain.decisions import DecisionType, Decision
from src.brain.behavior_tree import BehaviorTree, Selector, Sequence, Condition, Action
from src.brain.context_manager import ContextManager
from src.brain.autonomy_loop import AutonomyLoop
from src.brain.reasoning import ReasoningBridge
from src.vision.memory_buffer import MemoryBuffer
from src.vision.pipeline import VisionPipeline
from src.memory.long_term_memory import LongTermMemory
from src.persona.config import PersonaConfig, PersonaType
from src.persona.llm_synthesis import LLMSynthesisEngine, LLMSynthesisConfig
from src.interaction.handler import InteractionHandler, InteractionMessage, InteractionEvent
from src.actions.executor import SystemExecutor
from src.voice.tts_engine import TTSEngine, TTSConfig
from src.perf.profiler import PipelineProfiler
from src.perf.watchdog import WatchdogMonitor
from src.data.persistence import CheckpointManager
from src.sensors.terminal import TerminalSensor
from src.integrations.calendar import CalendarSensor
from src.brain.critic import CognitiveCritic
from src.actions.policy import ActionPolicy

# ── Logging Setup ──
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "mission_control.log",
            maxBytes=5_000_000, backupCount=5,
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mission_control")

# ── Global State ──
running = True
start_time = time.time()
profiler = PipelineProfiler()

USE_LLM = os.environ.get("BUBBY_USE_LLM", "1") == "1"
USE_TTS = os.environ.get("BUBBY_USE_TTS", "0") == "1"  # Off by default in daemon mode


def handle_shutdown(signum=None, frame=None):
    global running
    logger.info(f"Shutdown signal received (signal={signum}). Wrapping up...")
    running = False


def create_behavior_tree() -> BehaviorTree:
    """Create the behavior tree for autonomous decision-making."""
    def idle_action(context):
        return DecisionType.IDLE
    def wander_action(context):
        return DecisionType.WANDER
    def sit_action(context):
        return DecisionType.SIT
    def user_present(context):
        return context.user_present
    def idle_time_check(context):
        return context.user_idle_time > 5.0

    idle_sequence = Sequence("Idle Sequence")
    idle_sequence.add_child(Condition("User Present?", user_present))
    idle_sequence.add_child(Action("Idle", idle_action))
    wander_sequence = Sequence("Wander Sequence")
    wander_sequence.add_child(Condition("Idle > 5s?", idle_time_check))
    wander_sequence.add_child(Action("Wander", wander_action))
    root = Selector("Root")
    root.add_child(idle_sequence)
    root.add_child(wander_sequence)
    root.add_child(Action("Sit", sit_action))
    return BehaviorTree(root)


def build_cognition_stack(profiler, watchdog):
    """Construct the full autonomy pipeline (no Qt UI, cognition only)."""
    logger.info("=" * 60)
    logger.info("COGNITION STACK INITIALIZATION")
    logger.info("=" * 60)

    # ── Brain ──
    behavior_tree = create_behavior_tree()
    context_manager = ContextManager()

    # ── Vision & Memory ──
    vision_pipeline = VisionPipeline()
    memory_buffer = MemoryBuffer(max_observations=50, max_tokens=2048)
    long_term_memory = LongTermMemory()
    reasoning_bridge = ReasoningBridge(memory_buffer)

    # ── Persona & Synthesis (template-only in daemon mode to save RAM) ──
    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)
    synthesis_config = LLMSynthesisConfig(
        use_llm=USE_LLM,
        fallback_to_template=True,
        min_confidence_for_llm=0.5,
    )
    synthesis_engine = LLMSynthesisEngine(
        persona=persona,
        long_term_memory=long_term_memory,
        config=synthesis_config,
    )

    # ── TTS (optional, disabled by default in daemon) ──
    tts_engine = None
    if USE_TTS:
        tts_engine = TTSEngine(TTSConfig(use_subprocess=True))
        if tts_engine.is_ready():
            logger.info("TTS engine ready")

    # ── System Executor ──
    system_executor = SystemExecutor()
    logger.info(f"System executor: {len(system_executor.get_available_actions())} actions")

    # ── Cognitive Critic + Action Policy (audit remediation: live validation) ──
    critic = CognitiveCritic(
        action_executor=system_executor,
        action_policy=ActionPolicy(strict_mode=True),
    )
    logger.info(f"Critic initialized (groundedness + provenance gating)")

    # ── Interaction Handler (no display callback in daemon — logs only) ──
    def log_only_callback(message: InteractionMessage) -> None:
        if message and message.text:
            logger.info(f"[OUTPUT] {message.event.value}: {message.text[:80]}")

    interaction_handler = InteractionHandler(
        synthesis_engine=synthesis_engine,
        display_callback=log_only_callback,
        tts_engine=tts_engine,
        action_executor=system_executor,
        cognitive_critic=critic,
        action_policy=critic._action_policy,
    )

    # ── Autonomy Loop ──
    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree,
        context_manager=context_manager,
        decision_interval=2.0,
    )

    def on_decision(decision: Decision) -> None:
        profiler.stop("autonomy_decision")
        profiler.start("autonomy_decision")
        logger.debug(f"Decision: {decision.decision_type.value}")

    autonomy_loop.decision_made.connect(on_decision)

    # ── Vision → Reasoning → Synthesis pipeline ──
    def on_vision_observation(observation_text: str, confidence: float, metadata: dict) -> None:
        watchdog.heartbeat()
        memory_buffer.add_observation(description=observation_text, metadata=metadata)
        context_summary = memory_buffer.get_context_window(max_tokens=500)
        from src.brain.decisions import ScreenContext
        screen_context = ScreenContext(
            user_present=context_manager.user_present,
            user_idle_time=context_manager.user_idle_time,
            content_type=metadata.get("content_type", "unknown"),
            content_confidence=confidence,
        )
        reasoning = reasoning_bridge.reason(screen_context)
        if reasoning:
            msg = interaction_handler.on_observation(reasoning, context_text=context_summary)

    vision_pipeline.set_callback(on_vision_observation)

    logger.info("✓ Cognition stack initialized")
    return autonomy_loop, vision_pipeline, synthesis_engine, tts_engine, long_term_memory, system_executor, interaction_handler


def run_mission():
    """Main autonomous mission loop — FULL cognition stack."""
    global running

    logger.info("")
    logger.info("=" * 60)
    logger.info("MISSION STARTED — Autonomous Cognition Loop")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)
    logger.info("")

    # ── Qt Application (headless — QT_QPA_PLATFORM=offscreen required) ──
    app = QApplication(sys.argv)

    # ── Watchdog with real liveness check ──
    def real_llm_health_check():
        """
        Ping the synthesis engine with an actual inference call under timeout.
        Returns True only if the call completes within budget (no hang).
        Falls back to template synthesis if no LLM model loaded — still a real probe.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            def _probe():
                synth_engine._template_engine.synthesize(
                    reasoning=None,
                    context_text="health_check",
                    trigger_type="status",
                )
                return True
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_probe)
                return future.result(timeout=5.0)
        except Exception:
            return False

    watchdog = WatchdogMonitor(
        on_stall=lambda: logger.error("STALL DETECTED — Watchdog intervened"),
        check_interval=10.0,
        max_stall=60.0,
    )
    watchdog.register_service("llm_inference", real_llm_health_check)

    # ── Build cognition stack ──
    autonomy_loop, vision_pipeline, synth_engine, tts_engine, ltm, executor, handler = \
        build_cognition_stack(profiler, watchdog)

    # ── Sensors ──
    checkpoint = CheckpointManager()
    terminal = TerminalSensor()
    calendar = CalendarSensor()

    # ── Start everything ──
    watchdog.start()
    watchdog.heartbeat()  # Reset stall timer immediately after start
    autonomy_loop.start()
    vision_pipeline.start(capture_interval=5.0)
    handler.on_greeting(context="autonomous_startup")
    logger.info("All subsystems started — cognition loop active")

    # ── Periodic maintenance timer (60s) ──
    iteration = 0
    last_backup_hour = -1

    def periodic_tick():
        nonlocal iteration, last_backup_hour
        if not running:
            app.quit()
            return

        iteration += 1
        watchdog.heartbeat()
        profiler.start("sensor_polling")

        # Terminal sensor — route through handler for companion output
        try:
            term_state = terminal.poll()
            if term_state.event.value != "unknown" and term_state.urgency > 0.5:
                event_text = f"Terminal error detected: {term_state.event.value}. {term_state.error_summary}"
                handler.on_sensor_event(
                    event_text=event_text,
                    urgency=term_state.urgency,
                    source="terminal",
                )
        except Exception as e:
            logger.debug(f"Terminal poll: {e}")

        # Calendar sensor — route through handler for companion output
        try:
            cal_events = calendar.poll()
            for evt in cal_events:
                if evt.is_imminent:
                    event_text = f"Calendar alert: '{evt.title}' in {evt.minutes_until:.0f} minutes"
                    handler.on_sensor_event(
                        event_text=event_text,
                        urgency=evt.urgency,
                        source="calendar",
                    )
        except Exception as e:
            logger.debug(f"Calendar poll: {e}")

        profiler.stop("sensor_polling")

        # Hourly backup
        now = datetime.now()
        if now.hour != last_backup_hour:
            last_backup_hour = now.hour
            for path in ["data/knowledge/feedback.db", "data/knowledge/graph.db"]:
                p = Path(path)
                if p.exists():
                    checkpoint.snapshot(str(p), f"hourly_{now.strftime('%H%M')}")
            logger.info(f"Hourly backup at {now.strftime('%H:%M')}")

        # Log health every 15 iterations
        if iteration % 15 == 0:
            wd_stats = watchdog.get_stats()
            logger.info(
                f"Heartbeat #{iteration}: stalls={wd_stats['stalls_detected']}, "
                f"restarts={wd_stats['restarts']}, "
                f"pipeline={profiler.get_total_pipeline_ms():.0f}ms"
            )

    maintenance_timer = QTimer()
    maintenance_timer.timeout.connect(periodic_tick)
    maintenance_timer.start(60_000)  # Every 60 seconds

    # ── Shutdown handler ──
    def cleanup():
        global running
        running = False
        logger.info("")
        logger.info("=" * 60)
        logger.info("MISSION SHUTDOWN — Saving state...")
        watchdog.stop()
        vision_pipeline.stop()
        autonomy_loop.stop()
        synth_engine.shutdown()
        if tts_engine:
            tts_engine.shutdown()
        ltm.clear()
        for path in ["data/knowledge/feedback.db", "data/knowledge/graph.db"]:
            p = Path(path)
            if p.exists():
                checkpoint.snapshot(str(p), "shutdown")
        uptime_hours = (time.time() - start_time) / 3600
        logger.info(f"Total uptime: {uptime_hours:.1f} hours ({uptime_hours/24:.1f} days)")
        logger.info(f"Watchdog stalls: {watchdog.get_stats()['stalls_detected']}")
        logger.info(f"Profiler: {profiler.get_heatmap()}")
        logger.info("Mission complete. Logs: logs/mission_control.log")
        logger.info("=" * 60)

    app.aboutToQuit.connect(cleanup)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Cognition loop running — press Ctrl+C or send SIGTERM to stop")
    sys.exit(app.exec())


if __name__ == "__main__":
    run_mission()