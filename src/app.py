"""Main application entry point for Bubby desktop companion.

Architecture (decoupled):
┌─────────────────────────────────────────────────────┐
│  AutonomyLoop (Qt QThread)                          │
│    → decision_made Signal → on_decision()           │
│    (thread-safe: emitted from worker, slot on main) │
├─────────────────────────────────────────────────────┤
│  Vision Pipeline → Reasoning → Synthesis            │
│    → InteractionHandler.display_callback            │
│    → OverlayWindow.display_message_signal (main thd)│
│    → AvatarWidget.set_state() + TTS + Actions       │
└─────────────────────────────────────────────────────┘

All widget mutations go through Qt Signal/Slot — zero direct
calls from background threads to UI objects.
"""

import logging
import os
import sys
from typing import Optional

# Force X11 platform before PySide6 imports to avoid
# Wayland segfaults on headless/XWayland desktops.
if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from src.ui.overlay import OverlayWindow
from src.ui.avatar import AvatarWidget, AvatarConfig
from src.brain.decisions import DecisionType, Decision, make_wander_decision
from src.brain.behavior_tree import (
    BehaviorTree, Selector, Sequence, Condition, Action
)
from src.brain.context_manager import ContextManager
from src.brain.autonomy_loop import AutonomyLoop
from src.brain.reasoning import VisualReasoning, ReasoningBridge
from src.vision.memory_buffer import MemoryBuffer
from src.vision.pipeline import VisionPipeline
from src.memory.long_term_memory import LongTermMemory
from src.persona.config import PersonaConfig, PersonaType
from src.persona.llm_synthesis import LLMSynthesisEngine, LLMSynthesisConfig
from src.interaction.handler import InteractionHandler, InteractionMessage, InteractionEvent
from src.actions.executor import SystemExecutor
from src.voice.tts_engine import TTSEngine, TTSConfig

logger = logging.getLogger(__name__)

USE_LLM = os.environ.get("BUBBY_USE_LLM", "1") == "1"
USE_TTS = os.environ.get("BUBBY_USE_TTS", "0") == "1"


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


def main() -> None:
    """Initialize and launch the Bubby desktop companion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("BUBBY DESKTOP COMPANION — Starting")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # ── LAYER 1: UI Overlay ──
    overlay = OverlayWindow(size=(400, 400), click_through=False)

    # ── LAYER 2: Brain ──
    behavior_tree = create_behavior_tree()
    context_manager = ContextManager()

    # ── LAYER 3: Vision & Memory ──
    vision_pipeline = VisionPipeline()
    memory_buffer = MemoryBuffer(max_observations=50, max_tokens=2048)
    long_term_memory = LongTermMemory()

    # ── LAYER 4: Reasoning Bridge ──
    reasoning_bridge = ReasoningBridge(memory_buffer)

    # ── LAYER 5: Persona & Synthesis ──
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

    # ── LAYER 5.5: TTS Engine (Offline Voice) ──
    tts_config = TTSConfig(use_subprocess=True)
    tts_engine = TTSEngine(tts_config) if USE_TTS else None
    if tts_engine and tts_engine.is_ready():
        logger.info("TTS engine ready")
    elif USE_TTS:
        logger.info("TTS engine initialized (no voice model — download: scripts/download_voice.py)")

    # ── LAYER 5.6: System Executor (Secure Actions) ──
    system_executor = SystemExecutor()
    logger.info(f"System executor ready: {len(system_executor.get_available_actions())} actions")

    # ── LAYER 1.5: Avatar Widget (inside overlay) ──
    avatar_config = AvatarConfig(
        show_emotes=True, show_text=True, emote_size=64, bob_animation=True,
    )
    avatar_widget = AvatarWidget(parent=overlay, config=avatar_config)
    avatar_widget.set_state("idle")
    overlay.set_animation_widget(avatar_widget)
    logger.info("Avatar widget embedded in overlay")

    # ── LAYER 6: Interaction Handler ──
    #
    # THREAD-SAFETY: display_callback emits overlay signals which
    # are dispatched on the main Qt thread automatically.
    # Background threads (VisionPipeline, reasoning_bridge) never
    # call widget methods directly.
    def display_to_overlay(message: InteractionMessage) -> None:
        """Route synthesized messages to avatar UI, TTS, and actions.
           Called from background threads but safe — uses Qt Signal emit."""
        if not message or not message.text:
            return
        animation_map = {
            InteractionEvent.GREETING: "wave",
            InteractionEvent.OBSERVATION: "observe",
            InteractionEvent.RESPONSE: "talk",
            InteractionEvent.STATUS: "idle",
            InteractionEvent.ERROR: "confused",
        }
        animation = animation_map.get(message.event, "idle")
        logger.info(f"[UI→Avatar] {message.event.value}: {message.text[:60]}")
        # Thread-safe: use Signal emit (Qt delivers to main thread)
        overlay.display_message_signal.emit(
            message.text, animation, message.event.value
        )
        overlay.update_state_signal.emit(animation, message.text)

    def on_action_approval_needed(action_name: str, request) -> None:
        logger.info(f"ACTION APPROVAL NEEDED: {action_name}")
        overlay.update_state_signal.emit("think", f"May I {action_name}?")

    interaction_handler = InteractionHandler(
        synthesis_engine=synthesis_engine,
        display_callback=display_to_overlay,
        tts_engine=tts_engine,
        action_executor=system_executor,
        action_callback=on_action_approval_needed,
    )

    # ── Connect overlay signals to actual widget updates (main thread) ──
    def _on_display_message(text: str, animation: str, event_type: str):
        """Slot: called on main thread via Qt Signal dispatch."""
        if avatar_widget:
            avatar_widget.set_state(animation, text)
        overlay.show_message(text=text, animation=animation, event_type=event_type)

    def _on_update_state(state_name: str, message_text: str):
        """Slot: called on main thread via Qt Signal dispatch."""
        if avatar_widget:
            avatar_widget.set_state(state_name, message_text)

    overlay.display_message_signal.connect(_on_display_message)
    overlay.update_state_signal.connect(_on_update_state)

    # ── LAYER 7: Autonomy Loop (QThread — emits Qt signals) ──
    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree,
        context_manager=context_manager,
        decision_interval=2.0,
    )

    def on_decision(decision: Decision) -> None:
        """Slot: called on main thread via Qt Signal dispatch from AutonomyLoop."""
        logger.debug(f"Decision received: {decision}")
        # Thread-safe: use overlay's signal to update behavior state
        overlay.behavior_state_signal.emit(decision)
        if decision.decision_type == DecisionType.WANDER:
            import random
            screen = app.primaryScreen()
            if screen:
                screen_rect = screen.availableGeometry()
                target_x = random.randint(screen_rect.left() + 100, screen_rect.right() - 500)
                target_y = random.randint(screen_rect.top() + 100, screen_rect.bottom() - 500)
                overlay.wander_to(QPoint(target_x, target_y))

    autonomy_loop.decision_made.connect(on_decision)

    # Connect behavior_state_signal to overlay's update method
    overlay.behavior_state_signal.connect(overlay.update_behavior_state)

    # ── Show UI ──
    overlay.show()
    logger.info("Overlay visible")

    # ── START PIPELINES ──
    vision_pipeline.start(capture_interval=5.0)
    logger.info("Vision pipeline started")
    autonomy_loop.start()
    logger.info("Autonomy loop started")
    interaction_handler.on_greeting(context="startup")

    # ── SHUTDOWN ──
    def cleanup():
        logger.info("Shutting down...")
        vision_pipeline.stop()
        autonomy_loop.stop()
        synthesis_engine.shutdown()
        if tts_engine:
            tts_engine.shutdown()
        long_term_memory.clear()
        logger.info("Cleanup complete")

    app.aboutToQuit.connect(cleanup)
    logger.info("Application running — press Ctrl+C or close window to exit")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()