"""Main application entry point for Bubby desktop companion.

Architecture (decoupled, thread-safe via Qt Signal/Slot):
- AutonomyLoop (QThread) emits decision_made Signal → main thread slot
- Vision pipeline callbacks → InteractionHandler → overlay signal → main thread
- Zero direct widget calls from background threads
"""

# ── Faulthandler: produce C/C++ stack trace on segfault ──────────
# Must be the very first imports before any Qt libraries.
import faulthandler
import os
import signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

import logging
import sys
from typing import Optional

# Force X11 platform before PySide6 imports to avoid
# Wayland segfaults on headless/XWayland desktops.
if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QPoint

from src.ui.overlay import OverlayWindow
from src.ui.avatar import AvatarWidget, AvatarConfig
from src.brain.decisions import DecisionType, Decision
from src.brain.behavior_tree import (
    BehaviorTree, Selector, Sequence, Condition, Action
)
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

    overlay = OverlayWindow(size=(400, 400), click_through=False)
    behavior_tree = create_behavior_tree()
    context_manager = ContextManager()

    vision_pipeline = VisionPipeline()
    memory_buffer = MemoryBuffer(max_observations=50, max_tokens=2048)
    long_term_memory = LongTermMemory()
    reasoning_bridge = ReasoningBridge(memory_buffer)

    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)
    synthesis_config = LLMSynthesisConfig(
        use_llm=USE_LLM, fallback_to_template=True, min_confidence_for_llm=0.5,
    )
    synthesis_engine = LLMSynthesisEngine(
        persona=persona, long_term_memory=long_term_memory, config=synthesis_config,
    )

    tts_config = TTSConfig(use_subprocess=True)
    tts_engine = TTSEngine(tts_config) if USE_TTS else None
    system_executor = SystemExecutor()

    avatar_config = AvatarConfig(
        show_emotes=True, show_text=True, emote_size=64, bob_animation=True,
    )
    avatar_widget = AvatarWidget(parent=overlay, config=avatar_config)
    avatar_widget.set_state("idle")
    overlay.set_animation_widget(avatar_widget)

    # ── THREAD-SAFE UI dispatch ──
    def display_to_overlay(message: InteractionMessage) -> None:
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
        logger.info(f"[UI->Avatar] {message.event.value}: {message.text[:60]}")
        # Emit signals — Qt delivers to main thread automatically
        overlay.display_message_signal.emit(message.text, animation, message.event.value)
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

    def _on_display_message(text, animation, event_type):
        if avatar_widget:
            avatar_widget.set_state(animation, text)
        overlay.show_message(text=text, animation=animation, event_type=event_type)

    def _on_update_state(state_name, message_text):
        if avatar_widget:
            avatar_widget.set_state(state_name, message_text)

    overlay.display_message_signal.connect(_on_display_message)
    overlay.update_state_signal.connect(_on_update_state)

    # ── User poke → LLM response ──
    def _on_user_poked() -> None:
        """User clicked the character body — ask the LLM to say something."""
        logger.info("User poked Bubby — triggering LLM response")
        interaction_handler.on_user_input(
            "The user just clicked on you / poked you. Say something brief, witty, and in-character."
        )

    overlay.user_poked.connect(_on_user_poked)

    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree, context_manager=context_manager, decision_interval=2.0,
    )

    def on_decision(decision: Decision) -> None:
        logger.debug(f"Decision received: {decision}")
        overlay.behavior_state_signal.emit(decision)
        if decision.decision_type == DecisionType.WANDER:
            import random
            screen = app.primaryScreen()
            if screen:
                r = screen.availableGeometry()
                tx = random.randint(r.left() + 100, r.right() - 500)
                ty = random.randint(r.top() + 100, r.bottom() - 500)
                overlay.wander_to(QPoint(tx, ty))

    autonomy_loop.decision_made.connect(on_decision)
    overlay.behavior_state_signal.connect(overlay.update_behavior_state)

    overlay.show()
    logger.info("Overlay visible")

    vision_pipeline.start(capture_interval=5.0)
    logger.info("Vision pipeline started")
    autonomy_loop.start()
    logger.info("Autonomy loop started")
    interaction_handler.on_greeting(context="startup")

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