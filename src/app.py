"""Main application entry point for Bubby desktop companion.

Architecture (decoupled):
┌─────────────────────────────────────────────────────┐
│  AutonomyLoop (background thread)                   │
│    → decision_made Signal → on_decision()           │
│    (only for non-verbal UI state: animations,       │
│     wandering, etc.)                                │
├─────────────────────────────────────────────────────┤
│  Vision Pipeline → Reasoning → Synthesis            │
│    → InteractionHandler.display_callback            │
│    → AvatarWidget.set_state() + TTS + Actions       │
│    (for verbal/text output: observations,           │
│     greetings, responses + voice + system commands) │
└─────────────────────────────────────────────────────┘
PHASE 8: Omni-Integration — Avatar UI + TTS Voice + System Agency
"""

import logging
import sys
import os
from typing import Optional

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
from src.llm.inference import LLMInference, LLMConfig
from src.llm.model_manager import ModelManager
from src.voice.tts_engine import TTSEngine, TTSConfig
from src.actions.executor import SystemExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
USE_LLM = os.environ.get("BUBBY_USE_LLM", "1") == "1"
USE_TTS = os.environ.get("BUBBY_USE_TTS", "1") == "1"


def create_behavior_tree() -> BehaviorTree:
    """Create the behavior tree for autonomous decision-making. Returns BehaviorTree instance."""
    def idle_action(context):
        logger.info("Action: Idle")
        return DecisionType.IDLE
    def wander_action(context):
        logger.info("Action: Wander")
        return DecisionType.WANDER
    def sit_action(context):
        logger.info("Action: Sit")
        return DecisionType.SIT
    def user_present(context):
        return context.user_present
    def idle_time_check(context):
        return context.user_idle_time > 5.0

    user_present_cond = Condition("User Present?", user_present)
    idle_action_node = Action("Idle", idle_action)
    idle_sequence = Sequence("Idle Sequence")
    idle_sequence.add_child(user_present_cond)
    idle_sequence.add_child(idle_action_node)
    idle_time_cond = Condition("Idle > 5s?", idle_time_check)
    wander_action_node = Action("Wander", wander_action)
    wander_sequence = Sequence("Wander Sequence")
    wander_sequence.add_child(idle_time_cond)
    wander_sequence.add_child(wander_action_node)
    sit_action_node = Action("Sit", sit_action)
    root = Selector("Root")
    root.add_child(idle_sequence)
    root.add_child(wander_sequence)
    root.add_child(sit_action_node)
    return BehaviorTree(root)


def main() -> None:
    """Main application entry point with omni-integration architecture."""
    logger.info("=" * 60)
    logger.info("BUBBY - Autonomous Desktop Companion (Phase 8: Omni)")
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

    # ── LAYER 6: Interaction Handler (verbal/text/voice/action bridge) ──
    def display_to_overlay(message: InteractionMessage) -> None:
        """Route synthesized messages to avatar UI, TTS, and actions."""
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
        if avatar_widget:
            avatar_widget.set_state(animation, message.text)
        overlay.show_message(
            text=message.text,
            animation=animation,
            event_type=message.event.value,
        )

    def on_action_approval_needed(action_name: str, request) -> None:
        logger.info(f"ACTION APPROVAL NEEDED: {action_name}")
        if avatar_widget:
            avatar_widget.set_state("think", f"May I {action_name}?")

    interaction_handler = InteractionHandler(
        synthesis_engine=synthesis_engine,
        display_callback=display_to_overlay,
        tts_engine=tts_engine,
        action_executor=system_executor,
        action_callback=on_action_approval_needed,
    )

    # ── LAYER 1.5: Avatar Widget (inside overlay) ──
    avatar_config = AvatarConfig(
        show_emotes=True, show_text=True, emote_size=64, bob_animation=True,
    )
    avatar_widget = AvatarWidget(parent=overlay, config=avatar_config)
    avatar_widget.set_state("idle")
    overlay.set_animation_widget(avatar_widget)
    logger.info("Avatar widget embedded in overlay")

    # ── LAYER 7: Autonomy Loop ──
    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree,
        context_manager=context_manager,
        decision_interval=2.0,
    )

    def on_decision(decision: Decision) -> None:
        logger.debug(f"Decision received: {decision}")
        overlay.update_behavior_state(decision)
        if decision.decision_type == DecisionType.WANDER:
            import random
            screen = app.primaryScreen()
            if screen:
                screen_rect = screen.availableGeometry()
                margin = 100
                target_x = random.randint(
                    screen_rect.left() + margin,
                    screen_rect.right() - margin - 400
                )
                target_y = random.randint(
                    screen_rect.top() + margin,
                    screen_rect.bottom() - margin - 400
                )
                decision.params["target_x"] = target_x
                decision.params["target_y"] = target_y
                overlay.wander_to(target_x, target_y)

    autonomy_loop.decision_made.connect(on_decision)

    # ── VISION → REASONING → SYNTHESIS → INTERACTION pipeline ──
    def on_vision_observation(observation_text: str, confidence: float, metadata: dict) -> None:
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
            if msg.text:
                logger.debug(f"Observation: {msg.text[:50]}...")
            else:
                logger.debug("Observation suppressed by Interaction Budget")

    vision_pipeline.set_callback(on_vision_observation)

    # ── STARTUP ──
    overlay.show()
    logger.info("Overlay window shown")
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