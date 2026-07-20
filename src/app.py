"""Main application entry point for Bubby desktop companion.

Architecture (decoupled, thread-safe via Qt Signal/Slot):
- AutonomyLoop (QThread) emits decision_made Signal → main thread slot
- Vision pipeline callbacks → InteractionHandler → overlay signal → main thread
- Zero direct widget calls from background threads
- LLM async responses marshalled via Qt Signal to main thread
"""

# ── Faulthandler: produce C/C++ stack trace on segfault ──────────
import faulthandler
import os
import signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

# ── Load .env before ANY other imports ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
import sys
from pathlib import Path
from typing import Optional

# Force X11 platform before PySide6 imports
if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtCore import QTimer, QPoint, QObject, Signal

from src.ui.overlay import OverlayWindow
from src.ui.avatar import AvatarWidget, AvatarConfig
from src.ui.settings_window import SettingsWindow
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
from src.persona.synthesis import SynthesizedResponse
from src.interaction.handler import InteractionHandler, InteractionMessage, InteractionEvent
from src.actions.executor import SystemExecutor
from src.voice.tts_engine import TTSEngine, TTSConfig

logger = logging.getLogger(__name__)

USE_LLM = os.environ.get("BUBBY_USE_LLM", "1") == "1"
USE_TTS = os.environ.get("BUBBY_USE_TTS", "0") == "1"


# ══ Thread-safe LLM response bridge ══
class LLMResponseBridge(QObject):
    """
    Marshals LLM responses from background threads to the Qt main thread.

    The LLM generate_async callback runs on a background daemon thread.
    This bridge converts those callbacks into signals that Qt delivers
    to the main thread, where UI updates are safe.
    """
    response_ready = Signal(object)  # SynthesizedResponse


def create_behavior_tree() -> BehaviorTree:
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

    # ══ LLM Response Bridge (thread-safe signal) ══
    llm_bridge = LLMResponseBridge()

    def _on_llm_response(response: SynthesizedResponse) -> None:
        """SLOT: receives SynthesizedResponse on main thread via signal."""
        if not response or not response.text:
            logger.warning("LLM returned empty response — no UI update")
            return
        try:
            message = InteractionMessage(
                text=response.text,
                event=InteractionEvent.RESPONSE,
                animation=getattr(response, 'animation', 'talk'),
                source="synthesis",
            )
            # Update overlay and avatar
            overlay.display_message_signal.emit(
                message.text,
                message.animation,
                message.event.value,
            )
            overlay.update_state_signal.emit(message.animation, message.text)
            logger.info(f"[LLM→UI] {message.text[:60]}")
        except Exception as e:
            logger.error(f"Failed to dispatch LLM response to UI: {e}")

    # Connect bridge signal to main thread slot
    llm_bridge.response_ready.connect(_on_llm_response)

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
        overlay.show_message(text=text, animation=animation, event_type=event_type)

    def _on_update_state(state_name, message_text):
        if avatar_widget:
            avatar_widget.set_state(state_name, message_text)
        overlay.set_state(state_name, message_text)

    overlay.display_message_signal.connect(_on_display_message)
    overlay.update_state_signal.connect(_on_update_state)

    # ══ Async LLM dispatch (NEVER blocks the Qt event loop) ══
    import threading as _threading

    def _dispatch_llm_async(context_text: str, trigger_type: str = "user_input") -> None:
        """Run LLM synthesis in a background daemon thread, emit via bridge signal."""

        # Get template response immediately (instant)
        try:
            template = synthesis_engine._template_engine.synthesize(
                reasoning=None,
                context_text=context_text,
                trigger_type=trigger_type,
            )
        except Exception as e:
            logger.warning(f"Template synthesis failed: {e}")
            template = None

        def _bg_task():
            try:
                # If synthesis engine has generate_async, use it
                if hasattr(synthesis_engine, 'generate_async') and template:
                    def _on_result(response):
                        # This callback runs on bg thread — emit signal
                        llm_bridge.response_ready.emit(response)

                    synthesis_engine.generate_async(
                        reasoning=None,
                        context_text=context_text,
                        template_response=template,
                        trigger_type=trigger_type,
                        callback=_on_result,
                    )
                elif template:
                    # Direct response via bridge
                    llm_bridge.response_ready.emit(template)
                else:
                    # Emergency fallback
                    fallback = SynthesizedResponse(
                        text="Hey! I'm here. What's up? 😊",
                        animation="wave",
                    )
                    llm_bridge.response_ready.emit(fallback)
            except Exception as e:
                logger.error(f"LLM dispatch error: {e}", exc_info=True)
                # Emergency fallback on error
                fallback = SynthesizedResponse(
                    text="Sorry, I hit a snag. Try again?",
                    animation="confused",
                )
                llm_bridge.response_ready.emit(fallback)

        t = _threading.Thread(target=_bg_task, daemon=True)
        t.start()

    def _on_user_poked() -> None:
        logger.info("User poked Bubby — triggering async LLM response")
        _dispatch_llm_async(
            "The user just clicked on you / poked you. Say something brief, witty, and in-character.",
            trigger_type="user_input",
        )

    overlay.user_poked.connect(_on_user_poked)

    def _on_user_message(text: str) -> None:
        """User typed a message or dropped files — dispatch async LLM."""
        logger.info(f"User input from overlay: {text[:80]}")

        if text.startswith("[DROPPED FILES]"):
            # Extract file paths from the message
            file_part = text.replace("[DROPPED FILES]\n", "").strip()
            file_paths = [p.strip() for p in file_part.split("\n") if p.strip()]
            file_summary = "\n".join(file_paths[:5])  # Limit to 5 files
            file_count = len(file_paths)

            # ══ FIX: Route through interaction_handler for proper LLM processing ══
            context = (
                f"The user dropped {file_count} file(s) on you. "
                f"Files:\n{file_summary}\n\n"
                f"Acknowledge the files briefly and offer to help with them. "
                f"Keep your response to 1-2 sentences."
            )
            # Also show in chat
            _dispatch_llm_async(context, trigger_type="user_input")
        else:
            _dispatch_llm_async(text, trigger_type="user_input")

    overlay.user_message_submitted.connect(_on_user_message)

    # ── Settings button ──────────────────────────────────────────
    def _open_settings() -> None:
        dlg = SettingsWindow(overlay)
        dlg.exec()
    overlay.settings_requested.connect(_open_settings)

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

    # ── First-time setup wizard ──────────────────────────────────
    llm_path = os.environ.get("BUBBY_LLM_PATH", "").strip()
    if not llm_path or not Path(os.path.abspath(llm_path)).exists():
        logger.info("No LLM model configured — showing setup wizard")
        QMessageBox.information(
            overlay,
            "Welcome to Bubby! 🫧",
            (
                "Welcome to Bubby — your friendly desktop companion slime!\n\n"
                "Before we begin, please select your local LLM model (.gguf file).\n"
                "This will let Bubby talk to you naturally.\n\n"
                "If you don't have one yet, you can download one from:\n"
                "  https://huggingface.co/models?search=GGUF\n\n"
                "Click OK to select your model file."
            ),
        )
        path, _ = QFileDialog.getOpenFileName(
            overlay,
            "Select GGUF Model File",
            str(Path(__file__).parent.parent / "models" / "llm"),
            "GGUF Models (*.gguf);;All Files (*)",
        )
        if path:
            env_path = Path(__file__).parent.parent / ".env"
            lines = env_path.read_text().splitlines() if env_path.exists() else []
            found = False
            new_lines = []
            for line in lines:
                if line.startswith("BUBBY_LLM_PATH="):
                    new_lines.append(f"BUBBY_LLM_PATH={path}")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"BUBBY_LLM_PATH={path}")
            env_path.write_text("\n".join(new_lines) + "\n")
            os.environ["BUBBY_LLM_PATH"] = path
            logger.info(f"User selected model: {path}")
            QMessageBox.information(
                overlay,
                "Model Saved!",
                f"✅ LLM model configured!\n\nRestart Bubby to start chatting.\n\nModel: {path}",
            )
        else:
            logger.info("User skipped model selection — template mode only")

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