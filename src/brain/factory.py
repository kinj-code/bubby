"""
Shared cognition stack factory — builds the full autonomy pipeline
with CognitiveCritic, ActionPolicy, Vision, sensors, and safety gates.

Both `app.py` (GUI mode) and `run_autonomous.py` (headless daemon mode)
use this factory to ensure identical cognition wiring.
"""

import logging
from typing import Optional, Tuple, Any, Callable

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
from src.brain.critic import CognitiveCritic
from src.actions.policy import ActionPolicy

logger = logging.getLogger(__name__)

USE_LLM = __import__('os').environ.get("BUBBY_USE_LLM", "1") == "1"
USE_TTS = __import__('os').environ.get("BUBBY_USE_TTS", "0") == "1"


def create_behavior_tree() -> BehaviorTree:
    """Create the standard behavior tree for autonomous decision-making."""
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


def build_cognition_stack(
    display_callback: Optional[Callable[[InteractionMessage], None]] = None,
    action_callback: Optional[Callable[[str, Any], None]] = None,
    enable_critic: bool = True,
    enable_vision: bool = True,
) -> dict:
    """
    Build the FULL cognition stack with all safety gates.

    This is the shared factory used by both app.py (GUI) and run_autonomous.py (daemon).

    Args:
        display_callback: Called with InteractionMessage for UI display
        action_callback: Called when an action requires user approval
        enable_critic: Enable CognitiveCritic + ActionPolicy (default: True)
        enable_vision: Enable VisionPipeline (default: True)

    Returns:
        dict with keys:
            behavior_tree, context_manager, autonomy_loop, vision_pipeline,
            memory_buffer, long_term_memory, reasoning_bridge, persona,
            synthesis_engine, synthesis_config, tts_engine, system_executor,
            interaction_handler, critic, action_policy
    """
    logger.info("=" * 60)
    logger.info("COGNITION STACK INITIALIZATION")
    logger.info("=" * 60)

    # ── Brain ──
    behavior_tree = create_behavior_tree()
    context_manager = ContextManager()

    # ── Vision & Memory ──
    vision_pipeline = VisionPipeline() if enable_vision else None
    memory_buffer = MemoryBuffer(max_observations=50, max_tokens=2048)
    long_term_memory = LongTermMemory()
    reasoning_bridge = ReasoningBridge(memory_buffer)

    # ── Persona & Synthesis ──
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

    # ── TTS (optional) ──
    tts_engine = None
    if USE_TTS:
        try:
            tts_engine = TTSEngine(TTSConfig(use_subprocess=True))
            if tts_engine.is_ready():
                logger.info("TTS engine ready")
        except Exception as e:
            logger.warning(f"TTS engine init failed: {e}")

    # ── System Executor ──
    system_executor = SystemExecutor()
    logger.info(f"System executor: {len(system_executor.get_available_actions())} actions")

    # ── Cognitive Critic + Action Policy ──
    critic = None
    action_policy = None
    if enable_critic:
        action_policy = ActionPolicy(strict_mode=True)
        critic = CognitiveCritic(
            action_executor=system_executor,
            action_policy=action_policy,
        )
        logger.info(f"CognitiveCritic + ActionPolicy initialized (strict_mode=True)")

    # ── Interaction Handler ──
    interaction_handler = InteractionHandler(
        synthesis_engine=synthesis_engine,
        display_callback=display_callback,
        tts_engine=tts_engine,
        action_executor=system_executor,
        action_callback=action_callback,
        cognitive_critic=critic,
        action_policy=action_policy,
    )

    # ── Autonomy Loop ──
    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree,
        context_manager=context_manager,
        decision_interval=2.0,
    )

    logger.info("✓ Cognition stack initialized (critic={}, vision={})".format(
        enable_critic, enable_vision))

    return {
        "behavior_tree": behavior_tree,
        "context_manager": context_manager,
        "autonomy_loop": autonomy_loop,
        "vision_pipeline": vision_pipeline,
        "memory_buffer": memory_buffer,
        "long_term_memory": long_term_memory,
        "reasoning_bridge": reasoning_bridge,
        "persona": persona,
        "synthesis_engine": synthesis_engine,
        "synthesis_config": synthesis_config,
        "tts_engine": tts_engine,
        "system_executor": system_executor,
        "interaction_handler": interaction_handler,
        "critic": critic,
        "action_policy": action_policy,
    }