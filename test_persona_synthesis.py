"""Test script for Persona & Synthesis Engine (Phase 6)."""

import sys
import time
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_persona_config():
    """Test 1: Persona configuration and system prompt."""
    logger.info("=" * 60)
    logger.info("TEST 1: Persona Configuration")
    logger.info("=" * 60)
    
    from src.persona.config import PersonaConfig, PersonaType, PersonaTraits
    
    # Default persona (Witty Companion)
    config = PersonaConfig()
    assert config.name == "Bubby"
    assert config.persona_type == PersonaType.WITTY_COMPANION
    logger.info(f"✓ Default persona: {config.name} ({config.persona_type.value})")
    
    # System prompt generation
    prompt = config.build_system_prompt()
    assert "Bubby" in prompt
    assert "friendly" in prompt
    assert len(prompt) > 100
    logger.info(f"✓ System prompt generated ({len(prompt)} chars)")
    
    # Different persona types
    copilot = PersonaConfig(persona_type=PersonaType.HELPFUL_COPILOT)
    assert copilot.traits.formality > 0.5
    logger.info(f"✓ Different persona: {copilot.persona_type.value}")
    
    # Custom traits
    custom = PersonaConfig(
        name="TestBot",
        traits=PersonaTraits(warmth=0.9, humor=0.9, sassiness=0.8)
    )
    trait_str = custom.traits.to_prompt_segment()
    assert "Warm" in trait_str
    logger.info(f"✓ Custom traits: {trait_str[:60]}...")
    
    # Response rules
    assert len(config.response_rules) > 0
    assert config.never_reveal_internal_state == True
    logger.info("✓ Guardrails configured")
    
    logger.info("\n✓ Persona config test passed")
    return True


def test_synthesis_engine():
    """Test 2: Synthesis engine response generation."""
    logger.info("=" * 60)
    logger.info("TEST 2: Synthesis Engine")
    logger.info("=" * 60)
    
    from src.persona.synthesis import SynthesisEngine
    from src.persona.config import PersonaConfig
    from src.brain.reasoning import VisualReasoning
    
    engine = SynthesisEngine()
    
    # Test 2a: Code context
    reasoning = VisualReasoning(
        content_type="code", confidence=0.85,
        description="User writing Python in VS Code",
        should_interact=True, should_observe=False,
        reasoning="User writing Python"
    )
    response = engine.synthesize(reasoning, context_text="User coding in VS Code")
    assert len(response.text) > 0
    assert response.animation == "wave"
    assert "confidence" not in response.text.lower()
    logger.info(f"✓ Code: '{response.text[:50]}...'")
    
    # Test 2b: Video context (non-intrusive)
    reasoning2 = VisualReasoning(
        content_type="video", confidence=0.9,
        description="User watching YouTube",
        should_interact=False, should_observe=True,
        reasoning="Watching YouTube"
    )
    response2 = engine.synthesize(reasoning2)
    assert "quiet" in response2.text.lower() or "chill" in response2.text.lower()
    logger.info(f"✓ Video: '{response2.text[:50]}...'")
    
    # Test 2c: Low confidence (Wait-and-See)
    reasoning3 = VisualReasoning(
        content_type="unknown", confidence=0.3,
        description="Uncertain screen content",
        should_interact=False, should_observe=True,
        reasoning="Uncertain"
    )
    response3 = engine.synthesize(reasoning3)
    assert "not sure" in response3.text.lower()
    logger.info(f"✓ Uncertain: '{response3.text[:50]}...'")
    
    # Test 2d: Gaming
    reasoning4 = VisualReasoning(
        content_type="game", confidence=0.8,
        description="User playing a game",
        should_interact=False, should_observe=True,
        reasoning="User gaming"
    )
    response4 = engine.synthesize(reasoning4)
    logger.info(f"✓ Gaming: '{response4.text[:50]}...'")
    
    # Test 2e: Greeting
    greeting = engine.get_greeting("coding")
    assert len(greeting) > 0
    logger.info(f"✓ Greeting: '{greeting}'")
    
    logger.info("\n✓ Synthesis engine test passed")
    return True


def test_guardrails():
    """Test 3: Output guardrails."""
    logger.info("=" * 60)
    logger.info("TEST 3: Output Guardrails")
    logger.info("=" * 60)
    
    from src.persona.synthesis import SynthesisEngine
    from src.persona.config import PersonaConfig
    
    config = PersonaConfig(max_response_length=200)
    engine = SynthesisEngine(persona=config)
    
    # Test guardrail: max length
    guardrail_test = engine._apply_guardrails("A" * 500)
    assert len(guardrail_test) <= config.max_response_length + 10  # +10 for "..."
    logger.info(f"✓ Max length enforced ({len(guardrail_test)} <= {config.max_response_length})")
    
    # Test guardrail: internal state removal
    dirty_text = "Confidence: 0.85, DecisionType: INTERACT, memory_id: 42"
    clean_text = engine._apply_guardrails(dirty_text)
    assert "confidence" not in clean_text.lower()
    assert "decision" not in clean_text.lower()
    assert "memory_id" not in clean_text.lower()
    logger.info(f"✓ Internal state patterns stripped: '{clean_text}'")
    
    # Test guardrail: clean spaces
    assert "  " not in clean_text
    logger.info("✓ Double spaces removed")
    
    logger.info("\n✓ Guardrails test passed")
    return True


def test_interaction_handler():
    """Test 4: Interaction handler with display callback."""
    logger.info("=" * 60)
    logger.info("TEST 4: Interaction Handler")
    logger.info("=" * 60)
    
    from src.interaction.handler import InteractionHandler, InteractionMessage, InteractionEvent
    from src.persona.synthesis import SynthesisEngine
    from src.brain.reasoning import VisualReasoning
    
    # Track display messages
    displayed_messages = []
    
    def display_callback(msg: InteractionMessage):
        displayed_messages.append(msg)
    
    engine = SynthesisEngine()
    handler = InteractionHandler(
        synthesis_engine=engine,
        display_callback=display_callback
    )
    
    # Test observation
    reasoning = VisualReasoning(
        content_type="code", confidence=0.85,
        description="User coding in VS Code",
        should_interact=True, should_observe=False,
        reasoning="Coding"
    )
    msg = handler.on_observation(reasoning, "User coding")
    assert len(msg.text) > 0
    assert msg.event == InteractionEvent.OBSERVATION
    logger.info(f"✓ Observation: '{msg.text[:50]}...'")
    
    # Test greeting
    greeting = handler.on_greeting("coding")
    assert greeting.event == InteractionEvent.GREETING
    logger.info(f"✓ Greeting: '{greeting.text[:50]}...'")
    
    # Test greeting cooldown
    greeting2 = handler.on_greeting("coding")
    assert greeting2.text == ""  # Cooldown active
    logger.info("✓ Greeting cooldown working")
    
    # Test status
    status = handler.on_status("System running normally")
    assert status.event == InteractionEvent.STATUS
    logger.info(f"✓ Status: '{status.text}'")
    
    # Test message history
    history = handler.get_history()
    assert len(history) >= 3  # observation + greeting + status
    logger.info(f"✓ History stored: {len(history)} messages")
    
    # Test display callback was called
    assert len(displayed_messages) >= 3
    logger.info(f"✓ Display callback called: {len(displayed_messages)} times")
    
    # Test format
    formatted = msg.format_for_display()
    assert "[" in formatted and "]" in formatted
    logger.info(f"✓ Display format: '{formatted[:80]}...'")
    
    # Test stats
    stats = handler.get_stats()
    assert stats["total_messages"] >= 3
    logger.info(f"✓ Stats: {stats['total_messages']} messages")
    
    logger.info("\n✓ Interaction handler test passed")
    return True


def test_persona_memory_integration():
    """Test 5: Persona + LTM integration."""
    logger.info("=" * 60)
    logger.info("TEST 5: Persona + LTM Integration")
    logger.info("=" * 60)
    
    from src.persona.synthesis import SynthesisEngine
    from src.persona.config import PersonaConfig, PersonaType
    from src.brain.reasoning import VisualReasoning
    from src.memory.long_term_memory import LongTermMemory
    from src.interaction.handler import InteractionHandler
    
    # Setup with LTM
    ltm = LongTermMemory()
    ltm.archive("User likes Python and TypeScript", importance=0.9)
    ltm.archive("User prefers simple, clean code solutions", importance=0.8)
    
    engine = SynthesisEngine(long_term_memory=ltm)
    handler = InteractionHandler(synthesis_engine=engine)
    
    # Inject context: user struggling with coding
    reasoning = VisualReasoning(
        content_type="code",
        confidence=0.85,
        description="User struggling with Python code in VS Code",
        should_interact=True,
        should_observe=False,
        reasoning="User seems to be struggling with Python code"
    )
    
    # Get response
    response = engine.synthesize(
        reasoning=reasoning,
        context_text="User is struggling with a complex Python function"
    )
    
    # Verify response
    assert len(response.text) > 0
    assert "confidence" not in response.text.lower()
    assert response.animation == "wave"
    
    logger.info(f"Response: {response.text}")
    logger.info(f"Animation: {response.animation}")
    logger.info(f"Memory recall: {response.has_memory_recall}")
    
    # Verify through interaction handler
    msg = handler.on_observation(reasoning, "User struggling with Python")
    assert len(msg.text) > 0
    logger.info(f"Handler output: {msg.text}")
    
    # Stats
    stats = engine.get_stats()
    assert stats["has_ltm"] == True
    logger.info(f"✓ LTM integrated in synthesis")
    
    ltm.clear()
    
    logger.info("\n✓ Persona + LTM integration test passed")
    return True


def main():
    """Run all persona synthesis tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PERSONA & SYNTHESIS TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_persona_config()
        test_synthesis_engine()
        test_guardrails()
        test_interaction_handler()
        test_persona_memory_integration()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL PERSONA & SYNTHESIS TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 6 Complete!")
        logger.info("The companion now has a Voice!")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()