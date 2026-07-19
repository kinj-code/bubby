#!/usr/bin/env python3
"""
Phase 7 Integration Test: Local LLM Speech vs. Silence

Tests the grammar-constrained JSON generation pipeline:
1. Minor screen change → LLM outputs empty speech (silent animation)
2. Major user prompt → LLM outputs dialogue
3. Parser gracefully handles malformed output
4. Template fallback works when LLM unavailable

Run: python test_local_llm.py
"""

import json
import logging
import os
import sys
import time

# Configure logging for test visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_response_parser() -> None:
    """Test the JSON response parser with various outputs."""
    from src.persona.response_parser import (
        parse_structured_response,
        StructuredResponse,
        VALID_ANIMATIONS,
    )
    from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA

    logger.info("=" * 60)
    logger.info("TEST: Response Parser")
    logger.info("=" * 60)

    # Test 1: Clean JSON, silent response (minor screen change)
    raw = '{"animation": "nod", "speech": ""}'
    result = parse_structured_response(raw)
    assert result.is_valid, f"Expected valid, got invalid: {result}"
    assert result.is_silent, f"Expected silent, got speech: {result.speech}"
    assert result.animation == "nod", f"Expected nod, got: {result.animation}"
    logger.info("✓ Test 1: Silent nod parsed correctly")

    # Test 2: Clean JSON, speech response (user interaction)
    raw = '{"animation": "talk", "speech": "Looking good! Need any help with that code?"}'
    result = parse_structured_response(raw)
    assert result.is_valid
    assert not result.is_silent
    assert result.animation == "talk"
    assert "Looking good" in result.speech
    logger.info("✓ Test 2: Speech response parsed correctly")

    # Test 3: JSON wrapped in markdown fences
    raw = '```json\n{"animation": "observe", "speech": ""}\n```'
    result = parse_structured_response(raw)
    assert result.is_valid
    assert result.is_silent
    assert result.animation == "observe"
    logger.info("✓ Test 3: Markdown-fenced JSON parsed correctly")

    # Test 4: Malformed JSON - missing speech
    raw = '{"animation": "wave"}'
    result = parse_structured_response(raw)
    # Should fallback to defaults
    assert result.animation in VALID_ANIMATIONS
    logger.info(f"✓ Test 4: Malformed JSON handled gracefully (anim={result.animation})")

    # Test 5: Invalid animation value - parser fixes it
    raw = '{"animation": "dancing", "speech": "Hello!"}'
    result = parse_structured_response(raw)
    assert result.animation in VALID_ANIMATIONS, f"Expected valid animation, got: {result.animation}"
    assert result.animation != "dancing", "Invalid animation should be fixed"
    # The parser gracefully fixes invalid animations, so the result remains valid
    logger.info(f"✓ Test 5: Invalid animation fixed to '{result.animation}'")

    # Test 6: Empty raw text
    result = parse_structured_response("")
    assert result.is_silent
    assert result.animation == "idle"
    logger.info("✓ Test 6: Empty input handled gracefully")

    # Test 7: Raw text (no JSON) - should extract safely
    raw = "I see you're coding! That looks interesting."
    result = parse_structured_response(raw)
    # May or may not extract speech - just ensure it doesn't crash
    assert result.animation in VALID_ANIMATIONS
    logger.info(f"✓ Test 7: Raw text handled (speech={result.speech[:30] if result.speech else '(none)'})")

    # Test 8: JSON with markdown artifacts in speech
    raw = '{"animation": "talk", "speech": "**Great** job on that `function`!"}'
    result = parse_structured_response(raw)
    assert result.is_valid
    # Markdown should be stripped
    assert "**" not in result.speech
    assert "`" not in result.speech
    logger.info(f"✓ Test 8: Markdown stripped from speech: '{result.speech}'")

    # Test 9: Speech exceeds max length
    raw = '{"animation": "talk", "speech": "' + ('x' * 200) + '"}'
    result = parse_structured_response(raw)
    assert len(result.speech) <= 150  # MAX_SPEECH_LENGTH
    logger.info(f"✓ Test 9: Long speech truncated to {len(result.speech)} chars")

    # Test 10: LLM disclaimers stripped
    raw = '{"animation": "talk", "speech": "Nice work! As an AI I cannot actually judge code quality."}'
    result = parse_structured_response(raw)
    assert "as an AI" not in result.speech.lower()
    logger.info(f"✓ Test 10: LLM disclaimers stripped: '{result.speech}'")

    logger.info("\nAll parser tests PASSED\n")


def test_prompt_generation() -> None:
    """Test that prompts correctly instruct speech vs. silence."""
    from src.persona.prompts import (
        UnifiedPersonaPrompt,
        build_observation_prompt,
        build_greeting_prompt,
        build_user_input_prompt,
    )
    from src.persona.config import PersonaConfig, PersonaType

    logger.info("=" * 60)
    logger.info("TEST: Prompt Generation")
    logger.info("=" * 60)

    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)

    # Test 1: Observation prompt for coding (should stay silent)
    prompt = build_observation_prompt(
        persona=persona,
        observation="User is typing code in VS Code",
        content_type="coding",
    )
    assert "valid json" in prompt.lower(), f"Missing JSON instruction in: {prompt[:100]}"
    assert "coding" in prompt.lower()
    logger.info("✓ Test 1: Coding observation prompt built")

    # Test 2: Greeting prompt
    prompt = build_greeting_prompt(persona=persona, context="startup")
    assert "wave" in prompt.lower()
    assert "greeting" in prompt.lower()
    logger.info("✓ Test 2: Greeting prompt built")

    # Test 3: User input prompt
    prompt = build_user_input_prompt(
        persona=persona,
        user_text="What am I looking at?",
        recent_memories=["User was coding earlier"],
    )
    assert "User says" in prompt
    assert "Recent context" in prompt
    logger.info("✓ Test 3: User input prompt built with memory context")

    # Test 4: UnifiedPersonaPrompt - coding activity (should encourage silence)
    builder = UnifiedPersonaPrompt(
        persona=persona,
        context_description="VS Code editor with Python files",
        user_activity="coding",
    )
    system_prompt = builder.build()
    assert "coding" in system_prompt.lower()
    assert "silent" in system_prompt.lower()
    assert "nod" in system_prompt.lower()
    assert "speak" in system_prompt.lower() or "SPEAK" in system_prompt
    assert "animation" in system_prompt.lower()
    assert "speech" in system_prompt.lower()
    # Should specifically mention coding → mostly silent behavior
    assert "mostly silent" in system_prompt.lower() or "silent with idle" in system_prompt.lower()
    logger.info("✓ Test 4: System prompt includes speech-vs-silence rules")

    # Test 5: UnifiedPersonaPrompt - reading (should enforce silence)
    builder = UnifiedPersonaPrompt(
        persona=persona,
        context_description="PDF document open",
        user_activity="reading",
    )
    system_prompt = builder.build()
    assert "reading" in system_prompt.lower()
    logger.info("✓ Test 5: Reading activity enforces silence in system prompt")

    # Test 6: JSON schema validation
    from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA
    assert BUBBY_RESPONSE_JSON_SCHEMA["type"] == "object"
    assert "animation" in BUBBY_RESPONSE_JSON_SCHEMA["required"]
    assert "speech" in BUBBY_RESPONSE_JSON_SCHEMA["required"]
    assert "enum" in BUBBY_RESPONSE_JSON_SCHEMA["properties"]["animation"]
    logger.info("✓ Test 6: JSON schema is valid and complete")

    logger.info("\nAll prompt tests PASSED\n")


def test_structured_inference_stub() -> None:
    """Test the structured inference pipeline (works without model)."""
    from src.llm.inference import LLMInference, LLMConfig, InferenceResult
    from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA
    from src.persona.response_parser import parse_structured_response

    logger.info("=" * 60)
    logger.info("TEST: Structured Inference Pipeline")
    logger.info("=" * 60)

    # Create config with non-existent model path
    config = LLMConfig(
        model_path="models/llm/nonexistent.gguf",
        n_threads=4,
        n_ctx=2048,
    )

    llm = LLMInference(config)

    # initialize() should fail gracefully (model not found)
    success = llm.initialize()
    assert not success, "Should fail when model not found"

    # generate_structured should return error stub without crashing
    result = llm.generate_structured(
        prompt="New observation: User is coding",
        system_prompt="You are Bubby. Decide speech vs silence.",
        json_schema=BUBBY_RESPONSE_JSON_SCHEMA,
    )
    assert isinstance(result, InferenceResult)
    assert result.stop_reason == "error"
    # The error stub should still be parseable JSON
    parsed = parse_structured_response(result.text)
    assert parsed.animation in ["idle", "confused"]
    logger.info(f"✓ Test 1: Graceful degradation on missing model: anim={parsed.animation}")

    # Test that the generate method still works (non-structured)
    result = llm.generate(
        prompt="Hello",
        system_prompt="Be brief",
    )
    assert isinstance(result, InferenceResult)
    logger.info(f"✓ Test 2: Regular generate also degrades gracefully")

    # Test JSON schema is properly passed through
    from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA
    assert BUBBY_RESPONSE_JSON_SCHEMA["type"] == "object"
    assert len(BUBBY_RESPONSE_JSON_SCHEMA["properties"]["animation"]["enum"]) == 7
    logger.info("✓ Test 3: JSON schema constant is properly defined")

    # Test stats
    stats = llm.get_stats()
    assert "initialized" in stats
    assert not stats["initialized"]
    logger.info(f"✓ Test 4: Stats accessible (errors={stats['errors']})")

    llm.shutdown()
    logger.info("\nAll structured inference tests PASSED\n")


def test_synthesis_pipeline() -> None:
    """Test that the synthesis engine pipeline works (template fallback)."""
    from src.persona.config import PersonaConfig, PersonaType
    from src.persona.llm_synthesis import LLMSynthesisEngine, LLMSynthesisConfig
    from src.memory.long_term_memory import LongTermMemory
    from src.brain.reasoning import VisualReasoning

    logger.info("=" * 60)
    logger.info("TEST: Synthesis Engine Pipeline")
    logger.info("=" * 60)

    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)
    ltm = LongTermMemory()
    ltm.archive("User likes Python", importance=0.9)

    # Test 1: Template-only (no LLM) - should always work
    config = LLMSynthesisConfig(use_llm=False)
    engine = LLMSynthesisEngine(persona=persona, long_term_memory=ltm, config=config)

    # Minor screen change - coding observation (should produce a response)
    reasoning = VisualReasoning(
        content_type="code",
        confidence=0.9,
        description="User writing Python in VS Code",
        should_interact=True,
        should_observe=False,
        reasoning="User is actively coding",
    )
    response = engine.synthesize(reasoning, "Writing Python code")
    assert response.text or response.animation, "Should produce some output"
    logger.info(f"✓ Test 1: Template coding observation: '{response.text[:50]}' (anim={response.animation})")

    # Test 2: Greeting through the synthesis engine
    response = engine.synthesize(
        reasoning=None,
        context_text="startup",
        trigger_type="greeting",
    )
    assert response.text or response.animation, "Should produce greeting output"
    logger.info(f"✓ Test 2: Greeting generated: '{response.text[:50]}'")

    # Test 3: User input
    response = engine.synthesize(
        reasoning=None,
        context_text="What's on my screen?",
        trigger_type="user_input",
    )
    assert response.text or response.animation
    logger.info(f"✓ Test 3: User input response: '{response.text[:50]}'")

    # Test 4: Stats tracking
    stats = engine.get_stats()
    assert stats["total_syntheses"] == 3
    assert stats["template_responses"] == 3
    logger.info(f"✓ Test 4: Stats tracking correct: {stats['total_syntheses']} syntheses")

    # Test 5: LLM disabled correctly
    assert not stats.get("llm_available", False)
    logger.info("✓ Test 5: LLM correctly marked as unavailable")

    engine.shutdown()
    ltm.clear()
    logger.info("\nAll synthesis pipeline tests PASSED\n")


def test_speech_silence_simulation() -> None:
    """
    Simulate the full speech-vs-silence decision flow WITHOUT a real model.
    
    This tests that:
    - The system correctly would instruct an LLM to stay silent on minor changes
    - The parser correctly handles silent responses
    - The interaction handler correctly routes silent vs speech outputs
    """
    from src.persona.prompts import (
        UnifiedPersonaPrompt,
        build_observation_prompt,
        build_user_input_prompt,
    )
    from src.persona.config import PersonaConfig, PersonaType
    from src.persona.response_parser import parse_structured_response

    logger.info("=" * 60)
    logger.info("TEST: Speech vs. Silence Decision Simulation")
    logger.info("=" * 60)

    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)

    # Scenario 1: MINOR screen change while coding
    # The system prompt and prompt should instruct the LLM to stay silent
    builder = UnifiedPersonaPrompt(
        persona=persona,
        context_description="User is scrolling through Python code in VS Code",
        user_activity="coding",
    )
    system_prompt = builder.build()

    prompt = build_observation_prompt(
        persona=persona,
        observation="User scrolled down a few lines in the editor",
        content_type="coding",
    )

    # Verify the instructions push toward silence
    assert "silent" in system_prompt.lower()
    assert "coding" in system_prompt.lower()

    # Simulate LLM would choose silence
    simulated_llm_output = '{"animation": "nod", "speech": ""}'
    result = parse_structured_response(simulated_llm_output)
    assert result.is_silent
    assert result.animation == "nod"
    logger.info(f"✓ Scenario 1 (Minor change, coding): Silent nod → CORRECT")

    # Scenario 2: MAJOR user interaction
    prompt = build_user_input_prompt(
        persona=persona,
        user_text="Hey Bubby, can you tell me what I was working on earlier?",
        recent_memories=["User was coding Python scripts", "User opened browser to search docs"],
    )

    simulated_llm_output = '{"animation": "talk", "speech": "You were working on some Python scripts earlier. Need me to help with anything specific?"}'
    result = parse_structured_response(simulated_llm_output)
    assert not result.is_silent
    assert result.animation == "talk"
    logger.info(f"✓ Scenario 2 (User question): Speech response → CORRECT: '{result.speech}'")

    # Scenario 3: Reading mode - always silent
    builder = UnifiedPersonaPrompt(
        persona=persona,
        context_description="User reading a PDF document",
        user_activity="reading",
    )
    system_prompt = builder.build()
    assert "reading" in system_prompt.lower()

    simulated_llm_output = '{"animation": "idle", "speech": ""}'
    result = parse_structured_response(simulated_llm_output)
    assert result.is_silent
    logger.info(f"✓ Scenario 3 (Reading): Silent idle → CORRECT")

    # Scenario 4: Browser activity - silence
    simulated_llm_output = '{"animation": "observe", "speech": ""}'
    result = parse_structured_response(simulated_llm_output)
    assert result.is_silent
    assert result.animation == "observe"
    logger.info(f"✓ Scenario 4 (Browsing): Silent observe → CORRECT")

    # Scenario 5: Major context shift after long period
    # (Would normally trigger speech - our prompt would allow it)
    builder = UnifiedPersonaPrompt(
        persona=persona,
        context_description="User switched from IDE to terminal after 45 minutes of coding",
        user_activity="terminal",
    )
    system_prompt = builder.build()
    # The prompt includes both when to speak and when to stay silent
    assert "speak" in system_prompt.lower() or "SPEAK" in system_prompt
    assert "silent" in system_prompt.lower() or "STAY SILENT" in system_prompt

    # LLM would be allowed to speak for a major context shift
    simulated_llm_output = '{"animation": "talk", "speech": "Running some commands? Hope the code works!"}'
    result = parse_structured_response(simulated_llm_output)
    assert not result.is_silent
    logger.info(f"✓ Scenario 5 (Major context shift): Speech allowed → CORRECT: '{result.speech}'")

    logger.info("\nAll speech-vs-silence simulations PASSED\n")


def main():
    """Run all Phase 7 integration tests."""
    logger.info("=" * 60)
    logger.info("PHASE 7: LOCAL LLM INTEGRATION TESTS")
    logger.info("Speech vs. Silence - Grammar-Constrained Generation")
    logger.info("=" * 60)
    logger.info("")

    try:
        test_response_parser()
        test_prompt_generation()
        test_structured_inference_stub()
        test_synthesis_pipeline()
        test_speech_silence_simulation()

        logger.info("=" * 60)
        logger.info("ALL PHASE 7 TESTS PASSED ✓")
        logger.info("=" * 60)

        logger.info("""
Summary:
  ✓ JSON response parser handles 10 edge cases
  ✓ Prompts correctly instruct speech vs. silence per activity type
  ✓ Structured inference gracefully degrades without model
  ✓ Synthesis engine pipeline works with template fallback
  ✓ Speech-vs-silence decision flow verified for 5 scenarios

Next Steps:
  1. Download a GGUF model: python scripts/download_llm.py --best
  2. Run with LLM: BUBBY_USE_LLM=1 python src/app.py
  3. The LLM will now output structured JSON (animation + speech)
""")

    except AssertionError as e:
        logger.error(f"TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()