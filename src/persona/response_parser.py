"""Parse structured JSON responses from the LLM into typed response objects."""

import json
import logging
from typing import Optional
from dataclasses import dataclass

from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA

logger = logging.getLogger(__name__)

# Valid animation values from the schema
VALID_ANIMATIONS = set(
    BUBBY_RESPONSE_JSON_SCHEMA["properties"]["animation"]["enum"]
)

# Maximum speech length from the schema
MAX_SPEECH_LENGTH = BUBBY_RESPONSE_JSON_SCHEMA["properties"]["speech"]["maxLength"]


@dataclass(frozen=True)
class StructuredResponse:
    """
    Parsed and validated structured response from the LLM.

    Attributes:
        animation: One of the valid animation values
        speech: The dialogue text (empty string if silent)
        action: Optional system action name (empty string if none)
        is_silent: True if the LLM chose to remain silent
        has_action: True if a system action was requested
        is_valid: True if the response was successfully parsed
        raw_json: The raw JSON string from the LLM
    """

    animation: str
    speech: str
    action: str = ""
    is_silent: bool = True
    has_action: bool = False
    is_valid: bool = True
    raw_json: str = ""

    def __post_init__(self):
        """Validate on creation."""
        if self.animation not in VALID_ANIMATIONS:
            logger.warning(f"Invalid animation '{self.animation}', defaulting to 'idle'")
            object.__setattr__(self, "animation", "idle")
            object.__setattr__(self, "is_valid", False)

        if len(self.speech) > MAX_SPEECH_LENGTH:
            object.__setattr__(self, "speech", self.speech[:MAX_SPEECH_LENGTH])

        # Enforce: if speech is empty, animation cannot be "talk"
        if not self.speech and self.animation == "talk":
            object.__setattr__(self, "animation", "idle")
        
        # Normalize action
        if self.action:
            cleaned = self.action.strip().lower().replace(" ", "_")
            if len(cleaned) > 64:
                cleaned = cleaned[:64]
            object.__setattr__(self, "action", cleaned)
            object.__setattr__(self, "has_action", bool(cleaned))


def parse_structured_response(raw_text: str) -> StructuredResponse:
    """
    Parse a raw LLM output string into a StructuredResponse.

    Handles:
    - Clean JSON output
    - JSON wrapped in markdown fences
    - Malformed JSON (falls back to safe defaults)
    - Extra text around the JSON object

    Args:
        raw_text: Raw text from the LLM (may include markdown fences)

    Returns:
        StructuredResponse with parsed and validated fields
    """
    if not raw_text:
        return StructuredResponse(
            animation="idle",
            speech="",
            is_silent=True,
            is_valid=False,
            raw_json=raw_text,
        )

    # Clean markdown code fences
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Try to extract JSON object boundaries
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")

    if json_start == -1 or json_end == -1:
        logger.warning(f"No JSON object found in: {cleaned[:100]}")
        return StructuredResponse(
            animation="confused",
            speech="",
            is_silent=True,
            is_valid=False,
            raw_json=raw_text,
        )

    json_str = cleaned[json_start : json_end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e} → {json_str[:100]}")
        # Try simple regex extraction as fallback
        return _extract_fallback(json_str, raw_text)

    # Extract and validate fields
    animation = data.get("animation", "idle")
    speech = data.get("speech", "")

    # Type coercion
    if not isinstance(animation, str):
        animation = "idle"
    if not isinstance(speech, str):
        speech = str(speech) if speech else ""

    # Normalize animation
    animation = animation.lower().strip()
    if animation not in VALID_ANIMATIONS:
        # Fuzzy match
        animation = _fuzzy_match_animation(animation)

    # Normalize speech
    speech = speech.strip().strip('"').strip("'")
    if len(speech) > MAX_SPEECH_LENGTH:
        speech = speech[:MAX_SPEECH_LENGTH]

    # Clean speech of common LLM artifacts
    speech = _clean_speech(speech)

    # Determine silence
    is_silent = not speech or len(speech) < 2

    # Enforce talk animation only when speaking
    if is_silent and animation == "talk":
        animation = "idle"

    if not is_silent and animation in ("idle", "nod"):
        animation = "talk"

    # Extract action (optional, may not be present)
    action = data.get("action", "")
    if not isinstance(action, str):
        action = ""
    action = action.strip().lower().replace(" ", "_")[:64]

    return StructuredResponse(
        animation=animation,
        speech=speech if not is_silent else "",
        action=action,
        is_silent=is_silent,
        has_action=bool(action),
        is_valid=True,
        raw_json=raw_text,
    )


def _fuzzy_match_animation(value: str) -> str:
    """Fuzzy match an animation string to valid values."""
    value = value.lower().strip()
    for valid in VALID_ANIMATIONS:
        if valid in value or value in valid:
            return valid
    return "idle"


def _extract_fallback(json_str: str, raw_text: str) -> StructuredResponse:
    """
    Fallback extraction when JSON parsing fails.
    Uses simple regex/keyword matching.
    """
    import re

    # Try to find animation
    anim_match = re.search(
        r'"animation"\s*:\s*"([^"]+)"', json_str
    )
    animation = anim_match.group(1) if anim_match else "idle"
    animation = _fuzzy_match_animation(animation)

    # Try to find speech
    speech_match = re.search(
        r'"speech"\s*:\s*"([^"]*)"', json_str, re.DOTALL
    )
    speech = speech_match.group(1) if speech_match else ""

    # If no speech match found but there's text, try to extract it
    if not speech and not anim_match:
        # Maybe the LLM just output raw text
        speech = raw_text[:MAX_SPEECH_LENGTH].strip()
        if speech:
            animation = "talk"

    # Try to extract action from fallback
    action_match = re.search(r'"action"\s*:\s*"([^"]*)"', json_str)
    action = action_match.group(1) if action_match else ""
    action = action.strip().lower().replace(" ", "_")[:64]

    speech = _clean_speech(speech)
    is_silent = not speech

    return StructuredResponse(
        animation=animation,
        speech=speech if not is_silent else "",
        action=action,
        is_silent=is_silent,
        has_action=bool(action),
        is_valid=False,
        raw_json=raw_text,
    )


def _clean_speech(text: str) -> str:
    """Clean speech text of common LLM artifacts."""
    if not text:
        return ""

    import re

    # Remove markdown formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove common LLM disclaimers
    disclaimers = [
        r"as an AI.*",
        r"I am a language model.*",
        r"I don't have access.*",
        r"I cannot actually.*",
        r"I'm just a text.*",
    ]
    for pattern in disclaimers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove internal state leakage patterns
    internal_patterns = [
        r"confidence.*",
        r"NodeStatus.*",
        r"DecisionType.*",
        r"VisualReasoning.*",
        r"\b\d+\.\d{2,}\b",  # Float numbers (likely confidence scores)
    ]
    for pattern in internal_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip(' "\'\n\r\t.,;:!?')

    return text