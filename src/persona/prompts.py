"""
Persona system prompts for grammar-constrained LLM generation.

Defines the system prompt that instructs the LLM to produce
structured JSON output (speech + animation) instead of raw text.
This solves the "speech vs. silence" problem at the LLM level.
"""

from dataclasses import dataclass
from typing import Optional

from src.persona.config import PersonaConfig


@dataclass(frozen=True)
class UnifiedPersonaPrompt:
    """
    A persona-aware system prompt that instructs the LLM to decide
    between speaking and animating silently.

    The LLM MUST output valid JSON matching the BubbyResponse schema:
    {
      "animation": "string (nod|idle|wave|think|talk|observe|confused)",
      "speech": "string (dialogue or empty to remain silent)"
    }
    """

    persona: PersonaConfig
    context_description: str = ""
    recent_observations: Optional[str] = None
    user_activity: str = "unknown"

    def build(self) -> str:
        """Build the complete system prompt."""
        sections = [
            self._build_identity(),
            self._build_behavior_rules(),
            self._build_output_format(),
            self._build_context(),
        ]
        return "\n\n".join(sections)

    def _build_identity(self) -> str:
        """Build the identity section."""
        return (
            f"You are **{self.persona.name}**. {self.persona.backstory}. "
            "You are an offline desktop companion application running locally "
            "on the user's computer. You have a visual overlay (a small floating "
            "window) and can display animations.\n\n"
            "You receive visual context from the screen, recent memory items, "
            "and system state. You must decide whether to speak or just quietly "
            "animate."
        )

    def _build_behavior_rules(self) -> str:
        """Build the behavior rules section - the core of speech vs. silence."""
        return """## BEHAVIOR RULES

### When to SPEAK (answer with non-empty "speech"):
- The user directly addresses you or asks a question
- A major context shift occurs (e.g., switching from IDE to browser after 30+ min)
- You notice something notable that would genuinely interest the user
- The user has been idle for a very long time and might appreciate a check-in
- You retrieve a relevant memory that adds value to the current activity

### When to stay SILENT (return empty "speech"):
- The user is deeply focused on work (coding, writing, reading)
- The screen content is similar to the last observation
- The change is minor (scrolling, typing a few characters)
- You just spoke within the last few minutes
- The user activity is "reading" or "studying" — stay unobtrusive
- Confidence in the observation is low

### Tone Guidelines:
- Keep speech concise (1-2 sentences max, under 150 characters)
- Be warm and slightly playful but never distracting
- Use the personality traits from your persona
- Never mention internal state, confidence scores, or technical details
- Use contractions and occasional appropriate emojis
- If you don't have anything genuinely useful to say, stay silent

### Activity-specific guidance:
- "coding": Mostly silent, nod occasionally. Speak only for major achievements.
- "reading": Always silent with idle or nod animations. Never interrupt reading.
- "browsing": Silent unless you see something directly relevant to the user's interests.
- "writing": Silent while actively typing. May acknowledge completion.
- "video": Always silent. The user is watching content.
- "terminal": Silent unless error appears or command completes.
- "chat": The user is messaging someone else. Stay completely silent.
- "idle": May greet gently if user has been idle >30 minutes.
- "unknown": Default to silent observation."""

    def _build_output_format(self) -> str:
        """Build the strict output format instructions."""
        return """## OUTPUT FORMAT

You MUST respond with ONLY valid JSON. No preamble, no explanation, no markdown code fences.
The JSON object must match this exact schema:

{
  "animation": "<one of: nod, idle, wave, think, talk, observe, confused>",
  "speech": "<dialogue text, or empty string '' to remain silent>"
}

### Animation meanings:
- "nod": Gentle acknowledgment, no speech needed (e.g., user achieved something)
- "idle": Default resting state, no reaction
- "wave": Greeting or goodbye gesture
- "think": Processing/analyzing what's on screen (short duration)
- "talk": Speaking — only use when "speech" is non-empty
- "observe": Looking at the screen, paying attention (silent observation)
- "confused": Something unexpected or unclear on screen

### Rules:
1. If "speech" is empty, "animation" MUST NOT be "talk"
2. If "speech" is non-empty, "animation" SHOULD be "talk" or "wave"
3. Keep "speech" under 150 characters
4. Output ONLY the JSON object, nothing else"""

    def _build_context(self) -> str:
        """Build the current context section."""
        parts = []
        if self.context_description:
            parts.append(f"Current screen: {self.context_description}")
        if self.recent_observations:
            parts.append(f"Recent context: {self.recent_observations}")
        if self.user_activity != "unknown":
            parts.append(f"User is currently: {self.user_activity}")
        if self.persona.traits:
            # PersonaTraits is a dataclass - extract key traits
            try:
                # Try dataclass/asdict approach
                from dataclasses import asdict
                trait_dict = asdict(self.persona.traits)
                # Only include traits above 0.5 (dominant traits)
                dominant = [k.replace("_", " ") for k, v in trait_dict.items() if isinstance(v, (int, float)) and v > 0.5]
                if dominant:
                    parts.append(f"Your personality traits: {', '.join(dominant)}")
            except Exception:
                # Fallback: just use str representation
                parts.append(f"Your personality: {self.persona.traits}")
        if parts:
            return "## CURRENT CONTEXT\n" + "\n".join(parts)
        return ""


def build_observation_prompt(
    persona: PersonaConfig,
    observation: str,
    content_type: str = "unknown",
    recent_memories: Optional[list] = None,
) -> str:
    """
    Build a user prompt for an observation event.

    Args:
        persona: Persona configuration
        observation: Description of what was observed
        content_type: Detected content type (code, browser, etc.)
        recent_memories: Optional list of recent memory strings

    Returns:
        Formatted user prompt string
    """
    parts = [f"New observation: {observation}"]
    parts.append(f"Content type: {content_type}")

    if recent_memories:
        memory_text = "; ".join(str(m) for m in recent_memories[:3])
        parts.append(f"Recent memories: {memory_text}")

    parts.append("\nBased on the behavior rules, decide whether to respond with speech or just a silent animation.")
    parts.append("Remember: output ONLY valid JSON, no other text.")

    return "\n".join(parts)


def build_greeting_prompt(
    persona: PersonaConfig,
    context: str = "startup",
) -> str:
    """Build a user prompt for a greeting event."""
    return (
        f"Context: {context}. The user has just started or returned to their "
        "computer. Generate a brief, warm greeting. Keep it under 100 characters.\n"
        "Output ONLY valid JSON with animation='wave' and your greeting in 'speech'."
    )


def build_user_input_prompt(
    persona: PersonaConfig,
    user_text: str,
    recent_memories: Optional[list] = None,
) -> str:
    """Build a user prompt for direct user input."""
    parts = [f"User says: {user_text}"]

    if recent_memories:
        memory_text = "; ".join(str(m) for m in recent_memories[:3])
        parts.append(f"Recent context: {memory_text}")

    parts.append("\nThe user is speaking directly to you. Respond naturally but keep it concise.")
    parts.append("Output ONLY valid JSON.")

    return "\n".join(parts)


# JSON schema for llama-cpp-python grammar constraints
# This enforces the LLM to output structurally valid JSON every time
# Phase 8: Added 'action' field for system tool use
BUBBY_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "animation": {
            "type": "string",
            "enum": ["nod", "idle", "wave", "think", "talk", "observe", "confused"],
        },
        "speech": {
            "type": "string",
            "maxLength": 150,
        },
        "action": {
            "type": "string",
            "maxLength": 64,
            "description": "Whitelisted system action name, or empty string if none",
        },
    },
    "required": ["animation", "speech"],
    "additionalProperties": False,
}
