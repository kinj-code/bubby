"""Persona module for companion personality and response generation."""

from src.persona.config import PersonaConfig, PersonaTraits, PersonaType
from src.persona.synthesis import SynthesisEngine, SynthesizedResponse

__all__ = [
    "PersonaConfig",
    "PersonaTraits",
    "PersonaType",
    "SynthesisEngine",
    "SynthesizedResponse"
]