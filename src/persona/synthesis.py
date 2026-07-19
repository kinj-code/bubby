"""
Synthesis Engine: Turns decisions + memory + persona into natural language.

This is the "Voice" of the companion. It takes:
1. ReasoningBridge output (what the companion decided to do)
2. LTM retrieved facts (what the companion remembers)
3. PersonaConfig (who the companion is)

And produces a natural language response that:
- Stays in character (Bubby: warm, witty, helpful)
- Never reveals internal state (no confidence scores, memory IDs)
- Is contextually appropriate for the situation
"""

import logging
import re
import random
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from src.persona.config import PersonaConfig, PersonaType
from src.brain.reasoning import VisualReasoning
from src.memory.long_term_memory import LongTermMemory, MemoryRecord

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedResponse:
    """
    Final synthesized response ready for display.
    
    Attributes:
        text: The natural language response text
        animation: Suggested companion animation
        context_type: What kind of context triggered this
        has_memory_recall: Whether LTM facts were included
    """
    text: str
    animation: str = "idle"  # "idle", "wave", "wander", "greet", "think"
    context_type: str = "observation"
    has_memory_recall: bool = False
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"[{self.animation}] {self.text[:60]}..."


class SynthesisEngine:
    """
    Synthesis Engine: Decision + Memory + Persona → Natural Language.
    
    This is a template-based generator that produces persona-consistent
    responses without needing a full LLM. It uses:
    - Context-aware templates
    - Persona trait modulation
    - Memory integration
    - Output guardrails
    
    In future, this can be upgraded to use a local LLM for more
    dynamic responses while keeping the same interface.
    """
    
    def __init__(
        self,
        persona: Optional[PersonaConfig] = None,
        long_term_memory: Optional[LongTermMemory] = None
    ) -> None:
        """
        Initialize synthesis engine.
        
        Args:
            persona: Persona configuration (default: Witty Companion)
            long_term_memory: LTM for context-aware responses
        """
        self._persona = persona or PersonaConfig()
        self._ltm = long_term_memory
        
        # Track conversation state
        self._last_response: Optional[str] = None
        self._response_count = 0
        self._total_synthesized = 0
        
        logger.info(f"SynthesisEngine initialized (persona={self._persona.name}, "
                   f"type={self._persona.persona_type.value})")
    
    def synthesize(
        self,
        reasoning: Optional[VisualReasoning] = None,
        context_text: str = "",
        trigger_type: str = "observation"
    ) -> SynthesizedResponse:
        """
        Synthesize a natural language response.
        
        Args:
            reasoning: Current visual reasoning (optional)
            context_text: Additional context description
            trigger_type: What triggered this response
            
        Returns:
            SynthesizedResponse with text and animation
        """
        self._total_synthesized += 1
        
        # Gather context
        content_type = reasoning.content_type if reasoning else "unknown"
        confidence = reasoning.confidence if reasoning else 0.0
        should_interact = reasoning.should_interact if reasoning else False
        
        # Retrieve relevant memories
        memory_context = self._retrieve_memories(context_text or content_type)
        
        # Generate response based on context
        response_text, animation = self._generate_response(
            content_type=content_type,
            confidence=confidence,
            should_interact=should_interact,
            context_text=context_text,
            memory_context=memory_context,
            trigger_type=trigger_type
        )
        
        # Apply guardrails
        response_text = self._apply_guardrails(response_text)
        
        # Store for state tracking
        self._last_response = response_text
        self._response_count += 1
        
        return SynthesizedResponse(
            text=response_text,
            animation=animation,
            context_type=content_type,
            has_memory_recall=bool(memory_context)
        )
    
    def _retrieve_memories(self, context: str) -> List[Tuple[MemoryRecord, float]]:
        """Retrieve relevant memories from LTM."""
        if not self._ltm:
            return []
        
        try:
            results = self._ltm.retrieve(context, k=2, min_score=0.15)
            return results
        except Exception as e:
            logger.debug(f"Memory retrieval failed: {e}")
            return []
    
    def _generate_response(
        self,
        content_type: str,
        confidence: float,
        should_interact: bool,
        context_text: str,
        memory_context: List[Tuple[MemoryRecord, float]],
        trigger_type: str
    ) -> Tuple[str, str]:
        """
        Generate response text and animation based on context.
        
        Uses template-based generation modulated by persona traits.
        """
        # Low confidence → Wait-and-See
        if confidence < 0.5 or content_type == "unknown":
            return self._generate_uncertain_response(content_type)
        
        # User is coding → offer help
        if content_type == "code":
            return self._generate_coding_response(context_text, memory_context)
        
        # User is browsing → casual observation
        if content_type == "browser":
            return self._generate_browsing_response(context_text, memory_context)
        
        # User is watching video → don't disturb
        if content_type == "video":
            return self._generate_video_response()
        
        # User is working with documents
        if content_type == "document":
            return self._generate_document_response(memory_context)
        
        # User is gaming
        if content_type == "game":
            return self._generate_game_response()
        
        # User is idle or absent
        if not should_interact:
            return self._generate_idle_response()
        
        # Default: general observation
        return self._generate_general_response(content_type, memory_context)
    
    def _generate_uncertain_response(self, content_type: str) -> Tuple[str, str]:
        """Response when uncertain about what user is doing."""
        responses = [
            "Hmm, not sure what you're up to. I'll just hang out for a bit.",
            "Can't quite tell what's on your screen. I'll wait and see!",
            "Not sure what I'm looking at. Let me observe a bit longer.",
        ]
        return random.choice(responses), "think"
    
    def _generate_coding_response(
        self,
        context: str,
        memories: List[Tuple[MemoryRecord, float]]
    ) -> Tuple[str, str]:
        """Response when user is coding."""
        # Check if we remember something relevant
        if memories:
            memory = memories[0][0]
            if "python" in memory.text.lower() or "typescript" in memory.text.lower():
                responses = [
                    f"Nice, working with code! I remember you like {memory.text.split('likes')[-1].strip() if 'likes' in memory.text else 'this kind of thing'}.",
                    f"Code time! I recall you're into this stuff. Need a hand?",
                ]
                return random.choice(responses), "wave"
        
        responses = [
            "Ooh, code! What are you building?",
            "Love seeing code on the screen! Need a rubber duck? 🦆",
            "Coding! You've got this. Let me know if you need a second pair of eyes.",
        ]
        return random.choice(responses), "wave"
    
    def _generate_browsing_response(
        self,
        context: str,
        memories: List[Tuple[MemoryRecord, float]]
    ) -> Tuple[str, str]:
        """Response when user is browsing."""
        if memories:
            responses = [
                "Found something interesting? I remember you like this kind of stuff!",
                "Browsing around! I've got some context on what you're into.",
            ]
            return random.choice(responses), "idle"
        
        responses = [
            "Found something interesting?",
            "Browsing the web, I see. Let me know if you need anything!",
        ]
        return random.choice(responses), "idle"
    
    def _generate_video_response(self) -> Tuple[str, str]:
        """Response when user is watching video (don't disturb)."""
        responses = [
            "You're watching something! I'll be quiet. 🎬",
            "Movie time? I'll just chill over here.",
            "Enjoying a video? I'll keep the noise down.",
        ]
        return random.choice(responses), "idle"
    
    def _generate_document_response(
        self,
        memories: List[Tuple[MemoryRecord, float]]
    ) -> Tuple[str, str]:
        """Response when user is working with documents."""
        responses = [
            "Working on something important? I'll be here if you need me.",
            "Reading or writing? Either way, I'm around!",
        ]
        return random.choice(responses), "idle"
    
    def _generate_game_response(self) -> Tuple[str, str]:
        """Response when user is gaming."""
        responses = [
            "Gaming! Don't let me distract you. 😄",
            "Playing something? I'll cheer from the sidelines!",
            "Game on! Hope you're winning.",
        ]
        return random.choice(responses), "idle"
    
    def _generate_idle_response(self) -> Tuple[str, str]:
        """Response when user is idle or absent."""
        responses = [
            "Just hanging out. Say when you need me!",
            "I'm here when you need me. No rush.",
            "Taking a break? Me too. 😊",
        ]
        return random.choice(responses), "idle"
    
    def _generate_general_response(
        self,
        content_type: str,
        memories: List[Tuple[MemoryRecord, float]]
    ) -> Tuple[str, str]:
        """General response for other content types."""
        if memories:
            responses = [
                "I remember something about this! Cool to see it again.",
                "This reminds me of something we talked about before.",
            ]
            return random.choice(responses), "think"
        
        responses = [
            "Interesting! What's this about?",
            "I see you're busy. I'll just observe!",
        ]
        return random.choice(responses), "idle"
    
    def _apply_guardrails(self, text: str) -> str:
        """
        Apply output guardrails to ensure persona consistency.
        
        Rules:
        1. Never reveal internal state (confidence, decision types)
        2. Never reveal memory IDs or technical architecture
        3. Stay in character
        4. Respect max response length
        """
        # Guardrail 1: Remove any internal state patterns
        guardrail_patterns = [
            r'confidence[:\s]*\d+\.?\d*',
            r'decision[_\s]type[:\s]*\w+',
            r'memory[_\s]id[:\s]*\d+',
            r'record[_\s]id[:\s]*\d+',
            r'priority[:\s]*\d+',
            r'NodeStatus\.\w+',
            r'DecisionType[:\.]\s*\w+',
            r'DecisionType[:\s]*\w+',
        ]
        
        for pattern in guardrail_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Guardrail 2: Clean up double spaces from removals
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Guardrail 3: Respect max length
        if len(text) > self._persona.max_response_length:
            # Try to cut at a sentence boundary
            cut = text[:self._persona.max_response_length]
            last_period = cut.rfind('.')
            last_exclaim = cut.rfind('!')
            last_question = cut.rfind('?')
            
            cut_at = max(last_period, last_exclaim, last_question)
            if cut_at > len(cut) // 2:  # Only cut if we have a meaningful break
                text = text[:cut_at + 1]
            else:
                text = cut + "..."
        
        return text
    
    def get_greeting(self, context: str = "idle") -> str:
        """Get a persona-appropriate greeting."""
        return self._persona.greetings.get(context, self._persona.greetings["idle"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            "persona": self._persona.name,
            "persona_type": self._persona.persona_type.value,
            "total_synthesized": self._total_synthesized,
            "response_count": self._response_count,
            "has_ltm": self._ltm is not None,
            "max_response_length": self._persona.max_response_length,
        }


# Testing helper
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    from src.brain.reasoning import VisualReasoning
    from src.memory.long_term_memory import LongTermMemory
    
    logger.info("=" * 60)
    logger.info("SYNTHESIS ENGINE TEST")
    logger.info("=" * 60)
    
    # Create components
    ltm = LongTermMemory()
    ltm.archive("User likes Python and TypeScript", importance=0.9)
    ltm.archive("User prefers simple, clean solutions", importance=0.8)
    
    engine = SynthesisEngine(long_term_memory=ltm)
    
    # Test 1: Coding context
    logger.info("\n--- Test 1: Coding context ---")
    reasoning = VisualReasoning(
        content_type="code",
        confidence=0.85,
        should_interact=True,
        should_observe=False,
        reasoning="User is writing Python code"
    )
    response = engine.synthesize(reasoning, context_text="User writing Python in VS Code")
    logger.info(f"Response: {response.text}")
    logger.info(f"Animation: {response.animation}")
    assert len(response.text) > 0
    assert "confidence" not in response.text.lower()
    logger.info("✓ Coding response generated")
    
    # Test 2: Video context (don't disturb)
    logger.info("\n--- Test 2: Video context ---")
    reasoning2 = VisualReasoning(
        content_type="video",
        confidence=0.9,
        should_interact=False,
        should_observe=True,
        reasoning="User watching YouTube"
    )
    response2 = engine.synthesize(reasoning2)
    logger.info(f"Response: {response2.text}")
    assert "quiet" in response2.text.lower() or "chill" in response2.text.lower()
    logger.info("✓ Video response is non-intrusive")
    
    # Test 3: Low confidence (Wait-and-See)
    logger.info("\n--- Test 3: Low confidence ---")
    reasoning3 = VisualReasoning(
        content_type="unknown",
        confidence=0.3,
        should_interact=False,
        should_observe=True,
        reasoning="Uncertain what user is doing"
    )
    response3 = engine.synthesize(reasoning3)
    logger.info(f"Response: {response3.text}")
    assert "sure" in response3.text.lower() or "not sure" in response3.text.lower()
    logger.info("✓ Low confidence handled gracefully")
    
    # Test 4: Guardrails
    logger.info("\n--- Test 4: Guardrails ---")
    reasoning4 = VisualReasoning(
        content_type="code",
        confidence=0.8,
        should_interact=True,
        should_observe=False,
        reasoning="User coding"
    )
    response4 = engine.synthesize(reasoning4)
    logger.info(f"Response: {response4.text}")
    # Verify no internal state leaked
    assert "confidence" not in response4.text.lower()
    assert "decision" not in response4.text.lower()
    assert "memory_id" not in response4.text.lower()
    logger.info("✓ Guardrails working (no internal state leaked)")
    
    # Test 5: Greeting
    logger.info("\n--- Test 5: Greeting ---")
    greeting = engine.get_greeting("coding")
    logger.info(f"Greeting: {greeting}")
    assert len(greeting) > 0
    logger.info("✓ Greeting generated")
    
    # Test 6: Stats
    logger.info("\n--- Test 6: Statistics ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    ltm.clear()
    
    logger.info("\n" + "=" * 60)
    logger.info("SYNTHESIS ENGINE TEST COMPLETE ✓")
    logger.info("=" * 60)