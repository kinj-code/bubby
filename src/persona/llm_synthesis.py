"""LLM-aware synthesis engine: Combines template synthesis with dynamic LLM generation."""

import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.persona.config import PersonaConfig
from src.persona.synthesis import SynthesisEngine, SynthesizedResponse
from src.persona.prompts import (
    UnifiedPersonaPrompt,
    build_observation_prompt,
    build_greeting_prompt,
    build_user_input_prompt,
    BUBBY_RESPONSE_JSON_SCHEMA,
)
from src.persona.response_parser import parse_structured_response, StructuredResponse
from src.brain.reasoning import VisualReasoning
from src.memory.long_term_memory import LongTermMemory
from src.llm.inference import LLMInference, LLMConfig, InferenceResult
from src.llm.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class LLMSynthesisConfig:
    """Configuration for LLM-enhanced synthesis."""
    use_llm: bool = True                    # Enable LLM generation
    fallback_to_template: bool = True       # Fallback if LLM fails
    max_context_tokens: int = 1024          # Max tokens for context
    generation_timeout_ms: int = 60000      # Max generation time (60s for CPU inference)
    min_confidence_for_llm: float = 0.5     # Min reasoning confidence to use LLM
    
    # LLM model config
    model_id: Optional[str] = None          # Specific model ID (None = auto)
    temperature: float = 0.7                # Generation temperature
    max_response_tokens: int = 120          # Max response tokens


class LLMSynthesisEngine:
    """
    LLM-enhanced synthesis engine.
    
    Architecture:
    1. Template synthesis (fast, reliable) → always runs first
    2. LLM generation (dynamic, contextual) → runs async if enabled
    3. Response selection → LLM preferred if successful, else template
    
    This provides the best of both worlds:
    - Immediate response from templates (no perceived latency)
    - Rich, contextual responses from LLM when available
    - Graceful degradation if LLM unavailable/slow
    """
    
    def __init__(
        self,
        persona: PersonaConfig,
        long_term_memory: Optional[LongTermMemory] = None,
        config: Optional[LLMSynthesisConfig] = None,
        llm_inference: Optional[LLMInference] = None,
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        """
        Initialize LLM synthesis engine.
        
        Args:
            persona: Persona configuration
            long_term_memory: LTM for context retrieval
            config: LLM synthesis configuration
            llm_inference: Pre-initialized LLM inference (optional)
            model_manager: Model manager for auto-selection (optional)
        """
        self._persona = persona
        self._ltm = long_term_memory
        self._config = config or LLMSynthesisConfig()
        
        # Template engine (always available)
        self._template_engine = SynthesisEngine(persona, long_term_memory)
        
        # LLM components (lazy initialization)
        self._llm_inference = llm_inference
        self._model_manager = model_manager or ModelManager()
        self._llm_ready = False
        self._llm_lock = threading.RLock()
        
        # Stats
        self._stats = {
            "total_syntheses": 0,
            "template_responses": 0,
            "llm_responses": 0,
            "llm_failures": 0,
            "llm_timeouts": 0,
        }
        
        logger.info(f"LLMSynthesisEngine initialized (use_llm={self._config.use_llm})")
    
    def _ensure_llm(self) -> bool:
        """Lazy-initialize LLM inference if needed."""
        if not self._config.use_llm:
            return False
        
        with self._llm_lock:
            if self._llm_ready:
                return self._llm_inference is not None and self._llm_inference.is_ready()
            
            # Initialize model manager if not provided
            if self._llm_inference is None:
                # Priority 1: BUBBY_LLM_PATH env var (direct path override)
                local_path = self._model_manager.get_local_llm_path()
                if local_path:
                    model_path = local_path
                else:
                    # Priority 2: Auto-select best catalog model
                    model_id = self._config.model_id or self._model_manager.get_best_model()
                    if not model_id:
                        logger.warning("No LLM model available")
                        self._llm_ready = True  # Mark as checked to avoid retry
                        return False
                    model_path = self._model_manager.get_model_path(model_id)
                
                if not model_path or not model_path.exists():
                    path_str = str(model_path) if model_path else "unknown"
                    logger.warning(
                        f"No local GGUF model found at {path_str}. "
                        f"Please download a GGUF model and set BUBBY_LLM_PATH "
                        f"or use scripts/download_llm.py to download from catalog."
                    )
                    self._llm_ready = True
                    return False
                
                # Create LLM config with in-process llama.cpp settings
                llm_config = LLMConfig(
                    model_path=str(model_path),
                    n_threads=4,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_response_tokens,
                    persona=self._persona,
                )
                
                self._llm_inference = LLMInference(llm_config)
            
            # Initialize in background to avoid blocking
            if self._llm_inference.initialize():
                self._llm_ready = True
                logger.info(f"LLM inference ready (model: {self._llm_inference._config.model_path})")
                return True
            else:
                logger.warning("LLM initialization failed")
                self._llm_ready = True  # Mark as checked
                return False
    
    def synthesize(
        self,
        reasoning: Optional[VisualReasoning] = None,
        context_text: str = "",
        trigger_type: str = "observation",
    ) -> SynthesizedResponse:
        """
        Synthesize a response using template + optional LLM enhancement.
        
        Args:
            reasoning: Visual reasoning from ReasoningBridge
            context_text: Additional context description
            trigger_type: What triggered this response
            
        Returns:
            SynthesizedResponse with text, animation, metadata
        """
        self._stats["total_syntheses"] += 1
        
        # Always get template response first (instant)
        template_response = self._template_engine.synthesize(
            reasoning=reasoning,
            context_text=context_text,
            trigger_type=trigger_type,
        )
        
        # If LLM disabled or not ready, return template immediately
        if not self._config.use_llm or not self._ensure_llm():
            self._stats["template_responses"] += 1
            return template_response
        
        # Check if we should use LLM (confidence threshold)
        if reasoning and reasoning.confidence < self._config.min_confidence_for_llm:
            logger.debug(f"Confidence {reasoning.confidence:.2f} below threshold, using template")
            self._stats["template_responses"] += 1
            return template_response
        
        # Generate LLM response (with timeout, uses structured JSON output)
        structured = self._generate_llm_response(
            reasoning=reasoning,
            context_text=context_text,
            template_response=template_response,
            trigger_type=trigger_type,
        )
        
        if structured and structured.is_valid and structured.speech:
            # LLM chose to speak - apply guardrails and return
            cleaned_text = self._apply_guardrails(structured.speech)
            
            if cleaned_text and len(cleaned_text) > 2:
                self._stats["llm_responses"] += 1
                return SynthesizedResponse(
                    text=cleaned_text,
                    animation=structured.animation,
                    context_type=template_response.context_type,
                    has_memory_recall=template_response.has_memory_recall,
                )
        
        # LLM chose silence or failed - fall through to template
        # If LLM returned a valid silent response, use its animation preference
        if structured and structured.is_valid and structured.animation != "idle":
            # Use the LLM's animation even when silent (e.g., nod, observe)
            self._stats["llm_responses"] += 1
            return SynthesizedResponse(
                text="",
                animation=structured.animation,
                context_type=template_response.context_type,
                has_memory_recall=False,
            )
        
        # Fallback to template
        self._stats["template_responses"] += 1
        if structured:
            self._stats["llm_failures"] += 1
        return template_response
    
    def generate_async(
        self,
        reasoning: Optional[VisualReasoning],
        context_text: str,
        template_response: SynthesizedResponse,
        trigger_type: str,
        callback: callable,
    ) -> None:
        """
        Generate an LLM response in a background thread and call callback on completion.
        
        DOES NOT block the caller. The callback receives a SynthesizedResponse
        (which may be the template fallback if LLM fails).
        
        Args:
            reasoning: Visual reasoning context
            context_text: Raw context description
            template_response: Pre-generated template response (fallback)
            trigger_type: 'observation', 'greeting', 'user_input'
            callback: Called with final SynthesizedResponse on the background thread.
        """
        prompt = self._build_llm_prompt(
            reasoning=reasoning,
            context_text=context_text,
            template_response=template_response,
            trigger_type=trigger_type,
        )
        
        system_prompt = self._build_plain_system_prompt(
            reasoning=reasoning,
            context_text=context_text,
            trigger_type=trigger_type,
        )
        
        def _run():
            # If LLM is not initialized, fallback immediately (no warning spam)
            if not self._llm_inference or not self._llm_inference.is_ready():
                self._stats["template_responses"] += 1
                callback(template_response)
                return
            
            try:
                # Use plain text generation (no JSON schema — avoids timeout issues)
                result = self._llm_inference.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=self._config.max_response_tokens,
                    temperature=self._config.temperature,
                )
                if result and result.text and len(result.text.strip()) > 2:
                    cleaned = self._apply_guardrails(result.text.strip())
                    if cleaned:
                        self._stats["llm_responses"] += 1
                        response = SynthesizedResponse(
                            text=cleaned,
                            animation="talk",
                            context_type=template_response.context_type,
                            has_memory_recall=template_response.has_memory_recall,
                        )
                        callback(response)
                        return
            except Exception as e:
                logger.warning(f"Async LLM generation failed: {e}")
            
            # Fallback to template
            self._stats["template_responses"] += 1
            callback(template_response)
        
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _build_plain_system_prompt(
        self,
        reasoning: Optional[VisualReasoning],
        context_text: str,
        trigger_type: str,
    ) -> str:
        """Build a simple conversational system prompt (no JSON schema)."""
        name = self._persona.name
        backstory = self._persona.backstory.replace('\n', ' ').strip()
        traits_str = self._persona.traits.to_prompt_segment() if self._persona.traits else "friendly and helpful"
        
        return (
            f"{backstory}\n"
            f"Personality traits: {traits_str}\n"
            f"Be brief, warm, and conversational. Use 1-2 sentences max.\n"
            f"Never mention you are an AI, your internal state, or your decision logic.\n"
            f"Stay in character as {name} at all times."
        )

    def _generate_llm_response(
        self,
        reasoning: Optional[VisualReasoning],
        context_text: str,
        template_response: SynthesizedResponse,
        trigger_type: str,
    ) -> Optional[StructuredResponse]:
        """
        Synchronous plain-text LLM response (used for non-critical paths).
        
        Returns:
            StructuredResponse if successful, None on failure/timeout
        """
        prompt = self._build_llm_prompt(
            reasoning=reasoning,
            context_text=context_text,
            template_response=template_response,
            trigger_type=trigger_type,
        )
        
        system_prompt = self._build_plain_system_prompt(
            reasoning=reasoning,
            context_text=context_text,
            trigger_type=trigger_type,
        )
        
        try:
            result = self._llm_inference.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self._config.max_response_tokens,
                temperature=self._config.temperature,
            )
            if result and result.text and len(result.text.strip()) > 2:
                return StructuredResponse(
                    animation="talk",
                    speech=result.text.strip(),
                    is_valid=True,
                )
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
        
        return None
    
    def _build_structured_system_prompt(
        self,
        reasoning: Optional[VisualReasoning],
        context_text: str,
        trigger_type: str,
    ) -> str:
        """
        Build the persona-aware system prompt for structured generation.
        
        Uses UnifiedPersonaPrompt to instruct the LLM about speech vs. silence
        decisions based on user activity and context.
        """
        content_type = getattr(reasoning, 'content_type', 'unknown') if reasoning else 'unknown'
        description = getattr(reasoning, 'description', '') if reasoning else context_text
        
        prompt_builder = UnifiedPersonaPrompt(
            persona=self._persona,
            context_description=description or context_text,
            user_activity=content_type,
        )
        
        return prompt_builder.build()
    
    def _build_llm_prompt(
        self,
        reasoning: Optional[VisualReasoning],
        context_text: str,
        template_response: SynthesizedResponse,
        trigger_type: str,
    ) -> str:
        """
        Build the user prompt for structured LLM generation.
        
        Uses the prompt builders from prompts.py based on trigger type.
        """
        # Get memory context for prompt builders
        recent_memories = []
        if self._ltm:
            search_text = (
                context_text or
                (reasoning.description if reasoning else "") or
                ""
            )
            if search_text:
                memories = self._ltm.retrieve(search_text, k=2)
                recent_memories = [m.text for m in memories]
        
        content_type = getattr(reasoning, 'content_type', 'unknown') if reasoning else 'unknown'
        description = getattr(reasoning, 'description', '') if reasoning else ''
        
        if trigger_type == "greeting":
            return build_greeting_prompt(
                persona=self._persona,
                context=context_text or "startup",
            )
        
        elif trigger_type == "user_input":
            return build_user_input_prompt(
                persona=self._persona,
                user_text=context_text,
                recent_memories=recent_memories if recent_memories else None,
            )
        
        else:
            # observation or any other trigger
            return build_observation_prompt(
                persona=self._persona,
                observation=description or context_text or "Screen changed",
                content_type=content_type,
                recent_memories=recent_memories if recent_memories else None,
            )
    
    def _apply_guardrails(self, text: str) -> str:
        """Apply output guardrails to LLM response."""
        if not text:
            return ""
        
        # Length limit
        max_len = self._persona.max_response_length
        if len(text) > max_len:
            text = text[:max_len]
            # Try to end at sentence boundary
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > max_len // 2:
                text = text[:last_period + 1]
        
        # Remove internal state leaks
        forbidden = [
            "confidence", "decision_type", "memory_id", "record_id",
            "NodeStatus", "DecisionType", "VisualReasoning",
            "tokens", "generation_time", "stop_reason",
            "as an AI", "I am a language model", "I don't have access",
        ]
        
        for pattern in forbidden:
            import re
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up artifacts
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\[\(].*?[\]\)]', '', text)  # Remove [...]
        text = text.strip(' "\'')
        
        return text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        stats = dict(self._stats)
        stats["llm_ready"] = self._llm_ready
        stats["llm_available"] = self._llm_inference is not None and self._llm_inference.is_ready()
        if self._llm_inference:
            stats["llm_stats"] = self._llm_inference.get_stats()
        return stats
    
    def get_greeting(self, context: str = "idle") -> str:
        """Get a persona-appropriate greeting (delegates to template engine)."""
        return self._template_engine.get_greeting(context)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._llm_inference:
            self._llm_inference.shutdown()
        self._template_engine = None
        logger.info("LLMSynthesisEngine shutdown")


# Testing helper
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("LLM SYNTHESIS ENGINE TEST")
    logger.info("=" * 60)
    
    # Create components
    from src.persona.config import PersonaConfig, PersonaType
    
    persona = PersonaConfig(persona_type=PersonaType.WITTY_COMPANION)
    ltm = LongTermMemory()
    ltm.archive("User likes Python and TypeScript", importance=0.9)
    ltm.archive("User prefers simple, clean solutions", importance=0.8)
    
    # Test template-only (no LLM)
    config = LLMSynthesisConfig(use_llm=False)
    engine = LLMSynthesisEngine(persona=persona, long_term_memory=ltm, config=config)
    
    from src.brain.reasoning import VisualReasoning
    
    reasoning = VisualReasoning(
        content_type="code",
        confidence=0.9,
        description="User writing Python in VS Code",
        should_interact=True,
        should_observe=False,
        reasoning="User is actively coding",
    )
    
    response = engine.synthesize(reasoning, "User coding in VS Code")
    logger.info(f"Template response: {response.text}")
    logger.info(f"Animation: {response.animation}")
    
    # Test with LLM config (will use template fallback if no model)
    config_llm = LLMSynthesisConfig(use_llm=True)
    engine_llm = LLMSynthesisEngine(persona=persona, long_term_memory=ltm, config=config_llm)
    
    response2 = engine_llm.synthesize(reasoning, "User coding in VS Code")
    logger.info(f"LLM response: {response2.text}")
    logger.info(f"Animation: {response2.animation}")
    
    logger.info(f"Stats: {engine_llm.get_stats()}")
    
    engine.shutdown()
    engine_llm.shutdown()
    ltm.clear()
    
    logger.info("\n" + "=" * 60)
    logger.info("LLM SYNTHESIS ENGINE TEST COMPLETE")
    logger.info("=" * 60)