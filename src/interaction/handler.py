"""
Interaction handler: Bridges synthesis output to user display.

Connects the SynthesisEngine (persona-driven responses) to the
UI/terminal so the user actually "hears" from the companion.

Validation pipeline (after audit remediation):
  synthesis → CognitiveCritic.review() → ActionPolicy.check() → display/action
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.persona.synthesis import SynthesisEngine, SynthesizedResponse
from src.persona.config import PersonaConfig
from src.actions.policy import ActionSource

logger = logging.getLogger(__name__)


class InteractionEvent(str, Enum):
    """Types of interaction events."""
    OBSERVATION = "observation"
    GREETING = "greeting"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"


@dataclass
class InteractionMessage:
    """
    A single interaction message ready for display.
    
    Attributes:
        text: The display text
        event: What kind of event triggered this
        animation: Suggested companion animation
        timestamp: When this was generated
        source: Where this came from (synthesis/manual)
    """
    text: str
    event: InteractionEvent = InteractionEvent.OBSERVATION
    animation: str = "idle"
    timestamp: float = field(default_factory=time.time)
    source: str = "synthesis"
    
    def format_for_display(self) -> str:
        """Format message for terminal/UI display."""
        time_str = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        prefix = {
            InteractionEvent.GREETING: "👋",
            InteractionEvent.OBSERVATION: "👀",
            InteractionEvent.RESPONSE: "💬",
            InteractionEvent.STATUS: "ℹ️",
            InteractionEvent.ERROR: "⚠️",
        }.get(self.event, "💬")
        
        return f"{prefix} [{time_str}] {self.text}"
    
    def format_for_overlay(self) -> str:
        """Format message for overlay display (shorter)."""
        return self.text[:100]


class InteractionHandler:
    """
    Handles the flow of interaction between synthesis engine and display.
    
    This is the final bridge before the user sees anything. It:
    1. Receives synthesized responses
    2. Manages message history
    3. Provides formatted output for different display contexts
    4. Delegates to display callbacks (UI + TTS + System Actions)
    5. Enforces the Interaction Budget to prevent spam
    
    Guardrails:
    - GreetingCooldown: 60s between greetings
    - StateChangeCooldown: 180s (3 min) between verbal state-change acknowledgements
      This prevents rapid-fire commentary like "Coding!" → "Browser!" → "Terminal!"
    - User prompts always bypass cooldowns
    - System actions are always validated against whitelist before execution
    
    In production, display callbacks would update the Qt overlay.
    For testing, we use terminal output via display_callback.
    """
    
    # Cooldown constants (seconds)
    GREETING_COOLDOWN = 60.0      # Don't greet again within 60s
    STATE_CHANGE_COOLDOWN = 180.0  # 3 min silence between state-change observations
    USER_PROMPT_COOLDOWN = 0.0     # User prompts always allowed
    
    def __init__(
        self,
        synthesis_engine: SynthesisEngine,
        display_callback: Optional[Callable[[InteractionMessage], None]] = None,
        tts_engine: Optional[Any] = None,
        action_executor: Optional[Any] = None,
        action_callback: Optional[Callable[[str, Any], None]] = None,
        cognitive_critic: Optional[Any] = None,
        action_policy: Optional[Any] = None,
    ) -> None:
        """
        Initialize interaction handler.

        Args:
            synthesis_engine: Engine for generating responses
            display_callback: Function to call with display messages
            tts_engine: Optional TTSEngine for speech synthesis
            action_executor: Optional SystemExecutor for system commands
            action_callback: Optional callback for actions requiring approval
            cognitive_critic: Optional CognitiveCritic for validation (groundedness, provenance)
            action_policy: Optional ActionPolicy for provenance-based action gating
        """
        self._engine = synthesis_engine
        self._display_callback = display_callback
        self._tts_engine = tts_engine
        self._action_executor = action_executor
        self._action_callback = action_callback
        self._critic = cognitive_critic
        self._action_policy = action_policy

        # Message history
        self._messages: list[InteractionMessage] = []
        self._max_history = 50

        # State tracking
        self._last_greeting_time: Optional[float] = None
        self._greeting_cooldown = self.GREETING_COOLDOWN

        # Interaction Budget: State-change anti-spam
        self._last_interaction_time: Optional[float] = None
        self._state_change_cooldown = self.STATE_CHANGE_COOLDOWN
        self._last_content_type: Optional[str] = None  # Track context shifts

        logger.info(
            f"InteractionHandler initialized "
            f"(greeting_cd={self._greeting_cooldown}s, "
            f"state_change_cd={self._state_change_cooldown}s, "
            f"tts={'enabled' if tts_engine else 'disabled'}, "
            f"actions={'enabled' if action_executor else 'disabled'}, "
            f"critic={'enabled' if cognitive_critic else 'disabled'}, "
            f"policy={'enabled' if action_policy else 'disabled'})"
        )
    
    def on_observation(
        self,
        reasoning: Any,
        context_text: str = ""
    ) -> InteractionMessage:
        """
        Handle a new observation from the vision system.
        
        Enforces the Interaction Budget (StateChangeCooldown):
        - If the companion has spoken recently (within cooldown window),
          suppresses verbal output for context shifts
        - Always processes the observation internally (memory, reasoning)
        - Returns a silent STATUS message when suppressed
        
        Args:
            reasoning: VisualReasoning from ReasoningBridge
            context_text: Additional context
            
        Returns:
            Formatted interaction message (may be silent if cooldown active)
        """
        # Detect content type shift
        content_type = getattr(reasoning, 'content_type', None) if reasoning else None
        is_context_shift = (
            content_type is not None and
            content_type != self._last_content_type
        )
        
        # Check Interaction Budget for context shifts
        if is_context_shift and self._is_state_change_cooldown_active():
            logger.debug(
                f"StateChangeCooldown active: suppressing observation for "
                f"'{self._last_content_type}' → '{content_type}' "
                f"(last interaction was {self._seconds_since_last_interaction():.0f}s ago)"
            )
            # Still update tracking state
            self._last_content_type = content_type
            # Return silent message - no verbal output
            return InteractionMessage(
                text="",
                event=InteractionEvent.STATUS,
                animation="idle",
                source="synthesis"
            )
        
        # Synthesize response
        response = self._engine.synthesize(
            reasoning=reasoning,
            context_text=context_text,
            trigger_type="observation"
        )

        # ── Run response through cognitive critic ──
        response_dict = {
            "animation": response.animation,
            "speech": response.text,
            "action": getattr(response, 'action', '') or '',
        }
        if self._critic:
            # Autonomous observation source
            self._critic.set_rag_context([], source=ActionSource.AUTONOMOUS_OBSERVATION.value)
            verdict = self._critic.review(response_dict)
            response_dict = verdict.corrected_output
            self._critic.clear_rag_context()

        # Create message
        message = InteractionMessage(
            text=response_dict.get("speech", ""),
            event=InteractionEvent.OBSERVATION,
            animation=response_dict.get("animation", "idle"),
            source="synthesis"
        )

        # Update interaction budget tracker
        self._last_content_type = content_type
        self._last_interaction_time = time.time()

        self._add_message(message)
        return message
    
    def on_greeting(self, context: str = "idle") -> InteractionMessage:
        """
        Send a greeting (with cooldown).
        
        Args:
            context: Context for the greeting (morning/coding/etc.)
            
        Returns:
            Greeting message
        """
        # Check cooldown
        if self._last_greeting_time:
            elapsed = time.time() - self._last_greeting_time
            if elapsed < self._greeting_cooldown:
                logger.debug(f"Greeting cooldown active ({elapsed:.0f}s < {self._greeting_cooldown}s)")
                return InteractionMessage(
                    text="",
                    event=InteractionEvent.STATUS,
                    animation="idle"
                )
        
        greeting_text = self._engine.get_greeting(context)
        self._last_greeting_time = time.time()
        
        message = InteractionMessage(
            text=greeting_text,
            event=InteractionEvent.GREETING,
            animation="wave",
            source="synthesis"
        )
        
        self._add_message(message)
        return message
    
    def on_user_input(self, user_text: str) -> InteractionMessage:
        """
        Handle direct user input (text command).
        
        User prompts ALWAYS bypass cooldowns - the Interaction Budget
        only applies to autonomous observations.
        
        Args:
            user_text: What the user said/typed
            
        Returns:
            Response message
        """
        # Reset interaction timer on user input (resets cooldown)
        self._last_interaction_time = time.time()
        
        # Synthesize a response through the engine
        response = self._engine.synthesize(
            reasoning=None,
            context_text=user_text,
            trigger_type="user_input"
        )

        # ── Run response through cognitive critic ──
        response_dict = {
            "animation": response.animation,
            "speech": response.text,
            "action": getattr(response, 'action', '') or '',
        }
        if self._critic:
            # Voice command source for user input
            self._critic.set_rag_context([], source=ActionSource.VOICE_COMMAND.value)
            verdict = self._critic.review(response_dict)
            response_dict = verdict.corrected_output
            self._critic.clear_rag_context()

        message = InteractionMessage(
            text=response_dict.get("speech", ""),
            event=InteractionEvent.RESPONSE,
            animation="wave",
            source="synthesis"
        )

        self._add_message(message)
        return message
    
    def on_status(self, status_text: str) -> InteractionMessage:
        """
        Send a status message (non-intrusive).
        
        Args:
            status_text: Status information
            
        Returns:
            Status message
        """
        message = InteractionMessage(
            text=status_text,
            event=InteractionEvent.STATUS,
            animation="idle",
            source="manual"
        )
        
        self._add_message(message)
        return message
    
    def get_recent_messages(self, n: int = 5) -> list[InteractionMessage]:
        """Get recent messages for display."""
        return list(reversed(self._messages[-n:]))
    
    def get_history(self) -> list[InteractionMessage]:
        """Get full message history."""
        return list(self._messages)
    
    def clear_history(self) -> None:
        """Clear message history."""
        self._messages.clear()
        logger.debug("Interaction history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        stats = {
            "total_messages": len(self._messages),
            "last_greeting": (
                time.time() - self._last_greeting_time
                if self._last_greeting_time else None
            ),
            "last_interaction": (
                time.time() - self._last_interaction_time
                if self._last_interaction_time else None
            ),
            "current_content_type": self._last_content_type,
            "state_change_cooldown_active": self._is_state_change_cooldown_active(),
            "engine_stats": self._engine.get_stats(),
        }
        if self._action_executor:
            stats["action_stats"] = self._action_executor.get_stats()
        if self._critic:
            stats["critic_stats"] = self._critic.get_stats()
        if self._action_policy:
            stats["policy_stats"] = self._action_policy.get_stats()
        return stats
    
    def _route_side_effects(self, message: InteractionMessage) -> None:
        """
        Route TTS and system actions from a synthesized message.
        
        Called after every message is added to history. Handles:
        - TTS: Speak the message text if talk animation is set
        - Actions: Execute validated system commands
        """
        if not message or not message.text:
            return
        
        # TTS routing
        if self._tts_engine and self._tts_engine.is_ready():
            if message.animation in ("talk", "wave"):
                # Speak in background thread — doesn't block UI
                self._tts_engine.speak(message.text, blocking=False)
                logger.debug(f"TTS queued: {message.text[:40]}...")
        
        # Action routing (if message carries action metadata)
        action_name = getattr(message, 'action', '') or ''
        if action_name and self._action_executor:
            self._execute_action(action_name, message.text)
    
    def _execute_action(self, action_name: str, context: str = "") -> None:
        """
        Execute a system action through the whitelist executor.
        
        Args:
            action_name: The action key from the LLM response
            context: Optional context for the action callback
        """
        if not self._action_executor:
            return
        
        # Validate against whitelist
        request = self._action_executor.validate(action_name)
        if not request.is_valid:
            logger.warning(f"Action '{action_name}' rejected by whitelist")
            return
        
        # Check if approval is required
        if request.command and request.command.requires_approval:
            logger.info(f"Action '{action_name}' requires user approval")
            if self._action_callback:
                self._action_callback(action_name, request)
            return
        
        # ── Provenance policy check before execution ──
        if self._action_policy:
            source = ActionSource.AUTONOMOUS_OBSERVATION  # Default for handler-routed actions
            action_category = request.command.category.value if request.command else ""
            from src.actions.policy import PolicyDecision
            decision = self._action_policy.check(
                action_name=action_name,
                source=source,
                requires_approval=request.command.requires_approval if request.command else False,
                action_category=action_category,
            )
            if not decision:
                logger.warning(f"ActionPolicy blocked '{action_name}': {decision.reason}")
                return

        # Execute
        result = self._action_executor.execute(request)
        if result.success:
            logger.info(f"Action '{action_name}' succeeded: {result.output[:80]}")
            # Route result back through display callback
            if self._display_callback and result.output:
                result_msg = InteractionMessage(
                    text=f"[{action_name}] {result.output[:100]}",
                    event=InteractionEvent.STATUS,
                    animation="nod",
                    source="action",
                )
                self._display_callback(result_msg)
        else:
            logger.warning(f"Action '{action_name}' failed: {result.error}")
    
    def execute_action_by_name(self, action_name: str, params: Optional[list] = None) -> Any:
        """
        Public API for executing a system action from external code (e.g., user command).
        
        Args:
            action_name: The action key from the whitelist
            params: Optional parameters for the command
            
        Returns:
            ActionResult from the executor
        """
        if not self._action_executor:
            return None
        
        request = self._action_executor.validate(action_name, params or [])
        if not request.is_valid:
            return None
        
        return self._action_executor.execute(request)
    
    def _is_state_change_cooldown_active(self) -> bool:
        """Check if the state-change cooldown is currently active."""
        if self._last_interaction_time is None:
            return False
        elapsed = time.time() - self._last_interaction_time
        return elapsed < self._state_change_cooldown
    
    def _seconds_since_last_interaction(self) -> float:
        """Get seconds since the last verbal interaction."""
        if self._last_interaction_time is None:
            return float('inf')
        return time.time() - self._last_interaction_time
    
    def set_state_change_cooldown(self, seconds: float) -> None:
        """
        Adjust the state-change cooldown period.
        
        Args:
            seconds: New cooldown in seconds (min 30s, max 600s)
        """
        self._state_change_cooldown = max(30.0, min(600.0, seconds))
        logger.info(f"State change cooldown set to {self._state_change_cooldown}s")
    
    def reset_cooldowns(self) -> None:
        """Reset all cooldowns (useful for testing or manual override)."""
        self._last_greeting_time = None
        self._last_interaction_time = None
        self._last_content_type = None
        logger.info("All interaction cooldowns reset")
    
    def _add_message(self, message: InteractionMessage) -> None:
        """Add message to history and dispatch to display."""
        # Only add non-empty messages to history
        if message.text:
            self._messages.append(message)
        
        # Trim history
        if len(self._messages) > self._max_history:
            self._messages = self._messages[-self._max_history:]
        
        # Dispatch to display callback
        if self._display_callback and message.text:
            self._display_callback(message)
        
        # Log to terminal
        if message.text:
            logger.info(f"[{message.event.value}] {message.text}")