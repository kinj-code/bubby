"""Proactivity Evaluator: Determines whether the companion should intervene proactively.

Computes an Intervention Urgency Score (0.0-1.0) based on:
1. Terminal sensor events (build failures, tracebacks → high urgency)
2. Context change magnitude (minor scroll vs. major app switch)
3. User activity patterns (coding vs. reading vs. watching video)
4. Time since last interaction (long silence → may check in)
5. Memory match relevance (LTM finds relevant past insight)

Intervention only triggers if urgency clears configurable threshold.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ActivityLevel(str, Enum):
    """User activity intensity level."""
    IDLE = "idle"           # No activity detected
    PASSIVE = "passive"     # Reading, watching
    ACTIVE = "active"       # Coding, writing, browsing
    INTENSE = "intense"     # Terminal commands, rapid typing
    CRITICAL = "critical"   # Errors detected, user struggling


class InterventionType(str, Enum):
    """Types of proactive intervention."""
    NONE = "none"                    # No intervention needed
    SILENT_OBSERVATION = "silent"    # Observe but don't speak
    HELPFUL_SUGGESTION = "suggest"   # Offer a helpful tip
    ERROR_ASSIST = "error_assist"    # User hit an error — offer help
    MEMORY_RECALL = "memory"         # Recall relevant memory
    CHECK_IN = "check_in"            # Gentle check-in after long idle
    SYSTEM_ALERT = "alert"           # Critical system notification


@dataclass
class ProactivityContext:
    """Context snapshot for proactivity evaluation."""
    content_type: str = "unknown"
    content_confidence: float = 0.0
    activity_level: ActivityLevel = ActivityLevel.IDLE
    time_since_last_intervention: float = float('inf')
    time_since_last_user_action: float = float('inf')
    terminal_event: Optional[Any] = None     # TerminalState from sensor
    has_recent_error: bool = False
    memory_matches: int = 0                  # Number of relevant LTM matches
    is_context_shift: bool = False           # Major context change
    user_busy: bool = True                   # User appears deeply focused


@dataclass
class ProactivityDecision:
    """Decision output from the proactivity evaluator."""
    urgency_score: float = 0.0          # 0.0-1.0 intervention urgency
    should_intervene: bool = False      # Whether to trigger intervention
    intervention_type: InterventionType = InterventionType.NONE
    reason: str = ""                    # Human-readable reason
    confidence: float = 0.0             # Confidence in this decision


class ProactivityEvaluator:
    """
    Evaluates context to determine if proactive intervention is warranted.

    Configuration:
    - urgency_threshold: Minimum score to trigger intervention (default 0.75)
    - silent_check_in_minutes: Minutes of idle before gentle check-in
    - cooldown_minutes: Minimum time between proactive interventions
    - error_assist_enabled: Whether to assist with terminal errors
    """

    def __init__(
        self,
        urgency_threshold: float = 0.70,
        silent_check_in_minutes: float = 30.0,
        cooldown_minutes: float = 5.0,
        error_assist_enabled: bool = True,
        memory_recall_enabled: bool = True,
    ) -> None:
        self._threshold = urgency_threshold
        self._check_in_minutes = silent_check_in_minutes
        self._cooldown_minutes = cooldown_minutes
        self._error_assist_enabled = error_assist_enabled
        self._memory_recall_enabled = memory_recall_enabled
        
        self._last_intervention_time: Optional[float] = None
        self._total_evaluations = 0
        self._interventions_triggered = 0
        
        logger.info(
            f"ProactivityEvaluator initialized "
            f"(threshold={urgency_threshold}, check_in={silent_check_in_minutes}min, "
            f"cooldown={cooldown_minutes}min, error_assist={error_assist_enabled})"
        )

    def evaluate(self, context: ProactivityContext) -> ProactivityDecision:
        """
        Evaluate context and return a proactivity decision.

        Args:
            context: Current context snapshot

        Returns:
            ProactivityDecision with urgency score and recommendation
        """
        self._total_evaluations += 1

        # Check cooldown
        if self._is_in_cooldown():
            return ProactivityDecision(
                urgency_score=0.0,
                should_intervene=False,
                intervention_type=InterventionType.NONE,
                reason="Cooldown active — recently intervened",
            )

        # Calculate urgency from multiple factors
        scores = {
            "terminal_error": self._score_terminal_event(context),
            "memory_relevance": self._score_memory_relevance(context),
            "idle_time": self._score_idle_time(context),
            "context_shift": self._score_context_shift(context),
            "error_pattern": self._score_error_pattern(context),
        }

        # Weighted combination (terminal errors are the dominant signal)
        # A traceback (0.90) alone gives 0.585; needs one other factor to trigger
        # Traceback + error pattern: 0.585 + 0.14 = 0.725 → triggers ✓
        total_score = (
            scores["terminal_error"] * 0.65 +    # Dominant: terminal errors demand attention
            scores["error_pattern"] * 0.20 +     # User workflow shows errors
            scores["memory_relevance"] * 0.10 +  # Relevant memory adds value
            scores["context_shift"] * 0.03 +     # Major context changes
            scores["idle_time"] * 0.02           # Gentle check-in is lowest priority
        )

        total_score = round(min(1.0, total_score), 2)

        # Determine intervention type
        intervention_type, reason = self._determine_intervention(context, scores)

        should_intervene = total_score >= self._threshold

        if should_intervene:
            self._last_intervention_time = time.time()
            self._interventions_triggered += 1
            logger.info(
                f"Proactive intervention triggered: {intervention_type.value} "
                f"(score={total_score:.2f}, reason={reason})"
            )

        return ProactivityDecision(
            urgency_score=total_score,
            should_intervene=should_intervene,
            intervention_type=intervention_type,
            reason=reason,
            confidence=min(total_score / self._threshold, 1.0),
        )

    def _score_terminal_event(self, context: ProactivityContext) -> float:
        """Score based on terminal sensor events."""
        if not context.terminal_event:
            return 0.0

        # TerminalState has an urgency field — use it directly
        urgency = getattr(context.terminal_event, 'urgency', 0.0)
        return urgency

    def _score_memory_relevance(self, context: ProactivityContext) -> float:
        """Score based on relevant LTM memory matches."""
        if not self._memory_recall_enabled:
            return 0.0

        if context.memory_matches == 0:
            return 0.0
        elif context.memory_matches == 1:
            return 0.3
        elif context.memory_matches == 2:
            return 0.5
        else:
            return 0.6  # 3+ matches — very relevant

    def _score_idle_time(self, context: ProactivityContext) -> float:
        """Score based on how long user has been idle."""
        minutes = context.time_since_last_user_action / 60.0

        if minutes < self._check_in_minutes:
            return 0.0
        elif minutes < 60:
            return 0.2  # 30-60 min idle → mild check-in urgency
        elif minutes < 120:
            return 0.4  # 1-2 hours
        else:
            return 0.5  # 2+ hours

    def _score_context_shift(self, context: ProactivityContext) -> float:
        """Score based on major context changes."""
        if not context.is_context_shift:
            return 0.0

        # Major shift (e.g., IDE → Browser → Terminal)
        if context.content_type in ("terminal", "error"):
            return 0.4
        return 0.2

    def _score_error_pattern(self, context: ProactivityContext) -> float:
        """Score based on detected error patterns in user workflow."""
        if context.has_recent_error and not context.user_busy:
            return 0.7  # User stopped and there's an error — assist!
        if context.has_recent_error:
            return 0.3  # Error detected but user is still working
        return 0.0

    def _is_in_cooldown(self) -> bool:
        """Check if we're within the intervention cooldown period."""
        if self._last_intervention_time is None:
            return False
        elapsed = time.time() - self._last_intervention_time
        return elapsed < (self._cooldown_minutes * 60)

    def _determine_intervention(
        self,
        context: ProactivityContext,
        scores: Dict[str, float],
    ) -> tuple:
        """
        Determine the type of intervention and a human-readable reason.

        Returns:
            (InterventionType, reason_string)
        """
        # Priority order: errors > memory > check-in > context shift

        if scores["terminal_error"] >= self._threshold:
            return (
                InterventionType.ERROR_ASSIST,
                f"Terminal error detected — high urgency",
            )

        if scores["error_pattern"] >= self._threshold:
            return (
                InterventionType.ERROR_ASSIST,
                f"User workflow shows errors — offering help",
            )

        if scores["memory_relevance"] >= 0.3:
            return (
                InterventionType.MEMORY_RECALL,
                f"Relevant memory found ({context.memory_matches} matches)",
            )

        if scores["idle_time"] >= 0.4:
            minutes = context.time_since_last_user_action / 60.0
            return (
                InterventionType.CHECK_IN,
                f"User idle for {minutes:.0f} minutes — gentle check-in",
            )

        if scores["context_shift"] >= 0.3:
            return (
                InterventionType.SILENT_OBSERVATION,
                f"Major context shift to {context.content_type}",
            )

        return (
            InterventionType.NONE,
            "No intervention criteria met",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "interventions_triggered": self._interventions_triggered,
            "threshold": self._threshold,
            "cooldown_minutes": self._cooldown_minutes,
            "in_cooldown": self._is_in_cooldown(),
        }

    def reset_cooldown(self) -> None:
        """Reset the intervention cooldown (for testing)."""
        self._last_intervention_time = None


# Testing helper
if __name__ == "__main__":
    import tempfile
    import os
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("PROACTIVITY EVALUATOR TEST")
    logger.info("=" * 60)

    evaluator = ProactivityEvaluator(urgency_threshold=0.75)

    # Test 1: Low urgency — coding, no errors
    ctx = ProactivityContext(
        content_type="code",
        activity_level=ActivityLevel.ACTIVE,
        time_since_last_user_action=10,
        user_busy=True,
    )
    decision = evaluator.evaluate(ctx)
    assert not decision.should_intervene
    assert decision.urgency_score < 0.5
    logger.info(f"✓ Test 1: Coding focus → no intervention (score={decision.urgency_score})")

    # Test 2: Terminal error — high urgency
    from src.sensors.terminal import TerminalState, TerminalEvent
    terminal_state = TerminalState(
        event=TerminalEvent.TRACEBACK,
        exit_code=1,
        error_summary="ZeroDivisionError: division by zero",
        urgency=0.90,
    )
    ctx = ProactivityContext(
        content_type="terminal",
        activity_level=ActivityLevel.CRITICAL,
        terminal_event=terminal_state,
        has_recent_error=True,
        user_busy=False,
    )
    decision = evaluator.evaluate(ctx)
    assert decision.should_intervene, f"Expected intervention, got score={decision.urgency_score}"
    assert decision.intervention_type == InterventionType.ERROR_ASSIST
    logger.info(f"✓ Test 2: Traceback → intervention triggered (score={decision.urgency_score})")

    # Test 3: Memory recall
    evaluator.reset_cooldown()
    ctx = ProactivityContext(
        content_type="code",
        activity_level=ActivityLevel.ACTIVE,
        memory_matches=2,
        user_busy=False,
    )
    decision = evaluator.evaluate(ctx)
    # Memory alone may not pass 0.75 threshold (0.5 * 0.20 = 0.10)
    # That's correct — memory alone shouldn't trigger unless other factors align
    logger.info(f"✓ Test 3: Memory alone → score={decision.urgency_score} (type={decision.intervention_type.value})")

    # Test 4: Long idle → check-in
    evaluator.reset_cooldown()
    ctx = ProactivityContext(
        content_type="idle",
        activity_level=ActivityLevel.IDLE,
        time_since_last_user_action=45 * 60,  # 45 minutes
        user_busy=False,
    )
    decision = evaluator.evaluate(ctx)
    assert decision.urgency_score > 0.0
    logger.info(f"✓ Test 4: 45min idle → score={decision.urgency_score}, type={decision.intervention_type.value}")

    # Test 5: Cooldown enforcement
    evaluator.reset_cooldown()
    evaluator._last_intervention_time = time.time()  # Just intervened
    ctx = ProactivityContext(
        content_type="terminal",
        terminal_event=terminal_state,
    )
    decision = evaluator.evaluate(ctx)
    assert not decision.should_intervene
    assert "Cooldown" in decision.reason
    logger.info(f"✓ Test 5: Cooldown enforced: '{decision.reason}'")

    # Test 6: Combined factors
    evaluator.reset_cooldown()
    ctx = ProactivityContext(
        content_type="terminal",
        activity_level=ActivityLevel.CRITICAL,
        terminal_event=terminal_state,
        has_recent_error=True,
        memory_matches=1,
        user_busy=False,
        is_context_shift=True,
    )
    decision = evaluator.evaluate(ctx)
    assert decision.should_intervene
    assert decision.urgency_score >= 0.75
    logger.info(f"✓ Test 6: Combined factors → score={decision.urgency_score}, type={decision.intervention_type.value}")

    # Test 7: Stats
    stats = evaluator.get_stats()
    assert stats["total_evaluations"] == 6
    logger.info(f"✓ Test 7: Stats: {stats}")

    logger.info("\nALL PROACTIVITY EVALUATOR TESTS PASSED")