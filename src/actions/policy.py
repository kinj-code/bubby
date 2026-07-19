"""Action Provenance Policy — independent, non-LLM gate between persona output and action executor.

Ensures that actions triggered by RAG-retrieved document content cannot execute
approval-tier commands. Only explicit user commands and sensor-triggered events
may fire commands that require user approval.

This is the DevSecOps-recommended policy layer from the audit:
"Put an independent, non-LLM policy layer between the persona output and the
action executor that only allows the 'requires approval'-tier commands to fire
from sensor-triggered or explicit-voice-command paths — never from RAG-retrieved content."

RAM: negligible (enum + boolean checks, no model).
"""

import logging
from enum import Enum, auto
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ActionSource(str, Enum):
    """Where did the action request originate?"""
    VOICE_COMMAND = "voice_command"           # User explicitly spoke/typed a command
    SENSOR_TRIGGER = "sensor_trigger"         # Terminal error, battery low, calendar event
    RAG_CONTEXT = "rag_context"               # Action was suggested by ingested document content
    AUTONOMOUS_OBSERVATION = "autonomous"      # Vision/reasoning pipeline decided on its own
    UNKNOWN = "unknown"                        # Source not tracked (legacy / default deny)


# Only these sources may trigger approval-tier actions.
# RAG_CONTEXT is intentionally excluded — documents should not be able to
# sleep the system, lock the screen, etc.
APPROVAL_ALLOWED_SOURCES: Set[ActionSource] = {
    ActionSource.VOICE_COMMAND,
    ActionSource.SENSOR_TRIGGER,
}

# Sources that may trigger ANY action (even non-approval).
# RAG_CONTEXT is restricted — it can only trigger safe, non-approval actions.
SAFE_ACTION_ALLOWED_SOURCES: Set[ActionSource] = {
    ActionSource.VOICE_COMMAND,
    ActionSource.SENSOR_TRIGGER,
    ActionSource.AUTONOMOUS_OBSERVATION,
}

# RAG_CONTEXT can only trigger these categories (read-only, informational)
RAG_ALLOWED_CATEGORIES: Set[str] = {
    "system_info",
    "notification",
    "display",
    "utility",
}
# Explicitly blocked for RAG: power, file_ops (if we add write ops later)


class PolicyDecision:
    """Result of a policy check."""

    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        status = "ALLOW" if self.allowed else "DENY"
        return f"PolicyDecision({status}: {self.reason})"


class ActionPolicy:
    """
    Enforces provenance-based action authorization.

    Usage:
        policy = ActionPolicy()
        decision = policy.check(
            action_name="sleep_system",
            source=ActionSource.RAG_CONTEXT,
            requires_approval=True,
        )
        if not decision:
            logger.warning(f"Action blocked by policy: {decision.reason}")
            return  # Do not execute
    """

    def __init__(self, strict_mode: bool = True) -> None:
        self._strict_mode = strict_mode  # If True, UNKNOWN source = deny
        self._blocks = 0
        self._allows = 0
        logger.info(
            f"ActionPolicy initialized (strict={strict_mode}, "
            f"approval_sources={[s.value for s in APPROVAL_ALLOWED_SOURCES]})"
        )

    def check(
        self,
        action_name: str,
        source: ActionSource,
        requires_approval: bool = False,
        action_category: str = "",
    ) -> PolicyDecision:
        """
        Determine whether an action is authorized given its provenance.

        Args:
            action_name: The action key (e.g. 'sleep_system')
            source: Where the request came from
            requires_approval: Whether the whitelisted command requires user approval
            action_category: The ActionCategory value (e.g. 'power', 'system_info')

        Returns:
            PolicyDecision — truthy if allowed, falsy with reason if blocked
        """
        # UNKNOWN source in strict mode → deny
        if source == ActionSource.UNKNOWN:
            if self._strict_mode:
                self._blocks += 1
                return PolicyDecision(
                    False,
                    f"Action '{action_name}' blocked: unknown provenance (strict mode)"
                )
            # Non-strict: allow but warn
            logger.warning(f"Action '{action_name}' from UNKNOWN source — allowing in non-strict mode")
            self._allows += 1
            return PolicyDecision(True, "non-strict fallback")

        # If action requires approval, only approved sources may fire it
        if requires_approval:
            if source not in APPROVAL_ALLOWED_SOURCES:
                self._blocks += 1
                return PolicyDecision(
                    False,
                    f"Action '{action_name}' requires approval — "
                    f"source '{source.value}' is not authorized for approval-tier actions. "
                    f"Allowed sources: {[s.value for s in APPROVAL_ALLOWED_SOURCES]}"
                )

        # RAG_CONTEXT has restricted categories
        if source == ActionSource.RAG_CONTEXT:
            if action_category and action_category not in RAG_ALLOWED_CATEGORIES:
                self._blocks += 1
                return PolicyDecision(
                    False,
                    f"Action '{action_name}' (category='{action_category}') blocked: "
                    f"RAG documents cannot trigger actions in this category. "
                    f"RAG-allowed categories: {sorted(RAG_ALLOWED_CATEGORIES)}"
                )

        # If action doesn't require approval but source is RAG_CONTEXT,
        # only allow if the source is explicitly in safe sources
        # (RAG_CONTEXT is NOT in SAFE_ACTION_ALLOWED_SOURCES — it gets
        # category-level filtering above instead)
        if not requires_approval:
            if source == ActionSource.RAG_CONTEXT:
                # RAG can only fire if category passes the filter above
                # If we reach here, category is allowed
                pass
            elif source not in SAFE_ACTION_ALLOWED_SOURCES:
                # This shouldn't happen with current enum values
                self._blocks += 1
                return PolicyDecision(
                    False,
                    f"Action '{action_name}' blocked: source '{source.value}' not in safe sources"
                )

        self._allows += 1
        return PolicyDecision(True, f"Action '{action_name}' from '{source.value}' authorized")

    def get_stats(self):
        return {
            "blocks": self._blocks,
            "allows": self._allows,
            "strict_mode": self._strict_mode,
        }


# ── Testing ──
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logger.info("=" * 60)
    logger.info("ACTION POLICY TEST")
    logger.info("=" * 60)

    policy = ActionPolicy(strict_mode=True)

    # Test 1: Voice command can trigger approval-tier action
    decision = policy.check("sleep_system", ActionSource.VOICE_COMMAND, requires_approval=True, action_category="power")
    assert decision, f"Voice command should be allowed, got: {decision.reason}"
    logger.info(f"✓ Voice command → sleep_system: {decision}")

    # Test 2: RAG context CANNOT trigger approval-tier action
    decision = policy.check("lock_screen", ActionSource.RAG_CONTEXT, requires_approval=True, action_category="power")
    assert not decision, "RAG context should NOT be allowed for approval actions"
    logger.info(f"✓ RAG context → lock_screen: {decision}")

    # Test 3: RAG context CAN trigger safe actions (system_info)
    decision = policy.check("check_battery", ActionSource.RAG_CONTEXT, requires_approval=False, action_category="system_info")
    assert decision, f"RAG context should be allowed for system_info, got: {decision.reason}"
    logger.info(f"✓ RAG context → check_battery: {decision}")

    # Test 4: RAG context CANNOT trigger non-whitelisted RAG category
    decision = policy.check("check_battery", ActionSource.RAG_CONTEXT, requires_approval=False, action_category="power")
    assert not decision, "RAG should not trigger power actions even if non-approval"
    logger.info(f"✓ RAG context → power action: {decision}")

    # Test 5: Sensor trigger can fire approval-tier
    decision = policy.check("sleep_system", ActionSource.SENSOR_TRIGGER, requires_approval=True, action_category="power")
    assert decision, f"Sensor trigger should be allowed, got: {decision.reason}"
    logger.info(f"✓ Sensor trigger → sleep_system: {decision}")

    # Test 6: Autonomous observation can fire non-approval
    decision = policy.check("check_date", ActionSource.AUTONOMOUS_OBSERVATION, requires_approval=False, action_category="system_info")
    assert decision, f"Autonomous should be allowed for safe actions, got: {decision.reason}"
    logger.info(f"✓ Autonomous → check_date: {decision}")

    # Test 7: Autonomous observation CANNOT fire approval-tier
    decision = policy.check("lock_screen", ActionSource.AUTONOMOUS_OBSERVATION, requires_approval=True, action_category="power")
    assert not decision, "Autonomous should NOT fire approval actions"
    logger.info(f"✓ Autonomous → lock_screen: {decision}")

    # Test 8: UNKNOWN source → deny in strict mode
    decision = policy.check("check_date", ActionSource.UNKNOWN, requires_approval=False)
    assert not decision, "UNKNOWN source should be denied in strict mode"
    logger.info(f"✓ UNKNOWN → check_date: {decision}")

    # Test 9: Non-strict mode allows UNKNOWN with warning
    policy_loose = ActionPolicy(strict_mode=False)
    decision = policy_loose.check("check_date", ActionSource.UNKNOWN, requires_approval=False)
    assert decision, "UNKNOWN should be allowed in non-strict mode"
    logger.info(f"✓ UNKNOWN (non-strict) → check_date: {decision}")

    # Test 10: Stats
    stats = policy.get_stats()
    assert stats["blocks"] >= 3
    assert stats["allows"] >= 3
    logger.info(f"✓ Stats: {stats}")

    logger.info("\nALL ACTION POLICY TESTS PASSED")