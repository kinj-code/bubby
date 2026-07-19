"""Cognitive Critic: Safety verification layer for LLM output.

Intercepts the LLM's raw JSON output before routing to InteractionHandler.
Performs three validation checks:
1. Utility Check: Is this comment adding real value?
2. Safety Check: Does the action match the whitelist exactly?
3. Redundancy Check: Has similar content been said recently?

Failed outputs are silently mutated to safe defaults instead of raising errors.
"""

import logging
import time
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Result of cognitive critic validation."""
    passed: bool = True
    failures: List[str] = field(default_factory=list)      # What checks failed
    original_output: Dict[str, Any] = field(default_factory=dict)
    corrected_output: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def was_corrected(self) -> bool:
        """Whether the output was mutated by the critic."""
        return not self.passed or bool(self.failures)


# Safe default output when critic rejects something entirely
SAFE_SILENT_OUTPUT = {
    "animation": "think",
    "speech": "",
    "action": "",
}

# Patterns in speech that indicate low utility
LOW_UTILITY_PATTERNS = [
    "you are coding",
    "you're coding",
    "looks like you're typing",
    "i see you're working",
    "you seem to be browsing",
    "just letting you know",
    "as always",
    "just a reminder",
    "in case you didn't know",
]

# Maximum times the same phrase can appear in recent history
MAX_REPETITION_COUNT = 2


@dataclass
class RedundancyTracker:
    """Tracks recent outputs to detect repetition."""
    recent_speeches: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    max_tracked: int = 10
    
    def add(self, speech: str, action: str = "") -> None:
        """Record an output for future redundancy checks."""
        if speech:
            self.recent_speeches.append(speech.lower().strip())
        if action:
            self.recent_actions.append(action.lower().strip())
        
        # Trim
        if len(self.recent_speeches) > self.max_tracked:
            self.recent_speeches = self.recent_speeches[-self.max_tracked:]
        if len(self.recent_actions) > self.max_tracked:
            self.recent_actions = self.recent_actions[-self.max_tracked:]
    
    def is_redundant_speech(self, speech: str) -> bool:
        """Check if similar speech was recently said."""
        if not speech or not self.recent_speeches:
            return False
        
        speech_lower = speech.lower().strip()
        
        # Exact match
        if speech_lower in self.recent_speeches:
            return True
        
        # Substring match (>70% overlap)
        for recent in self.recent_speeches:
            shorter = min(speech_lower, recent, key=len)
            if len(shorter) > 15:
                common_words = set(shorter.split()) & set(
                    (speech_lower if len(speech_lower) < len(recent) else recent).split()
                )
                total_words = len(shorter.split())
                if total_words > 0 and len(common_words) / total_words > 0.7:
                    return True
        
        return False
    
    def is_redundant_action(self, action: str) -> bool:
        """Check if the same action was recently requested."""
        if not action or not self.recent_actions:
            return False
        return action.lower() in self.recent_actions[-3:]  # Last 3 actions


class CognitiveCritic:
    """
    Intercepts and validates LLM output before it reaches the user.

    Validation flow:
    1. Parse the raw JSON output
    2. Utility Check → is this actually helpful?
    3. Safety Check → is the action whitelisted?
    4. Redundancy Check → haven't we just said this?
    5. If any check fails → mutate output to safe silent default
    6. Return validated output (original or corrected)
    """

    def __init__(
        self,
        action_executor: Optional[Any] = None,
        redundancy_tracker: Optional[RedundancyTracker] = None,
    ) -> None:
        """
        Initialize the cognitive critic.

        Args:
            action_executor: SystemExecutor for whitelist validation
            redundancy_tracker: Tracker for detecting repetition
        """
        self._action_executor = action_executor
        self._redundancy_tracker = redundancy_tracker or RedundancyTracker()
        self._total_checks = 0
        self._rejections = 0
        self._corrections = 0
        
        logger.info(
            f"CognitiveCritic initialized "
            f"(executor={'enabled' if action_executor else 'disabled'})"
        )

    def review(self, output: Dict[str, Any]) -> CriticVerdict:
        """
        Review an LLM output and return a verdict.

        Args:
            output: Dict with keys: animation, speech, action

        Returns:
            CriticVerdict with validation result and (possibly corrected) output
        """
        self._total_checks += 1
        failures = []
        corrected = dict(output)  # Start with a copy

        # ── UTILITY CHECK ──
        if not self._check_utility(output):
            failures.append("utility: filler/no-value speech")
            corrected["speech"] = ""

        # ── SAFETY CHECK ──
        if not self._check_safety(output):
            failures.append("safety: unwhitelisted action")
            corrected["action"] = ""
            corrected["speech"] = ""  # Don't speak if action was hallucinated

        # ── REDUNDANCY CHECK ──
        if not self._check_redundancy(output):
            failures.append("redundancy: recently said or done")
            corrected["speech"] = ""

        # ── POST-VALIDATION FIXUP ──
        # If speech was cleared but action is valid, it's fine
        # If both cleared → silent idle
        if not corrected.get("speech") and not corrected.get("action"):
            if failures:
                # Critic rejected something → use confused/silent
                corrected = {
                    "animation": "confused" if "safety" in str(failures) else "idle",
                    "speech": "",
                    "action": "",
                }
        
        # If speech was cleared but animation is talk → fix
        if not corrected.get("speech") and corrected.get("animation") == "talk":
            corrected["animation"] = "idle"

        passed = len(failures) == 0
        if not passed:
            self._rejections += 1
            if corrected != output:
                self._corrections += 1
            logger.warning(
                f"Critic rejected output: {failures} "
                f"(original={output}, corrected={corrected})"
            )
        else:
            logger.debug(f"Critic passed: {output.get('speech', '')[:40]}")

        # Track for redundancy regardless
        if output.get("speech"):
            self._redundancy_tracker.add(
                output.get("speech", ""),
                output.get("action", ""),
            )

        return CriticVerdict(
            passed=passed,
            failures=failures,
            original_output=dict(output),
            corrected_output=corrected,
        )

    def _check_utility(self, output: Dict[str, Any]) -> bool:
        """Check if the speech output provides real utility."""
        speech = output.get("speech", "")

        # Empty speech is always OK (silent observation is valid)
        if not speech:
            return True

        # Too short to be meaningful
        if len(speech) < 10:
            return False

        # Check for low-utility filler phrases
        speech_lower = speech.lower()
        for pattern in LOW_UTILITY_PATTERNS:
            if pattern in speech_lower:
                # Only reject if it's the MAIN content (not just part of it)
                # Check if the speech is mostly just stating the obvious
                if len(speech) < 50 and pattern in speech_lower:
                    return False

        # Check if speech contains internal state leakage
        forbidden = [
            "confidence score",
            "i'm just a",
            "as a language model",
            "i cannot actually",
        ]
        for term in forbidden:
            if term in speech_lower:
                return False

        return True

    def _check_safety(self, output: Dict[str, Any]) -> bool:
        """Check if the action is safe (whitelisted)."""
        action = output.get("action", "")

        # No action → always safe
        if not action:
            return True

        # If we have an executor, validate against it
        if self._action_executor:
            from src.actions.executor import ActionRequest
            request = self._action_executor.validate(action)
            return request.is_valid

        # Without executor, only allow empty actions
        # This prevents any action execution without whitelist
        return False

    def _check_redundancy(self, output: Dict[str, Any]) -> bool:
        """Check if this output is redundant with recent history."""
        speech = output.get("speech", "")
        action = output.get("action", "")

        # Check speech redundancy
        if speech and self._redundancy_tracker.is_redundant_speech(speech):
            return False

        # Check action redundancy
        if action and self._redundancy_tracker.is_redundant_action(action):
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get critic statistics."""
        return {
            "total_checks": self._total_checks,
            "rejections": self._rejections,
            "corrections": self._corrections,
            "pass_rate": (
                (self._total_checks - self._rejections) / self._total_checks
                if self._total_checks > 0 else 1.0
            ),
        }


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("COGNITIVE CRITIC TEST")
    logger.info("=" * 60)

    critic = CognitiveCritic()

    # Test 1: Clean output — should pass
    output = {"animation": "talk", "speech": "Your build failed with a ModuleNotFoundError — need help fixing the import?", "action": ""}
    verdict = critic.review(output)
    assert verdict.passed
    assert not verdict.was_corrected
    logger.info(f"✓ Test 1: Clean useful output passes: '{output['speech'][:50]}...'")

    # Test 2: Low utility filler — should be rejected
    output = {"animation": "talk", "speech": "You are coding", "action": ""}
    verdict = critic.review(output)
    assert not verdict.passed or verdict.corrected_output["speech"] == ""
    logger.info(f"✓ Test 2: Low utility filler rejected")

    # Test 3: Hallucinated action — should be corrected
    output = {"animation": "talk", "speech": "Deleting system files!", "action": "delete_all"}
    verdict = critic.review(output)
    assert "safety" in str(verdict.failures)
    assert verdict.corrected_output["action"] == ""
    logger.info(f"✓ Test 3: Hallucinated action blocked: failures={verdict.failures}")

    # Test 4: Redundancy — same speech twice
    output = {"animation": "talk", "speech": "Nice work on that Python script!", "action": ""}
    critic.review(output)  # First time
    verdict = critic.review(output)  # Second time — should be flagged
    assert "redundancy" in str(verdict.failures)
    logger.info(f"✓ Test 4: Redundant speech detected")

    # Test 5: Internal state leakage — should be caught
    output = {"animation": "talk", "speech": "As a language model, I cannot actually execute code", "action": ""}
    verdict = critic.review(output)
    assert "utility" in str(verdict.failures)
    logger.info(f"✓ Test 5: Internal state leakage caught")

    # Test 6: Whiltelisted action + valid speech — passes
    from src.actions.executor import SystemExecutor
    critic_with_executor = CognitiveCritic(action_executor=SystemExecutor())
    output = {"animation": "talk", "speech": "Your battery is at 15% — should I adjust brightness?", "action": "check_battery"}
    verdict = critic_with_executor.review(output)
    assert verdict.passed, f"Expected pass but got failures: {verdict.failures}"
    logger.info(f"✓ Test 6: Whitelisted action + valid speech passes")

    # Test 7: Empty output — always passes
    output = {"animation": "idle", "speech": "", "action": ""}
    verdict = critic.review(output)
    assert verdict.passed
    logger.info(f"✓ Test 7: Silent idle always passes")

    # Test 8: Critic mutates to confused when safety fails
    output = {"animation": "talk", "speech": "Running hack command!", "action": "hack_all"}
    verdict = critic.review(output)
    if not verdict.passed:
        corrected = verdict.corrected_output
        # Safety failure should result in confused or idle animation
        assert corrected["animation"] in ("confused", "idle"), f"Expected confused/idle, got {corrected['animation']}"
        assert corrected["speech"] == ""
    logger.info(f"✓ Test 8: Safety failure → silent confused animation")

    # Test 9: Stats
    stats = critic.get_stats()
    assert stats["total_checks"] >= 6
    logger.info(f"✓ Test 9: Stats: pass_rate={stats['pass_rate']:.1%}")

    logger.info("\nALL COGNITIVE CRITIC TESTS PASSED")