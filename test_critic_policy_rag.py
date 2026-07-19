#!/usr/bin/env python3
"""
Punch List Items P2+P3 — Integration Verification Test:
Prove the full critic → policy → groundedness chain works end-to-end.

Tests:
1. Critic groundedness check catches RAG hallucination (claims not in retrieved chunks)
2. Critic groundedness check passes when claims ARE in retrieved chunks
3. Critic provenance check blocks RAG-triggered approval-tier actions
4. Critic provenance check allows RAG-triggered safe actions (system_info)
5. Empty RAG context → groundedness check is a no-op (doesn't crash)
6. Critic stats include groundedness_rejections and policy_stats
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_groundedness_catches_hallucination():
    """Groundedness: claims not in RAG chunks → rejected."""
    from src.brain.critic import CognitiveCritic

    critic = CognitiveCritic()

    # Set RAG context with known content
    rag_chunks = [
        "The Salomon v Salomon & Co Ltd [1897] AC 22 case established separate legal personality.",
        "Exceptions to the rule include fraud, agency, and statutory exceptions under the Companies Act.",
    ]
    critic.set_rag_context(rag_chunks, source="rag_context")

    # Speech makes a claim NOT in the chunks — "Donoghue v Stevenson"
    output = {
        "animation": "talk",
        "speech": "According to Donoghue v Stevenson, the tort of negligence requires a duty of care.",
        "action": "",
    }
    verdict = critic.review(output)

    assert not verdict.passed, f"Should reject hallucinated claim, got passed={verdict.passed}"
    assert "groundedness" in str(verdict.failures), f"Expected groundedness failure, got {verdict.failures}"
    assert verdict.corrected_output["speech"] == "", "Should clear speech on groundedness failure"
    logger.info("✓ Groundedness catches hallucination (Donoghue v Stevenson not in Salomon chunks)")

    critic.clear_rag_context()
    logger.info("✓ Test 1 PASSED")


def test_groundedness_passes_valid_claim():
    """Groundedness: claims present in RAG chunks → passes."""
    from src.brain.critic import CognitiveCritic

    critic = CognitiveCritic()

    rag_chunks = [
        "The Salomon v Salomon & Co Ltd [1897] AC 22 case established separate legal personality.",
        "Exceptions: fraud (Gilford Motor Co v Horne), agency, statutory exceptions.",
    ]
    critic.set_rag_context(rag_chunks, source="rag_context")

    # Speech makes a claim that IS in the chunks — "Salomon v Salomon" and "Gilford Motor Co"
    output = {
        "animation": "talk",
        "speech": "Under Salomon v Salomon, the company is separate from its members. The fraud exception comes from Gilford Motor Co v Horne.",
        "action": "",
    }
    verdict = critic.review(output)

    assert verdict.passed, f"Should pass when claims are in RAG chunks, got failures: {verdict.failures}"
    logger.info("✓ Groundedness passes when claims appear in RAG context")

    critic.clear_rag_context()
    logger.info("✓ Test 2 PASSED")


def test_groundedness_noop_without_rag():
    """Groundedness: no RAG context → check is a no-op."""
    from src.brain.critic import CognitiveCritic

    critic = CognitiveCritic()
    # No RAG context set

    output = {
        "animation": "talk",
        "speech": "Some random factual claim about quantum physics and string theory.",
        "action": "",
    }
    verdict = critic.review(output)

    # Should pass because no RAG context = no groundedness check
    assert verdict.passed, f"Should pass without RAG context, got failures: {verdict.failures}"
    logger.info("✓ Groundedness is no-op when no RAG context set")

    logger.info("✓ Test 3 PASSED")


def test_provenance_blocks_rag_approval_action():
    """Provenance policy: RAG source cannot trigger approval-tier actions."""
    from src.brain.critic import CognitiveCritic
    from src.actions.executor import SystemExecutor
    from src.actions.policy import ActionPolicy

    executor = SystemExecutor()
    policy = ActionPolicy(strict_mode=True)
    critic = CognitiveCritic(action_executor=executor, action_policy=policy)

    # Set RAG context
    critic.set_rag_context([], source="rag_context")

    # RAG tries to trigger lock_screen (requires approval)
    output = {
        "animation": "talk",
        "speech": "According to the security policy, I should lock the screen now.",
        "action": "lock_screen",
    }
    verdict = critic.review(output)

    assert not verdict.passed, f"Should block approval action from RAG, got passed={verdict.passed}"
    assert "provenance" in str(verdict.failures), f"Expected provenance failure, got {verdict.failures}"
    assert verdict.corrected_output["action"] == "", "Should clear action on provenance failure"
    assert verdict.corrected_output["speech"] == "", "Should clear speech on provenance failure"
    logger.info("✓ Provenance blocks RAG → lock_screen (approval-tier)")

    critic.clear_rag_context()
    logger.info("✓ Test 4 PASSED")


def test_provenance_allows_rag_safe_action():
    """Provenance policy: RAG source CAN trigger safe actions (system_info)."""
    from src.brain.critic import CognitiveCritic
    from src.actions.executor import SystemExecutor
    from src.actions.policy import ActionPolicy

    executor = SystemExecutor()
    policy = ActionPolicy(strict_mode=True)
    critic = CognitiveCritic(action_executor=executor, action_policy=policy)

    # Set RAG context
    critic.set_rag_context([], source="rag_context")

    # RAG triggers check_battery (non-approval, system_info)
    output = {
        "animation": "talk",
        "speech": "Running a battery check.",
        "action": "check_battery",
    }
    verdict = critic.review(output)

    assert verdict.passed, f"Should allow safe action from RAG, got failures: {verdict.failures}"
    logger.info("✓ Provenance allows RAG → check_battery (system_info is RAG-safe)")

    critic.clear_rag_context()
    logger.info("✓ Test 5 PASSED")


def test_provenance_allows_voice_command_approval():
    """Provenance policy: Voice command CAN trigger approval-tier actions."""
    from src.brain.critic import CognitiveCritic
    from src.actions.executor import SystemExecutor
    from src.actions.policy import ActionPolicy

    executor = SystemExecutor()
    policy = ActionPolicy(strict_mode=True)
    critic = CognitiveCritic(action_executor=executor, action_policy=policy)

    # Simulate voice command source
    critic.set_rag_context([], source="voice_command")

    output = {
        "animation": "talk",
        "speech": "Locking the screen as requested.",
        "action": "lock_screen",
    }
    verdict = critic.review(output)

    # Voice command should be allowed for approval-tier actions
    # But note: lock_screen requires approval, and even voice commands
    # still need the user_approval flow — what the policy controls is
    # whether the source is *authorized* to trigger approval actions.
    # The actual approval still happens in SystemExecutor.execute()
    assert verdict.passed, f"Voice command should be authorized for approval actions, got failures: {verdict.failures}"
    logger.info("✓ Provenance allows voice_command → lock_screen (authorized source)")

    critic.clear_rag_context()
    logger.info("✓ Test 6 PASSED")


def test_critic_stats_include_groundedness_and_policy():
    """Critic stats should include groundedness_rejections and policy_stats."""
    from src.brain.critic import CognitiveCritic
    from src.actions.executor import SystemExecutor
    from src.actions.policy import ActionPolicy

    executor = SystemExecutor()
    policy = ActionPolicy(strict_mode=True)
    critic = CognitiveCritic(action_executor=executor, action_policy=policy)

    # Trigger a groundedness rejection
    critic.set_rag_context(["Salomon v Salomon establishes separate legal personality."], source="rag_context")
    output = {
        "animation": "talk",
        "speech": "Donoghue v Stevenson created the tort of negligence.",
        "action": "",
    }
    critic.review(output)

    # Trigger a provenance rejection
    critic.set_rag_context([], source="rag_context")
    output = {
        "animation": "talk",
        "speech": "Locking screen.",
        "action": "lock_screen",
    }
    critic.review(output)

    stats = critic.get_stats()
    assert "groundedness_rejections" in stats, f"Stats should include groundedness_rejections, got {stats.keys()}"
    assert stats["groundedness_rejections"] >= 1, f"Should have at least 1 groundedness rejection"
    assert "policy_stats" in stats, "Stats should include policy_stats"
    assert "blocks" in stats["policy_stats"], "Policy stats should have blocks"
    logger.info(f"✓ Stats include groundedness_rejections={stats['groundedness_rejections']}, policy_stats={stats['policy_stats']}")

    critic.clear_rag_context()
    logger.info("✓ Test 7 PASSED")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PUNCH LIST P2+P3 — CRITIC + POLICY + RAG INTEGRATION TESTS")
    logger.info("Groundedness (entailment) + Provenance (policy gate)")
    logger.info("=" * 60)
    logger.info("")

    test_groundedness_catches_hallucination()
    logger.info("")
    test_groundedness_passes_valid_claim()
    logger.info("")
    test_groundedness_noop_without_rag()
    logger.info("")
    test_provenance_blocks_rag_approval_action()
    logger.info("")
    test_provenance_allows_rag_safe_action()
    logger.info("")
    test_provenance_allows_voice_command_approval()
    logger.info("")
    test_critic_stats_include_groundedness_and_policy()
    logger.info("")

    logger.info("=" * 60)
    logger.info("ALL P2+P3 CRITIC/POLICY/RAG TESTS PASSED ✓")
    logger.info("=" * 60)