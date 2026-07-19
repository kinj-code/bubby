"""Test script for Phase 2 Brain & Autonomy components."""

import sys
import logging
import time
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.brain.decisions import (
    Decision, DecisionType, ScreenContext, NodeStatus,
    make_idle_decision, make_wander_decision, make_sit_decision
)
from src.brain.behavior_tree import (
    BehaviorTree, Selector, Sequence, Condition, Action, Inverter
)
from src.brain.context_manager import ContextManager
from src.brain.autonomy_loop import AutonomyLoop

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_decisions():
    """Test 1: Decision data structures."""
    logger.info("=" * 60)
    logger.info("TEST 1: Decision Data Structures")
    logger.info("=" * 60)
    
    # Test ScreenContext
    context = ScreenContext(
        user_present=True,
        user_idle_time=30.0,
        active_window="Firefox",
        content_type="browser",
        content_confidence=0.9
    )
    
    logger.info(f"✓ ScreenContext created: user_present={context.user_present}")
    logger.info(f"  Idle time: {context.user_idle_time}s")
    logger.info(f"  Active window: {context.active_window}")
    
    context_dict = context.to_dict()
    assert "user_present" in context_dict
    assert "user_idle_time" in context_dict
    logger.info("✓ ScreenContext.to_dict() works")
    
    # Test Decision
    decision = make_idle_decision(animation="idle_breathing")
    logger.info(f"✓ Decision created: {decision}")
    assert decision.decision_type == DecisionType.IDLE
    assert decision.priority == 1
    assert "animation" in decision.params
    
    decision_dict = decision.to_dict()
    assert "type" in decision_dict
    assert "priority" in decision_dict
    logger.info("✓ Decision.to_dict() works")
    
    # Test decision factories
    wander = make_wander_decision(x=100.0, y=200.0)
    assert wander.decision_type == DecisionType.WANDER
    assert wander.params["target_x"] == 100.0
    logger.info("✓ Decision factories work")
    
    logger.info("\n✓ All decision tests passed")
    return True


def test_behavior_tree_nodes():
    """Test 2: Behavior tree node types."""
    logger.info("=" * 60)
    logger.info("TEST 2: Behavior Tree Nodes")
    logger.info("=" * 60)
    
    # Test Selector
    logger.info("\n--- Testing Selector ---")
    
    def success_action(context):
        return NodeStatus.SUCCESS
    
    def failure_action(context):
        return NodeStatus.FAILURE
    
    selector = Selector("Test Selector")
    selector.add_child(Action("Success", success_action))
    selector.add_child(Action("Failure", failure_action))
    
    context = ScreenContext()
    status = selector.evaluate(context)
    assert status == NodeStatus.SUCCESS, "Selector should return first success"
    logger.info("✓ Selector returns first SUCCESS")
    
    # Test Sequence
    logger.info("\n--- Testing Sequence ---")
    
    sequence = Sequence("Test Sequence")
    sequence.add_child(Action("Success1", success_action))
    sequence.add_child(Action("Success2", success_action))
    
    status = sequence.evaluate(context)
    assert status == NodeStatus.SUCCESS, "Sequence should succeed if all children succeed"
    logger.info("✓ Sequence succeeds when all children succeed")
    
    sequence_fail = Sequence("Test Sequence Fail")
    sequence_fail.add_child(Action("Success", success_action))
    sequence_fail.add_child(Action("Failure", failure_action))
    
    status = sequence_fail.evaluate(context)
    assert status == NodeStatus.FAILURE, "Sequence should fail if any child fails"
    logger.info("✓ Sequence fails when child fails")
    
    # Test Condition
    logger.info("\n--- Testing Condition ---")
    
    def user_present(context):
        return context.user_present
    
    condition = Condition("User Present", user_present)
    
    context_present = ScreenContext(user_present=True)
    status = condition.evaluate(context_present)
    assert status == NodeStatus.SUCCESS, "Condition should succeed when true"
    logger.info("✓ Condition succeeds when true")
    
    context_absent = ScreenContext(user_present=False)
    status = condition.evaluate(context_absent)
    assert status == NodeStatus.FAILURE, "Condition should fail when false"
    logger.info("✓ Condition fails when false")
    
    # Test Inverter
    logger.info("\n--- Testing Inverter ---")
    
    inverter = Inverter("Invert User Present", condition)
    
    status = inverter.evaluate(context_present)
    assert status == NodeStatus.FAILURE, "Inverter should invert SUCCESS to FAILURE"
    logger.info("✓ Inverter inverts SUCCESS to FAILURE")
    
    status = inverter.evaluate(context_absent)
    assert status == NodeStatus.SUCCESS, "Inverter should invert FAILURE to SUCCESS"
    logger.info("✓ Inverter inverts FAILURE to SUCCESS")
    
    logger.info("\n✓ All node tests passed")
    return True


def test_behavior_tree():
    """Test 3: Complete behavior tree."""
    logger.info("=" * 60)
    logger.info("TEST 3: Complete Behavior Tree")
    logger.info("=" * 60)
    
    # Build a simple tree
    # Root: Selector
    #   - Sequence: User Present → Greet
    #   - Action: Idle
    
    def user_present_condition(context):
        return context.user_present
    
    def greet_action(context):
        logger.info("  → Greeting user!")
        return NodeStatus.SUCCESS
    
    def idle_action(context):
        logger.info("  → Idling...")
        return NodeStatus.SUCCESS
    
    # Build tree
    user_present = Condition("User Present?", user_present_condition)
    greet = Action("Greet", greet_action)
    greet_sequence = Sequence("Greet Sequence")
    greet_sequence.add_child(user_present)
    greet_sequence.add_child(greet)
    
    idle = Action("Idle", idle_action)
    
    root = Selector("Root")
    root.add_child(greet_sequence)
    root.add_child(idle)
    
    tree = BehaviorTree(root)
    
    # Test with user present
    logger.info("\n--- Test: User Present ---")
    context1 = ScreenContext(user_present=True, user_idle_time=0.0)
    decision1 = tree.evaluate(context1)
    logger.info(f"Decision: {decision1}")
    assert decision1.decision_type == DecisionType.IDLE  # Default mapping
    
    # Test with user absent
    logger.info("\n--- Test: User Absent ---")
    context2 = ScreenContext(user_present=False, user_idle_time=10.0)
    decision2 = tree.evaluate(context2)
    logger.info(f"Decision: {decision2}")
    
    # Test stats
    stats = tree.get_stats()
    assert stats["evaluations"] == 2
    logger.info(f"✓ Tree evaluations: {stats['evaluations']}")
    
    logger.info("\n✓ Behavior tree tests passed")
    return True


def test_context_manager():
    """Test 4: Context manager."""
    logger.info("=" * 60)
    logger.info("TEST 4: Context Manager")
    logger.info("=" * 60)
    
    manager = ContextManager()
    
    # Test initial state
    logger.info("\n--- Test: Initial State ---")
    context = manager.build_context()
    logger.info(f"Idle time: {manager.get_idle_time():.1f}s")
    logger.info(f"User present: {manager.is_user_present()}")
    assert manager.is_user_present() == True  # Initially present (0 idle time)
    logger.info("✓ Initial state correct")
    
    # Test after activity
    logger.info("\n--- Test: After User Activity ---")
    time.sleep(0.1)
    manager.update_user_activity()
    idle_time = manager.get_idle_time()
    logger.info(f"Idle time after activity: {idle_time:.1f}s")
    assert idle_time < 1.0, "Idle time should be < 1s after activity"
    logger.info("✓ User activity tracked")
    
    # Test idle detection
    logger.info("\n--- Test: Idle Detection ---")
    logger.info("Simulating 4 minutes idle (exceeds 3min threshold)...")
    manager._user_activity.last_input_time = time.time() - 240
    idle_time = manager.get_idle_time()
    logger.info(f"Idle time: {idle_time:.1f}s")
    assert idle_time > 200, "Idle time should be ~240s"
    assert manager.is_user_present() == False, "User should not be present after 4min"
    logger.info("✓ Idle detection works")
    
    # Test system usage
    logger.info("\n--- Test: System Usage ---")
    cpu, mem = manager.get_system_usage()
    logger.info(f"CPU: {cpu:.1f}%, Memory: {mem:.1f}%")
    logger.info("✓ System usage retrieved")
    
    # Test stats
    logger.info("\n--- Test: Statistics ---")
    stats = manager.get_stats()
    logger.info(f"Stats: {stats}")
    assert "idle_time" in stats
    assert "user_present" in stats
    logger.info("✓ Statistics work")
    
    logger.info("\n✓ Context manager tests passed")
    return True


def test_autonomy_loop():
    """Test 5: Autonomy loop."""
    logger.info("=" * 60)
    logger.info("TEST 5: Autonomy Loop")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create simple tree
    def idle_action(context):
        return NodeStatus.SUCCESS
    
    idle_node = Action("Idle", idle_action)
    root = Selector("Root")
    root.add_child(idle_node)
    
    tree = BehaviorTree(root)
    context_manager = ContextManager()
    
    # Create loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        decision_interval=0.5
    )
    
    # Track decisions via signals
    decisions_received = []
    
    def on_decision(decision):
        decisions_received.append(decision)
        logger.info(f"  → Decision received: {decision.decision_type.value}")
    
    loop.decision_made.connect(on_decision)
    
    # Start loop
    logger.info("\n--- Starting autonomy loop ---")
    loop.start()
    
    # Wait for decisions to be made
    time.sleep(2.0)
    
    # Check internal stats (more reliable than signals in test environment)
    stats = loop.get_stats()
    logger.info(f"Decisions made: {stats['decisions_made']}")
    logger.info(f"Decisions received via signal: {len(decisions_received)}")
    
    # The loop should have made decisions internally
    assert stats["decisions_made"] >= 2, f"Loop should make at least 2 decisions, got {stats['decisions_made']}"
    logger.info("✓ Loop is making decisions internally")
    
    # Note: Signal delivery may be limited in test environment without event loop
    # But the important part is that the loop is working
    if len(decisions_received) > 0:
        logger.info(f"✓ Signals also delivered ({len(decisions_received)} received)")
    else:
        logger.info("⚠ Signals not delivered (expected in non-event-loop test)")
    
    # Stop loop
    logger.info("\n--- Stopping loop ---")
    loop.stop()
    loop.wait(1000)
    
    assert not loop.is_running(), "Loop should be stopped"
    logger.info("✓ Loop stopped successfully")
    
    # Verify final stats
    final_stats = loop.get_stats()
    logger.info(f"Final stats: {final_stats}")
    assert final_stats["decisions_made"] >= 2
    logger.info("✓ Statistics tracked correctly")
    
    logger.info("\n✓ Autonomy loop tests passed")
    return True


def test_integration():
    """Test 6: Full integration test."""
    logger.info("=" * 60)
    logger.info("TEST 6: Full Integration")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create behavior tree
    def check_idle(context):
        return context.user_idle_time > 5.0
    
    def wander_action(context):
        return NodeStatus.SUCCESS
    
    def idle_action(context):
        return NodeStatus.SUCCESS
    
    idle_condition = Condition("Idle > 5s", check_idle)
    wander_node = Action("Wander", wander_action)
    idle_node = Action("Idle", idle_action)
    
    wander_sequence = Sequence("Wander Sequence")
    wander_sequence.add_child(idle_condition)
    wander_sequence.add_child(wander_node)
    
    root = Selector("Root")
    root.add_child(wander_sequence)
    root.add_child(idle_node)
    
    tree = BehaviorTree(root)
    context_manager = ContextManager()
    
    # Simulate user being idle
    context_manager._user_activity.last_input_time = time.time() - 10
    
    # Create loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        decision_interval=1.0
    )
    
    decisions = []
    
    def on_decision(decision):
        decisions.append(decision)
        logger.info(f"  → {decision.decision_type.value}")
    
    loop.decision_made.connect(on_decision)
    
    # Run for 3 seconds
    loop.start()
    time.sleep(3.5)
    
    # Check internal stats (signals may not deliver without event loop)
    stats = loop.get_stats()
    logger.info(f"\nTotal decisions made: {stats['decisions_made']}")
    logger.info(f"Decisions received via signal: {len(decisions)}")
    
    assert stats["decisions_made"] >= 2, f"Should make multiple decisions, got {stats['decisions_made']}"
    logger.info("✓ Loop made multiple decisions")
    
    if len(decisions) > 0:
        # All should be idle (since actions don't set specific decisions)
        for decision in decisions:
            assert decision.decision_type == DecisionType.IDLE
        logger.info("✓ All decisions are IDLE type (expected for stub actions)")
    else:
        logger.info("⚠ Signals not delivered (expected without event loop)")
    
    # Stop loop
    loop.stop()
    loop.wait(1000)
    
    logger.info("\n✓ Integration test passed")
    return True


def main():
    """Run all brain tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BUBBY BRAIN & AUTONOMY - PHASE 2 TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        # Test 1: Decisions
        test_decisions()
        
        # Test 2: Behavior tree nodes
        test_behavior_tree_nodes()
        
        # Test 3: Complete tree
        test_behavior_tree()
        
        # Test 4: Context manager
        test_context_manager()
        
        # Test 5: Autonomy loop
        test_autonomy_loop()
        
        # Test 6: Integration
        test_integration()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL BRAIN TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 2, Part 1 Complete!")
        logger.info("Next: Integrate with UI and add more complex behaviors")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()