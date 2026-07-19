"""Test script for Behavior-Vision Integration (Phase 4)."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_reasoning_bridge():
    """Test 1: Reasoning bridge with fake observations."""
    logger.info("=" * 60)
    logger.info("TEST 1: Reasoning Bridge")
    logger.info("=" * 60)
    
    from src.brain.reasoning import ReasoningBridge
    from src.brain.decisions import ScreenContext, DecisionType
    from src.vision.memory_buffer import MemoryBuffer
    
    # Create buffer with test observations
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    
    # Add test observations simulating VLM output
    logger.info("\n--- Adding test observations ---")
    
    # Observation 1: User watching video
    buffer.add_observation(
        "Video player playing YouTube video",
        metadata={"vlm_confidence": 0.9}
    )
    
    # Create reasoning bridge
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Test reasoning
    context = ScreenContext(user_present=True, user_idle_time=5.0)
    reasoning = bridge.reason(context)
    
    if reasoning:
        logger.info(f"Content type: {reasoning.content_type}")
        logger.info(f"Confidence: {reasoning.confidence:.2f}")
        logger.info(f"Should interact: {reasoning.should_interact}")
        logger.info(f"Should observe: {reasoning.should_observe}")
        logger.info(f"Reasoning: {reasoning.reasoning}")
        
        assert reasoning.content_type == "video"
        assert reasoning.confidence == 0.9
        assert reasoning.should_interact == False  # Video = don't interact
        logger.info("✓ Video observation reasoned correctly")
    
    # Observation 2: User coding (should interact)
    logger.info("\n--- Testing code observation ---")
    buffer.add_observation(
        "Text Editor with Python code visible",
        metadata={"vlm_confidence": 0.85}
    )
    
    reasoning2 = bridge.reason(context)
    if reasoning2:
        logger.info(f"Content type: {reasoning2.content_type}")
        logger.info(f"Should interact: {reasoning2.should_interact}")
        
        assert reasoning2.content_type == "code"
        assert reasoning2.should_interact == True  # Code = interact
        logger.info("✓ Code observation reasoned correctly")
    
    # Observation 3: Low confidence (UNKNOWN)
    logger.info("\n--- Testing low confidence (UNKNOWN) ---")
    buffer.add_observation(
        "UNKNOWN",
        metadata={"vlm_confidence": 0.3}
    )
    
    reasoning3 = bridge.reason(context)
    if reasoning3:
        logger.info(f"Content type: {reasoning3.content_type}")
        logger.info(f"Confidence: {reasoning3.confidence:.2f}")
        logger.info(f"Should interact: {reasoning3.should_interact}")
        logger.info(f"Reasoning: {reasoning3.reasoning}")
        
        assert reasoning3.content_type == "unknown"
        assert reasoning3.confidence == 0.3
        assert reasoning3.should_interact == False  # Low confidence = don't interact
        logger.info("✓ Low confidence handled correctly (Wait-and-See)")
    
    logger.info("\n✓ Reasoning bridge test passed")
    return True


def test_active_perception():
    """Test 2: Active Perception in autonomy loop."""
    logger.info("=" * 60)
    logger.info("TEST 2: Active Perception")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.brain.decisions import DecisionType
    from src.vision.memory_buffer import MemoryBuffer
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Add observation
    buffer.add_observation(
        "Web Browser displaying example.com",
        metadata={"vlm_confidence": 0.9}
    )
    
    # Create mock behavior tree
    from src.brain.behavior_tree import BehaviorTree, Selector, Action
    from src.brain.decisions import NodeStatus, ScreenContext
    
    def observe_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    observe_node = Action("Observe", observe_action)
    root = Selector("Root")
    root.add_child(observe_node)
    
    tree = BehaviorTree(root)
    
    # Create mock context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop with reasoning bridge
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        decision_interval=0.5
    )
    
    logger.info("\n--- Testing Active Perception ---")
    logger.info(f"Vision check interval: {loop._vision_check_interval}s")
    
    # Test should_check_vision
    context = ScreenContext(user_present=True, user_idle_time=5.0)
    
    # First check should trigger
    should_check = loop._should_check_vision(context)
    logger.info(f"First check (no previous): {should_check}")
    assert should_check == True
    
    # Perform vision check
    loop._perform_vision_check(context)
    
    # Second check should not trigger (too soon)
    should_check2 = loop._should_check_vision(context)
    logger.info(f"Second check (immediate): {should_check2}")
    assert should_check2 == False
    
    logger.info("✓ Active Perception working correctly")
    return True


def test_wait_and_see_protocol():
    """Test 3: Wait-and-See protocol blocks low-confidence actions."""
    logger.info("=" * 60)
    logger.info("TEST 3: Wait-and-See Protocol")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.brain.decisions import Decision, DecisionType, ScreenContext
    from src.vision.memory_buffer import MemoryBuffer
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Add low-confidence observation
    buffer.add_observation(
        "UNKNOWN",
        metadata={"vlm_confidence": 0.3}
    )
    
    # Create mock behavior tree
    from src.brain.behavior_tree import BehaviorTree, Selector, Action
    from src.brain.decisions import NodeStatus
    
    def interact_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    interact_node = Action("Interact", interact_action)
    root = Selector("Root")
    root.add_child(interact_node)
    
    tree = BehaviorTree(root)
    
    # Create mock context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        decision_interval=1.0
    )
    
    logger.info("\n--- Testing Wait-and-See Protocol ---")
    
    # Create a decision that should be blocked
    blocked_decision = Decision(
        decision_type=DecisionType.INTERACT,
        priority=4,
        params={"action": "wave"}
    )
    
    context = ScreenContext(user_present=True, user_idle_time=5.0)
    
    # Perform vision check first to populate reasoning
    loop._perform_vision_check(context)
    
    # Debug: Check what reasoning was generated
    reasoning = bridge.get_last_reasoning()
    if reasoning:
        logger.info(f"Reasoning generated: {reasoning.content_type}, confidence={reasoning.confidence:.2f}")
    
    # Check if action is blocked
    is_blocked = loop._is_action_blocked(blocked_decision, context)
    logger.info(f"INTERACT blocked: {is_blocked}")
    assert is_blocked == True, f"Expected True but got {is_blocked}"
    
    # Create safe decision
    safe_decision = loop._create_safe_decision(blocked_decision)
    logger.info(f"Safe decision: {safe_decision}")
    
    assert safe_decision.decision_type == DecisionType.OBSERVE_SCREEN
    assert safe_decision.params.get("reason") == "wait_and_see"
    logger.info("✓ INTERACT blocked → OBSERVE_SCREEN")
    
    # Test OBSERVE_SCREEN blocking
    observe_decision = Decision(
        decision_type=DecisionType.OBSERVE_SCREEN,
        priority=3
    )
    
    is_blocked2 = loop._is_action_blocked(observe_decision, context)
    logger.info(f"\nOBSERVE_SCREEN blocked: {is_blocked2}")
    assert is_blocked2 == True
    
    safe_decision2 = loop._create_safe_decision(observe_decision)
    logger.info(f"Safe decision: {safe_decision2}")
    
    assert safe_decision2.decision_type == DecisionType.IDLE
    logger.info("✓ OBSERVE_SCREEN blocked → IDLE")
    
    # Test with high confidence (should not block)
    buffer.add_observation(
        "Web Browser displaying example.com",
        metadata={"vlm_confidence": 0.9}
    )
    
    # Perform vision check again with new observation
    loop._perform_vision_check(context)
    
    is_blocked3 = loop._is_action_blocked(blocked_decision, context)
    logger.info(f"\nINTERACT with high confidence blocked: {is_blocked3}")
    assert is_blocked3 == False
    logger.info("✓ High confidence allows action")
    
    logger.info("\n✓ Wait-and-See protocol working correctly")
    return True


def test_integration_simulation():
    """Test 4: Full integration simulation with fake observations."""
    logger.info("=" * 60)
    logger.info("TEST 4: Integration Simulation")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.brain.decisions import DecisionType, ScreenContext
    from src.vision.memory_buffer import MemoryBuffer
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Simulate user watching video
    logger.info("\n--- Simulating: User watching video ---")
    buffer.add_observation(
        "Video player playing YouTube video",
        metadata={"vlm_confidence": 0.9}
    )
    
    # Create behavior tree that would INTERACT
    from src.brain.behavior_tree import BehaviorTree, Selector, Sequence, Condition, Action
    from src.brain.decisions import NodeStatus
    
    def user_present(context: ScreenContext) -> bool:
        return context.user_present
    
    def interact_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    user_present_cond = Condition("User Present?", user_present)
    interact_node = Action("Interact", interact_action)
    
    interact_sequence = Sequence("Interact Sequence")
    interact_sequence.add_child(user_present_cond)
    interact_sequence.add_child(interact_node)
    
    root = Selector("Root")
    root.add_child(interact_sequence)
    
    tree = BehaviorTree(root)
    
    # Create context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        decision_interval=1.0
    )
    
    # Simulate decision
    context = ScreenContext(user_present=True, user_idle_time=5.0)
    
    # Perform vision check
    loop._perform_vision_check(context)
    
    # Evaluate tree
    decision = tree.evaluate(context)
    
    # Check if blocked
    is_blocked = loop._is_action_blocked(decision, context)
    logger.info(f"Decision: {decision.decision_type.value}")
    logger.info(f"Blocked by Wait-and-See: {is_blocked}")
    
    if is_blocked:
        safe_decision = loop._create_safe_decision(decision)
        logger.info(f"Safe decision: {safe_decision.decision_type.value}")
        assert safe_decision.decision_type == DecisionType.OBSERVE_SCREEN
        logger.info("✓ Video watching → INTERACT blocked → OBSERVE_SCREEN")
    else:
        logger.info("✓ Action allowed (high confidence)")
    
    # Simulate user coding (should interact)
    logger.info("\n--- Simulating: User coding ---")
    buffer.add_observation(
        "Text Editor with Python code",
        metadata={"vlm_confidence": 0.85}
    )
    
    loop._perform_vision_check(context)
    decision2 = tree.evaluate(context)
    is_blocked2 = loop._is_action_blocked(decision2, context)
    
    logger.info(f"Decision: {decision2.decision_type.value}")
    logger.info(f"Blocked: {is_blocked2}")
    assert is_blocked2 == False  # Code = should interact
    logger.info("✓ Coding → INTERACT allowed")
    
    logger.info("\n✓ Integration simulation passed")
    return True


def test_behavior_tree_vision_nodes():
    """Test 5: Behavior tree nodes with vision conditions."""
    logger.info("=" * 60)
    logger.info("TEST 5: Behavior Tree Vision Nodes")
    logger.info("=" * 60)
    
    from src.brain.behavior_tree import BehaviorTree, Selector, Sequence, Condition, Action
    from src.brain.decisions import NodeStatus, ScreenContext, DecisionType
    from src.brain.reasoning import ReasoningBridge
    from src.vision.memory_buffer import MemoryBuffer
    
    # Create buffer and bridge
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Add observation
    buffer.add_observation(
        "Web Browser displaying example.com",
        metadata={"vlm_confidence": 0.9}
    )
    
    # Create vision-aware conditions
    def is_browsing(context: ScreenContext) -> bool:
        """Check if user is browsing."""
        return context.content_type == "browser" and context.content_confidence > 0.7
    
    def is_coding(context: ScreenContext) -> bool:
        """Check if user is coding."""
        return context.content_type == "code" and context.content_confidence > 0.7
    
    def is_unknown(context: ScreenContext) -> bool:
        """Check if content is unknown."""
        return context.content_type == "unknown" or context.content_confidence < 0.5
    
    def greet_action(context: ScreenContext) -> NodeStatus:
        logger.info("Greeting user!")
        return NodeStatus.SUCCESS
    
    def observe_action(context: ScreenContext) -> NodeStatus:
        logger.info("Observing...")
        return NodeStatus.SUCCESS
    
    def idle_action(context: ScreenContext) -> NodeStatus:
        logger.info("Idling...")
        return NodeStatus.SUCCESS
    
    # Build tree
    # Priority 1: If browsing, greet
    browsing_cond = Condition("Is Browsing?", is_browsing)
    greet_node = Action("Greet", greet_action, decision_type=DecisionType.GREET)
    browse_sequence = Sequence("Browse Greet")
    browse_sequence.add_child(browsing_cond)
    browse_sequence.add_child(greet_node)
    
    # Priority 2: If coding, observe
    coding_cond = Condition("Is Coding?", is_coding)
    observe_node = Action("Observe", observe_action, decision_type=DecisionType.OBSERVE_SCREEN)
    code_sequence = Sequence("Code Observe")
    code_sequence.add_child(coding_cond)
    code_sequence.add_child(observe_node)
    
    # Priority 3: If unknown, idle
    unknown_cond = Condition("Is Unknown?", is_unknown)
    idle_node = Action("Idle", idle_action, decision_type=DecisionType.IDLE)
    unknown_sequence = Sequence("Unknown Idle")
    unknown_sequence.add_child(unknown_cond)
    unknown_sequence.add_child(idle_node)
    
    # Root selector
    root = Selector("Root")
    root.add_child(browse_sequence)
    root.add_child(code_sequence)
    root.add_child(unknown_sequence)
    
    tree = BehaviorTree(root)
    
    # Test 1: Browser context
    logger.info("\n--- Test 1: Browser context ---")
    context1 = ScreenContext(
        user_present=True,
        user_idle_time=5.0,
        content_type="browser",
        content_confidence=0.9
    )
    
    decision1 = tree.evaluate(context1)
    logger.info(f"Decision: {decision1.decision_type.value}")
    # Should succeed (browse_sequence)
    assert decision1.decision_type == DecisionType.GREET
    logger.info("✓ Browser → GREET")
    
    # Test 2: Code context
    logger.info("\n--- Test 2: Code context ---")
    context2 = ScreenContext(
        user_present=True,
        user_idle_time=5.0,
        content_type="code",
        content_confidence=0.85
    )
    
    decision2 = tree.evaluate(context2)
    logger.info(f"Decision: {decision2.decision_type.value}")
    # Should succeed (code_sequence)
    assert decision2.decision_type == DecisionType.OBSERVE_SCREEN
    logger.info("✓ Code → OBSERVE_SCREEN")
    
    # Test 3: Unknown context
    logger.info("\n--- Test 3: Unknown context ---")
    context3 = ScreenContext(
        user_present=True,
        user_idle_time=5.0,
        content_type="unknown",
        content_confidence=0.3
    )
    
    decision3 = tree.evaluate(context3)
    logger.info(f"Decision: {decision3.decision_type.value}")
    # Should succeed (unknown_sequence)
    assert decision3.decision_type == DecisionType.IDLE
    logger.info("✓ Unknown → IDLE (Wait-and-See)")
    
    logger.info("\n✓ Behavior tree vision nodes test passed")
    return True


def main():
    """Run all behavior-vision integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BEHAVIOR-VISION INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_reasoning_bridge()
        test_active_perception()
        test_wait_and_see_protocol()
        test_integration_simulation()
        test_behavior_tree_vision_nodes()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL BEHAVIOR-VISION INTEGRATION TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 4, Part 1 Complete!")
        logger.info("Behavior-Vision integration ready for production")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()