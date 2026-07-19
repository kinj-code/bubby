"""Test script for Smart Sampling (Phase 4, Part 2)."""

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


def test_change_detector():
    """Test 1: Screen change detector basics."""
    logger.info("=" * 60)
    logger.info("TEST 1: Change Detector Basics")
    logger.info("=" * 60)
    
    from src.vision.change_detector import ScreenChangeDetector
    
    # Create detector
    detector = ScreenChangeDetector(
        change_threshold=10.0,  # 10% change
        mse_threshold=100.0,
        min_frame_interval=0.1
    )
    
    # Test static screen
    logger.info("\n--- Static Screen Test ---")
    static_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    
    # First frame (force to bypass min_frame_interval)
    result1 = detector.detect_change(static_frame, force_check=True)
    logger.info(f"Frame 1: {result1}")
    assert result1.has_change == False
    
    # Second frame (identical) - force to bypass min_frame_interval
    result2 = detector.detect_change(static_frame, force_check=True)
    logger.info(f"Frame 2 (identical): {result2}")
    assert result2.has_change == False
    assert result2.change_percentage < 10.0
    
    # Test changing screen
    logger.info("\n--- Changing Screen Test ---")
    changing_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result3 = detector.detect_change(changing_frame, force_check=True)
    logger.info(f"Frame 3 (completely different): {result3}")
    assert result3.has_change == True
    assert result3.avg_mse > 100.0  # MSE threshold exceeded
    logger.info(f"✓ Change detected via MSE: {result3.avg_mse:.2f} (threshold: 100.0)")
    
    logger.info("\n✓ Change detector basics working")
    return True


def test_smart_sampling_static():
    """Test 2: Smart sampling with static screen (low VLM count)."""
    logger.info("=" * 60)
    logger.info("TEST 2: Smart Sampling - Static Screen")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.vision.change_detector import ScreenChangeDetector
    from src.vision.memory_buffer import MemoryBuffer
    from src.brain.decisions import ScreenContext
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    change_detector = ScreenChangeDetector(change_threshold=10.0)
    
    # Create mock behavior tree
    from src.brain.behavior_tree import BehaviorTree, Selector, Action
    from src.brain.decisions import NodeStatus
    
    def idle_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    idle_node = Action("Idle", idle_action)
    root = Selector("Root")
    root.add_child(idle_node)
    tree = BehaviorTree(root)
    
    # Create mock context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop with smart sampling
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        change_detector=change_detector,
        decision_interval=0.5  # Fast decisions for testing
    )
    
    logger.info("\n--- Simulating Static Screen (10 seconds) ---")
    
    # Static frame
    static_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    
    # Track vision checks
    initial_mode = loop.get_vision_mode()
    logger.info(f"Initial vision mode: {initial_mode}")
    
    # Simulate 10 seconds of static screen
    vision_check_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 10.0:
        # Update change detection
        change_result = loop.update_change_detection(static_frame)
        
        # Simulate decision
        context = ScreenContext(user_present=True, user_idle_time=5.0)
        
        # Check if vision would be triggered
        if loop._should_check_vision(context):
            # Simulate the actual vision check (sets _last_vision_check_time)
            loop._perform_vision_check(context)
            vision_check_count += 1
            logger.debug(f"Vision check #{vision_check_count} at {time.time() - start_time:.1f}s")
        
        # Sleep briefly
        time.sleep(0.5)
    
    final_mode = loop.get_vision_mode()
    logger.info(f"Final vision mode: {final_mode}")
    logger.info(f"Vision checks in 10s: {vision_check_count}")
    
    # With static screen, should be in idle mode
    assert final_mode == "idle", f"Expected idle mode but got {final_mode}"
    
    # In idle mode, checks should be infrequent (every 30s)
    # In 10 seconds, we should have at most 1 check
    assert vision_check_count <= 1, f"Too many vision checks for static screen: {vision_check_count}"
    
    logger.info(f"✓ Static screen: {vision_check_count} vision checks (efficient!)")
    return True


def test_smart_sampling_active():
    """Test 3: Smart sampling with active screen (high VLM count)."""
    logger.info("=" * 60)
    logger.info("TEST 3: Smart Sampling - Active Screen")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.vision.change_detector import ScreenChangeDetector
    from src.vision.memory_buffer import MemoryBuffer
    from src.brain.decisions import ScreenContext
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    change_detector = ScreenChangeDetector(change_threshold=10.0)
    
    # Create mock behavior tree
    from src.brain.behavior_tree import BehaviorTree, Selector, Action
    from src.brain.decisions import NodeStatus
    
    def idle_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    idle_node = Action("Idle", idle_action)
    root = Selector("Root")
    root.add_child(idle_node)
    tree = BehaviorTree(root)
    
    # Create mock context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        change_detector=change_detector,
        decision_interval=0.5
    )
    
    logger.info("\n--- Simulating Active Screen (10 seconds) ---")
    
    # Simulate active screen (changing every 0.5s)
    vision_check_count = 0
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 10.0:
        # Generate new frame (simulating screen changes)
        active_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_count += 1
        
        # Update change detection
        change_result = loop.update_change_detection(active_frame)
        
        # Simulate decision
        context = ScreenContext(user_present=True, user_idle_time=5.0)
        
        # Check if vision would be triggered
        if loop._should_check_vision(context):
            # Simulate the actual vision check
            loop._perform_vision_check(context)
            vision_check_count += 1
            logger.debug(f"Vision check #{vision_check_count} at {time.time() - start_time:.1f}s")
        
        # Sleep briefly
        time.sleep(0.5)
    
    final_mode = loop.get_vision_mode()
    logger.info(f"Final vision mode: {final_mode}")
    logger.info(f"Frames generated: {frame_count}")
    logger.info(f"Vision checks in 10s: {vision_check_count}")
    
    # With active screen, should be in alert or normal mode
    assert final_mode in ["alert", "normal"], f"Expected alert/normal mode but got {final_mode}"
    
    # In alert mode, checks should be frequent (every 2s)
    # In 10 seconds, we should have ~5 checks
    assert vision_check_count >= 3, f"Too few vision checks for active screen: {vision_check_count}"
    
    logger.info(f"✓ Active screen: {vision_check_count} vision checks (responsive!)")
    return True


def test_vision_mode_transitions():
    """Test 4: Vision mode transitions."""
    logger.info("=" * 60)
    logger.info("TEST 4: Vision Mode Transitions")
    logger.info("=" * 60)
    
    from src.brain.autonomy_loop import AutonomyLoop
    from src.brain.reasoning import ReasoningBridge
    from src.vision.change_detector import ScreenChangeDetector
    from src.vision.memory_buffer import MemoryBuffer
    from src.brain.decisions import ScreenContext
    
    # Create components
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Use a very small min_frame_interval to avoid caching issues in tests
    change_detector = ScreenChangeDetector(change_threshold=10.0, min_frame_interval=0.001)
    
    # Create mock behavior tree
    from src.brain.behavior_tree import BehaviorTree, Selector, Action
    from src.brain.decisions import NodeStatus
    
    def idle_action(context: ScreenContext) -> NodeStatus:
        return NodeStatus.SUCCESS
    
    idle_node = Action("Idle", idle_action)
    root = Selector("Root")
    root.add_child(idle_node)
    tree = BehaviorTree(root)
    
    # Create mock context manager
    from src.brain.context_manager import ContextManager
    context_manager = ContextManager()
    
    # Create autonomy loop
    loop = AutonomyLoop(
        behavior_tree=tree,
        context_manager=context_manager,
        reasoning_bridge=bridge,
        change_detector=change_detector,
        decision_interval=1.0
    )
    
    logger.info("\n--- Testing Mode Transitions ---")
    
    # Start in idle mode
    assert loop.get_vision_mode() == "idle"
    logger.info("✓ Initial mode: idle")
    
    # Step 1: First frame establishes baseline (no change)
    frame1 = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    loop.update_change_detection(frame1, force_check=True)
    assert loop.get_vision_mode() == "idle", f"Expected idle for baseline, got {loop.get_vision_mode()}"
    logger.info("✓ Step 1: baseline frame → idle")
    
    # Step 2: Different frame triggers change → alert
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    loop.update_change_detection(frame2, force_check=True)
    assert loop.get_vision_mode() == "alert", f"Expected alert after change, got {loop.get_vision_mode()}"
    logger.info("✓ Step 2: screen changed → alert")
    
    # Step 3: New stable frame differs from old baseline → still alert
    stable_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    loop.update_change_detection(stable_frame, force_check=True)
    assert loop.get_vision_mode() == "alert", f"Expected alert (stable vs old baseline), got {loop.get_vision_mode()}"
    logger.info(f"✓ Step 3: new stable frame → alert (differs from changing baseline)")
    
    # Step 4: Same stable frame again → normal (stable compared to itself)
    loop.update_change_detection(stable_frame, force_check=True)
    assert loop.get_vision_mode() == "normal", f"Expected normal (stable vs stable), got {loop.get_vision_mode()}"
    logger.info("✓ Step 4: same stable → normal")
    
    # Step 5: Same stable frame again → idle
    loop.update_change_detection(stable_frame, force_check=True)
    assert loop.get_vision_mode() == "idle", f"Expected idle (still stable), got {loop.get_vision_mode()}"
    logger.info("✓ Step 5: still stable → idle")
    
    logger.info("\n✓ Vision mode transitions working correctly")
    return True


def test_sampling_efficiency():
    """Test 5: Verify sampling efficiency (static vs active)."""
    logger.info("=" * 60)
    logger.info("TEST 5: Sampling Efficiency")
    logger.info("=" * 60)
    
    from src.vision.change_detector import ScreenChangeDetector
    
    # Create detector
    detector = ScreenChangeDetector(change_threshold=10.0, min_frame_interval=0.1)
    
    logger.info("\n--- Static Screen (20 frames) ---")
    static_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    
    static_checks = 0
    static_changes = 0
    
    for i in range(20):
        result = detector.detect_change(static_frame, force_check=True)
        if result.has_change:
            static_changes += 1
        static_checks += 1
    
    logger.info(f"Total checks: {static_checks}")
    logger.info(f"Changes detected: {static_changes}")
    logger.info(f"Change rate: {static_changes/static_checks*100:.1f}%")
    
    assert static_changes == 0, "Static screen should have no changes"
    
    logger.info("\n--- Active Screen (20 frames) ---")
    
    # Reset detector
    detector.reset()
    
    active_checks = 0
    active_changes = 0
    
    for i in range(20):
        # Generate different frame each time
        active_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect_change(active_frame, force_check=True)
        if result.has_change:
            active_changes += 1
        active_checks += 1
    
    logger.info(f"Total checks: {active_checks}")
    logger.info(f"Changes detected: {active_changes}")
    logger.info(f"Change rate: {active_changes/active_checks*100:.1f}%")
    
    assert active_changes > 15, f"Active screen should have many changes, got {active_changes}"
    
    # Compare efficiency
    logger.info("\n--- Efficiency Comparison ---")
    logger.info(f"Static screen: {static_changes} changes / {static_checks} checks")
    logger.info(f"Active screen: {active_changes} changes / {active_checks} checks")
    logger.info(f"Efficiency gain: {active_changes - static_changes} fewer checks for static content")
    
    logger.info("\n✓ Sampling efficiency verified")
    return True


def main():
    """Run all smart sampling tests."""
    logger.info("\n" + "=" * 60)
    logger.info("SMART SAMPLING TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        test_change_detector()
        test_smart_sampling_static()
        test_smart_sampling_active()
        test_vision_mode_transitions()
        test_sampling_efficiency()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL SMART SAMPLING TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 4, Part 2 Complete!")
        logger.info("Smart sampling ready for production")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()