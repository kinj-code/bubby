"""Test script for AnimationEngine Phase 1."""

import sys
import logging
import time
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
from src.ui.animation_engine import AnimationEngine, Animation, AnimationFrame

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_animation_basic():
    """Test 1: Basic animation engine initialization."""
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Animation Engine Initialization")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    logger.info(f"✓ AnimationEngine created")
    logger.info(f"✓ Initial state: {engine.get_state()}")
    logger.info(f"✓ Current animation: {engine.get_current_animation()}")
    
    stats = engine.get_stats()
    logger.info(f"✓ Loaded animations: {stats['loaded_animations']}")
    
    return engine, app


def test_animation_loading():
    """Test 2: Loading and validating animations."""
    logger.info("=" * 60)
    logger.info("TEST 2: Animation Loading")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    # Create test animation
    test_frames = []
    for i in range(5):
        from PySide6.QtGui import QImage
        img = QImage(100, 100, QImage.Format.Format_ARGB32)
        img.fill(QColor(i * 50, i * 50, i * 50, 128))
        
        frame = AnimationFrame(
            image=img,
            duration_ms=100,
            x=0,
            y=0
        )
        test_frames.append(frame)
    
    # Load animation
    animation = Animation(
        name="test_anim",
        frames=test_frames,
        loop=True,
        fps=10,
        frame_duration_ms=100
    )
    
    engine.load_animation(animation)
    
    stats = engine.get_stats()
    logger.info(f"✓ Loaded animation: {stats['loaded_animations']}")
    assert "test_anim" in stats['loaded_animations'], "Animation should be loaded"
    
    return engine, app


def test_animation_playback():
    """Test 3: Animation playback control."""
    logger.info("=" * 60)
    logger.info("TEST 3: Animation Playback")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    # Create test animation
    test_frames = []
    for i in range(10):
        from PySide6.QtGui import QImage
        img = QImage(200, 200, QImage.Format.Format_ARGB32)
        img.fill(QColor(
            (i * 25) % 256,
            (i * 50) % 256,
            (i * 75) % 256,
            128
        ))
        
        frame = AnimationFrame(
            image=img,
            duration_ms=100,
            x=50,
            y=50
        )
        test_frames.append(frame)
    
    animation = Animation(
        name="color_cycle",
        frames=test_frames,
        loop=True,
        fps=10,
        frame_duration_ms=100
    )
    
    engine.load_animation(animation)
    
    # Test play
    logger.info("Playing animation...")
    success = engine.play("color_cycle", loop=True)
    assert success, "Play should succeed"
    logger.info("✓ Animation playing")
    
    # Let it play for 1 second
    time.sleep(1)
    
    stats = engine.get_stats()
    logger.info(f"✓ State: {stats['state']}")
    logger.info(f"✓ Current animation: {stats['current_animation']}")
    logger.info(f"✓ Frames rendered: {stats['frames_rendered']}")
    
    assert stats['state'] == 'playing', "Should be playing"
    assert stats['current_animation'] == 'color_cycle', "Should be correct animation"
    assert stats['frames_rendered'] > 0, "Should have rendered frames"
    
    # Test pause
    logger.info("\nPausing animation...")
    engine.pause()
    stats = engine.get_stats()
    logger.info(f"✓ State after pause: {stats['state']}")
    assert stats['state'] == 'paused', "Should be paused"
    
    # Test resume
    logger.info("\nResuming animation...")
    engine.resume()
    stats = engine.get_stats()
    logger.info(f"✓ State after resume: {stats['state']}")
    assert stats['state'] == 'playing', "Should be playing again"
    
    # Test stop
    logger.info("\nStopping animation...")
    engine.stop()
    stats = engine.get_stats()
    logger.info(f"✓ State after stop: {stats['state']}")
    logger.info(f"✓ Current animation: {stats['current_animation']}")
    assert stats['state'] == 'stopped', "Should be stopped"
    assert stats['current_animation'] is None, "Should have no animation"
    
    return engine, app


def test_animation_states():
    """Test 4: Character state management."""
    logger.info("=" * 60)
    logger.info("TEST 4: Character State Management")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    # Create animations for different states
    for state in ["idle", "walk", "sit", "interact"]:
        test_frames = []
        for i in range(5):
            from PySide6.QtGui import QImage
            img = QImage(100, 100, QImage.Format.Format_ARGB32)
            img.fill(QColor(100, 100, 100, 128))
            
            frame = AnimationFrame(
                image=img,
                duration_ms=100,
                x=0,
                y=0
            )
            test_frames.append(frame)
        
        animation = Animation(
            name=state,
            frames=test_frames,
            loop=True,
            fps=5,
            frame_duration_ms=200
        )
        engine.load_animation(animation)
    
    # Test state changes
    logger.info("Testing state: idle")
    success = engine.set_state("idle")
    assert success, "Idle state should work"
    logger.info("✓ Idle state set")
    
    time.sleep(0.5)
    
    logger.info("Testing state: walk")
    success = engine.set_state("walk")
    assert success, "Walk state should work"
    logger.info("✓ Walk state set")
    
    time.sleep(0.5)
    
    logger.info("Testing state: sit")
    success = engine.set_state("sit")
    assert success, "Sit state should work"
    logger.info("✓ Sit state set")
    
    time.sleep(0.5)
    
    logger.info("Testing state: interact")
    success = engine.set_state("interact")
    assert success, "Interact state should work"
    logger.info("✓ Interact state set")
    
    engine.stop()
    
    logger.info("\n✓ All state transitions successful")
    
    return engine, app


def test_animation_looping():
    """Test 5: Animation looping behavior."""
    logger.info("=" * 60)
    logger.info("TEST 5: Animation Looping")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    # Create short non-looping animation
    test_frames = []
    for i in range(3):
        from PySide6.QtGui import QImage
        img = QImage(100, 100, QImage.Format.Format_ARGB32)
        img.fill(QColor(255, 0, 0, 128))
        
        frame = AnimationFrame(
            image=img,
            duration_ms=100,
            x=0,
            y=0
        )
        test_frames.append(frame)
    
    animation = Animation(
        name="once",
        frames=test_frames,
        loop=False,
        fps=10,
        frame_duration_ms=100
    )
    
    engine.load_animation(animation)
    
    logger.info("Playing non-looping animation...")
    engine.play("once", loop=False)
    
    # Wait for animation to complete
    time.sleep(1)
    
    stats = engine.get_stats()
    logger.info(f"✓ State after completion: {stats['state']}")
    assert stats['state'] == 'stopped', "Should auto-stop after non-looping animation"
    
    # Test looping animation
    logger.info("\nPlaying looping animation...")
    engine.play("once", loop=True)
    
    time.sleep(0.5)
    
    stats = engine.get_stats()
    logger.info(f"✓ State with loop: {stats['state']}")
    assert stats['state'] == 'playing', "Should continue playing with loop"
    
    engine.stop()
    
    logger.info("\n✓ Looping behavior correct")
    
    return engine, app


def test_animation_stats():
    """Test 6: Statistics and monitoring."""
    logger.info("=" * 60)
    logger.info("TEST 6: Statistics & Monitoring")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    engine = AnimationEngine()
    
    # Load multiple animations
    for i in range(3):
        test_frames = []
        for j in range(5):
            from PySide6.QtGui import QImage
            img = QImage(50, 50, QImage.Format.Format_ARGB32)
            img.fill(QColor(i * 80, j * 50, 128))
            
            frame = AnimationFrame(
                image=img,
                duration_ms=100,
                x=0,
                y=0
            )
            test_frames.append(frame)
        
        animation = Animation(
            name=f"anim_{i}",
            frames=test_frames,
            loop=True,
            fps=5,
            frame_duration_ms=200
        )
        engine.load_animation(animation)
    
    # Play one animation
    engine.play("anim_0", loop=True)
    time.sleep(0.5)
    
    # Get stats
    stats = engine.get_stats()
    
    logger.info("Animation Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Validate stats
    assert stats['state'] == 'playing', "Should be playing"
    assert stats['current_animation'] == 'anim_0', "Should be correct animation"
    assert stats['total_frames'] == 5, "Should have 5 frames"
    assert stats['frames_rendered'] > 0, "Should have rendered frames"
    assert len(stats['loaded_animations']) == 3, "Should have 3 animations"
    
    engine.stop()
    
    logger.info("\n✓ Statistics test passed")


def main():
    """Run all animation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BUBBY ANIMATION ENGINE - PHASE 1 TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        # Test 1: Basic initialization
        test_animation_basic()
        
        # Test 2: Loading
        test_animation_loading()
        
        # Test 3: Playback
        test_animation_playback()
        
        # Test 4: States
        test_animation_states()
        
        # Test 5: Looping
        test_animation_looping()
        
        # Test 6: Stats
        test_animation_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL ANIMATION TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Run 'python3 -m src.ui.animation_engine' for visual test")
        logger.info("2. Verify colored frames appear and animate")
        logger.info("3. Check logs for any errors")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()