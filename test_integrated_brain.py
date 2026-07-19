"""Integrated test script for Brain + UI system."""

import sys
import logging
import time
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.app import main

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_integrated_system():
    """Run the integrated brain + UI system for 20 seconds."""
    logger.info("=" * 60)
    logger.info("INTEGRATED BRAIN + UI TEST")
    logger.info("=" * 60)
    logger.info("This will launch the transparent window with autonomous behavior")
    logger.info("Watch for:")
    logger.info("  - State text changes (IDLE, WANDER, SIT)")
    logger.info("  - Window movement when WANDERing")
    logger.info("  - Color tint changes")
    logger.info("  - Close zone in bottom-right corner")
    logger.info("=" * 60)
    
    # Create Qt application
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Import after QApplication created
    from src.app import create_behavior_tree, ContextManager, AutonomyLoop
    from src.ui.overlay import OverlayWindow
    from src.brain.decisions import DecisionType
    
    # Create components
    logger.info("\nInitializing components...")
    overlay = OverlayWindow(size=(400, 400), click_through=False)
    behavior_tree = create_behavior_tree()
    context_manager = ContextManager()
    
    # Create autonomy loop
    autonomy_loop = AutonomyLoop(
        behavior_tree=behavior_tree,
        context_manager=context_manager,
        decision_interval=2.0
    )
    
    # Track decisions
    decisions = []
    
    def on_decision(decision):
        """Handle decision from brain."""
        decisions.append(decision)
        logger.info(f"  → Decision #{len(decisions)}: {decision.decision_type.value}")
        
        # Update overlay
        overlay.update_behavior_state(decision)
        
        # Handle wandering with random target
        if decision.decision_type == DecisionType.WANDER:
            import random
            screen = app.primaryScreen()
            if screen:
                screen_rect = screen.availableGeometry()
                margin = 100
                target_x = random.randint(
                    screen_rect.left() + margin,
                    screen_rect.right() - margin - 400
                )
                target_y = random.randint(
                    screen_rect.top() + margin,
                    screen_rect.bottom() - margin - 400
                )
                
                logger.info(f"  → Moving to: ({target_x}, {target_y})")
                overlay.wander_to(target_x, target_y)
    
    # Connect signals
    autonomy_loop.decision_made.connect(on_decision)
    
    # Show overlay
    overlay.show()
    logger.info("✓ Overlay window shown")
    
    # Start autonomy loop
    autonomy_loop.start()
    logger.info("✓ Autonomy loop started")
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING TEST - Watch the window for 20 seconds")
    logger.info("=" * 60)
    
    # Setup auto-stop after 20 seconds
    def stop_test():
        """Stop the test after duration."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)
        
        # Show summary
        logger.info(f"Total decisions: {len(decisions)}")
        
        # Count decision types
        decision_counts = {}
        for d in decisions:
            dtype = d.decision_type.value
            decision_counts[dtype] = decision_counts.get(dtype, 0) + 1
        
        logger.info("Decision breakdown:")
        for dtype, count in sorted(decision_counts.items()):
            logger.info(f"  {dtype}: {count}")
        
        # Stop loop
        autonomy_loop.stop()
        
        # Quit app
        app.quit()
    
    # Stop after 20 seconds
    QTimer.singleShot(20000, stop_test)
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    test_integrated_system()