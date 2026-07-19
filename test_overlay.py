"""Test script for OverlayWindow Phase 1."""

import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.ui.overlay import OverlayWindow

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_overlay_basic():
    """Test 1: Basic window creation and visibility."""
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Window Creation")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = OverlayWindow(size=(400, 400), click_through=False)
    
    logger.info(f"✓ Window created: {window.size()}")
    logger.info(f"✓ Window flags: Frameless={window.windowFlags() & Qt.WindowType.FramelessWindowHint != 0}")
    logger.info(f"✓ Click-through: {window.is_click_through()}")
    logger.info(f"✓ Close zone enabled: {window._close_zone_enabled}")
    
    return window, app


def test_overlay_click_through():
    """Test 2: Click-through toggle."""
    logger.info("=" * 60)
    logger.info("TEST 2: Click-Through Toggle")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = OverlayWindow(click_through=False)
    
    window.set_click_through(True)
    assert window.is_click_through() == True, "Click-through should be enabled"
    logger.info("✓ Click-through enabled")
    
    window.set_click_through(False)
    assert window.is_click_through() == False, "Click-through should be disabled"
    logger.info("✓ Click-through disabled")
    
    window.close()
    return None, app


def test_overlay_drag_detection():
    """Test 3: Drag and close zone detection."""
    logger.info("=" * 60)
    logger.info("TEST 3: Drag & Close Zone Detection")
    logger.info("=" * 60)
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = OverlayWindow(click_through=False)
    window.show()
    
    logger.info("✓ Window shown - try dragging it")
    logger.info("✓ Try dropping in bottom-right corner (red X zone)")
    logger.info("✓ Watch logs for drag/click events")
    
    return window, app


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BUBBY OVERLAY WINDOW - PHASE 1 TESTS")
    logger.info("=" * 60 + "\n")
    
    # Test 1: Basic creation
    window1, app1 = test_overlay_basic()
    window1.show()
    logger.info("\n→ Window 1 visible for 3 seconds...")
    QApplication.processEvents()
    import time
    time.sleep(3)
    window1.close()
    
    # Test 2: Click-through
    test_overlay_click_through()
    
    # Test 3: Interactive drag test
    window3, app3 = test_overlay_drag_detection()
    
    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE TEST - Close window with Alt+F4 or Ctrl+C")
    logger.info("=" * 60)
    
    sys.exit(app3.exec())


if __name__ == "__main__":
    main()