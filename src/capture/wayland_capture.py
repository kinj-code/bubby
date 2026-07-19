"""Wayland-compatible screen capture using xdg-desktop-portal and PipeWire."""

import logging
import time
import struct
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event
import numpy as np

try:
    from dbus_next import DBusError, Message, MessageType
    from dbus_next.aio import MessageBus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    logging.warning("dbus-next not available, capture will use stub mode")

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Represents a captured screen frame."""
    data: np.ndarray  # RGB array
    width: int
    height: int
    timestamp: float
    frame_number: int
    
    @property
    def size_mb(self) -> float:
        """Return frame size in megabytes."""
        return self.data.nbytes / (1024 * 1024)


class WaylandCapture:
    """
    Wayland screen capture using xdg-desktop-portal and PipeWire.
    
    Architecture:
    1. Connects to xdg-desktop-portal via DBus
    2. Requests screen/window share session
    3. Negotiates PipeWire stream
    4. Reads frames from PipeWire node
    5. Provides frames via callback or queue
    
    Falls back to stub mode if Wayland/portal not available.
    """
    
    # xdg-desktop-portal interface names
    PORTAL_BUS_NAME = "org.freedesktop.portal.Desktop"
    PORTAL_OBJECT_PATH = "/org/freedesktop/portal/desktop"
    PORTAL_INTERFACE = "org.freedesktop.portal.ScreenCast"
    
    # Capture settings
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    DEFAULT_FPS = 1  # Low FPS for lightweight operation
    FRAME_QUEUE_SIZE = 5  # Small buffer to limit memory usage
    
    def __init__(
        self,
        source: str = "desktop",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS
    ) -> None:
        """
        Initialize Wayland capture.
        
        Args:
            source: "desktop" for full screen, "window" for window picker
            width: Capture width (will be adjusted to match screen)
            height: Capture height (will be adjusted to match screen)
            fps: Frames per second (keep low for performance)
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # Capture state
        self._is_capturing = False
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._frame_queue: Queue = Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._frame_counter = 0
        
        # DBus/PipeWire state
        self._bus: Optional[Any] = None
        self._portal_proxy: Optional[Any] = None
        self._session_path: Optional[str] = None
        self._pipewire_node: Optional[int] = None
        
        # Performance tracking
        self._frames_captured = 0
        self._capture_start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        
        logger.info(f"WaylandCapture initialized: {source} {width}x{height} @ {fps}fps")
    
    def start(self) -> bool:
        """
        Start screen capture session.
        
        Returns:
            True if capture started successfully, False otherwise
        """
        if self._is_capturing:
            logger.warning("Capture already running")
            return False
        
        logger.info("Starting Wayland capture...")
        
        # Try to initialize DBus/portal capture
        if DBUS_AVAILABLE:
            success = self._start_portal_capture()
            if success:
                self._is_capturing = True
                self._capture_start_time = time.time()
                self._stop_event.clear()
                
                # Start capture thread
                self._capture_thread = Thread(
                    target=self._capture_loop,
                    daemon=True,
                    name="WaylandCaptureThread"
                )
                self._capture_thread.start()
                
                logger.info("Portal capture started successfully")
                return True
        
        # Fallback to stub mode
        logger.warning("Using stub capture mode (no actual frames)")
        self._is_capturing = True
        self._capture_start_time = time.time()
        self._stop_event.clear()
        
        self._capture_thread = Thread(
            target=self._stub_capture_loop,
            daemon=True,
            name="WaylandCaptureStubThread"
        )
        self._capture_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop screen capture and cleanup resources."""
        if not self._is_capturing:
            return
        
        logger.info("Stopping capture...")
        self._stop_event.set()
        self._is_capturing = False
        
        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Cleanup portal session
        self._cleanup_portal()
        
        # Log stats
        self._log_capture_stats()
        
        logger.info("Capture stopped")
    
    def grab_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """
        Get the next frame from the capture stream.
        
        Args:
            timeout: Maximum time to wait for frame (seconds)
            
        Returns:
            Frame object or None if timeout/error
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
            self._frames_captured += 1
            self._last_frame_time = time.time()
            return frame
        except Empty:
            return None
    
    def get_stats(self) -> dict:
        """
        Get capture statistics.
        
        Returns:
            Dictionary with capture stats
        """
        elapsed = time.time() - self._capture_start_time if self._capture_start_time else 0
        actual_fps = self._frames_captured / elapsed if elapsed > 0 else 0
        
        return {
            "is_capturing": self._is_capturing,
            "frames_captured": self._frames_captured,
            "elapsed_time": elapsed,
            "actual_fps": actual_fps,
            "target_fps": self.fps,
            "queue_size": self._frame_queue.qsize(),
            "source": self.source,
            "resolution": f"{self.width}x{self.height}"
        }
    
    def _start_portal_capture(self) -> bool:
        """
        Initialize xdg-desktop-portal screen capture.
        
        Returns:
            True if portal capture initialized, False otherwise
        """
        try:
            # Note: Full async DBus implementation would go here
            # For Phase 1, we're creating the structure with stub fallback
            
            logger.info("Portal capture requested (stub implementation)")
            logger.info("Full PipeWire negotiation will be implemented in next iteration")
            
            # TODO: Implement actual DBus calls:
            # 1. Connect to session bus
            # 2. Create screen cast session
            # 3. Request sources (monitor/window)
            # 4. Start session
            # 5. Get PipeWire node ID
            # 6. Connect to PipeWire stream
            
            return False  # Use stub mode for now
            
        except Exception as e:
            logger.error(f"Portal capture failed: {e}")
            return False
    
    def _cleanup_portal(self) -> None:
        """Cleanup portal session and PipeWire resources."""
        # TODO: Close portal session
        # TODO: Disconnect PipeWire stream
        logger.debug("Portal cleanup (stub)")
    
    def _capture_loop(self) -> None:
        """Main capture loop (portal mode)."""
        logger.info("Capture loop started (portal mode)")
        
        frame_interval = 1.0 / self.fps
        
        while not self._stop_event.is_set():
            try:
                # TODO: Read frame from PipeWire node
                # frame_data = self._read_pipewire_frame()
                
                # Stub: Create dummy frame
                frame = self._create_stub_frame()
                
                # Add to queue (non-blocking)
                if not self._frame_queue.full():
                    self._frame_queue.put(frame)
                else:
                    logger.warning("Frame queue full, dropping frame")
                
                # Sleep to maintain FPS
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)
    
    def _stub_capture_loop(self) -> None:
        """Stub capture loop for testing without portal."""
        logger.info("Stub capture loop started")
        
        frame_interval = 1.0 / self.fps
        
        while not self._stop_event.is_set():
            try:
                # Create dummy frame
                frame = self._create_stub_frame()
                
                # Add to queue
                if not self._frame_queue.full():
                    self._frame_queue.put(frame)
                
                # Sleep to maintain FPS
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Stub capture loop error: {e}")
                time.sleep(0.1)
    
    def _create_stub_frame(self) -> Frame:
        """
        Create a stub frame for testing.
        
        Returns:
            Frame with dummy data
        """
        # Create a simple gradient pattern
        data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add timestamp-based pattern to verify frames are updating
        t = time.time() % 10  # 0-10 cycle
        value = int((t / 10) * 255)
        
        # Create gradient
        for y in range(self.height):
            for x in range(self.width):
                data[y, x] = [
                    (x * 255) // self.width,  # Red gradient
                    (y * 255) // self.height,  # Green gradient
                    value  # Blue changes over time
                ]
        
        return Frame(
            data=data,
            width=self.width,
            height=self.height,
            timestamp=time.time(),
            frame_number=self._frame_counter
        )
    
    def _log_capture_stats(self) -> None:
        """Log capture statistics."""
        stats = self.get_stats()
        
        logger.info("=" * 60)
        logger.info("CAPTURE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Frames captured: {stats['frames_captured']}")
        logger.info(f"Elapsed time: {stats['elapsed_time']:.2f}s")
        logger.info(f"Actual FPS: {stats['actual_fps']:.2f}")
        logger.info(f"Target FPS: {stats['target_fps']}")
        logger.info(f"Resolution: {stats['resolution']}")
        logger.info(f"Source: {stats['source']}")
        logger.info("=" * 60)


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("WAYLAND CAPTURE TEST")
    logger.info("=" * 60)
    
    # Create capture instance
    capture = WaylandCapture(
        source="desktop",
        width=640,  # Low res for testing
        height=480,
        fps=1
    )
    
    # Start capture
    logger.info("Starting capture...")
    capture.start()
    
    # Capture frames for 10 seconds
    logger.info("Capturing frames for 10 seconds...")
    frame_count = 0
    
    try:
        for i in range(10):
            time.sleep(1)
            frame = capture.grab_frame(timeout=0.5)
            
            if frame:
                frame_count += 1
                logger.info(
                    f"Frame {frame.frame_number}: "
                    f"{frame.width}x{frame.height} "
                    f"({frame.size_mb:.2f}MB) "
                    f"@ {frame.timestamp:.2f}"
                )
            else:
                logger.warning(f"No frame received at iteration {i}")
        
        # Print stats
        stats = capture.get_stats()
        logger.info("=" * 60)
        logger.info("FINAL STATS")
        logger.info("=" * 60)
        logger.info(f"Frames received: {frame_count}")
        logger.info(f"Frames captured by engine: {stats['frames_captured']}")
        logger.info(f"Actual FPS: {stats['actual_fps']:.2f}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Stop capture
        capture.stop()
        logger.info("Test complete")