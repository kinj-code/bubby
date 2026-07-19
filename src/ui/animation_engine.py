"""Animation engine for Lottie and skeletal animations."""

import logging
import json
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, QPoint, QSize, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor

logger = logging.getLogger(__name__)


class AnimationState(Enum):
    """Animation playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    LOOPING = "looping"


@dataclass
class AnimationFrame:
    """Single frame of animation."""
    image: QImage
    duration_ms: int
    x: int = 0
    y: int = 0
    scale: float = 1.0
    rotation: int = 0


@dataclass
class Animation:
    """Complete animation sequence."""
    name: str
    frames: List[AnimationFrame]
    loop: bool
    fps: int
    frame_duration_ms: int


class AnimationEngine(QWidget):
    """
    Lightweight animation engine for Lottie and skeletal animations.
    
    Features:
    - Lottie JSON loader (structure ready, full parser in future)
    - Frame-based animation playback
    - State management (idle, walk, sit, interact)
    - Paper doll layer system
    - Smooth transitions between animations
    """
    
    # Animation state signals
    animation_started = Signal(str)  # animation name
    animation_finished = Signal(str)
    frame_changed = Signal(int)  # frame number
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Animation state
        self._animations: Dict[str, Animation] = {}
        self._current_animation: Optional[Animation] = None
        self._current_frame_index: int = 0
        self._state: AnimationState = AnimationState.STOPPED
        
        # Playback control
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_frame)
        self._loop_enabled: bool = False
        
        # Paper doll layers
        self._layers: Dict[str, QWidget] = {}
        self._active_layers: List[str] = []
        
        # Performance
        self._last_frame_time: float = 0
        self._frame_count: int = 0
        
        # Setup widget
        self.setMouseTracking(True)
        
        logger.info("AnimationEngine initialized")
    
    def load_animation(self, animation: Animation) -> None:
        """
        Load an animation into the engine.
        
        Args:
            animation: Animation object with frames and metadata
        """
        self._animations[animation.name] = animation
        logger.info(f"Loaded animation: {animation.name} ({len(animation.frames)} frames)")
    
    def play(self, animation_name: str, loop: bool = False) -> bool:
        """
        Play an animation.
        
        Args:
            animation_name: Name of animation to play
            loop: Whether to loop the animation
            
        Returns:
            True if animation started, False if not found
        """
        if animation_name not in self._animations:
            logger.warning(f"Animation not found: {animation_name}")
            return False
        
        # Stop current animation
        if self._state != AnimationState.STOPPED:
            self.stop()
        
        # Set new animation
        self._current_animation = self._animations[animation_name]
        self._current_frame_index = 0
        self._loop_enabled = loop
        self._state = AnimationState.PLAYING
        
        # Start playback timer
        frame_interval = 1000 // self._current_animation.fps
        self._playback_timer.start(frame_interval)
        
        # Render first frame
        self._render_frame()
        
        self.animation_started.emit(animation_name)
        logger.info(f"Playing animation: {animation_name} (loop={loop})")
        
        return True
    
    def stop(self) -> None:
        """Stop current animation."""
        if self._state == AnimationState.STOPPED:
            return
        
        self._playback_timer.stop()
        
        if self._current_animation:
            anim_name = self._current_animation.name
            self.animation_finished.emit(anim_name)
        
        self._state = AnimationState.STOPPED
        self._current_animation = None
        self._current_frame_index = 0
        
        logger.debug("Animation stopped")
    
    def pause(self) -> None:
        """Pause current animation."""
        if self._state != AnimationState.PLAYING:
            return
        
        self._playback_timer.stop()
        self._state = AnimationState.PAUSED
        logger.debug("Animation paused")
    
    def resume(self) -> None:
        """Resume paused animation."""
        if self._state != AnimationState.PAUSED:
            return
        
        if self._current_animation:
            frame_interval = 1000 // self._current_animation.fps
            self._playback_timer.start(frame_interval)
            self._state = AnimationState.PLAYING
            logger.debug("Animation resumed")
    
    def set_state(self, state: str) -> bool:
        """
        Set character state (triggers corresponding animation).
        
        Args:
            state: State name (idle, walk, sit, interact, etc.)
            
        Returns:
            True if state animation found and started
        """
        # Map states to animation names
        state_animation_map = {
            "idle": "idle",
            "walk": "walk",
            "sit": "sit",
            "interact": "interact",
            "sleep": "sleep"
        }
        
        animation_name = state_animation_map.get(state, state)
        
        # Determine if this state should loop
        loop = state in ["idle", "walk", "sit", "sleep"]
        
        return self.play(animation_name, loop=loop)
    
    def _advance_frame(self) -> None:
        """Advance to next frame in animation."""
        if not self._current_animation:
            return
        
        self._current_frame_index += 1
        
        # Check if animation ended
        if self._current_frame_index >= len(self._current_animation.frames):
            if self._loop_enabled:
                self._current_frame_index = 0
            else:
                self.stop()
                return
        
        # Render frame
        self._render_frame()
        
        # Emit signal
        self.frame_changed.emit(self._current_frame_index)
    
    def _render_frame(self) -> None:
        """Render current frame."""
        if not self._current_animation:
            return
        
        frame = self._current_animation.frames[self._current_frame_index]
        
        # Trigger repaint
        self.update()
        
        self._frame_count += 1
        self._last_frame_time = time.time()
    
    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Paint current animation frame."""
        if not self._current_animation or self._state == AnimationState.STOPPED:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        frame = self._current_animation.frames[self._current_frame_index]
        
        # Draw frame
        if not frame.image.isNull():
            painter.drawImage(frame.x, frame.y, frame.image)
    
    def get_current_animation(self) -> Optional[str]:
        """Get name of currently playing animation."""
        return self._current_animation.name if self._current_animation else None
    
    def get_state(self) -> AnimationState:
        """Get current animation state."""
        return self._state
    
    def get_stats(self) -> dict:
        """Get animation statistics."""
        return {
            "state": self._state.value,
            "current_animation": self.get_current_animation(),
            "frame_index": self._current_frame_index,
            "total_frames": len(self._current_animation.frames) if self._current_animation else 0,
            "frames_rendered": self._frame_count,
            "loaded_animations": list(self._animations.keys())
        }


# Lottie loader stub (full implementation in future phase)
class LottieLoader:
    """Stub for Lottie JSON animation loader."""
    
    @staticmethod
    def load(filepath: str) -> Optional[Animation]:
        """
        Load Lottie animation from JSON file.
        
        Args:
            filepath: Path to Lottie JSON file
            
        Returns:
            Animation object or None if loading fails
        """
        logger.info(f"Lottie loader stub: {filepath}")
        logger.warning("Full Lottie parser not yet implemented")
        
        # TODO: Implement Lottie JSON parsing
        # 1. Parse JSON structure
        # 2. Extract layers, shapes, transforms
        # 3. Rasterize frames using Qt or custom renderer
        # 4. Return Animation object
        
        return None


# Paper doll asset manager stub
class PaperDollManager:
    """Stub for paper doll asset management."""
    
    def __init__(self, assets_path: str = "assets/sprites") -> None:
        self.assets_path = assets_path
        self._loaded_assets: Dict[str, QImage] = {}
        
        logger.info(f"PaperDollManager initialized: {assets_path}")
        logger.warning("Full asset loading not yet implemented")
    
    def load_component(self, component_type: str, variant: str) -> Optional[QImage]:
        """
        Load a paper doll component.
        
        Args:
            component_type: Type of component (head, body, accessory)
            variant: Variant name
            
        Returns:
            QImage of component or None
        """
        # TODO: Implement sprite loading
        # 1. Load PNG from assets/sprites/{component_type}/{variant}.png
        # 2. Apply transparency mask
        # 3. Cache in _loaded_assets
        # 4. Return QImage
        
        logger.debug(f"Paper doll component requested: {component_type}/{variant}")
        return None
    
    def compose_character(self, components: Dict[str, str]) -> Optional[QImage]:
        """
        Compose character from paper doll components.
        
        Args:
            components: Dict of component_type -> variant
            
        Returns:
            Composed character image or None
        """
        # TODO: Implement layer composition
        # 1. Load each component
        # 2. Layer in correct order (base, body, head, accessories)
        # 3. Apply transforms
        # 4. Return composed image
        
        logger.debug(f"Character composition requested: {components}")
        return None


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create animation engine
    engine = AnimationEngine()
    engine.setFixedSize(400, 400)
    
    # Create a simple test animation
    test_frames = []
    for i in range(10):
        # Create simple colored frames
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
            x=100,
            y=100
        )
        test_frames.append(frame)
    
    # Load animation
    test_anim = Animation(
        name="test",
        frames=test_frames,
        loop=True,
        fps=10,
        frame_duration_ms=100
    )
    engine.load_animation(test_anim)
    
    # Play animation
    engine.play("test", loop=True)
    engine.show()
    
    logger.info("AnimationEngine test - watch for colored frames")
    logger.info("Close window to exit")
    
    sys.exit(app.exec())