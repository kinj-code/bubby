"""
Avatar animation engine for the desktop companion overlay.

Maps the 'animation' string from the LLM's JSON output to visual states
in the overlay window. Supports sprite-based animations using QMovie (GIF)
or programmatic animations using Qt property animations.

Memory: ~100MB overhead (mostly Qt rendering buffers).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QWidget, QLabel, QApplication
from PySide6.QtCore import (
    Qt, QTimer, QSize, QPoint, QPropertyAnimation, QEasingCurve, Signal
)
from PySide6.QtGui import QMovie, QPixmap, QPainter, QColor, QFont

logger = logging.getLogger(__name__)


# Animation state definitions — comprehensive Genshin-quality emotion set
ANIMATION_STATES = {
    "idle": {
        "label": "Idle",
        "description": "Default resting state — companion is waiting, gentle breathing",
        "tint": None,
        "duration_ms": 0,
        "loop": True,
        "emote": "😊",
    },
    "nod": {
        "label": "Nodding",
        "description": "Silent acknowledgment (e.g., user did something good)",
        "tint": QColor(100, 255, 100, 30),
        "duration_ms": 2000,
        "loop": False,
        "emote": "👍",
    },
    "wave": {
        "label": "Waving",
        "description": "Greeting or goodbye gesture",
        "tint": QColor(255, 200, 150, 30),
        "duration_ms": 3000,
        "loop": True,
        "emote": "👋",
    },
    "think": {
        "label": "Thinking",
        "description": "Processing or analyzing screen content",
        "tint": QColor(200, 200, 255, 30),
        "duration_ms": 2500,
        "loop": True,
        "emote": "🤔",
    },
    "talk": {
        "label": "Talking",
        "description": "Actively speaking — only used when speech is non-empty",
        "tint": QColor(255, 255, 200, 40),
        "duration_ms": 5000,
        "loop": True,
        "emote": "💬",
    },
    "observe": {
        "label": "Observing",
        "description": "Watching the screen, silent observation",
        "tint": QColor(200, 180, 255, 25),
        "duration_ms": 0,
        "loop": True,
        "emote": "👀",
    },
    "confused": {
        "label": "Confused",
        "description": "Something unexpected on screen",
        "tint": QColor(255, 200, 100, 35),
        "duration_ms": 3000,
        "loop": True,
        "emote": "😕",
    },
    "curious": {
        "label": "Curious",
        "description": "Screen changed — investigating what the user is doing",
        "tint": QColor(200, 220, 255, 30),
        "duration_ms": 0,
        "loop": True,
        "emote": "🧐",
    },
    "success": {
        "label": "Satisfied",
        "description": "Command executed successfully",
        "tint": QColor(100, 255, 130, 25),
        "duration_ms": 2000,
        "loop": False,
        "emote": "✅",
    },
    "frustrated": {
        "label": "Frustrated",
        "description": "Command failed or inference error",
        "tint": QColor(255, 130, 100, 30),
        "duration_ms": 2500,
        "loop": False,
        "emote": "😤",
    },
    # ── Extended Emotions (Genshin-quality) ──
    "sleep": {
        "label": "Sleeping",
        "description": "Resting — Zzz, closed eyes, curled up",
        "tint": QColor(100, 100, 150, 50),
        "duration_ms": 0,
        "loop": True,
        "emote": "😴",
    },
    "excited": {
        "label": "Excited",
        "description": "High energy — bouncing, stars in eyes",
        "tint": QColor(255, 200, 50, 35),
        "duration_ms": 3000,
        "loop": True,
        "emote": "🤩",
    },
    "sad": {
        "label": "Sad",
        "description": "Disappointed — droopy posture, tear",
        "tint": QColor(100, 100, 200, 30),
        "duration_ms": 2500,
        "loop": False,
        "emote": "😢",
    },
    "surprised": {
        "label": "Surprised",
        "description": "Sudden event — jump, wide eyes",
        "tint": QColor(255, 255, 180, 35),
        "duration_ms": 1500,
        "loop": False,
        "emote": "😮",
    },
    "love": {
        "label": "Loving",
        "description": "Affectionate — heart eyes, warm glow",
        "tint": QColor(255, 150, 150, 30),
        "duration_ms": 3000,
        "loop": True,
        "emote": "😍",
    },
    "dance": {
        "label": "Dancing",
        "description": "Celebratory — happy jig, music notes",
        "tint": QColor(200, 255, 200, 35),
        "duration_ms": 4000,
        "loop": True,
        "emote": "💃",
    },
    "point": {
        "label": "Pointing",
        "description": "Directing attention — pointing at screen element",
        "tint": QColor(200, 200, 255, 30),
        "duration_ms": 3000,
        "loop": False,
        "emote": "👆",
    },
    "hide": {
        "label": "Hiding",
        "description": "Peekaboo — peeking from behind window edge",
        "tint": QColor(150, 150, 200, 25),
        "duration_ms": 0,
        "loop": True,
        "emote": "👀",
    },
    "stretch": {
        "label": "Stretching",
        "description": "After idle — yawn, stretch arms",
        "tint": QColor(200, 200, 150, 25),
        "duration_ms": 2000,
        "loop": False,
        "emote": "🙆",
    },
    "blush": {
        "label": "Blushing",
        "description": "Embarrassed — shy, rosy cheeks",
        "tint": QColor(255, 150, 150, 30),
        "duration_ms": 2000,
        "loop": False,
        "emote": "😊",
    },
    "angry": {
        "label": "Angry",
        "description": "Annoyed — furrowed brow, steam",
        "tint": QColor(255, 80, 80, 35),
        "duration_ms": 2000,
        "loop": False,
        "emote": "😠",
    },
    "plead": {
        "label": "Pleading",
        "description": "Requesting — puppy eyes, clasped hands",
        "tint": QColor(200, 180, 255, 30),
        "duration_ms": 2500,
        "loop": False,
        "emote": "🥺",
    },
    "celebrate": {
        "label": "Celebrating",
        "description": "Big win — confetti, party popper",
        "tint": QColor(255, 220, 100, 35),
        "duration_ms": 4000,
        "loop": False,
        "emote": "🎉",
    },
    "sneeze": {
        "label": "Sneezing",
        "description": "Cute sneeze — achoo!",
        "tint": QColor(255, 200, 200, 25),
        "duration_ms": 1500,
        "loop": False,
        "emote": "🤧",
    },
    "yawn": {
        "label": "Yawning",
        "description": "Tired — big yawn, sleepy eyes",
        "tint": QColor(150, 150, 180, 25),
        "duration_ms": 2000,
        "loop": False,
        "emote": "🥱",
    },
    "reading": {
        "label": "Reading",
        "description": "Focused on text — glasses, book",
        "tint": QColor(180, 200, 220, 25),
        "duration_ms": 0,
        "loop": True,
        "emote": "📖",
    },
    "listening": {
        "label": "Listening",
        "description": "Attentive — ear perked, head tilted",
        "tint": QColor(180, 220, 180, 25),
        "duration_ms": 0,
        "loop": True,
        "emote": "👂",
    },
    "shrug": {
        "label": "Shrugging",
        "description": "Uncertain — shoulder shrug, palms up",
        "tint": QColor(200, 200, 200, 25),
        "duration_ms": 1500,
        "loop": False,
        "emote": "🤷",
    },
    "facepalm": {
        "label": "Facepalming",
        "description": "User did something silly — palm to face",
        "tint": QColor(200, 180, 150, 30),
        "duration_ms": 1500,
        "loop": False,
        "emote": "🤦",
    },
    "victory": {
        "label": "Victory",
        "description": "Major achievement — trophy pose",
        "tint": QColor(255, 215, 0, 35),
        "duration_ms": 3000,
        "loop": False,
        "emote": "🏆",
    },
}


@dataclass
class AvatarConfig:
    """Configuration for the avatar engine."""
    show_emotes: bool = True           # Show emoji icons
    emote_size: int = 64               # Emote font size
    emote_color: str = "white"         # Emote color

    # Animation
    default_duration_ms: int = 2000    # Default state duration
    bob_animation: bool = True         # Gentle floating motion
    bob_range_px: int = 8              # Bob vertical range

    # Sprite support
    sprite_dir: Optional[str] = None   # Directory for GIF/sprite assets
    use_sprites: bool = False          # Use sprite files (overrides emotes)

    # Text display
    show_text: bool = True             # Show speech text
    text_max_chars: int = 80           # Max characters in overlay text
    text_font_size: int = 14           # Text label font size


class AvatarWidget(QWidget):
    """
    Widget that displays the companion's animated avatar.

    Renders directly on the transparent overlay window. Supports:
    - Emoji-based avatars (zero dependencies, always works)
    - GIF/sprite sheet animations (when assets are available)
    - Gentle floating bob animation
    - State-based tint coloring
    - Speech text display

    All animations are Qt-native — no external dependencies.
    """

    # Signals
    animation_changed = Signal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        config: Optional[AvatarConfig] = None,
    ) -> None:
        super().__init__(parent)

        self._config = config or AvatarConfig()
        self._current_state = "idle"
        self._speech_text = ""
        self._state_timer: Optional[QTimer] = None
        self._bob_timer: Optional[QTimer] = None
        self._bob_offset = 0
        self._bob_direction = 1
        self._gif_movie: Optional[QMovie] = None
        self._gif_label: Optional[QLabel] = None
        self._emote_label: Optional[QLabel] = None
        self._text_label: Optional[QLabel] = None

        self._setup_ui()
        self._start_bob_animation()

        logger.info("AvatarWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the avatar UI components."""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        width = self.parent().width() if self.parent() else 400
        height = self.parent().height() if self.parent() else 400
        self.setFixedSize(width, height)

        # Emote label (emoji avatar)
        if self._config.show_emotes:
            self._emote_label = QLabel(self)
            self._emote_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._emote_label.setGeometry(0, 0, width, height - 80)
            self._emote_label.setStyleSheet(
                f"background: transparent; font-size: {self._config.emote_size}px;"
            )
            self._emote_label.setText(ANIMATION_STATES["idle"]["emote"])
            self._emote_label.show()

        # Text/speech label
        if self._config.show_text:
            self._text_label = QLabel(self)
            self._text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._text_label.setGeometry(20, height - 100, width - 40, 80)
            self._text_label.setWordWrap(True)
            self._text_label.setStyleSheet(
                f"background: rgba(0, 0, 0, 40); color: white; "
                f"font-size: {self._config.text_font_size}px; "
                "border-radius: 8px; padding: 8px;"
            )
            self._text_label.hide()

        # GIF label (for sprite-based animations)
        self._gif_label = QLabel(self)
        self._gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_label.setGeometry(50, 50, width - 100, height - 130)
        self._gif_label.hide()

    def _start_bob_animation(self) -> None:
        """Start the gentle bobbing/floating animation."""
        if not self._config.bob_animation:
            return

        self._bob_timer = QTimer(self)
        self._bob_timer.timeout.connect(self._update_bob)
        self._bob_timer.start(50)  # ~20 FPS for smooth motion

    def _update_bob(self) -> None:
        """Update the bobbing offset for gentle floating motion."""
        self._bob_offset += 0.05 * self._bob_direction
        if abs(self._bob_offset) >= self._config.bob_range_px:
            self._bob_direction *= -1

        if self._emote_label:
            self._emote_label.move(
                self._emote_label.x(),
                int(self._bob_offset),
            )

    def set_state(self, animation: str, speech_text: str = "") -> None:
        """
        Set the current animation state.

        Args:
            animation: Animation name from the comprehensive state list
            speech_text: Optional speech text to display
        """
        if animation not in ANIMATION_STATES:
            logger.warning(f"Unknown animation '{animation}', defaulting to idle")
            animation = "idle"

        state = ANIMATION_STATES[animation]
        self._current_state = animation
        self._speech_text = speech_text
        self.animation_changed.emit(animation)

        # Update emote
        if self._emote_label:
            self._emote_label.setText(state["emote"])

        # Update speech text
        if self._text_label and self._config.show_text:
            if speech_text:
                display_text = speech_text[:self._config.text_max_chars]
                self._text_label.setText(display_text)
                self._text_label.show()
            else:
                self._text_label.hide()

        # Update GIF if available
        sprite_path = self._get_sprite_path(animation)
        if sprite_path and self._config.use_sprites:
            self._play_gif(sprite_path)

        # Schedule state timeout for non-looping states
        if not state["loop"] and state["duration_ms"] > 0:
            if self._state_timer:
                self._state_timer.stop()
            self._state_timer = QTimer(self)
            self._state_timer.setSingleShot(True)
            self._state_timer.timeout.connect(lambda: self.set_state("idle"))
            self._state_timer.start(state["duration_ms"])

        # Get tint for overlay parent
        tint = state["tint"]
        if tint and self.parent() and hasattr(self.parent(), '_current_tint'):
            self.parent()._current_tint = tint
            self.parent().update()

        logger.debug(f"Avatar state: {animation} ({state['label']})")

    def show_speech(self, text: str, duration_ms: int = 5000) -> None:
        """
        Display speech text on the avatar and auto-clear.

        Args:
            text: Speech text to display
            duration_ms: How long to show the text
        """
        if not text:
            return

        self.set_state("talk", text)

        # Auto-clear after duration
        QTimer.singleShot(duration_ms, self._clear_speech)

    def _clear_speech(self) -> None:
        """Clear speech text display."""
        self._speech_text = ""
        if self._text_label:
            self._text_label.hide()

    def _get_sprite_path(self, animation: str) -> Optional[str]:
        """Get the path to a sprite/GIF file for an animation."""
        if not self._config.sprite_dir:
            return None

        sprite_dir = Path(self._config.sprite_dir)
        if not sprite_dir.exists():
            return None

        # Look for gif with animation name
        candidates = [
            sprite_dir / f"{animation}.gif",
            sprite_dir / f"bubby_{animation}.gif",
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        return None

    def _play_gif(self, path: str) -> None:
        """Play a GIF animation."""
        if self._gif_movie:
            self._gif_movie.stop()

        self._gif_movie = QMovie(path)
        self._gif_label.setMovie(self._gif_movie)
        self._gif_label.show()
        self._gif_movie.start()

        # Hide emote while GIF plays
        if self._emote_label:
            self._emote_label.hide()

    def get_current_state(self) -> str:
        """Get the current animation state."""
        return self._current_state

    def paintEvent(self, event) -> None:
        """Paint the avatar (handles transparency)."""
        # The overlay parent handles tint painting
        # Avatar renders child widgets (emotes, text, GIFs)
        super().paintEvent(event)


# Testing helper
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    app = QApplication(sys.argv)

    # Create a simple test window
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    window.setWindowTitle("Avatar Test")
    window.setStyleSheet("background: #1a1a2e;")

    avatar = AvatarWidget(
        config=AvatarConfig(
            show_emotes=True,
            show_text=True,
            emote_size=80,
        )
    )
    window.setCentralWidget(avatar)
    window.resize(400, 400)

    # Cycle through states for testing
    states = list(ANIMATION_STATES.keys())
    current_idx = [0]

    def cycle_state():
        state = states[current_idx[0] % len(states)]
        speech = ""
        if state == "talk":
            speech = "Hello! I'm Bubby, your desktop companion."
        elif state == "wave":
            speech = "Hi there! 👋"

        avatar.set_state(state, speech)
        logger.info(f"State: {state} ({ANIMATION_STATES[state]['label']})")
        current_idx[0] += 1

    # Cycle every 2.5 seconds
    timer = QTimer()
    timer.timeout.connect(cycle_state)
    timer.start(2500)

    # Start immediately
    cycle_state()

    window.show()
    logger.info("Avatar test window shown — cycling through states")

    sys.exit(app.exec())