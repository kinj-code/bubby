"""Decision types and data structures for the behavior tree."""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Behavior tree node execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class DecisionType(Enum):
    """Types of decisions the companion can make."""
    IDLE = "idle"
    WANDER = "wander"
    PACE = "pace"
    SIT = "sit"
    OBSERVE_SCREEN = "observe_screen"
    INTERACT = "interact"
    GREET = "greet"
    SLEEP = "sleep"


@dataclass
class ScreenContext:
    """
    Current screen and user context.
    
    This is passed to the behavior tree for decision-making.
    """
    # User presence
    user_present: bool = False
    user_idle_time: float = 0.0  # Seconds since last input
    
    # Window/application info
    active_window: str = ""
    active_window_class: str = ""
    
    # Screen content (from vision pipeline in future)
    content_type: str = "unknown"  # "browser", "code", "video", "game", etc.
    content_confidence: float = 0.0
    
    # System state
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    timestamp: float = field(default_factory=datetime.now().timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "user_present": self.user_present,
            "user_idle_time": f"{self.user_idle_time:.1f}s",
            "active_window": self.active_window,
            "content_type": self.content_type,
            "cpu_usage": f"{self.cpu_usage:.1f}%",
            "memory_usage": f"{self.memory_usage:.1f}%",
            "timestamp": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        }


@dataclass
class Decision:
    """
    Represents a decision made by the behavior tree.
    
    This is emitted by the autonomy loop and consumed by the UI.
    """
    decision_type: DecisionType
    priority: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=datetime.now().timestamp)
    
    def __str__(self) -> str:
        """Human-readable decision string."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Decision({self.decision_type.value}, priority={self.priority}, params=[{params_str}])"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "type": self.decision_type.value,
            "priority": self.priority,
            "params": self.params,
            "confidence": f"{self.confidence:.2f}",
            "timestamp": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        }


# Common decision factory functions
def make_idle_decision(animation: str = "idle") -> Decision:
    """Create an idle decision."""
    return Decision(
        decision_type=DecisionType.IDLE,
        priority=1,
        params={"animation": animation}
    )


def make_wander_decision(x: float = 0.0, y: float = 0.0) -> Decision:
    """Create a wander decision."""
    return Decision(
        decision_type=DecisionType.WANDER,
        priority=2,
        params={"target_x": x, "target_y": y}
    )


def make_pace_decision(direction: str = "horizontal") -> Decision:
    """Create a pace decision."""
    return Decision(
        decision_type=DecisionType.PACE,
        priority=2,
        params={"direction": direction}
    )


def make_sit_decision(duration: float = 10.0) -> Decision:
    """Create a sit decision."""
    return Decision(
        decision_type=DecisionType.SIT,
        priority=2,
        params={"duration": duration}
    )


def make_observe_decision() -> Decision:
    """Create an observe screen decision."""
    return Decision(
        decision_type=DecisionType.OBSERVE_SCREEN,
        priority=3,
        params={}
    )


def make_interact_decision(action: str = "wave") -> Decision:
    """Create an interact decision."""
    return Decision(
        decision_type=DecisionType.INTERACT,
        priority=4,
        params={"action": action}
    )


def make_greet_decision(message: str = "Hello!") -> Decision:
    """Create a greet decision."""
    return Decision(
        decision_type=DecisionType.GREET,
        priority=5,
        params={"message": message}
    )


def make_sleep_decision() -> Decision:
    """Create a sleep decision."""
    return Decision(
        decision_type=DecisionType.SLEEP,
        priority=1,
        params={}
    )