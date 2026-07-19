"""In-process event bus — single pub/sub mechanism replacing callback lists and polling.

Sensors publish events. Subscribers (autonomy loop, avatar, action executor, TTS)
subscribe by topic. No external dependencies — pure Python dict + threading.Lock.

RAM: negligible (in-memory dict of subscriber lists).
"""

import threading
import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)

# Topic names as constants to prevent typos
TOPIC_MOBILE_EVENT = "mobile_event"
TOPIC_TERMINAL_EVENT = "terminal_event"
TOPIC_CALENDAR_EVENT = "calendar_event"
TOPIC_DECISION_MADE = "decision_made"
TOPIC_INTERACTION_OUTPUT = "interaction_output"
TOPIC_SYSTEM_ACTION = "system_action"


class EventBus:
    """Thread-safe in-process pub/sub event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._publish_count = 0

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Register a subscriber for a topic."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)
        logger.debug(f"Subscribed to '{topic}' (total: {len(self._subscribers.get(topic, []))})")

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Remove a subscriber from a topic."""
        with self._lock:
            if topic in self._subscribers:
                try:
                    self._subscribers[topic].remove(callback)
                except ValueError:
                    pass

    def publish(self, topic: str, event: Any) -> None:
        """Publish an event to all subscribers of a topic. Errors are isolated."""
        self._publish_count += 1
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in subscriber for '{topic}': {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "topics": {k: len(v) for k, v in self._subscribers.items()},
                "publish_count": self._publish_count,
            }