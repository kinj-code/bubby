"""Mobile device sensor for offline phone-to-desktop synchronization.

Receives normalized MobileEvent payloads from the LocalBridgeServer
and maps them into the ProactivityEvaluator framework for contextual
interventions (e.g., low battery alert, incoming call notification).

RAM: Negligible (event passthrough + state tracking).
"""

import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from src.network.local_server import MobileEvent, MobileEventType
from src.brain.proactivity import (
    ProactivityContext, ProactivityDecision, ActivityLevel, InterventionType
)

logger = logging.getLogger(__name__)


@dataclass
class MobileDeviceState:
    """Tracked state of the paired mobile device."""
    device_id: str = ""
    battery: int = 100
    charging: bool = False
    active_app: str = ""
    last_event_type: MobileEventType = MobileEventType.CUSTOM
    last_event_time: float = 0.0
    is_connected: bool = False
    low_battery_alerted: bool = False
    events_received: int = 0


class MobileSensor:
    """
    Ingests mobile device events and maps them to proactivity triggers.
    
    Flow:
    1. LocalBridgeServer receives TCP data from Android client
    2. MobileSensor.on_mobile_event() is called via callback
    3. Event is normalized and stored in device state
    4. evaluate_proactivity() converts state to ProactivityContext
    5. ProactivityEvaluator decides whether to intervene
    
    The sensor itself doesn't trigger actions — it just prepares
    context for the evaluator.
    """

    # Thresholds
    LOW_BATTERY_THRESHOLD = 15       # Percentage
    CRITICAL_BATTERY_THRESHOLD = 5   # Percentage
    ALERT_COOLDOWN_SECONDS = 300     # 5 minutes between alerts

    def __init__(self) -> None:
        self._device_state = MobileDeviceState()
        self._alert_callbacks: List[Callable[[str, float], None]] = []
        self._last_alert_time: Dict[str, float] = {}
        logger.info("MobileSensor initialized")

    def on_mobile_event(self, event: MobileEvent) -> None:
        """
        Callback for incoming mobile events from the bridge server.
        
        Args:
            event: Normalized MobileEvent from LocalBridgeServer
        """
        self._device_state.device_id = event.device_id or self._device_state.device_id
        self._device_state.battery = event.device_battery
        self._device_state.charging = event.device_charging
        self._device_state.active_app = event.active_app or self._device_state.active_app
        self._device_state.last_event_type = event.event_type
        self._device_state.last_event_time = time.time()
        self._device_state.is_connected = True
        self._device_state.events_received += 1

        logger.debug(
            f"Mobile event: type={event.event_type.value}, "
            f"battery={event.device_battery}%, urgency={event.urgency:.2f}"
        )

        # Notify alert callbacks if urgency is significant
        self._check_alert_conditions(event)

    def evaluate_proactivity(self) -> ProactivityContext:
        """
        Build a ProactivityContext from current mobile device state.
        
        Returns:
            ProactivityContext ready for ProactivityEvaluator
        """
        state = self._device_state

        # Determine activity level
        activity_level = ActivityLevel.IDLE
        if state.last_event_type == MobileEventType.CALL_STATE:
            activity_level = ActivityLevel.CRITICAL
        elif state.last_event_type == MobileEventType.BATTERY_UPDATE and state.battery <= self.LOW_BATTERY_THRESHOLD:
            activity_level = ActivityLevel.CRITICAL
        elif state.last_event_type == MobileEventType.SMS_RECEIVED:
            activity_level = ActivityLevel.ACTIVE
        elif state.last_event_type == MobileEventType.APP_OPENED:
            activity_level = ActivityLevel.PASSIVE

        # Check if we should alert about battery
        has_error = (
            state.battery <= self.LOW_BATTERY_THRESHOLD
            and not state.charging
        )
        user_busy = False  # Mobile events are always worth checking

        return ProactivityContext(
            content_type="mobile",
            activity_level=activity_level,
            has_recent_error=has_error,
            user_busy=user_busy,
            is_context_shift=(state.last_event_type != MobileEventType.CUSTOM),
        )

    def get_device_summary(self) -> str:
        """Get a human-readable summary of the mobile device state."""
        state = self._device_state
        if not state.is_connected:
            return "No mobile device connected"
        
        status = "charging" if state.charging else "on battery"
        app_name = state.active_app.split(".")[-1] if state.active_app else "unknown"
        
        return (
            f"Phone: {state.battery}% ({status}), "
            f"app: {app_name}"
        )

    def is_connected(self) -> bool:
        """Check if a mobile device is connected."""
        return self._device_state.is_connected

    def _check_alert_conditions(self, event: MobileEvent) -> None:
        """Check if this event should trigger an alert to the user."""
        now = time.time()
        
        # Battery alert
        if event.is_low_battery:
            alert_key = "low_battery"
            last = self._last_alert_time.get(alert_key, 0)
            if now - last > self.ALERT_COOLDOWN_SECONDS:
                self._last_alert_time[alert_key] = now
                self._device_state.low_battery_alerted = True
                msg = (
                    f"Your phone battery is at {event.device_battery}%! "
                    "You might want to plug it in."
                )
                for cb in self._alert_callbacks:
                    cb(msg, event.urgency)
                logger.info(f"Mobile alert: {msg}")
        
        # Critical battery
        if event.device_battery <= self.CRITICAL_BATTERY_THRESHOLD and not event.device_charging:
            alert_key = "critical_battery"
            last = self._last_alert_time.get(alert_key, 0)
            if now - last > self.ALERT_COOLDOWN_SECONDS:
                self._last_alert_time[alert_key] = now
                msg = (
                    f"URGENT: Your phone battery is critically low "
                    f"({event.device_battery}%) and not charging!"
                )
                for cb in self._alert_callbacks:
                    cb(msg, max(event.urgency, 0.90))
                logger.warning(f"Mobile critical alert: {msg}")

    def register_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Register a callback for mobile alerts.
        
        Args:
            callback: Function that receives (message, urgency_score)
        """
        self._alert_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get sensor statistics."""
        state = self._device_state
        return {
            "device_connected": state.is_connected,
            "device_id": state.device_id,
            "battery": state.battery,
            "charging": state.charging,
            "events_received": state.events_received,
            "low_battery_alerted": state.low_battery_alerted,
        }

    def reset(self) -> None:
        """Reset sensor state (for testing)."""
        self._device_state = MobileDeviceState()
        self._last_alert_time.clear()
        logger.debug("MobileSensor reset")


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("MOBILE SENSOR TEST")
    logger.info("=" * 60)

    sensor = MobileSensor()

    # Test 1: Process a normal battery event (85%)
    from src.network.local_server import MobileEvent, MobileEventType
    event = MobileEvent(
        event_type=MobileEventType.BATTERY_UPDATE,
        device_id="infinix_hot60i",
        device_battery=85,
        device_charging=False,
        active_app="com.spotify.music",
    )
    sensor.on_mobile_event(event)
    assert sensor.is_connected()
    assert sensor._device_state.battery == 85
    logger.info(f"✓ Test 1: Normal battery processed: {sensor.get_device_summary()}")

    # Test 2: Low battery triggers alert callback
    alerts = []
    def on_alert(msg, urgency):
        alerts.append((msg, urgency))
    
    sensor.register_alert_callback(on_alert)
    event = MobileEvent(
        event_type=MobileEventType.BATTERY_UPDATE,
        device_id="infinix_hot60i",
        device_battery=14,
        device_charging=False,
    )
    sensor.on_mobile_event(event)
    assert sensor._device_state.battery == 14
    assert sensor._device_state.low_battery_alerted
    assert len(alerts) >= 1
    logger.info(f"✓ Test 2: Low battery alert triggered: '{alerts[0][0][:50]}...'")

    # Test 3: Proactivity context generation
    ctx = sensor.evaluate_proactivity()
    assert ctx.content_type == "mobile"
    assert ctx.has_recent_error  # Battery is low
    assert ctx.activity_level == ActivityLevel.CRITICAL
    logger.info(f"✓ Test 3: Proactivity context: level={ctx.activity_level.value}, error={ctx.has_recent_error}")

    # Test 4: Critical battery (1%)
    event = MobileEvent(
        event_type=MobileEventType.BATTERY_UPDATE,
        device_id="infinix_hot60i",
        device_battery=1,
        device_charging=False,
    )
    sensor._last_alert_time.clear()  # Reset cooldown
    sensor.on_mobile_event(event)
    assert sensor._device_state.battery == 1
    logger.info(f"✓ Test 4: Critical battery alert triggered (should log WARNING)")

    # Test 5: Cooldown prevents spam
    alerts_before = len(alerts)
    event = MobileEvent(
        event_type=MobileEventType.BATTERY_UPDATE,
        device_id="infinix_hot60i",
        device_battery=10,
        device_charging=False,
    )
    sensor.on_mobile_event(event)
    assert len(alerts) == alerts_before  # No new alert (cooldown)
    logger.info(f"✓ Test 5: Alert cooldown enforced (no new alert)")

    # Test 6: Device summary when connected
    summary = sensor.get_device_summary()
    assert "10%" in summary or "Phone" in summary
    logger.info(f"✓ Test 6: Device summary: '{summary}'")

    # Test 7: Reset
    sensor.reset()
    assert not sensor.is_connected()
    assert sensor._device_state.battery == 100
    logger.info(f"✓ Test 7: Reset works")

    # Test 8: Stats
    sensor.on_mobile_event(event)
    stats = sensor.get_stats()
    assert stats["events_received"] == 1
    logger.info(f"✓ Test 8: Stats: {stats}")

    logger.info("\nALL MOBILE SENSOR TESTS PASSED")