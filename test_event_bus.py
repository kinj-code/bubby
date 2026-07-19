#!/usr/bin/env python3
"""Punch List Item 4 — Verification Test: Event bus pub/sub with mobile bridge integration."""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def test_event_bus_pub_sub():
    """Test that multiple subscribers receive the same published event."""
    from src.network.event_bus import EventBus, TOPIC_MOBILE_EVENT
    
    bus = EventBus()
    received_1 = []; received_2 = []
    bus.subscribe(TOPIC_MOBILE_EVENT, lambda e: received_1.append(e))
    bus.subscribe(TOPIC_MOBILE_EVENT, lambda e: received_2.append(e))
    
    event = {"type": "battery_update", "device_battery": 42}
    bus.publish(TOPIC_MOBILE_EVENT, event)
    
    assert len(received_1) == 1, f"Subscriber 1 should receive 1 event, got {len(received_1)}"
    assert len(received_2) == 1, f"Subscriber 2 should receive 1 event, got {len(received_2)}"
    assert received_1[0]["device_battery"] == 42
    logger.info("✓ Multiple subscribers receive the same event")

def test_event_bus_isolation():
    """Test that errors in one subscriber don't break others."""
    from src.network.event_bus import EventBus, TOPIC_MOBILE_EVENT
    
    bus = EventBus()
    received = []
    def crashy(event): raise RuntimeError("simulated crash")
    def good(event): received.append(event)
    
    bus.subscribe(TOPIC_MOBILE_EVENT, crashy)
    bus.subscribe(TOPIC_MOBILE_EVENT, good)
    bus.publish(TOPIC_MOBILE_EVENT, {"type": "battery_update"})
    
    assert len(received) == 1, f"Good subscriber should still receive event despite crashy subscriber"
    logger.info("✓ Error isolation: bad subscriber doesn't break good ones")

def test_event_bus_stats():
    """Test that stats correctly report topic subscriber counts."""
    from src.network.event_bus import EventBus, TOPIC_MOBILE_EVENT, TOPIC_TERMINAL_EVENT
    
    bus = EventBus()
    bus.subscribe(TOPIC_MOBILE_EVENT, lambda e: None)
    bus.subscribe(TOPIC_MOBILE_EVENT, lambda e: None)
    bus.subscribe(TOPIC_TERMINAL_EVENT, lambda e: None)
    bus.publish(TOPIC_MOBILE_EVENT, {})
    
    stats = bus.get_stats()
    assert stats["topics"][TOPIC_MOBILE_EVENT] == 2
    assert stats["topics"][TOPIC_TERMINAL_EVENT] == 1
    assert stats["publish_count"] == 1
    logger.info(f"✓ Stats correct: {stats}")

def test_mobile_sensor_via_event_bus():
    """Test that the mobile sensor callback works when routed through the event bus."""
    from src.network.event_bus import EventBus, TOPIC_MOBILE_EVENT
    from src.sensors.mobile import MobileSensor
    from src.network.local_server import LocalBridgeServer, MobileEvent, MobileEventType
    
    bus = EventBus()
    sensor = MobileSensor()
    bus.subscribe(TOPIC_MOBILE_EVENT, sensor.on_mobile_event)
    
    # Simulate the LocalBridgeServer publishing a battery event
    server = LocalBridgeServer(host="127.0.0.1", port=19879)
    event = server._parse_event({"type": "battery_update", "device_battery": 14, "device_charging": False})
    assert event is not None
    
    bus.publish(TOPIC_MOBILE_EVENT, event)
    
    assert sensor.is_connected()
    assert sensor._device_state.battery == 14
    logger.info(f"✓ Mobile sensor receives event via event bus: battery={sensor._device_state.battery}%")

def test_existing_mobile_bridge_test_still_passes():
    """Confirm the original test_mobile_bridge.py still imports and runs its core logic."""
    from src.network.local_server import LocalBridgeServer, MobileEventType
    
    server = LocalBridgeServer(host="127.0.0.1", port=19880)
    event = server._parse_event({"type": "battery_update", "device_battery": 10, "device_charging": False})
    assert event is not None
    assert event.is_low_battery
    logger.info("✓ Original mobile bridge parsing still works after event bus addition")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PUNCH LIST ITEM 4 — VERIFICATION TEST")
    logger.info("Event bus pub/sub + mobile bridge integration")
    logger.info("=" * 60)
    logger.info("")
    
    test_event_bus_pub_sub()
    logger.info("")
    test_event_bus_isolation()
    logger.info("")
    test_event_bus_stats()
    logger.info("")
    test_mobile_sensor_via_event_bus()
    logger.info("")
    test_existing_mobile_bridge_test_still_passes()
    logger.info("")
    
    logger.info("=" * 60)
    logger.info("ALL PUNCH LIST ITEM 4 TESTS PASSED ✓")
    logger.info("=" * 60)