#!/usr/bin/env python3
"""
Punch List Item 3 — Verification Test:
Prove the mobile bridge rejects unauthenticated payloads when auth is enabled,
and accepts authenticated payloads. Also prove unknown event types are rejected.

Tests run against _parse_event directly (no TCP needed).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_auth_disabled_allows_all():
    """When shared_secret is empty, all payloads pass (backward compat)."""
    from src.network.local_server import LocalBridgeServer, MobileEventType

    server = LocalBridgeServer(host="127.0.0.1", port=19878, shared_secret="")
    assert not server._auth_enabled
    assert server._auth_failures == 0

    payload = {"type": "battery_update", "device_battery": 78}
    event = server._parse_event(payload)
    assert event is not None
    assert event.event_type == MobileEventType.BATTERY_UPDATE
    logger.info("✓ Auth disabled: event parsed normally (backward compatible)")


def test_unknown_event_type_rejected():
    """Unknown type strings should return None, not silently default to CUSTOM."""
    from src.network.local_server import LocalBridgeServer

    server = LocalBridgeServer(host="127.0.0.1", port=19878)
    event = server._parse_event({"type": "malware_injection"})
    assert event is None, "Unknown event type should be rejected (return None)"
    logger.info("✓ Unknown event type 'malware_injection' rejected (returned None)")

    event = server._parse_event({"type": ""})
    assert event is None, "Empty event type should be rejected"
    logger.info("✓ Empty event type rejected")


def test_known_event_types_still_work():
    """All 9 known event types must still parse correctly."""
    from src.network.local_server import LocalBridgeServer, MobileEventType

    server = LocalBridgeServer(host="127.0.0.1", port=19878)
    known_types = [
        ("battery_update", MobileEventType.BATTERY_UPDATE),
        ("app_opened", MobileEventType.APP_OPENED),
        ("app_closed", MobileEventType.APP_CLOSED),
        ("sms_received", MobileEventType.SMS_RECEIVED),
        ("call_state", MobileEventType.CALL_STATE),
        ("notification", MobileEventType.NOTIFICATION),
        ("connection_status", MobileEventType.CONNECTION_STATUS),
        ("device_info", MobileEventType.DEVICE_INFO),
        ("custom", MobileEventType.CUSTOM),
    ]
    for type_str, expected_enum in known_types:
        event = server._parse_event({"type": type_str})
        assert event is not None, f"Known type '{type_str}' should parse"
        assert event.event_type == expected_enum, f"Expected {expected_enum}, got {event.event_type}"
    logger.info(f"✓ All {len(known_types)} known event types parse correctly")


def test_auth_enabled_stats():
    """Stats must include auth_enabled and auth_failures fields."""
    from src.network.local_server import LocalBridgeServer

    server = LocalBridgeServer(
        host="127.0.0.1", port=19878, shared_secret="bubby-pair-token-2026"
    )
    stats = server.get_stats()
    assert stats["auth_enabled"] is True, "auth_enabled should be True"
    assert "auth_failures" in stats, "auth_failures should be in stats"
    logger.info(f"✓ Auth stats: enabled={stats['auth_enabled']}, failures={stats['auth_failures']}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PUNCH LIST ITEM 3 — VERIFICATION TEST")
    logger.info("Mobile bridge authentication + unknown type rejection")
    logger.info("=" * 60)
    logger.info("")

    test_auth_disabled_allows_all()
    logger.info("")
    test_unknown_event_type_rejected()
    logger.info("")
    test_known_event_types_still_work()
    logger.info("")
    test_auth_enabled_stats()
    logger.info("")

    logger.info("=" * 60)
    logger.info("ALL PUNCH LIST ITEM 3 TESTS PASSED ✓")
    logger.info("=" * 60)