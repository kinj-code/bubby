#!/usr/bin/env python3
"""
Punch List Item 3 — Verification Test:
Prove the mobile bridge rejects unauthenticated payloads when auth is enabled,
and accepts authenticated payloads. Also prove unknown event types are rejected.

Tests include a real socket-level auth handshake exercising _handle_client directly.
"""

import json
import logging
import socket
import sys
import threading
import time
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


def test_socket_auth_bad_secret_rejected():
    """
    REAL SOCKET TEST: Connect to a running server with auth enabled,
    send a bad secret, verify the server returns 'unauthorized' and closes.
    """
    import asyncio
    from src.network.local_server import LocalBridgeServer

    TEST_PORT = 19881
    SHARED_SECRET = "test-auth-token-abc123"

    # Start the server in a background thread
    server = LocalBridgeServer(
        host="127.0.0.1", port=TEST_PORT, shared_secret=SHARED_SECRET,
    )
    assert server._auth_enabled, "Auth should be enabled when secret is provided"

    server_running = threading.Event()

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start_server(server, loop, server_running))

    async def _start_server(srv, loop, ready):
        srv._server = await asyncio.start_server(
            srv._handle_client, srv._host, TEST_PORT,
        )
        ready.set()
        async with srv._server:
            await srv._server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True, name="test-bridge")
    thread.start()
    server_running.wait(timeout=2.0)

    async def _stop_server(srv):
        srv._server.close()
        await srv._server.wait_closed()

    try:
        # Connect with a BAD secret
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(("127.0.0.1", TEST_PORT))

        # Send bad auth
        bad_auth = json.dumps({"auth": "wrong-secret"}) + "\n"
        sock.sendall(bad_auth.encode("utf-8"))

        # Read response — should be unauthorized
        response = sock.recv(1024).decode("utf-8")
        resp_json = json.loads(response.split("\n")[0])
        assert resp_json.get("status") == "error", f"Expected error status, got {resp_json}"
        assert resp_json.get("reason") == "unauthorized", f"Expected unauthorized, got {resp_json}"
        logger.info("✓ Socket test: bad auth token → 'unauthorized' response")

        # Connection should be closed by server
        sock.settimeout(0.5)
        try:
            sock.recv(1024)
            # If we didn't get an error, the connection may still be open — check data
            logger.warning("Server may not have closed connection after bad auth")
        except (ConnectionResetError, BrokenPipeError, socket.timeout):
            pass  # Expected: server closed or timed out
        sock.close()

        # Check stats reflect the failure
        stats = server.get_stats()
        assert stats["auth_failures"] >= 1, f"Expected auth_failures≥1, got {stats}"
        logger.info(f"✓ Auth failures tracked: {stats['auth_failures']}")

    finally:
        # Shutdown
        if server._server:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_stop_server(server))

    logger.info("✓ Socket auth handshake test PASSED")


def test_socket_auth_good_secret_accepted():
    """
    REAL SOCKET TEST: Connect with the correct secret, then send a valid
    battery_update event. Verify the event is parsed and dispatched.
    """
    import asyncio
    from src.network.local_server import LocalBridgeServer

    TEST_PORT = 19882
    SHARED_SECRET = "good-auth-token"

    server = LocalBridgeServer(
        host="127.0.0.1", port=TEST_PORT, shared_secret=SHARED_SECRET,
    )
    assert server._auth_enabled
    server._running = True  # Required for the client handler's while loop

    received_events = []
    server.register_callback(lambda e: received_events.append(e))

    server_running = threading.Event()

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run(server, loop, server_running))

    async def _run(srv, loop, ready):
        srv._server = await asyncio.start_server(
            srv._handle_client, srv._host, TEST_PORT,
        )
        ready.set()
        async with srv._server:
            await srv._server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True, name="test-bridge-good")
    thread.start()
    server_running.wait(timeout=2.0)

    try:
        # Connect with GOOD secret
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(("127.0.0.1", TEST_PORT))

        # Send correct auth
        good_auth = json.dumps({"auth": SHARED_SECRET}) + "\n"
        sock.sendall(good_auth.encode("utf-8"))
        time.sleep(0.05)  # Let server process auth before event

        # Send a valid battery event
        payload = json.dumps({
            "type": "battery_update",
            "device_battery": 14,
            "device_charging": False,
        }) + "\n"
        sock.sendall(payload.encode("utf-8"))

        # Read ACK — server sends response after event is processed
        sock.settimeout(2.0)
        response = sock.recv(1024).decode("utf-8")
        if not response.strip():
            raise AssertionError(f"Empty response from server — connection may have closed prematurely")
        resp_json = json.loads(response.split("\n")[0])
        assert resp_json.get("status") == "ok", f"Expected ok, got {resp_json}"
        logger.info("✓ Socket test: good auth token → accepted, event processed")

        # Verify the callback received the event
        time.sleep(0.3)  # Allow async dispatch
        assert len(received_events) >= 1, f"Callback should have received at least 1 event, got {len(received_events)}"
        evt = received_events[0]
        assert evt.device_battery == 14, f"Expected battery=14, got {evt.device_battery}"
        logger.info(f"✓ Callback received event: battery={evt.device_battery}%, type={evt.event_type.value}")

        stats = server.get_stats()
        assert stats["events_received"] >= 1, f"Expected events_received≥1, got {stats}"
        logger.info(f"✓ Events tracked: {stats['events_received']}")

        sock.close()

    finally:
        server._running = False
        if server._server:
            server._server.close()

    logger.info("✓ Socket good-auth + event dispatch test PASSED")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PUNCH LIST ITEM 3 — VERIFICATION TEST")
    logger.info("Mobile bridge authentication + socket-level handshake")
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
    test_socket_auth_bad_secret_rejected()
    logger.info("")
    test_socket_auth_good_secret_accepted()
    logger.info("")

    logger.info("=" * 60)
    logger.info("ALL PUNCH LIST ITEM 3 TESTS PASSED ✓")
    logger.info("=" * 60)
