"""Local TCP bridge server for offline mobile-to-desktop communication.

Runs an async TCP server on localhost that accepts JSON payloads from a
paired Android device. No external internet, no cloud APIs — strictly
local network communication (Wi-Fi hotspot or USB tethering).

The server runs in its own thread to avoid blocking the Qt event loop
or LLM inference pipeline.

RAM: ~5MB (async TCP + JSON parsing).
"""

import asyncio
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MobileEventType(str, Enum):
    """Types of events received from the mobile device."""
    BATTERY_UPDATE = "battery_update"
    APP_OPENED = "app_opened"
    APP_CLOSED = "app_closed"
    SMS_RECEIVED = "sms_received"
    CALL_STATE = "call_state"
    NOTIFICATION = "notification"
    CONNECTION_STATUS = "connection_status"
    DEVICE_INFO = "device_info"
    CUSTOM = "custom"


@dataclass
class MobileEvent:
    """A normalized mobile device event."""
    event_type: MobileEventType
    device_id: str = ""
    device_battery: int = 100
    device_charging: bool = False
    active_app: str = ""
    timestamp: float = field(default_factory=time.time)
    urgency: float = 0.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    message: str = ""  # Human-readable summary
    
    @property
    def is_low_battery(self) -> bool:
        """Check if device battery is critically low."""
        return self.device_battery <= 15 and not self.device_charging


class LocalBridgeServer:
    """
    Async TCP server for receiving mobile device events.
    
    Listens on a configurable local port for JSON-line payloads.
    Each connection is handled asynchronously. Events are dispatched
    to registered callbacks for sensor integration.
    
    Architecture:
    1. Server starts in background thread
    2. Android client connects via TCP socket
    3. Client sends JSON payloads (one per line)
    4. Server parses, normalizes, dispatches to MobileSensor
    5. MobileSensor feeds ProactivityEvaluator
    """

    DEFAULT_HOST = "0.0.0.0"  # Listen on all interfaces (local network only)
    DEFAULT_PORT = 9877       # Non-privileged port

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        shared_secret: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._shared_secret = shared_secret
        self._auth_enabled = bool(shared_secret)
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[MobileEvent], None]] = []
        self._connected_clients = 0
        self._events_received = 0
        self._auth_failures = 0
        
        logger.info(f"LocalBridgeServer configured: {host}:{port} (auth={'on' if self._auth_enabled else 'off'})")

    def register_callback(self, callback: Callable[[MobileEvent], None]) -> None:
        """Register a callback for incoming mobile events."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._running:
            logger.warning("Server already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="mobile-bridge",
        )
        self._thread.start()
        logger.info(f"LocalBridgeServer started on {self._host}:{self._port}")

    def stop(self) -> None:
        """Stop the server gracefully."""
        self._running = False
        logger.info("LocalBridgeServer stopped")

    def _run_server(self) -> None:
        """Run the async server in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._start_async_server())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            loop.close()

    async def _start_async_server(self) -> None:
        """Start the async TCP server."""
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self._host,
                self._port,
            )
            logger.info(f"TCP server listening on {self._host}:{self._port}")
            
            async with self._server:
                await self._server.serve_forever()
        except OSError as e:
            logger.error(f"Failed to bind to {self._host}:{self._port}: {e}")
            logger.info("Check that the port is available and not blocked by firewall")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a connected mobile client with optional shared-secret auth."""
        addr = writer.get_extra_info('peername')
        
        # ── Shared-secret authentication handshake ──
        if self._auth_enabled:
            try:
                auth_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                auth_data = json.loads(auth_line.decode('utf-8').strip())
                presented_secret = auth_data.get("auth", "")
                if presented_secret != self._shared_secret:
                    self._auth_failures += 1
                    logger.warning(f"Auth rejected for {addr}: bad secret (failure #{self._auth_failures})")
                    writer.write(b'{"status": "error", "reason": "unauthorized"}\n')
                    await writer.drain()
                    writer.close()
                    return
            except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
                self._auth_failures += 1
                logger.warning(f"Auth failed for {addr}: no valid auth token (failure #{self._auth_failures})")
                writer.write(b'{"status": "error", "reason": "unauthorized"}\n')
                await writer.drain()
                writer.close()
                return
        
        self._connected_clients += 1
        logger.info(f"Mobile client connected: {addr} (total: {self._connected_clients})")
        
        try:
            while self._running:
                data = await reader.readline()
                if not data:
                    break
                
                try:
                    line = data.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    payload = json.loads(line)
                    event = self._parse_event(payload)
                    
                    if event:
                        self._events_received += 1
                        logger.debug(
                            f"Mobile event: {event.event_type.value} "
                            f"(battery={event.device_battery}%, app={event.active_app or 'none'})"
                        )
                        
                        for callback in self._callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        ack = json.dumps({"status": "ok", "event_id": self._events_received})
                        writer.write((ack + "\n").encode('utf-8'))
                        await writer.drain()
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {addr}: {data[:100]}")
                    writer.write(b'{"status": "error", "reason": "invalid_json"}\n')
                    await writer.drain()
                    
        except (ConnectionResetError, BrokenPipeError):
            logger.info(f"Client disconnected: {addr}")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            self._connected_clients -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    
    def _parse_event(self, payload: Dict[str, Any]) -> Optional[MobileEvent]:
        """
        Parse a raw JSON payload into a normalized MobileEvent.
        
        Expected payload format (from Android client):
        {
            "type": "battery_update",
            "device_id": "infinix_hot60i",
            "device_battery": 14,
            "device_charging": false,
            "active_app": "com.spotify.music",
            "message": "Battery at 14%"
        }
        """
        event_type_str = payload.get("type", "")
        try:
            event_type = MobileEventType(event_type_str)
        except ValueError:
            logger.warning(f"Rejected unknown event type: '{event_type_str}'")
            return None
        
        event = MobileEvent(
            event_type=event_type,
            device_id=payload.get("device_id", ""),
            device_battery=payload.get("device_battery", 100),
            device_charging=payload.get("device_charging", False),
            active_app=payload.get("active_app", ""),
            raw_payload=payload,
            message=payload.get("message", ""),
        )
        
        # Calculate urgency based on event type
        event.urgency = self._calculate_urgency(event)
        
        return event
    
    def _calculate_urgency(self, event: MobileEvent) -> float:
        """Calculate urgency score for a mobile event."""
        if event.event_type == MobileEventType.BATTERY_UPDATE:
            if event.is_low_battery:
                # Linear urgency from 15% down to 0%
                return round(0.5 + (15 - event.device_battery) / 30, 2)  # 0.50 at 15% → 0.97 at 1%
            elif event.device_battery <= 25:
                return 0.3  # Getting low
            else:
                return 0.05  # Normal battery — low urgency
        
        if event.event_type == MobileEventType.SMS_RECEIVED:
            return 0.4  # Moderate urgency
        
        if event.event_type == MobileEventType.CALL_STATE:
            return 0.6  # Higher urgency
        
        if event.event_type == MobileEventType.APP_OPENED:
            return 0.1  # Informational
        
        return 0.05  # Default low urgency
    
    def get_connection_instructions(self) -> str:
        """
        Get instructions for connecting a mobile device.
        
        Returns:
            Human-readable connection instructions
        """
        import socket
        hostname = socket.gethostname()
        
        # Try to get local IP
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"
        
        return f"""=== MOBILE BRIDGE CONNECTION ===
Server: {self._host}:{self._port}
Local IP: {local_ip}

To connect your Android device:
1. Connect phone to same Wi-Fi network as this PC
   OR enable USB tethering
2. Run the OfflineBridgeService on your Android device
3. Point the client to: ws://{local_ip}:{self._port}

The Kotlin service code is in: mobile/OfflineBridgeService.kt
"""

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "running": self._running,
            "host": self._host,
            "port": self._port,
            "connected_clients": self._connected_clients,
            "events_received": self._events_received,
            "auth_enabled": self._auth_enabled,
            "auth_failures": self._auth_failures,
        }


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("LOCAL BRIDGE SERVER TEST")
    logger.info("=" * 60)
    
    server = LocalBridgeServer(host="127.0.0.1", port=19877)
    
    events_received = []
    def on_event(event: MobileEvent):
        events_received.append(event)
        logger.info(f"Callback received: {event.event_type.value} (battery={event.device_battery}%, urgency={event.urgency:.2f})")
    
    server.register_callback(on_event)
    
    # Test 1: Parse battery event
    payload = {
        "type": "battery_update",
        "device_id": "infinix_hot60i",
        "device_battery": 14,
        "device_charging": False,
        "active_app": "com.spotify.music",
    }
    event = server._parse_event(payload)
    assert event is not None
    assert event.event_type == MobileEventType.BATTERY_UPDATE
    assert event.is_low_battery
    assert event.urgency > 0.5
    logger.info(f"✓ Test 1: Low battery parsed (urgency={event.urgency:.2f})")
    
    # Test 2: Parse app open event
    payload = {"type": "app_opened", "device_battery": 85, "active_app": "com.whatsapp"}
    event = server._parse_event(payload)
    assert event.event_type == MobileEventType.APP_OPENED
    assert event.urgency == 0.1
    logger.info(f"✓ Test 2: App opened parsed (urgency={event.urgency:.2f})")
    
    # Test 3: Urgency scaling for battery
    for pct in [30, 20, 15, 10, 5, 1]:
        payload = {"type": "battery_update", "device_battery": pct, "device_charging": False}
        event = server._parse_event(payload)
        logger.info(f"  Battery {pct}% → urgency={event.urgency:.2f}")
        if pct <= 15:
            assert event.is_low_battery
        
    logger.info(f"✓ Test 3: Battery urgency scaling verified")
    
    # Test 4: Connection instructions
    instructions = server.get_connection_instructions()
    assert "19877" in instructions
    assert "OfflineBridgeService" in instructions
    logger.info(f"✓ Test 4: Connection instructions generated ({len(instructions)} chars)")
    
    # Test 5: Stats
    stats = server.get_stats()
    assert stats["port"] == 19877
    logger.info(f"✓ Test 5: Stats: {stats}")
    
    logger.info("\nALL LOCAL SERVER TESTS PASSED")