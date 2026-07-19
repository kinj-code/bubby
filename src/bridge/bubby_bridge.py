#!/usr/bin/env python3
"""
Bubby IPC Bridge — Python client for the Rust core kernel.

Protocol: MessagePack over TCP with 4-byte LE length prefix.
Connects to the Rust `ipc` server on a local port and exposes the
same API as the old Python memory classes: add, search, get_recent, etc.

Usage:
    from src.bridge.bubby_bridge import BubbyBridge

    bridge = BubbyBridge(port=9500)
    bridge.ping()              # -> {"version": "0.2.0"}
    bridge.add("hello world")  # -> 0
    bridge.search("hello", k=5) # -> [MemoryRecord, ...]
"""

import socket
import struct
import time
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MessagePack cross-compatibility
# ---------------------------------------------------------------------------
try:
    import msgpack
except ImportError:
    # Fallback: JSON over the wire (will be slower but works without msgpack)
    import json as msgpack  # type: ignore[no-redef]
    logger.warning("msgpack not installed — falling back to JSON. "
                   "Install with: pip install msgpack")

# ---------------------------------------------------------------------------
# Python-side data classes mirroring Rust structs
# ---------------------------------------------------------------------------

class MemoryRecord:
    """Mirrors the Rust `MemoryRecord` struct."""
    def __init__(self, id: int, text: str, timestamp: float,
                 importance: float, metadata: str):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.importance = importance
        self.metadata = metadata

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryRecord':
        return cls(
            id=d["id"],
            text=d["text"],
            timestamp=d["timestamp"],
            importance=d["importance"],
            metadata=d.get("metadata", "{}"),
        )

    def __repr__(self):
        return (f"MemoryRecord(id={self.id}, text={self.text[:40]!r}, "
                f"imp={self.importance:.2f})")


class SearchResult:
    """Mirrors the Rust `SearchResult` struct."""
    def __init__(self, record: MemoryRecord, similarity: float,
                 weighted_score: float):
        self.record = record
        self.similarity = similarity
        self.weighted_score = weighted_score

    @classmethod
    def from_dict(cls, d: dict) -> 'SearchResult':
        return cls(
            record=MemoryRecord.from_dict(d["record"]),
            similarity=d["similarity"],
            weighted_score=d.get("weighted_score", d["similarity"]),
        )

    def __repr__(self):
        return (f"SearchResult({self.record.text[:30]!r}, "
                f"sim={self.similarity:.3f})")


class MemoryStats:
    """Mirrors the Rust `MemoryStats` struct."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"MemoryStats({self.__dict__})"


# ---------------------------------------------------------------------------
# TCP client with MessagePack framing
# ---------------------------------------------------------------------------

class BubbyBridge:
    """
    High-performance Python bridge to the Rust core.

    Maintains a persistent TCP connection for low-latency RPC.
    Supports all `MemoryStore` methods including batch search.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 9500

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._id_counter = 0
        self._connected = False

    # ── connection management ─────────────────────────────────────

    def connect(self) -> None:
        """Open a persistent TCP connection to the Rust server."""
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self._host, self._port))
        self._connected = True
        logger.debug(f"Connected to Rust core at {self._host}:{self._port}")

    def close(self) -> None:
        """Close the connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._connected = False

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _send_and_recv(self, method: str, payload: Any) -> Any:
        """Send a request and receive the parsed response payload."""
        if self._sock is None:
            self.connect()

        req_id = self._next_id()
        request = {
            "id": req_id,
            "method": method,
            "payload": msgpack.packb(payload) if not isinstance(payload, bytes) else payload,
        }

        # Serialize request with MessagePack
        req_bytes = msgpack.packb(request)

        # Send: [4-byte LE length] [msgpack body]
        length = struct.pack("<I", len(req_bytes))
        try:
            self._sock.sendall(length + req_bytes)
        except (BrokenPipeError, ConnectionResetError) as e:
            self._sock = None
            raise ConnectionError(f"Lost connection to Rust core: {e}")

        # Receive: [4-byte LE length] [msgpack body]
        try:
            len_bytes = self._recv_exact(4)
            payload_len = struct.unpack("<I", len_bytes)[0]
            resp_bytes = self._recv_exact(payload_len)
        except (BrokenPipeError, ConnectionResetError) as e:
            self._sock = None
            raise ConnectionError(f"Lost connection to Rust core: {e}")

        response = msgpack.unpackb(resp_bytes)

        if response.get("status") != "ok":
            raise RuntimeError(f"RPC error [{method}]: {response.get('error', 'unknown')}")

        return msgpack.unpackb(response["payload"])

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket."""
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError("Connection closed by Rust core")
            buf += chunk
        return buf

    # ── health check ──────────────────────────────────────────────

    def ping(self) -> dict:
        """Ping the Rust server and get version info."""
        return self._send_and_recv("ping", {})

    # ── write operations ──────────────────────────────────────────

    def add(self, text: str, timestamp: Optional[float] = None,
            importance: float = 0.5, metadata: Optional[dict] = None) -> int:
        """
        Add a memory record. Returns the assigned ID.

        Args:
            text: The memory text content.
            timestamp: Unix timestamp (default: now).
            importance: Importance score 0-1.
            metadata: Additional JSON-serializable dict.

        Returns:
            Integer record ID.
        """
        ts = timestamp if timestamp is not None else time.time()
        meta_str = msgpack.packb(metadata or {}) if hasattr(msgpack, 'packb') else '{}'
        payload = {
            "text": text,
            "timestamp": ts,
            "importance": importance,
            "metadata": meta_str if isinstance(meta_str, str) else metadata or {},
        }
        result = self._send_and_recv("add", payload)
        return result["id"]

    def add_with_embedding(self, text: str, embedding: List[float],
                           timestamp: Optional[float] = None,
                           importance: float = 0.5,
                           metadata: Optional[dict] = None) -> int:
        """
        Add a memory record with an associated embedding vector.

        Args:
            text: Memory text.
            embedding: 384-dim float vector.
            timestamp: Unix timestamp (default: now).
            importance: Importance score 0-1.
            metadata: Additional JSON-serializable dict.

        Returns:
            Integer record ID.
        """
        ts = timestamp if timestamp is not None else time.time()
        meta_str = msgpack.packb(metadata or {}) if hasattr(msgpack, 'packb') else '{}'
        payload = {
            "text": text,
            "timestamp": ts,
            "importance": importance,
            "metadata": meta_str if isinstance(meta_str, str) else metadata or {},
            "embedding": embedding,
        }
        result = self._send_and_recv("add_with_embedding", payload)
        return result["id"]

    def update_importance(self, id: int, importance: float) -> bool:
        """Update the importance of an existing record."""
        result = self._send_and_recv("update_importance", {
            "id": id,
            "importance": importance,
        })
        return result.get("ok", False)

    def delete(self, id: int) -> bool:
        """Delete a memory record and its embedding. Returns True if found."""
        result = self._send_and_recv("delete", {"id": id})
        return result.get("ok", False)

    def clear(self) -> bool:
        """Delete all records."""
        result = self._send_and_recv("clear", {})
        return result.get("ok", False)

    # ── read operations ───────────────────────────────────────────

    def get(self, id: int) -> Optional[MemoryRecord]:
        """Get a single record by ID, or None if not found."""
        result = self._send_and_recv("get", {"id": id})
        if result is None:
            return None
        return MemoryRecord.from_dict(result)

    def search_by_text(self, query: str, k: int = 5) -> List[MemoryRecord]:
        """Search memories by text substring (LIKE query)."""
        result = self._send_and_recv("search_by_text", {
            "query": query,
            "k": k,
        })
        return [MemoryRecord.from_dict(r) for r in result]

    def search_by_embedding(self, embedding: List[float], k: int = 5,
                            min_score: float = 0.0) -> List[SearchResult]:
        """Search by embedding vector (cosine similarity)."""
        result = self._send_and_recv("search_by_embedding", {
            "embedding": embedding,
            "k": k,
            "min_score": min_score,
        })
        return [SearchResult.from_dict(r) for r in result]

    def batch_search_by_embedding(
            self, queries: List[dict]) -> List[List[SearchResult]]:
        """
        Execute multiple embedding searches in a single IPC call.

        Args:
            queries: List of dicts with keys: embedding, k, min_score.

        Returns:
            List of lists of SearchResult.
        """
        result = self._send_and_recv("batch_search_by_embedding", {
            "queries": queries,
        })
        return [
            [SearchResult.from_dict(r) for r in (batch or [])]
            for batch in result
        ]

    def search_by_metadata(self, key: str, value: str, k: int = 5) -> List[MemoryRecord]:
        """Search by metadata key-value pair."""
        result = self._send_and_recv("search_by_metadata", {
            "key": key,
            "value": value,
            "k": k,
        })
        return [MemoryRecord.from_dict(r) for r in result]

    def get_recent(self, n: int = 10) -> List[MemoryRecord]:
        """Get the N most recent memories (newest first)."""
        result = self._send_and_recv("get_recent", n)
        return [MemoryRecord.from_dict(r) for r in result]

    def get_by_importance(self, n: int = 10,
                          min_importance: float = 0.0) -> List[MemoryRecord]:
        """Get top N memories by importance, filtered by threshold."""
        result = self._send_and_recv("get_by_importance", {
            "n": n,
            "min_importance": min_importance,
        })
        return [MemoryRecord.from_dict(r) for r in result]

    # ── stats ─────────────────────────────────────────────────────

    def count(self) -> int:
        """Get total number of stored memories."""
        result = self._send_and_recv("count", {})
        return result["count"]

    def stats(self) -> MemoryStats:
        """Get store statistics."""
        result = self._send_and_recv("stats", {})
        return MemoryStats(**result)


# ---------------------------------------------------------------------------
# Latency verification (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("=" * 60)
    logger.info("BUBBY BRIDGE — Latency Verification")
    logger.info("=" * 60)

    bridge = BubbyBridge(port=9500)

    # 1. Ping
    logger.info("\n>>> Ping")
    version = bridge.ping()
    logger.info(f"  Server version: {version}")

    # 2. Add some records
    logger.info("\n>>> Adding records")
    ids = []
    for i in range(10):
        rid = bridge.add(f"Test record {i}", importance=0.5 + i * 0.05)
        ids.append(rid)
    logger.info(f"  Added {len(ids)} records")

    # 3. Text search
    logger.info("\n>>> Text search")
    results = bridge.search_by_text("record", k=5)
    logger.info(f"  Found {len(results)} records: {[r.text for r in results]}")

    # 4. Embedding search
    logger.info("\n>>> Embedding search")
    embedding = [0.0] * 384
    embedding[0] = 1.0
    bridge.add_with_embedding("coffee test", embedding, importance=0.9)
    results = bridge.search_by_embedding([1.0] + [0.0] * 383, k=3)
    logger.info(f"  Found {len(results)} results")

    # 5. Batch latency test
    logger.info("\n>>> Batch latency: 1000 queries")
    queries = [{"embedding": [1.0] + [0.0] * 383, "k": 3, "min_score": 0.0}] * 1000

    start = time.monotonic()
    batch_results = bridge.batch_search_by_embedding(queries)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    logger.info(f"  Completed {len(batch_results)} queries")
    logger.info(f"  Total time: {elapsed_ms:.3f} ms")
    logger.info(f"  Per query:  {elapsed_ms / 1000:.4f} ms ({elapsed_ms / 1000 * 1000:.3f} µs)")

    assert len(batch_results) == 1000, f"Expected 1000 results, got {len(batch_results)}"
    assert elapsed_ms < 50, f"FAILED: {elapsed_ms:.1f}ms exceeds 50ms limit"

    # 6. Stats
    logger.info("\n>>> Store statistics")
    stats = bridge.stats()
    logger.info(f"  Records: {stats.total_records}")
    logger.info(f"  Embeddings: {stats.total_embeddings}")

    logger.info("\n" + "=" * 60)
    logger.info(f"✅ ALL TESTS PASSED (latency: {elapsed_ms:.3f}ms < 50ms)")
    logger.info("=" * 60)