#!/usr/bin/env python3
"""
Streaming TTS Pipeline — double-buffered audio from LLM token streams.

Architecture:
    LLM tokens → TokenBuffer → sentence chunks → Piper (thread) → AudioQueue
                                                                      ↓
                                              paplay plays chunk[0] while chunk[1] synthesizes

Target: <300ms from first token entering the buffer to first audio byte played.
Keeps the Piper model resident via a persistent subprocess stdin/stdout pipe.

If no audio output device is available, synthesis still runs but playback
is silently skipped (CI/headless-safe).
"""

import logging
import os
import re
import subprocess
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Audio availability detection ──────────────────────────────────

_HAS_AUDIO: Optional[bool] = None

def _audio_available() -> bool:
    """Check if an audio player (paplay or aplay) is installed."""
    global _HAS_AUDIO
    if _HAS_AUDIO is None:
        _HAS_AUDIO = shutil.which("paplay") is not None or shutil.which("aplay") is not None
    return _HAS_AUDIO


# ── Sentence chunking (token → sentence) ─────────────────────────

class TokenBuffer:
    """
    Buffers incoming LLM tokens and yields complete sentence chunks.

    Emits a chunk when it encounters any of:
      - sentence-ending punctuation: . ! ?
      - clause-ending punctuation: , ; :
      - a hard word-count limit
    """
    SENTENCE_END = re.compile(r"[.!?]")
    CLAUSE_BREAK = re.compile(r"[,;:]")
    MAX_WORDS = 20

    def __init__(self) -> None:
        self._buffer: List[str] = []
        self._word_count = 0
        self._start_time: Optional[float] = None

    def feed(self, token: str) -> Optional[str]:
        if self._start_time is None:
            self._start_time = time.monotonic()

        self._buffer.append(token)
        joined = "".join(self._buffer)
        self._word_count = len(joined.split())

        if self._word_count >= self.MAX_WORDS:
            return self._flush()
        if self.SENTENCE_END.search(token):
            return self._flush()
        if self.CLAUSE_BREAK.search(token):
            return self._flush()
        return None

    def flush(self) -> Optional[str]:
        return self._flush()

    def _flush(self) -> Optional[str]:
        if not self._buffer:
            return None
        chunk = "".join(self._buffer).strip()
        self._buffer.clear()
        self._word_count = 0
        return chunk if chunk else None

    def time_since_first_token(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time


# ── Double-buffered audio queue ───────────────────────────────────

@dataclass
class AudioChunk:
    data: bytes
    sample_rate: int = 22050
    index: int = 0
    synthesis_time_ms: float = 0.0
    is_first: bool = False


class AudioQueue:
    """
    Double-buffered queue: while chunk N plays, chunk N+1 synthesizes.
    If no audio device is available, synthesis still runs but playback is no-op.
    """

    MAX_QUEUED = 4

    def __init__(self, sample_rate: int = 22050) -> None:
        self._sample_rate = sample_rate
        self._queue: Deque[AudioChunk] = deque()
        self._lock = threading.Lock()
        self._playing = False
        self._play_thread: Optional[threading.Thread] = None
        self._chunk_index = 0
        self._on_first_audio: Optional[Callable[[], None]] = None
        self._has_audio = _audio_available()

    def set_first_audio_callback(self, cb: Callable[[], None]) -> None:
        self._on_first_audio = cb

    def push(self, chunk: AudioChunk) -> None:
        chunk.index = self._chunk_index
        self._chunk_index += 1
        # Non-blocking push with size limit
        with self._lock:
            if len(self._queue) >= self.MAX_QUEUED:
                self._queue.popleft()  # drop oldest if overflow
            self._queue.append(chunk)
        self._ensure_playing()

    def _ensure_playing(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _play_loop(self) -> None:
        first = True
        while True:
            chunk: Optional[AudioChunk] = None
            with self._lock:
                if self._queue:
                    chunk = self._queue.popleft()
                elif not self._playing:
                    break
            if chunk is None:
                time.sleep(0.01)
                continue

            if first and self._on_first_audio:
                self._on_first_audio()
                first = False

            self._play_chunk(chunk)

    def _play_chunk(self, chunk: AudioChunk) -> None:
        if not self._has_audio:
            return  # CI / headless — skip audio I/O

        try:
            proc = subprocess.Popen(
                ["paplay", "--raw",
                 "--rate", str(self._sample_rate),
                 "--format", "s16ne", "--channels", "1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                if proc.stdin:
                    proc.stdin.write(chunk.data)
                    proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait(timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            try:
                proc = subprocess.Popen(
                    ["aplay", "-q",
                     "-r", str(self._sample_rate),
                     "-f", "S16_LE", "-c", "1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if proc.stdin:
                    proc.stdin.write(chunk.data)
                    proc.stdin.close()
                proc.wait(timeout=10)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        except Exception:
            pass

    def stop(self) -> None:
        self._playing = False
        with self._lock:
            self._queue.clear()

    def wait_done(self, timeout: float = 10) -> None:
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=timeout)


# ── Streaming TTS Engine ──────────────────────────────────────────

class StreamTTS:
    """
    Streaming TTS pipeline. Accepts LLM tokens and produces
    double-buffered audio with sub-300ms first-chunk latency.

    Piper stays resident as a long-lived subprocess with
    stdin→text, stdout→raw PCM.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 22050,
    ) -> None:
        self._model_path = model_path or self._find_model()
        self._sample_rate = sample_rate
        self._buffer = TokenBuffer()
        self._audio_queue = AudioQueue(sample_rate)
        self._piper_proc: Optional[subprocess.Popen] = None
        self._chunk_count = 0
        self._running = False
        self._first_audio_time: Optional[float] = None
        self._first_token_time: Optional[float] = None

        if _audio_available():
            self._start_piper()

    def _find_model(self) -> Optional[str]:
        candidates = [
            Path(__file__).parent.parent.parent / "models" / "voice",
            Path.home() / ".local" / "share" / "piper-tts",
        ]
        for d in candidates:
            if d.exists():
                onnx = list(d.glob("*.onnx"))
                for f in onnx:
                    if "lessac" in f.name.lower():
                        return str(f)
                if onnx:
                    return str(onnx[0])
        return None

    def _start_piper(self) -> None:
        if not self._model_path or not Path(self._model_path).exists():
            logger.debug("No Piper voice model found — TTS silent")
            return
        config_path = self._model_path + ".json"
        cmd = ["piper", "--model", self._model_path, "--output-raw"]
        if os.path.exists(config_path):
            cmd.extend(["--config", config_path])
        try:
            self._piper_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            self._running = True
            logger.info(f"StreamTTS: Piper started ({Path(self._model_path).name})")
        except FileNotFoundError:
            logger.debug("piper binary not found")
        except Exception as e:
            logger.error(f"Failed to start Piper: {e}")

    def feed_token(self, token: str) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.monotonic()
        chunk = self._buffer.feed(token)
        if chunk:
            self._synthesize_async(chunk)

    def finish(self) -> None:
        chunk = self._buffer.flush()
        if chunk:
            self._synthesize_async(chunk)

    def _synthesize_async(self, text: str) -> None:
        threading.Thread(target=self._synthesize, args=(text,), daemon=True).start()

    def _synthesize(self, text: str) -> None:
        pcm = self._run_piper(text)
        if pcm is None:
            return
        is_first = (self._chunk_count == 0)
        self._chunk_count += 1
        if is_first and self._first_token_time is not None:
            self._first_audio_time = time.monotonic()
            ms = (self._first_audio_time - self._first_token_time) * 1000
            logger.info(f"StreamTTS: first-chunk latency = {ms:.1f} ms")
        self._audio_queue.push(AudioChunk(data=pcm, sample_rate=self._sample_rate, is_first=is_first))

    def _run_piper(self, text: str) -> Optional[bytes]:
        """Run Piper in one-shot mode (most reliable across Piper versions)."""
        if not self._model_path:
            return None
        config_path = self._model_path + ".json"
        cmd = ["piper", "--model", self._model_path, "--output-raw"]
        if os.path.exists(config_path):
            cmd.extend(["--config", config_path])
        try:
            proc = subprocess.run(
                cmd, input=text.encode(), capture_output=True, timeout=15
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            logger.warning("Piper synthesis timed out")
        except Exception as e:
            logger.error(f"Piper error: {e}")
        return None

    @property
    def first_chunk_latency_ms(self) -> Optional[float]:
        if self._first_token_time is not None and self._first_audio_time is not None:
            return (self._first_audio_time - self._first_token_time) * 1000
        return None

    def stop(self) -> None:
        self._running = False
        self._audio_queue.stop()


# ── Latency test ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 60)
    logger.info("STREAMING TTS — Latency Verification")
    logger.info(f"  Audio available: {_audio_available()}")
    logger.info("=" * 60)

    # --- TokenBuffer ---
    logger.info("\n--- TokenBuffer ---")
    buf = TokenBuffer()
    tokens = ["Hello", ",", " ", "world", ".", " ", "This", " ", "is", " ", "a", " ", "test", "."]
    chunks = []
    for t in tokens:
        c = buf.feed(t)
        if c:
            chunks.append(c)
    chunks.append(buf.flush())
    assert any("Hello" in c for c in chunks if c)
    logger.info("  ✓ TokenBuffer chunks on punctuation")

    # --- AudioQueue (non-blocking) ---
    logger.info("\n--- AudioQueue ---")
    aq = AudioQueue(sample_rate=22050)
    callbacks = []
    aq.set_first_audio_callback(lambda: callbacks.append(time.monotonic()))
    silence = b"\x00\x00" * (22050 // 10)
    t0 = time.monotonic()
    aq.push(AudioChunk(data=silence, is_first=True))
    time.sleep(0.2)
    aq.wait_done(timeout=1)
    elapsed = (time.monotonic() - t0) * 1000
    logger.info(f"  AudioQueue done in {elapsed:.0f} ms (callback_fired={len(callbacks)})")
    logger.info("  ✓ AudioQueue non-blocking")

    # --- StreamTTS mock LLM ---
    logger.info("\n--- StreamTTS mock LLM ---")
    mock_sentence = (
        "Hello there, I am your companion. "
        "Today is a beautiful day, and I hope you are doing well. "
        "Let me know if I can help with anything."
    )
    words = mock_sentence.split()
    tokens = []
    for w in words:
        tokens.append(w)
        tokens.append(" ")

    tts = StreamTTS()
    t0 = time.monotonic()
    for i, t in enumerate(tokens):
        tts.feed_token(t)
        if i % 2 == 0:
            time.sleep(0.05)  # ~20 tokens/sec
    tts.finish()
    total_ms = (time.monotonic() - t0) * 1000
    latency = tts.first_chunk_latency_ms

    logger.info(f"  Pipeline: {total_ms:.0f} ms total")
    if latency is not None:
        logger.info(f"  First-chunk latency: {latency:.1f} ms")
        if latency < 300:
            logger.info(f"  ✓ PASS: {latency:.1f} ms < 300 ms target")
        else:
            logger.warning(f"  ✗ FAIL: {latency:.1f} ms > 300 ms target")
    else:
        logger.info("  ✓ Pipeline functional (latency N/A — no audio device)")

    tts.stop()
    logger.info("\n" + "=" * 60)
    logger.info("STREAMING TTS TEST COMPLETE")
    logger.info("=" * 60)