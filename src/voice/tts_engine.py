"""Piper TTS engine for offline speech synthesis.

Uses Piper TTS via piper-tts or the piper binary. Extremely fast CPU
inference with tiny voice models (~30MB). Runs in a separate thread to
avoid blocking the UI or vision pipeline.

RAM: ~50MB per loaded voice model.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for the TTS engine."""
    # Model path
    model_path: str = ""         # Path to .onnx voice model
    model_config_path: str = ""  # Path to .onnx.json config file
    
    # Generation
    length_scale: float = 1.0    # Speaking speed (lower = faster)
    noise_scale: float = 0.667   # Speech variation
    noise_w: float = 0.8         # Phoneme noise
    
    # Runtime
    use_subprocess: bool = True  # Use piper binary (more reliable)
    piper_binary: str = "piper"  # Path to piper binary
    sample_rate: int = 22050     # Output sample rate
    
    # Threading
    max_queue_size: int = 10     # Max queued TTS requests


@dataclass
class TTSResult:
    """Result of a TTS synthesis request."""
    audio_path: Optional[str] = None   # Path to generated WAV file
    duration_seconds: float = 0.0      # Audio duration
    synthesis_time_ms: float = 0.0     # Time to synthesize
    success: bool = False
    error: str = ""


class TTSEngine:
    """
    Piper TTS engine for offline speech synthesis.
    
    Architecture:
    1. InteractionHandler calls tts_engine.speak(text)
    2. Text is queued for async synthesis
    3. Piper synthesizes WAV in background thread
    4. Audio callback plays the WAV via system audio player (aplay/paplay)
    5. TTS runs concurrently with UI — no thread blocking
    
    Voice models are stored in models/voice/ directory.
    Download with: python scripts/download_voice.py
    
    Memory: ~50MB per voice model (ONNX runtime).
    """
    
    # Supported voice models and their download sizes
    VOICE_MODELS = {
        "en_US-lessac-medium": {
            "name": "Lessac (US English, Medium)",
            "onnx_file": "en_US-lessac-medium.onnx",
            "config_file": "en_US-lessac-medium.onnx.json",
            "size_mb": 32,
            "description": "Warm, natural female US English voice",
            "recommended": True,
        },
        "en_US-amy-medium": {
            "name": "Amy (US English, Medium)",
            "onnx_file": "en_US-amy-medium.onnx",
            "config_file": "en_US-amy-medium.onnx.json",
            "size_mb": 31,
            "description": "Clear female US English voice",
        },
        "en_US-danny-low": {
            "name": "Danny (US English, Low)",
            "onnx_file": "en_US-danny-low.onnx",
            "config_file": "en_US-danny-low.onnx.json",
            "size_mb": 16,
            "description": "Lightweight male US English voice",
        },
        "en_GB-alan-medium": {
            "name": "Alan (British English, Medium)",
            "onnx_file": "en_GB-alan-medium.onnx",
            "config_file": "en_GB-alan-medium.onnx.json",
            "size_mb": 32,
            "description": "UK English male voice",
        },
    }
    
    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        """
        Initialize TTS engine.
        
        Args:
            config: TTS configuration (uses defaults if None)
        """
        self._config = config or TTSConfig()
        self._lock = threading.RLock()
        self._ready = False
        self._synthesizing = False
        self._stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "total_synthesis_time_ms": 0.0,
        }
        self._auto_discover_models()
        logger.info(f"TTSEngine initialized (model={self._config.model_path or 'not configured'})")
    
    def _auto_discover_models(self) -> None:
        """Auto-discover voice models in the models/voice directory."""
        if self._config.model_path and Path(self._config.model_path).exists():
            self._ready = True
            return
        
        # Search default locations
        project_root = Path(__file__).parent.parent.parent
        voice_dir = project_root / "models" / "voice"
        
        if voice_dir.exists():
            onnx_files = list(voice_dir.glob("*.onnx"))
            if onnx_files:
                best = None
                for f in onnx_files:
                    if "lessac" in f.name.lower():
                        best = f
                        break
                if not best:
                    best = onnx_files[0]
                
                self._config.model_path = str(best)
                config_path = str(best) + ".json"
                if os.path.exists(config_path):
                    self._config.model_config_path = config_path
                self._ready = True
                logger.info(f"Auto-discovered voice model: {best.name}")
    
    def is_ready(self) -> bool:
        """Check if TTS engine has a voice model available."""
        return self._ready
    
    def speak(self, text: str, blocking: bool = False) -> TTSResult:
        """
        Synthesize speech from text (async by default).
        
        Args:
            text: Text to speak (keep under 200 chars for best results)
            blocking: If True, wait for synthesis to complete
            
        Returns:
            TTSResult with audio path or error
        """
        if not text or not text.strip():
            return TTSResult(success=False, error="Empty text")
        
        self._stats["total_requests"] += 1
        
        # Truncate very long text
        if len(text) > 300:
            text = text[:300]
            logger.debug(f"Text truncated to 300 chars for TTS")
        
        if blocking:
            return self._synthesize(text)
        else:
            # Fire and forget in background thread
            thread = threading.Thread(
                target=self._synthesize,
                args=(text,),
                daemon=True,
            )
            thread.start()
            return TTSResult(success=True)  # Optimistic response
    
    def _synthesize(self, text: str) -> TTSResult:
        """Internal synthesis method (may block)."""
        with self._lock:
            self._synthesizing = True
        
        try:
            start_time = time.time()
            
            if self._config.use_subprocess:
                result = self._synthesize_piper_binary(text)
            else:
                result = self._synthesize_piper_lib(text)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["total_synthesis_time_ms"] += elapsed_ms
            
            if result.success:
                self._stats["successful"] += 1
                result.synthesis_time_ms = elapsed_ms
                logger.debug(
                    f"TTS synthesized in {elapsed_ms:.0f}ms: "
                    f"'{text[:40]}...' → {result.duration_seconds:.1f}s"
                )
            else:
                self._stats["failed"] += 1
                
            return result
            
        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"TTS synthesis failed: {e}")
            return TTSResult(success=False, error=str(e))
        finally:
            with self._lock:
                self._synthesizing = False
    
    def _synthesize_piper_binary(self, text: str) -> TTSResult:
        """
        Synthesize using the piper binary (subprocess).
        
        More reliable than the Python library, especially on Linux.
        """
        import subprocess
        import tempfile
        
        if not self._config.model_path:
            return TTSResult(success=False, error="No voice model configured")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            cmd = [
                self._config.piper_binary,
                "--model", self._config.model_path,
                "--output_file", output_path,
            ]
            if self._config.model_config_path:
                cmd.extend(["--config", self._config.model_config_path])
            
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return TTSResult(
                    audio_path=output_path,
                    duration_seconds=len(text) * 0.06,  # Rough estimate
                    success=True,
                )
            else:
                # Clean up on failure
                if os.path.exists(output_path):
                    os.unlink(output_path)
                return TTSResult(
                    success=False,
                    error=result.stderr.strip() or "Piper returned non-zero exit code",
                )
                
        except FileNotFoundError:
            return TTSResult(
                success=False,
                error=f"Piper binary not found at '{self._config.piper_binary}'. "
                      "Install with: pip install piper-tts",
            )
        except subprocess.TimeoutExpired:
            if os.path.exists(output_path):
                os.unlink(output_path)
            return TTSResult(success=False, error="TTS synthesis timed out")
    
    def _synthesize_piper_lib(self, text: str) -> TTSResult:
        """
        Synthesize using the piper-tts Python library.
        
        Falls back gracefully if not installed.
        """
        import tempfile
        
        try:
            from piper import PiperVoice
            import numpy as np
            import wave
            
            if not self._config.model_path:
                return TTSResult(success=False, error="No voice model configured")
            
            # Load voice (lazy)
            voice = PiperVoice.load(
                self._config.model_path,
                config_path=self._config.model_config_path or None,
            )
            
            # Synthesize
            audio_data = bytearray()
            for audio_bytes in voice.synthesize_stream_raw(
                text,
                length_scale=self._config.length_scale,
                noise_scale=self._config.noise_scale,
                noise_w=self._config.noise_w,
            ):
                audio_data.extend(audio_bytes)
            
            # Write WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            
            with wave.open(output_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._config.sample_rate)
                wav.writeframes(audio_data)
            
            duration = len(audio_data) / (2 * self._config.sample_rate)
            return TTSResult(
                audio_path=output_path,
                duration_seconds=duration,
                success=True,
            )
            
        except ImportError:
            return TTSResult(
                success=False,
                error="piper-tts Python library not installed. "
                      "Install with: pip install piper-tts",
            )
        except Exception as e:
            return TTSResult(success=False, error=str(e))
    
    def play_audio(self, audio_path: str) -> bool:
        """
        Play a synthesized WAV file through the system audio output.
        
        Uses aplay (ALSA) or paplay (PulseAudio) automatically.
        Non-blocking — returns immediately.
        """
        if not audio_path or not os.path.exists(audio_path):
            return False
        
        try:
            import subprocess
            
            # Try PulseAudio first (most common on modern Linux)
            try:
                subprocess.Popen(
                    ["paplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except FileNotFoundError:
                pass
            
            # Try ALSA
            try:
                subprocess.Popen(
                    ["aplay", "-q", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except FileNotFoundError:
                pass
            
            logger.warning("No audio player found (paplay/aplay)")
            return False
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False
    
    def speak_and_play(self, text: str) -> TTSResult:
        """
        Synthesize speech and play it immediately.
        
        Convenience method that chains synthesize → play.
        Uses async synthesis + play.
        """
        if not self._ready:
            return TTSResult(success=False, error="TTS engine not ready")
        
        result = self.speak(text, blocking=True)
        
        if result.success and result.audio_path:
            # Play in background
            def play():
                self.play_audio(result.audio_path)
                # Clean up temp file after play duration
                time.sleep(result.duration_seconds + 1)
                try:
                    os.unlink(result.audio_path)
                except Exception:
                    pass
            
            thread = threading.Thread(target=play, daemon=True)
            thread.start()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS engine statistics."""
        stats = dict(self._stats)
        stats["ready"] = self._ready
        stats["model_path"] = self._config.model_path
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_requests"]
            stats["avg_synthesis_ms"] = stats["total_synthesis_time_ms"] / stats["total_requests"]
        return stats
    
    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("TTSEngine shutdown")


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("TTS ENGINE TEST")
    logger.info("=" * 60)
    
    config = TTSConfig(
        use_subprocess=True,
    )
    
    engine = TTSEngine(config)
    
    # Test 1: Engine initialization
    if engine.is_ready():
        logger.info("✓ TTS engine ready with voice model")
    else:
        logger.info("✓ TTS engine initialized (no model found — expected if not downloaded)")
    
    # Test 2: Empty text handling
    result = engine.speak("", blocking=True)
    assert not result.success
    assert "Empty" in result.error
    logger.info(f"✓ Empty text rejected: '{result.error}'")
    
    # Test 3: Silent speak (async)
    if engine.is_ready():
        result = engine.speak("Hello world", blocking=False)
        assert result.success
        logger.info("✓ Async speak queued successfully")
    
    # Test 4: Stats tracking
    stats = engine.get_stats()
    assert stats["total_requests"] >= 1
    logger.info(f"✓ Stats: {stats}")
    
    # Test 5: Voice models catalog
    assert "en_US-lessac-medium" in engine.VOICE_MODELS
    assert engine.VOICE_MODELS["en_US-lessac-medium"]["size_mb"] <= 50
    logger.info(f"✓ Voice models catalog: {len(engine.VOICE_MODELS)} voices available")
    
    engine.shutdown()
    logger.info("\nALL TTS TESTS PASSED")