#!/usr/bin/env python3
"""
Download script for Piper TTS voice models.

Downloads lightweight ONNX voice models (~16-32MB each) from the
Hugging Face Piper voices repository.

Usage:
    python scripts/download_voice.py --list          # List available voices
    python scripts/download_voice.py --download en_US-lessac-medium
    python scripts/download_voice.py --best           # Download recommended voice
"""

import sys
import os
import logging
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Piper voice model URLs (Hugging Face)
PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

VOICE_CATALOG = {
    "en_US-lessac-medium": {
        "name": "Lessac (US English, Medium)",
        "onnx_url": f"{PIPER_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "size_mb": 32,
        "description": "Warm, natural female US English voice — RECOMMENDED",
        "recommended": True,
    },
    "en_US-amy-medium": {
        "name": "Amy (US English, Medium)",
        "onnx_url": f"{PIPER_BASE_URL}/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "size_mb": 31,
        "description": "Clear female US English voice",
    },
    "en_US-danny-low": {
        "name": "Danny (US English, Low)",
        "onnx_url": f"{PIPER_BASE_URL}/en/en_US/danny/low/en_US-danny-low.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_US/danny/low/en_US-danny-low.onnx.json",
        "size_mb": 16,
        "description": "Lightweight male US English voice (~16MB)",
    },
    "en_GB-alan-medium": {
        "name": "Alan (British English, Medium)",
        "onnx_url": f"{PIPER_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "config_url": f"{PIPER_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        "size_mb": 32,
        "description": "UK English male voice",
    },
}


def download_file(url: str, dest: Path, progress_label: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        req = Request(url, headers={"User-Agent": "Bubby/1.0"})
        with urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  [{pct:5.1f}%] {progress_label} "
                              f"({downloaded // 1024 // 1024}/{total_size // 1024 // 1024} MB)",
                              end="", flush=True)
        print()
        return True
    except HTTPError as e:
        logger.error(f"HTTP error: {e.code} {e.reason}")
    except URLError as e:
        logger.error(f"URL error: {e.reason}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
    return False


def download_voice(voice_id: str, models_dir: Path) -> bool:
    """Download a voice model and its config."""
    voice_info = VOICE_CATALOG.get(voice_id)
    if not voice_info:
        logger.error(f"Unknown voice: {voice_id}")
        return False

    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {voice_info['name']}...")
    logger.info(f"  Size: ~{voice_info['size_mb']} MB")

    # Download ONNX model
    onnx_path = models_dir / f"{voice_id}.onnx"
    print(f"\n  Model:", end=" ")
    if not download_file(
        voice_info["onnx_url"],
        onnx_path,
        f"ONNX model ({voice_info['size_mb']} MB)",
    ):
        if onnx_path.exists():
            onnx_path.unlink()
        return False

    # Download config
    config_path = models_dir / f"{voice_id}.onnx.json"
    print("  Config:", end=" ")
    if not download_file(
        voice_info["config_url"],
        config_path,
        "Config",
    ):
        logger.warning("Config download failed — voice may still work without it")

    logger.info(f"✓ Voice model downloaded: {onnx_path}")
    return True


def check_piper_installed() -> bool:
    """Check if Piper TTS is available."""
    import shutil
    if shutil.which("piper"):
        return True
    try:
        import piper
        return True
    except ImportError:
        pass
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Piper TTS voice models")
    parser.add_argument("--list", action="store_true", help="List available voices")
    parser.add_argument("--download", type=str, help="Download voice by ID")
    parser.add_argument("--best", action="store_true", help="Download recommended voice")
    parser.add_argument("--models-dir", type=str, help="Custom models directory")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else project_root / "models" / "voice"

    if args.list:
        print("\n" + "=" * 70)
        print("AVAILABLE PIPER TTS VOICES")
        print("=" * 70)
        for voice_id, info in VOICE_CATALOG.items():
            rec = " ★ RECOMMENDED" if info.get("recommended") else ""
            downloaded = (models_dir / f"{voice_id}.onnx").exists()
            status = "✓ DOWNLOADED" if downloaded else "  AVAILABLE"
            print(f"\n  {voice_id} {status}{rec}")
            print(f"    Name:  {info['name']}")
            print(f"    Size:  ~{info['size_mb']} MB")
            print(f"    Desc:  {info['description']}")
        print("\n" + "=" * 70)

        if not check_piper_installed():
            logger.warning("Piper TTS is not installed!")
            logger.info("Install with: pip install piper-tts")
        return 0

    if args.best:
        for voice_id, info in VOICE_CATALOG.items():
            if info.get("recommended"):
                success = download_voice(voice_id, models_dir)
                if success:
                    logger.info(f"✓ Recommended voice '{voice_id}' downloaded")
                    if not check_piper_installed():
                        logger.info("Install piper-tts: pip install piper-tts")
                    return 0
        logger.error("No recommended voice found")
        return 1

    if args.download:
        success = download_voice(args.download, models_dir)
        if success:
            logger.info(f"✓ Voice '{args.download}' downloaded")
            if not check_piper_installed():
                logger.info("Install piper-tts: pip install piper-tts")
        else:
            logger.error(f"Failed to download voice '{args.download}'")
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())