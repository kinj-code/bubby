#!/usr/bin/env python3
"""
Download script for local LLM models.

Usage:
    python scripts/download_llm.py --list          # List available models
    python scripts/download_llm.py --download qwen2.5-1.5b-q4
    python scripts/download_llm.py --best           # Download best recommended model
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download local LLM models for Bubby")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--downloaded", action="store_true", help="List downloaded models")
    parser.add_argument("--download", type=str, help="Download model by ID")
    parser.add_argument("--remove", type=str, help="Remove downloaded model")
    parser.add_argument("--best", action="store_true", help="Download best recommended model")
    parser.add_argument("--models-dir", type=str, help="Custom models directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir) if args.models_dir else None
    manager = ModelManager(models_dir)
    
    if args.list:
        manager.print_catalog()
        return 0
    
    if args.downloaded:
        downloaded = manager.list_downloaded_models()
        if downloaded:
            print("\nDownloaded models:")
            for m in downloaded:
                status = "✓" if m["exists"] else "✗ MISSING"
                print(f"  {status} {m['model_id']} ({m['size_mb']} MB)")
                print(f"      Path: {m['path']}")
        else:
            print("No models downloaded")
        return 0
    
    if args.remove:
        def progress(pct, msg):
            print(f"\r  [{pct:5.1f}%] {msg}", end="", flush=True)
        
        success = manager.remove_model(args.remove)
        print()
        if success:
            print(f"✓ Removed {args.remove}")
        else:
            print(f"✗ Failed to remove {args.remove}")
            return 1
        return 0
    
    if args.best:
        # Download best available recommended model
        recommended = manager.list_recommended_models()
        if not recommended:
            print("No recommended models in catalog")
            return 1
        
        # Try each recommended model until one downloads
        for model in recommended:
            print(f"\nTrying recommended model: {model.name} ({model_id})...")
            
            # Find model_id
            model_id = None
            for mid, info in manager._downloaded_models.items():
                pass
            # Get from catalog
            for mid, info in manager.list_available_models():
                if info.name == model.name:
                    model_id = mid
                    break
            
            # Actually just use the known IDs
            for mid in ["qwen2.5-1.5b-q4", "llama-3.2-1b-q4", "phi-3.5-mini-q4"]:
                if mid in [m for m in manager._downloaded_models]:
                    continue
                print(f"Downloading {mid}...")
                def progress(pct, msg):
                    print(f"\r  [{pct:5.1f}%] {msg}", end="", flush=True)
                
                success = manager.download_model(mid, progress_callback=progress)
                print()
                if success:
                    print(f"✓ Successfully downloaded {mid}")
                    return 0
            
        print("Failed to download any recommended model")
        return 1
    
    if args.download:
        def progress(pct, msg):
            print(f"\r  [{pct:5.1f}%] {msg}", end="", flush=True)
        
        success = manager.download_model(args.download, force=args.force, progress_callback=progress)
        print()
        if success:
            print(f"✓ Successfully downloaded {args.download}")
        else:
            print(f"✗ Failed to download {args.download}")
            return 1
        return 0
    
    # Default: show catalog
    manager.print_catalog()
    return 0


if __name__ == "__main__":
    sys.exit(main())