#!/usr/bin/env python3
"""Download and setup Moondream2 VLM model for offline use."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check if required packages are installed."""
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        logger.info("✓ All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("\nPlease install required packages:")
        logger.error("  pip install torch transformers huggingface_hub pillow")
        return False


def download_model(
    model_name: str = "vikhyatk/moondream2",
    output_dir: Path = Path("./models/moondream2"),
    force_download: bool = False
) -> bool:
    """
    Download Moondream2 model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save model
        force_download: Re-download even if model exists
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download, HfApi
        import torch
        
        # Check if already downloaded
        if output_dir.exists() and list(output_dir.glob("*")) and not force_download:
            logger.info(f"Model already exists at {output_dir}")
            logger.info("Use --force to re-download")
            return True
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("DOWNLOADING MOONDREAM2 MODEL")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Size: ~1.8GB (1.8B parameters, 4-bit quantized)")
        logger.info("")
        
        # Download model
        logger.info("Downloading model files (this may take a few minutes)...")
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {output_dir.absolute()}")
        logger.info("")
        logger.info("Model files:")
        for file in sorted(output_dir.glob("*")):
            size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
            logger.info(f"  {file.name}: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return False


def verify_model(model_dir: Path = Path("./models/moondream2")) -> bool:
    """
    Verify model files are complete and valid.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        True if model is valid, False otherwise
    """
    logger.info("Verifying model files...")
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return False
    
    # Check for essential files
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors"  # or model-00001-of-00002.safetensors
    ]
    
    missing_files = []
    for filename in essential_files:
        # Check for exact match or partial match (for sharded models)
        files = list(model_dir.glob(filename))
        if not files:
            # Try alternative patterns
            if filename == "model.safetensors":
                files = list(model_dir.glob("model-*.safetensors"))
        
        if not files:
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing essential files: {missing_files}")
        return False
    
    logger.info("✓ Model files verified")
    
    # Try to load model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Testing model load...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
            local_files_only=True
        )
        
        logger.info("✓ Model loads successfully")
        
        # Clean up
        del model
        del tokenizer
        
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False


def print_model_info(model_dir: Path = Path("./models/moondream2")) -> None:
    """Print information about the downloaded model."""
    if not model_dir.exists():
        logger.error(f"Model not found: {model_dir}")
        return
    
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    
    # Calculate total size
    total_size = 0
    file_count = 0
    
    for file in model_dir.rglob("*"):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            file_count += 1
    
    logger.info(f"Location: {model_dir.absolute()}")
    logger.info(f"Files: {file_count}")
    logger.info(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
    logger.info(f"Model: Moondream2 (1.8B parameters)")
    logger.info(f"Quantization: 4-bit (CPU-optimized)")
    logger.info(f"Offline-ready: Yes")
    logger.info("")
    logger.info("Usage:")
    logger.info("  from src.vision.vlm_engine import VLMEngine")
    logger.info("  engine = VLMEngine()")
    logger.info("  engine.load_model()")
    logger.info("  description = engine.describe_frame(frame_tensor)")


def main():
    """Main download script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Moondream2 VLM model for offline use"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./models/vlm"),
        help="Output directory for model (default: ./models/vlm)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing model"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MOONDREAM2 MODEL DOWNLOADER")
    logger.info("=" * 60)
    logger.info("")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Show info
    if args.info:
        print_model_info(args.output)
        sys.exit(0)
    
    # Verify model
    if args.verify:
        if verify_model(args.output):
            logger.info("✓ Model is valid and ready to use")
            sys.exit(0)
        else:
            logger.error("✗ Model verification failed")
            sys.exit(1)
    
    # Download model
    if download_model(
        model_name="vikhyatk/moondream2",
        output_dir=args.output,
        force_download=args.force
    ):
        logger.info("")
        
        # Verify after download
        if verify_model(args.output):
            logger.info("")
            print_model_info(args.output)
            logger.info("")
            logger.info("=" * 60)
            logger.info("READY TO USE")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.error("Download completed but verification failed")
            sys.exit(1)
    else:
        logger.error("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()