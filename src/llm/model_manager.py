"""Model manager for downloading, verifying, and managing local LLM models."""

import logging
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str                          # Display name
    repo_id: str                       # Hugging Face repo ID
    filename: str                      # GGUF filename
    size_mb: int                       # Approximate file size in MB
    quantization: str                  # Quantization type (Q4_K_M, Q5_K_M, etc.)
    context_length: int                # Max context tokens
    description: str                   # Model description
    sha256: Optional[str] = None       # Expected SHA256 (if known)
    recommended: bool = False          # Whether this is a recommended model
    
    @property
    def estimated_ram_mb(self) -> int:
        """
        Estimated total RAM footprint at runtime (model + context).
        
        At Q4_K_M quantization, runtime RAM ≈ file size + ~15-20% overhead
        for KV cache and scratch buffers with n_ctx=2048.
        """
        overhead_mb = int(self.size_mb * 0.20) + 64  # 20% + 64MB for context
        return self.size_mb + overhead_mb


# Predefined model catalog - optimized for 16GB RAM systems
MODEL_CATALOG: Dict[str, ModelInfo] = {
    "qwen2.5-1.5b-q4": ModelInfo(
        name="Qwen2.5-1.5B-Instruct (Q4_K_M)",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_mb=1100,
        quantization="Q4_K_M",
        context_length=32768,
        description="Excellent multilingual, strong instruction following, very fast",
        recommended=True,
    ),
    "qwen2.5-1.5b-q5": ModelInfo(
        name="Qwen2.5-1.5B-Instruct (Q5_K_M)",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q5_k_m.gguf",
        size_mb=1300,
        quantization="Q5_K_M",
        context_length=32768,
        description="Higher quality than Q4, slightly slower",
        recommended=False,
    ),
    "llama-3.2-1b-q4": ModelInfo(
        name="Llama-3.2-1B-Instruct (Q4_K_M)",
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size_mb=850,
        quantization="Q4_K_M",
        context_length=131072,
        description="Meta's latest small model, huge context window",
        recommended=True,
    ),
    "llama-3.2-3b-q4": ModelInfo(
        name="Llama-3.2-3B-Instruct (Q4_K_M)",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_mb=1900,
        quantization="Q4_K_M",
        context_length=131072,
        description="Better quality than 1B, still fits in RAM",
        recommended=False,
    ),
    "phi-3.5-mini-q4": ModelInfo(
        name="Phi-3.5-mini-Instruct (Q4_K_M)",
        repo_id="microsoft/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        size_mb=2200,
        quantization="Q4_K_M",
        context_length=131072,
        description="Microsoft's strong small model, great reasoning",
        recommended=True,
    ),
    "gemma-2-2b-q4": ModelInfo(
        name="Gemma-2-2B-Instruct (Q4_K_M)",
        repo_id="bartowski/Gemma-2-2B-Instruct-GGUF",
        filename="Gemma-2-2B-Instruct-Q4_K_M.gguf",
        size_mb=1500,
        quantization="Q4_K_M",
        context_length=8192,
        description="Google's efficient 2B model",
        recommended=False,
    ),
}


class ModelManager:
    """
    Manages local LLM models - download, verify, list, and select.
    
    Handles:
    - Model catalog with metadata
    - Downloading from Hugging Face Hub
    - SHA256 verification
    - Disk space checking
    - Model selection for inference
    
    RAM Budget:
    - Moondream2 (VLM):      ~1,800 MB
    - Sub-3B LLM (Q4_K_M):   ~1,000-1,500 MB
    - Embedding model:         ~300 MB
    - Python/Qt overhead:      ~500 MB
    ─────────────────────────────────────
    TOTAL AI Stack:           ~3,600-4,100 MB
    Remaining (16GB system):  ~12,000 MB for OS + IDEs + apps
    """
    
    # Total system RAM assumption
    SYSTEM_RAM_MB = 16 * 1024  # 16GB
    # Reserve for OS, IDE, browser
    SYSTEM_RESERVE_MB = 4 * 1024  # 4GB base reserve
    # Estimated other AI components
    MOONDREAM_RAM_MB = 1800
    EMBEDDING_RAM_MB = 300
    QT_OVERHEAD_MB = 500
    
    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store models (default: ./models/llm)
        """
        if models_dir is None:
            # Default to project_root/models/llm
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models" / "llm"
        
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for downloaded model metadata
        self._cache_file = self._models_dir / ".model_cache.json"
        self._downloaded_models: Dict[str, Dict[str, Any]] = self._load_cache()
        
        # Check for user-specified local model path
        self._local_llm_path = os.environ.get("BUBBY_LLM_PATH", "").strip()
        if self._local_llm_path:
            local_path = Path(self._local_llm_path)
            logger.info(f"BUBBY_LLM_PATH env var is set to: {local_path}")
            if local_path.exists():
                logger.info(f"Loading local LLM from {local_path}")
            else:
                logger.warning(
                    f"No local GGUF model found at {local_path}. "
                    f"Please download a GGUF model and place it there, "
                    f"or set BUBBY_LLM_PATH to point to an existing .gguf file."
                )
        else:
            logger.info("BUBBY_LLM_PATH not set — will auto-select from catalog if a model is downloaded")
        
        logger.info(f"ModelManager initialized (dir: {self._models_dir})")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load downloaded models cache."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save downloaded models cache."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._downloaded_models, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    def list_available_models(self) -> List[ModelInfo]:
        """Get list of all available models in catalog."""
        return list(MODEL_CATALOG.values())
    
    def list_recommended_models(self) -> List[ModelInfo]:
        """Get list of recommended models."""
        return [m for m in MODEL_CATALOG.values() if m.recommended]
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """Get list of downloaded models with metadata."""
        result = []
        for model_id, info in self._downloaded_models.items():
            model_path = self._models_dir / info["filename"]
            result.append({
                "model_id": model_id,
                "name": info.get("name", model_id),
                "filename": info["filename"],
                "path": str(model_path),
                "exists": model_path.exists(),
                "size_mb": info.get("size_mb", 0),
                "quantization": info.get("quantization", "unknown"),
                "context_length": info.get("context_length", 0),
            })
        return result
    
    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        if model_id not in self._downloaded_models:
            return False
        model_path = self._models_dir / self._downloaded_models[model_id]["filename"]
        return model_path.exists()
    
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info from catalog."""
        return MODEL_CATALOG.get(model_id)
    
    def check_disk_space(self, required_mb: int) -> bool:
        """Check if there's enough disk space."""
        try:
            stat = shutil.disk_usage(self._models_dir)
            free_mb = stat.free / (1024 * 1024)
            return free_mb >= required_mb * 1.2  # 20% buffer
        except Exception:
            return True  # Assume OK if can't check
    
    def download_model(
        self,
        model_id: str,
        force: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_id: Model ID from catalog
            force: Re-download even if exists
            progress_callback: Optional callback(progress_pct, status_msg)
            
        Returns:
            True if download successful
        """
        model_info = MODEL_CATALOG.get(model_id)
        if not model_info:
            logger.error(f"Unknown model: {model_id}")
            return False
        
        # Check if already downloaded
        if not force and self.is_downloaded(model_id):
            logger.info(f"Model {model_id} already downloaded")
            if progress_callback:
                progress_callback(100, "Already downloaded")
            return True
        
        # Check disk space
        if not self.check_disk_space(model_info.size_mb):
            logger.error(f"Insufficient disk space (need {model_info.size_mb} MB)")
            return False
        
        model_path = self._models_dir / model_info.filename
        
        # Download using huggingface-hub or direct HTTP
        success = self._download_from_hf(
            model_info.repo_id,
            model_info.filename,
            model_path,
            progress_callback,
        )
        
        if success:
            # Verify file
            if self._verify_model(model_path, model_info):
                # Update cache
                self._downloaded_models[model_id] = {
                    "name": model_info.name,
                    "filename": model_info.filename,
                    "size_mb": model_info.size_mb,
                    "quantization": model_info.quantization,
                    "context_length": model_info.context_length,
                }
                self._save_cache()
                logger.info(f"Model {model_id} downloaded and verified")
                if progress_callback:
                    progress_callback(100, "Download complete and verified")
                return True
            else:
                logger.error(f"Model verification failed for {model_id}")
                if model_path.exists():
                    model_path.unlink()
        
        return False
    
    def _download_from_hf(
        self,
        repo_id: str,
        filename: str,
        dest_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Download file from Hugging Face Hub."""
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        
        logger.info(f"Downloading {filename} from {repo_id}")
        
        try:
            # Create request with user agent
            req = Request(url, headers={"User-Agent": "Bubby/1.0"})
            
            # Get file size for progress
            with urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                
                # Download with progress
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            pct = (downloaded / total_size) * 100
                            progress_callback(pct, f"Downloading... {downloaded//1024//1024}/{total_size//1024//1024} MB")
            
            logger.info(f"Download complete: {dest_path.name}")
            return True
            
        except HTTPError as e:
            logger.error(f"HTTP error downloading {filename}: {e.code} {e.reason}")
        except URLError as e:
            logger.error(f"URL error downloading {filename}: {e.reason}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
        
        # Cleanup on failure
        if dest_path.exists():
            dest_path.unlink()
        return False
    
    def _verify_model(self, model_path: Path, model_info: ModelInfo) -> bool:
        """Verify downloaded model file."""
        # Check file exists and has reasonable size
        if not model_path.exists():
            return False
        
        actual_size_mb = model_path.stat().st_size / (1024 * 1024)
        expected_size_mb = model_info.size_mb
        
        # Allow 20% variance
        if abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.2:
            logger.warning(
                f"Model size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.0f}MB"
            )
            # Don't fail - size can vary slightly
        
        # TODO: Add SHA256 verification when hashes are available
        if model_info.sha256:
            logger.info("Verifying SHA256...")
            actual_hash = self._compute_sha256(model_path)
            if actual_hash != model_info.sha256:
                logger.error(f"SHA256 mismatch: expected {model_info.sha256[:16]}..., got {actual_hash[:16]}...")
                return False
        
        logger.info(f"Model verified: {model_path.name} ({actual_size_mb:.0f} MB)")
        return True
    
    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model."""
        if not self.is_downloaded(model_id):
            logger.warning(f"Model {model_id} not downloaded")
            return False
        
        info = self._downloaded_models[model_id]
        model_path = self._models_dir / info["filename"]
        
        try:
            model_path.unlink()
            del self._downloaded_models[model_id]
            self._save_cache()
            logger.info(f"Removed model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            return False
    
    def get_best_model(self) -> Optional[str]:
        """Get the best available downloaded model (prefers recommended).
        
        Priority: BUBBY_LLM_PATH env var > recommended catalog model > any downloaded.
        """
        # Check BUBBY_LLM_PATH env var first
        if self._local_llm_path and Path(self._local_llm_path).exists():
            return None  # Signal to use direct path, not catalog lookup
        
        downloaded = self.list_downloaded_models()
        if not downloaded:
            return None
        
        # Prefer recommended models
        for model_id in ["qwen2.5-1.5b-q4", "llama-3.2-1b-q4", "phi-3.5-mini-q4"]:
            if self.is_downloaded(model_id):
                return model_id
        
        # Fallback: first downloaded
        return downloaded[0]["model_id"]

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to downloaded model.
        
        If BUBBY_LLM_PATH is set and the file exists, returns that path directly
        regardless of model_id.
        """
        # Priority: environment variable override
        if self._local_llm_path:
            env_path = Path(self._local_llm_path)
            if env_path.exists():
                return env_path
            else:
                logger.warning(
                    f"No local GGUF model found at {env_path}. "
                    f"Please download a GGUF model and place it there."
                )
                return None

        if not self.is_downloaded(model_id):
            return None
        info = self._downloaded_models[model_id]
        return self._models_dir / info["filename"]

    def get_local_llm_path(self) -> Optional[Path]:
        """Get the BUBBY_LLM_PATH value if set and the file exists."""
        if self._local_llm_path:
            p = Path(self._local_llm_path)
            if p.exists():
                return p
        return None
    
    def print_catalog(self) -> None:
        """Print model catalog to console."""
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)
        
        for model_id, info in MODEL_CATALOG.items():
            status = "✓ DOWNLOADED" if self.is_downloaded(model_id) else "  AVAILABLE"
            rec = " ★ RECOMMENDED" if info.recommended else ""
            
            print(f"\n{model_id} {status}{rec}")
            print(f"  Name:        {info.name}")
            print(f"  Size:        ~{info.size_mb} MB ({info.quantization})")
            print(f"  Context:     {info.context_length:,} tokens")
            print(f"  Description: {info.description}")
        
        print("\n" + "=" * 80)
        print("Downloaded models:")
        for m in self.list_downloaded_models():
            print(f"  ✓ {m['model_id']} ({m['size_mb']} MB)")
        
        best = self.get_best_model()
        if best:
            print(f"\nBest available: {best}")
        print("=" * 80)


# CLI helper
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage local LLM models")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--downloaded", action="store_true", help="List downloaded models")
    parser.add_argument("--download", type=str, help="Download model by ID")
    parser.add_argument("--remove", type=str, help="Remove downloaded model")
    parser.add_argument("--best", action="store_true", help="Show best available model")
    parser.add_argument("--models-dir", type=str, help="Custom models directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    models_dir = Path(args.models_dir) if args.models_dir else None
    manager = ModelManager(models_dir)
    
    if args.list:
        manager.print_catalog()
    elif args.downloaded:
        downloaded = manager.list_downloaded_models()
        if downloaded:
            for m in downloaded:
                print(f"✓ {m['model_id']} ({m['size_mb']} MB) - {m['path']}")
        else:
            print("No models downloaded")
    elif args.download:
        def progress(pct, msg):
            print(f"\r  [{pct:5.1f}%] {msg}", end="", flush=True)
        
        success = manager.download_model(args.download, progress_callback=progress)
        print()
        if success:
            print(f"✓ Successfully downloaded {args.download}")
        else:
            print(f"✗ Failed to download {args.download}")
            sys.exit(1)
    elif args.remove:
        success = manager.remove_model(args.remove)
        if success:
            print(f"✓ Removed {args.remove}")
        else:
            print(f"✗ Failed to remove {args.remove}")
    elif args.best:
        best = manager.get_best_model()
        if best:
            print(f"Best model: {best}")
        else:
            print("No models downloaded")
    else:
        manager.print_catalog()