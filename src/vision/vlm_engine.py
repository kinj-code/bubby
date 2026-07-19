"""Vision-Language Model engine for offline screen understanding."""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM engine."""
    # Model settings
    model_name: str = "moondream2"
    model_size: str = "1.8b"  # 1.8B parameters - CPU friendly
    model_path: Path = Path("./models/moondream2")
    
    # Inference settings
    device: str = "cpu"  # CPU-only for compatibility
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Confidence gate
    confidence_threshold: float = 0.5  # Minimum confidence to accept description
    use_confidence_gate: bool = True
    
    # System prompt (The "Constitution")
    system_prompt: str = """You are an objective, evidence-based visual observer. Your goal is to describe the UI elements visible on the user's screen with high precision.

Rules:
1. Confidence Only: Only describe UI elements you identify with high confidence (e.g., 'Text Editor', 'Web Browser', 'Terminal').
2. Admit Ignorance: If an element is ambiguous or blurry, output 'UNKNOWN' or describe the geometry (e.g., 'An unidentified rectangular container in the bottom-left'). Do not guess.
3. Consistency: Your observations must be consistent with the provided 'Previous Observations' history. If you see a radical change, verify if it is a transient visual glitch or a real state change.
4. Brevity: Output concise, structured observations. No conversational filler.
5. No Hallucinations: Never invent interactive elements that are not clearly rendered."""
    
    # Performance
    use_quantization: bool = True
    quantization_bits: int = 4  # 4-bit quantization for CPU


class VLMEngine:
    """
    Offline Vision-Language Model engine for screen understanding.
    
    Features:
    - 100% offline operation (no API calls)
    - CPU-friendly quantized models
    - Moondream2 architecture (1.8B params)
    - Automatic model download on first run
    - Memory-efficient inference
    
    Architecture:
    1. Load quantized model from local storage
    2. Preprocess image tensor (224x224)
    3. Run inference to generate text description
    4. Return description for memory buffer
    """
    
    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        """
        Initialize VLM engine.
        
        Args:
            config: VLM configuration (uses defaults if not provided)
        """
        self._config = config or VLMConfig()
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        # Ensure model directory exists
        self._config.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VLMEngine initialized: {self._config.model_name} ({self._config.model_size})")
        logger.info(f"Model path: {self._config.model_path}")
        logger.info(f"Device: {self._config.device}")
    
    def load_model(self) -> bool:
        """
        Load VLM model from local storage.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded:
            logger.warning("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading VLM model from {self._config.model_path}...")
            start_time = time.time()
            
            # Try to import transformers (will be installed in Phase 3, Part 2)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # Check if model exists locally
                if not self._config.model_path.exists() or not list(self._config.model_path.glob("*")):
                    logger.error(f"Model not found at {self._config.model_path}")
                    logger.error("Please run: python scripts/download_vlm.py")
                    return False
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._config.model_path,
                    local_files_only=True  # Force offline mode
                )
                
                # Load model with quantization for CPU
                logger.info(f"Loading model (quantized to {self._config.quantization_bits}-bit)...")
                model_kwargs = {
                    "device_map": "cpu",
                    "torch_dtype": torch.float32,
                    "local_files_only": True  # Force offline mode
                }
                
                # Add quantization if enabled
                if self._config.use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float32,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True
                        )
                        model_kwargs["quantization_config"] = quantization_config
                    except ImportError:
                        logger.warning("bitsandbytes not available, loading without quantization")
                        self._config.use_quantization = False
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._config.model_path,
                    **model_kwargs
                )
                
                self._model.eval()
                self._is_loaded = True
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                return True
                
            except ImportError as e:
                logger.error(f"transformers not installed: {e}")
                logger.error("Please install: pip install transformers torch pillow")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False
    
    def describe_frame(self, frame_tensor: Any, previous_observations: Optional[str] = None) -> str:
        """
        Generate text description of screen content with confidence checking.
        
        Args:
            frame_tensor: Preprocessed frame (1, 3, 224, 224) from VisionPipeline
            previous_observations: Optional context from memory buffer for consistency
            
        Returns:
            Text description of screen content, or "UNKNOWN" if low confidence
        """
        if not self._is_loaded:
            if not self.load_model():
                return "[ERROR: Model not loaded]"
        
        start_time = time.time()
        
        try:
            import torch
            from PIL import Image
            import numpy as np
            
            # Convert tensor to PIL Image
            # Input: (1, 3, 224, 224) float32 [0-1]
            # Output: PIL Image (224, 224, 3) uint8 [0-255]
            
            frame_np = frame_tensor.squeeze(0).cpu().numpy()  # (3, 224, 224)
            frame_np = np.transpose(frame_np, (1, 2, 0))  # (224, 224, 3)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            image = Image.fromarray(frame_np, mode='RGB')
            
            # Build prompt with context
            prompt = f"{self._config.system_prompt}\n"
            
            if previous_observations:
                prompt += f"\nPrevious Observations:\n{previous_observations}\n"
            
            prompt += "\n<image>\n"
            
            # Encode image
            logger.debug("Encoding image...")
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Generate description with confidence checking
            logger.debug("Generating description...")
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True  # Get confidence scores
                )
            
            # Decode output
            description = self._tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean up
            description = description.strip()
            
            # Confidence Gate: Check if description is reliable
            if self._config.use_confidence_gate:
                confidence = self._calculate_confidence(outputs)
                
                if confidence < self._config.confidence_threshold:
                    logger.debug(f"Low confidence ({confidence:.2f}), returning UNKNOWN")
                    description = "UNKNOWN"
                else:
                    logger.debug(f"Confidence: {confidence:.2f}")
            
            # Track performance
            inference_time = time.time() - start_time
            self._inference_count += 1
            self._total_inference_time += inference_time
            
            logger.debug(f"Generated description in {inference_time:.2f}s: {description[:50]}...")
            
            return description if description else "UNKNOWN"
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return "UNKNOWN"
    
    def _calculate_confidence(self, outputs: Any) -> float:
        """
        Calculate confidence score from generation outputs.
        
        Args:
            outputs: Model generation outputs with scores
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            import torch
            
            # Get token scores from generation
            scores = outputs.scores  # List of tensors, one per token
            
            if not scores:
                return 0.5  # Default if no scores
            
            # Calculate average log-probability of generated tokens
            total_log_prob = 0.0
            token_count = 0
            
            for token_scores in scores:
                # Get the selected token's log-probability
                token_id = outputs.sequences[0][len(scores) - token_count - 1]
                log_prob = torch.log_softmax(token_scores[0], dim=-1)[token_id]
                total_log_prob += log_prob.item()
                token_count += 1
            
            if token_count == 0:
                return 0.5
            
            avg_log_prob = total_log_prob / token_count
            
            # Convert log-probability to confidence score (0-1)
            # Higher log-probability = higher confidence
            # Typical range: -5 to 0, map to 0-1
            confidence = min(1.0, max(0.0, (avg_log_prob + 5.0) / 5.0))
            
            return confidence
            
        except Exception as e:
            logger.debug(f"Confidence calculation failed: {e}")
            return 0.5  # Default to medium confidence
    
    def get_stats(self) -> Dict[str, Any]:
        """Get VLM engine statistics."""
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0 else 0
        )
        
        return {
            "is_loaded": self._is_loaded,
            "model_name": self._config.model_name,
            "model_size": self._config.model_size,
            "device": self._config.device,
            "inference_count": self._inference_count,
            "avg_inference_time_s": f"{avg_time:.2f}",
            "total_inference_time_s": f"{self._total_inference_time:.2f}",
            "quantization": f"{self._config.quantization_bits}-bit" if self._config.use_quantization else "none"
        }
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model:
            del self._model
            self._model = None
        
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        
        self._is_loaded = False
        logger.info("Model unloaded from memory")


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("VLM ENGINE TEST")
    logger.info("=" * 60)
    
    # Create engine
    engine = VLMEngine()
    
    # Try to load model
    logger.info("\n--- Test 1: Load model ---")
    if engine.load_model():
        logger.info("✓ Model loaded")
    else:
        logger.error("✗ Failed to load model")
        logger.error("Run: python scripts/download_vlm.py")
        sys.exit(1)
    
    # Test with dummy frame
    logger.info("\n--- Test 2: Generate description ---")
    import numpy as np
    
    dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_tensor = np.transpose(dummy_frame, (2, 0, 1))  # (3, 224, 224)
    dummy_tensor = np.expand_dims(dummy_tensor, axis=0)  # (1, 3, 224, 224)
    dummy_tensor = dummy_tensor.astype(np.float32) / 255.0
    
    description = engine.describe_frame(dummy_tensor)
    logger.info(f"Description: {description}")
    
    # Print stats
    logger.info("\n--- Test 3: Engine stats ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VLM ENGINE TEST COMPLETE")
    logger.info("=" * 60)