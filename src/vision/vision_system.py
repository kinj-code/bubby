"""Complete vision system integrating pipeline, VLM, and memory buffer."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .pipeline import VisionPipeline, PipelineConfig
from .vlm_engine import VLMEngine, VLMConfig
from .memory_buffer import MemoryBuffer, Observation

logger = logging.getLogger(__name__)


@dataclass
class VisionSystemConfig:
    """Configuration for complete vision system."""
    # Pipeline config
    target_width: int = 224
    target_height: int = 224
    normalize: bool = True
    
    # VLM config
    vlm_model_path: Path = Path("./models/moondream2")
    vlm_max_tokens: int = 100
    vlm_temperature: float = 0.7
    
    # Memory buffer config
    max_observations: int = 50
    max_tokens: int = 2048
    max_age_seconds: float = 300.0
    
    # System settings
    auto_describe: bool = True  # Auto-generate descriptions from frames
    min_confidence: float = 0.5  # Minimum confidence to store observation


class VisionSystem:
    """
    Complete vision system: Capture → Pipeline → VLM → Memory Buffer.
    
    This is the main entry point for the companion's visual perception.
    It integrates all vision components into a unified interface.
    
    Flow:
    1. Accept raw frame from capture system
    2. Downsample and preprocess via VisionPipeline
    3. Generate description via VLMEngine (if enabled)
    4. Store in MemoryBuffer for temporal awareness
    
    All processing is 100% offline - no API calls.
    """
    
    def __init__(self, config: Optional[VisionSystemConfig] = None) -> None:
        """
        Initialize vision system.
        
        Args:
            config: System configuration (uses defaults if not provided)
        """
        self._config = config or VisionSystemConfig()
        
        # Initialize components
        logger.info("Initializing Vision System...")
        
        # Pipeline: Frame preprocessing
        pipeline_config = PipelineConfig(
            target_width=self._config.target_width,
            target_height=self._config.target_height,
            normalize=self._config.normalize,
            discard_raw_frame=True
        )
        self._pipeline = VisionPipeline(config=pipeline_config)
        
        # VLM: Description generation
        vlm_config = VLMConfig(
            model_path=self._config.vlm_model_path,
            max_tokens=self._config.vlm_max_tokens,
            temperature=self._config.vlm_temperature
        )
        self._vlm = VLMEngine(config=vlm_config)
        
        # Buffer: Short-term memory
        self._buffer = MemoryBuffer(
            max_observations=self._config.max_observations,
            max_tokens=self._config.max_tokens,
            max_age_seconds=self._config.max_age_seconds
        )
        
        self._is_initialized = False
        self._frames_processed = 0
        
        logger.info("Vision System initialized")
        logger.info(f"  Pipeline: {self._config.target_width}x{self._config.target_height}")
        logger.info(f"  VLM: {self._config.vlm_model_path}")
        logger.info(f"  Buffer: {self._config.max_observations} obs, {self._config.max_tokens} tokens")
    
    def initialize(self, load_vlm: bool = True) -> bool:
        """
        Initialize all components.
        
        Args:
            load_vlm: If True, load VLM model (requires model download)
            
        Returns:
            True if initialization successful
        """
        logger.info("Initializing vision system components...")
        
        # Pipeline is always ready (no model needed)
        logger.info("✓ Pipeline ready")
        
        # Load VLM if requested
        if load_vlm:
            if self._vlm.load_model():
                logger.info("✓ VLM loaded")
            else:
                logger.warning("✗ VLM failed to load - descriptions will be generic")
                logger.warning("  Run: python scripts/download_vlm.py")
        else:
            logger.info("⊘ VLM loading skipped")
        
        self._is_initialized = True
        logger.info("Vision System initialization complete")
        return True
    
    def process_frame(
        self,
        raw_frame: Any,
        generate_description: bool = True
    ) -> Optional[Observation]:
        """
        Process a single frame through the complete vision pipeline.
        
        Args:
            raw_frame: Raw screen capture (numpy array, PIL Image, etc.)
            generate_description: If True, use VLM to generate description
                                 If False, use generic description
            
        Returns:
            Observation object with description and metadata
        """
        if not self._is_initialized:
            logger.error("Vision system not initialized. Call initialize() first.")
            return None
        
        try:
            # Step 1: Preprocess frame
            logger.debug("Processing frame through pipeline...")
            processed = self._pipeline.preprocess_frame(raw_frame)
            
            # Step 2: Generate description
            if generate_description and self._vlm.get_stats()["is_loaded"]:
                logger.debug("Generating description with VLM...")
                description = self._vlm.describe_frame(processed)
            else:
                description = f"Screen capture #{self._frames_processed}"
            
            # Step 3: Create observation with metadata
            metadata = {
                "frame_number": self._frames_processed,
                "processed_shape": list(processed.shape),
                "vlm_model": self._vlm.get_stats().get("model_name", "none"),
                "pipeline_avg_time_ms": self._pipeline.get_stats().get("avg_processing_time_ms", "0")
            }
            
            # Step 4: Store in memory buffer
            observation = self._buffer.add_observation(
                description=description,
                metadata=metadata
            )
            
            self._frames_processed += 1
            
            logger.debug(f"Frame {self._frames_processed} processed and stored")
            
            return observation
            
        except Exception as e:
            logger.error(f"Failed to process frame: {e}", exc_info=True)
            return None
    
    def get_recent_context(self, n: int = 5) -> str:
        """
        Get recent observations as context string.
        
        Args:
            n: Number of recent observations to include
            
        Returns:
            Formatted context string
        """
        return self._buffer.get_context_window(max_tokens=512)
    
    def get_recent_observations(self, n: int = 5):
        """
        Get recent observations.
        
        Args:
            n: Number of recent observations
            
        Returns:
            List of Observation objects (newest first)
        """
        return self._buffer.get_recent(n=n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        pipeline_stats = self._pipeline.get_stats()
        vlm_stats = self._vlm.get_stats()
        buffer_stats = self._buffer.get_stats()
        
        return {
            "frames_processed": self._frames_processed,
            "is_initialized": self._is_initialized,
            "pipeline": pipeline_stats,
            "vlm": vlm_stats,
            "buffer": buffer_stats
        }
    
    def reset(self) -> None:
        """Reset all components."""
        self._pipeline.reset_stats()
        self._buffer.clear()
        self._frames_processed = 0
        logger.info("Vision System reset")


# Convenience function for quick integration
def create_vision_system(
    load_vlm: bool = True,
    vlm_model_path: Path = Path("./models/moondream2")
) -> VisionSystem:
    """
    Create and initialize vision system with sensible defaults.
    
    Args:
        load_vlm: If True, load VLM model
        vlm_model_path: Path to VLM model directory
        
    Returns:
        Initialized VisionSystem
    """
    config = VisionSystemConfig(vlm_model_path=vlm_model_path)
    system = VisionSystem(config=config)
    system.initialize(load_vlm=load_vlm)
    return system


# Testing helper
if __name__ == "__main__":
    import sys
    import numpy as np
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("VISION SYSTEM TEST")
    logger.info("=" * 60)
    
    # Create system (without VLM for quick test)
    logger.info("\n--- Test 1: Initialize without VLM ---")
    system = VisionSystem()
    system.initialize(load_vlm=False)
    logger.info("✓ System initialized")
    
    # Process dummy frames
    logger.info("\n--- Test 2: Process frames ---")
    for i in range(3):
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        obs = system.process_frame(dummy_frame, generate_description=False)
        
        if obs:
            logger.info(f"Frame {i+1}: {obs.description}")
    
    # Get stats
    logger.info("\n--- Test 3: System stats ---")
    stats = system.get_stats()
    logger.info(f"Frames processed: {stats['frames_processed']}")
    logger.info(f"Buffer size: {stats['buffer']['total_observations']}")
    logger.info(f"VLM loaded: {stats['vlm']['is_loaded']}")
    
    # Get context
    logger.info("\n--- Test 4: Recent context ---")
    context = system.get_recent_context(n=3)
    logger.info(f"Context:\n{context}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VISION SYSTEM TEST COMPLETE")
    logger.info("=" * 60)
    logger.info("\nTo use with VLM:")
    logger.info("  1. python scripts/download_vlm.py")
    logger.info("  2. system = create_vision_system(load_vlm=True)")
    logger.info("  3. obs = system.process_frame(frame, generate_description=True)")