"""Short-term memory buffer for vision observations."""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """
    Single observation entry in memory buffer.
    
    Attributes:
        timestamp: Unix timestamp when observation was made
        description: Text description of what was observed
        metadata: Additional context (window, app, confidence, etc.)
        tokens: Estimated token count for budget management
    """
    timestamp: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S"),
            "description": self.description,
            "metadata": self.metadata,
            "tokens": self.tokens
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        dt = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f"[{dt}] {self.description[:50]}..."


class MemoryBuffer:
    """
    Rolling memory buffer for short-term temporal awareness.
    
    Features:
    - Stores text descriptions (not raw images) to save RAM
    - Rolling window: automatically prunes old observations
    - Token-limited: prevents RAM overflow
    - Time-limited: removes observations older than max_age
    - Chronological ordering maintained
    
    Memory Budget:
    - Max 50 observations (~10-50KB)
    - Max 2048 tokens (~1.5KB)
    - Max age 300s (5 minutes)
    """
    
    def __init__(
        self,
        max_observations: int = 50,
        max_tokens: int = 2048,
        max_age_seconds: float = 300.0
    ) -> None:
        """
        Initialize memory buffer.
        
        Args:
            max_observations: Maximum number of observations to store
            max_tokens: Maximum total tokens across all observations
            max_age_seconds: Maximum age of observations in seconds
        """
        self._max_observations = max_observations
        self._max_tokens = max_tokens
        self._max_age_seconds = max_age_seconds
        
        self._observations: List[Observation] = []
        self._total_tokens = 0
        
        logger.info(f"MemoryBuffer initialized: max_obs={max_observations}, "
                   f"max_tokens={max_tokens}, max_age={max_age_seconds}s")
    
    def add_observation(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        tokens: Optional[int] = None
    ) -> Observation:
        """
        Add new observation to buffer.
        
        Args:
            description: Text description of observation
            metadata: Additional context (window, app, confidence, etc.)
            tokens: Token count (estimated if not provided)
            
        Returns:
            Observation that was added
        """
        # Estimate tokens if not provided (rough: chars / 4)
        if tokens is None:
            tokens = max(1, len(description) // 4)
        
        # Create observation
        observation = Observation(
            timestamp=time.time(),
            description=description,
            metadata=metadata or {},
            tokens=tokens
        )
        
        # Add to buffer
        self._observations.append(observation)
        self._total_tokens += tokens
        
        logger.debug(f"Added observation: {str(observation)}")
        
        # Prune if necessary
        self._prune()
        
        return observation
    
    def get_recent(self, n: int = 5) -> List[Observation]:
        """
        Get last N observations.
        
        Args:
            n: Number of recent observations to return
            
        Returns:
            List of most recent observations (newest first)
        """
        return list(reversed(self._observations[-n:]))
    
    def get_context_window(self, max_tokens: int = 512) -> str:
        """
        Get recent observations within token limit as formatted text.
        
        Args:
            max_tokens: Maximum tokens to include
            
        Returns:
            Formatted text summary of recent observations
        """
        if not self._observations:
            return "No observations yet."
        
        # Start from most recent and work backwards
        selected = []
        total_tokens = 0
        
        for obs in reversed(self._observations):
            if total_tokens + obs.tokens > max_tokens:
                break
            
            selected.append(obs)
            total_tokens += obs.tokens
        
        # Format as text
        lines = []
        for obs in reversed(selected):  # Reverse back to chronological
            dt = datetime.fromtimestamp(obs.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{dt}] {obs.description}")
        
        return "\n".join(lines)
    
    def get_timeline(self, seconds: int = 60) -> List[Observation]:
        """
        Get observations from last N seconds.
        
        Args:
            seconds: Time window in seconds
            
        Returns:
            List of observations within time window (newest first)
        """
        cutoff_time = time.time() - seconds
        
        # Filter by time (most recent first)
        timeline = [
            obs for obs in reversed(self._observations)
            if obs.timestamp >= cutoff_time
        ]
        
        return timeline
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_observations": len(self._observations),
            "total_tokens": self._total_tokens,
            "oldest_age_s": (
                time.time() - self._observations[0].timestamp
                if self._observations else 0
            ),
            "newest_age_s": (
                time.time() - self._observations[-1].timestamp
                if self._observations else 0
            ),
            "max_observations": self._max_observations,
            "max_tokens": self._max_tokens,
            "max_age_seconds": self._max_age_seconds
        }
    
    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
        self._total_tokens = 0
        logger.debug("Buffer cleared")
    
    def _prune(self) -> None:
        """Prune old observations to stay within limits."""
        pruned_count = 0
        current_time = time.time()
        
        # Step 1: Remove by age
        cutoff_time = current_time - self._max_age_seconds
        while self._observations and self._observations[0].timestamp < cutoff_time:
            obs = self._observations.pop(0)
            self._total_tokens -= obs.tokens
            pruned_count += 1
        
        # Step 2: Remove by token limit (oldest first)
        while self._total_tokens > self._max_tokens and self._observations:
            obs = self._observations.pop(0)
            self._total_tokens -= obs.tokens
            pruned_count += 1
        
        # Step 3: Remove by count limit (oldest first)
        while len(self._observations) > self._max_observations:
            obs = self._observations.pop(0)
            self._total_tokens -= obs.tokens
            pruned_count += 1
        
        if pruned_count > 0:
            logger.debug(f"Pruned {pruned_count} observations")
    
    def __len__(self) -> int:
        """Return number of observations in buffer."""
        return len(self._observations)
    
    def __iter__(self):
        """Iterate over observations."""
        return iter(self._observations)


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("MEMORY BUFFER TEST")
    logger.info("=" * 60)
    
    # Create buffer
    buffer = MemoryBuffer(max_observations=10, max_tokens=100, max_age_seconds=60)
    
    # Test 1: Add observations
    logger.info("\n--- Test 1: Add observations ---")
    for i in range(5):
        buffer.add_observation(f"User is using application {i}", metadata={"app": f"app{i}"})
    
    assert len(buffer) == 5
    logger.info(f"✓ Added 5 observations, buffer size: {len(buffer)}")
    
    # Test 2: Get recent
    logger.info("\n--- Test 2: Get recent observations ---")
    recent = buffer.get_recent(3)
    assert len(recent) == 3
    logger.info(f"✓ Got {len(recent)} recent observations")
    for obs in recent:
        logger.info(f"  {obs}")
    
    # Test 3: Get context window
    logger.info("\n--- Test 3: Get context window ---")
    context = buffer.get_context_window(max_tokens=50)
    logger.info(f"Context window ({len(context)} chars):")
    logger.info(context)
    assert len(context) > 0
    logger.info("✓ Context window generated")
    
    # Test 4: Get timeline
    logger.info("\n--- Test 4: Get timeline (last 30s) ---")
    timeline = buffer.get_timeline(seconds=30)
    assert len(timeline) == 5
    logger.info(f"✓ Got {len(timeline)} observations from last 30s")
    
    # Test 5: Max observations limit
    logger.info("\n--- Test 5: Max observations limit ---")
    for i in range(20):
        buffer.add_observation(f"Observation {i}")
    
    assert len(buffer) <= 10, f"Buffer exceeded max: {len(buffer)}"
    logger.info(f"✓ Buffer limited to {len(buffer)} observations (max=10)")
    
    # Test 6: Token limit
    logger.info("\n--- Test 6: Token limit ---")
    stats = buffer.get_stats()
    logger.info(f"Total tokens: {stats['total_tokens']} (max={stats['max_tokens']})")
    assert stats["total_tokens"] <= 100
    logger.info("✓ Token limit respected")
    
    # Test 7: Age limit
    logger.info("\n--- Test 7: Age limit (simulated) ---")
    buffer2 = MemoryBuffer(max_observations=10, max_tokens=100, max_age_seconds=2)
    
    # Add observation
    buffer2.add_observation("Recent observation")
    assert len(buffer2) == 1
    logger.info("Added observation, waiting 3 seconds...")
    
    time.sleep(3)
    
    # Add another to trigger pruning
    buffer2.add_observation("New observation")
    # Old one should be pruned
    logger.info(f"Buffer size after age pruning: {len(buffer2)}")
    assert len(buffer2) == 1
    logger.info("✓ Old observation pruned by age")
    
    # Test 8: Clear
    logger.info("\n--- Test 8: Clear buffer ---")
    buffer.clear()
    assert len(buffer) == 0
    logger.info("✓ Buffer cleared")
    
    # Test 9: Stats
    logger.info("\n--- Test 9: Buffer statistics ---")
    for i in range(5):
        buffer.add_observation(f"Test observation {i}")
    
    stats = buffer.get_stats()
    logger.info(f"Stats: {stats}")
    assert stats["total_observations"] == 5
    logger.info("✓ Statistics correct")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL MEMORY BUFFER TESTS PASSED ✓")
    logger.info("=" * 60)