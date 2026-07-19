"""Reasoning bridge between vision system and behavior tree."""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.vision.memory_buffer import MemoryBuffer, Observation
from src.brain.decisions import ScreenContext, DecisionType

logger = logging.getLogger(__name__)


@dataclass
class VisualReasoning:
    """
    Reasoning result from visual observation.
    
    This bridges the gap between raw VLM output and behavior tree decisions.
    """
    content_type: str  # "browser", "code", "video", "unknown", etc.
    confidence: float  # 0.0 to 1.0
    description: str  # Raw VLM description
    should_observe: bool  # Should companion continue observing?
    should_interact: bool  # Should companion interact with user?
    reasoning: str  # Human-readable reasoning
    
    def to_context_update(self) -> Dict[str, Any]:
        """Convert to ScreenContext update dict."""
        return {
            "content_type": self.content_type,
            "content_confidence": self.confidence
        }


class ReasoningBridge:
    """
    Bridges vision observations to behavior tree decisions.
    
    This module:
    1. Takes latest observation from MemoryBuffer
    2. Analyzes it with reasoning rules
    3. Returns VisualReasoning for behavior tree
    4. Enforces "Wait-and-See" protocol for low confidence
    """
    
    def __init__(self, memory_buffer: MemoryBuffer) -> None:
        """
        Initialize reasoning bridge.
        
        Args:
            memory_buffer: MemoryBuffer with visual observations
        """
        self._buffer = memory_buffer
        self._last_reasoning: Optional[VisualReasoning] = None
        
        logger.info("ReasoningBridge initialized")
    
    def reason(self, context: ScreenContext) -> Optional[VisualReasoning]:
        """
        Analyze latest visual observation and produce reasoning.
        
        Args:
            context: Current screen context
            
        Returns:
            VisualReasoning or None if no observations available
        """
        # Get latest observation
        recent = self._buffer.get_recent(n=1)
        
        if not recent:
            logger.debug("No observations available for reasoning")
            return None
        
        observation = recent[0]
        
        # Parse observation description
        reasoning = self._analyze_observation(observation, context)
        
        # Store for debugging
        self._last_reasoning = reasoning
        
        logger.debug(f"Reasoning: {reasoning.reasoning}")
        
        return reasoning
    
    def _analyze_observation(self, observation: Observation, context: ScreenContext) -> VisualReasoning:
        """
        Analyze a single observation and produce reasoning.
        
        Args:
            observation: Latest observation from buffer
            context: Current screen context
            
        Returns:
            VisualReasoning with analysis
        """
        description = observation.description.lower()
        confidence = observation.metadata.get("vlm_confidence", 0.5)
        
        # "Wait-and-See" Protocol: If confidence is low, don't act
        if confidence < 0.5 or description == "unknown":
            logger.debug(f"Low confidence ({confidence:.2f}) or UNKNOWN - aborting action")
            return VisualReasoning(
                content_type="unknown",
                confidence=confidence,
                description=observation.description,
                should_observe=True,  # Keep observing
                should_interact=False,  # Don't interact
                reasoning=f"Low confidence ({confidence:.2f}) - waiting for clearer observation"
            )
        
        # Classify content type
        content_type = self._classify_content(description)
        
        # Determine if companion should interact
        should_interact = self._should_interact(content_type, confidence, context)
        
        # Determine if companion should continue observing
        should_observe = self._should_continue_observing(content_type, context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(content_type, confidence, should_interact, context)
        
        return VisualReasoning(
            content_type=content_type,
            confidence=confidence,
            description=observation.description,
            should_observe=should_observe,
            should_interact=should_interact,
            reasoning=reasoning
        )
    
    def _classify_content(self, description: str) -> str:
        """
        Classify screen content from description.
        
        Args:
            description: VLM description text
            
        Returns:
            Content type string
        """
        # Keywords for classification
        browser_keywords = ["browser", "web", "website", "firefox", "chrome", "safari"]
        code_keywords = ["code", "editor", "vs code", "vim", "terminal", "programming"]
        video_keywords = ["video", "youtube", "netflix", "playing", "movie"]
        document_keywords = ["document", "pdf", "text", "writing", "word"]
        game_keywords = ["game", "playing", "steam", "gaming"]
        
        description_lower = description.lower()
        
        # Check each category
        if any(kw in description_lower for kw in browser_keywords):
            return "browser"
        elif any(kw in description_lower for kw in code_keywords):
            return "code"
        elif any(kw in description_lower for kw in video_keywords):
            return "video"
        elif any(kw in description_lower for kw in document_keywords):
            return "document"
        elif any(kw in description_lower for kw in game_keywords):
            return "game"
        else:
            return "unknown"
    
    def _should_interact(self, content_type: str, confidence: float, context: ScreenContext) -> bool:
        """
        Determine if companion should interact with user.
        
        Args:
            content_type: Classified content type
            confidence: VLM confidence
            context: Current screen context
            
        Returns:
            True if companion should interact
        """
        # Don't interact if user is idle
        if context.user_idle_time > 60.0:
            return False
        
        # Don't interact if confidence is low
        if confidence < 0.7:
            return False
        
        # Interact for specific content types
        interactive_content = ["browser", "document", "code"]
        
        return content_type in interactive_content
    
    def _should_continue_observing(self, content_type: str, context: ScreenContext) -> bool:
        """
        Determine if companion should continue observing.
        
        Args:
            content_type: Classified content type
            context: Current screen context
            
        Returns:
            True if companion should continue observing
        """
        # Continue observing if content is unknown
        if content_type == "unknown":
            return True
        
        # Continue observing if user is actively present
        if context.user_present and context.user_idle_time < 30.0:
            return True
        
        # Stop observing if user is idle or absent
        return False
    
    def _generate_reasoning(self, content_type: str, confidence: float, 
                           should_interact: bool, context: ScreenContext) -> str:
        """
        Generate human-readable reasoning.
        
        Args:
            content_type: Classified content type
            confidence: VLM confidence
            should_interact: Whether companion should interact
            context: Current screen context
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        # Content type
        reasons.append(f"Detected {content_type}")
        
        # Confidence
        reasons.append(f"confidence={confidence:.2f}")
        
        # User state
        if context.user_present:
            reasons.append(f"user present (idle {context.user_idle_time:.1f}s)")
        else:
            reasons.append("user absent")
        
        # Action
        if should_interact:
            reasons.append("→ will interact")
        else:
            reasons.append("→ will observe/wait")
        
        return ", ".join(reasons)
    
    def get_last_reasoning(self) -> Optional[VisualReasoning]:
        """Get last reasoning result."""
        return self._last_reasoning
    
    def should_trigger_vision_check(self, context: ScreenContext) -> bool:
        """
        Determine if vision check should be triggered.
        
        Active Perception: Only check vision when needed.
        
        Args:
            context: Current screen context
            
        Returns:
            True if vision check should be triggered
        """
        # Always check if in OBSERVE or INTERACT state
        if context.content_type in ["browser", "code", "document"]:
            return True
        
        # Check if user is present and active
        if context.user_present and context.user_idle_time < 30.0:
            return True
        
        return False


# Testing helper
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("REASONING BRIDGE TEST")
    logger.info("=" * 60)
    
    from src.vision.memory_buffer import MemoryBuffer, Observation
    
    # Create buffer with test observations
    buffer = MemoryBuffer(max_observations=10, max_tokens=500)
    
    # Add test observations
    buffer.add_observation(
        "Web Browser displaying example.com",
        metadata={"vlm_confidence": 0.9}
    )
    buffer.add_observation(
        "Text Editor with Python code",
        metadata={"vlm_confidence": 0.8}
    )
    buffer.add_observation(
        "UNKNOWN",
        metadata={"vlm_confidence": 0.3}
    )
    
    # Create reasoning bridge
    bridge = ReasoningBridge(memory_buffer=buffer)
    
    # Test reasoning
    logger.info("\n--- Test 1: Browser observation ---")
    context1 = ScreenContext(user_present=True, user_idle_time=5.0)
    reasoning1 = bridge.reason(context1)
    if reasoning1:
        logger.info(f"Content: {reasoning1.content_type}")
        logger.info(f"Confidence: {reasoning1.confidence:.2f}")
        logger.info(f"Should interact: {reasoning1.should_interact}")
        logger.info(f"Reasoning: {reasoning1.reasoning}")
    
    logger.info("\n--- Test 2: Code observation ---")
    buffer.add_observation(
        "Terminal with command line",
        metadata={"vlm_confidence": 0.85}
    )
    reasoning2 = bridge.reason(context1)
    if reasoning2:
        logger.info(f"Content: {reasoning2.content_type}")
        logger.info(f"Should interact: {reasoning2.should_interact}")
    
    logger.info("\n--- Test 3: Low confidence (UNKNOWN) ---")
    buffer.add_observation(
        "UNKNOWN",
        metadata={"vlm_confidence": 0.3}
    )
    reasoning3 = bridge.reason(context1)
    if reasoning3:
        logger.info(f"Content: {reasoning3.content_type}")
        logger.info(f"Should interact: {reasoning3.should_interact}")
        logger.info(f"Reasoning: {reasoning3.reasoning}")
    
    logger.info("\n" + "=" * 60)
    logger.info("REASONING BRIDGE TEST COMPLETE")
    logger.info("=" * 60)