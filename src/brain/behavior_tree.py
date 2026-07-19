"""Minimal behavior tree implementation for autonomous decision-making."""

import logging
from typing import List, Optional, Callable, Any
from abc import ABC, abstractmethod

from src.brain.decisions import NodeStatus, ScreenContext, Decision

logger = logging.getLogger(__name__)


class Node(ABC):
    """
    Base class for behavior tree nodes.
    
    All nodes must implement the evaluate() method which returns
    a NodeStatus (SUCCESS, FAILURE, or RUNNING).
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize node.
        
        Args:
            name: Human-readable name for debugging
        """
        self.name = name
        self._children: List['Node'] = []
    
    def add_child(self, child: 'Node') -> None:
        """Add a child node."""
        self._children.append(child)
        logger.debug(f"Added child '{child.name}' to '{self.name}'")
    
    @abstractmethod
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """
        Evaluate the node and return status.
        
        Args:
            context: Current screen and user context
            
        Returns:
            NodeStatus: SUCCESS, FAILURE, or RUNNING
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}({self.name})"


class Selector(Node):
    """
    Selector node (also called "Fallback").
    
    Tries children in order until one succeeds.
    Returns SUCCESS if any child succeeds.
    Returns FAILURE if all children fail.
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Try children in order until one succeeds."""
        logger.debug(f"Selector '{self.name}' evaluating {len(self._children)} children")
        
        for child in self._children:
            status = child.evaluate(context)
            
            if status == NodeStatus.SUCCESS:
                logger.debug(f"Selector '{self.name}' → SUCCESS (child '{child.name}' succeeded)")
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                logger.debug(f"Selector '{self.name}' → RUNNING (child '{child.name}' running)")
                return NodeStatus.RUNNING
        
        logger.debug(f"Selector '{self.name}' → FAILURE (all children failed)")
        return NodeStatus.FAILURE


class Sequence(Node):
    """
    Sequence node.
    
    Executes children in order until one fails.
    Returns SUCCESS if all children succeed.
    Returns FAILURE if any child fails.
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Execute children in order until one fails."""
        logger.debug(f"Sequence '{self.name}' evaluating {len(self._children)} children")
        
        for child in self._children:
            status = child.evaluate(context)
            
            if status == NodeStatus.FAILURE:
                logger.debug(f"Sequence '{self.name}' → FAILURE (child '{child.name}' failed)")
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                logger.debug(f"Sequence '{self.name}' → RUNNING (child '{child.name}' running)")
                return NodeStatus.RUNNING
        
        logger.debug(f"Sequence '{self.name}' → SUCCESS (all children succeeded)")
        return NodeStatus.SUCCESS


class Condition(Node):
    """
    Condition node.
    
    Checks a condition and returns SUCCESS or FAILURE.
    """
    
    def __init__(self, name: str, condition_func: Callable[[ScreenContext], bool]) -> None:
        """
        Initialize condition node.
        
        Args:
            name: Human-readable name
            condition_func: Function that takes context and returns bool
        """
        super().__init__(name)
        self.condition_func = condition_func
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Evaluate the condition."""
        try:
            result = self.condition_func(context)
            status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
            logger.debug(f"Condition '{self.name}' → {status.value} (result={result})")
            return status
        except Exception as e:
            logger.error(f"Condition '{self.name}' raised exception: {e}")
            return NodeStatus.FAILURE


class Action(Node):
    """
    Action node.
    
    Executes an action and returns SUCCESS or FAILURE.
    Can return RUNNING if the action takes time.
    
    Actions can optionally return a Decision by setting it on the context.
    """
    
    def __init__(
        self,
        name: str,
        action_func: Callable[[ScreenContext], NodeStatus],
        decision_type: Optional[Any] = None  # Optional DecisionType
    ) -> None:
        """
        Initialize action node.
        
        Args:
            name: Human-readable name
            action_func: Function that takes context and returns NodeStatus
            decision_type: Optional DecisionType to return on success
        """
        super().__init__(name)
        self.action_func = action_func
        self.decision_type = decision_type
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Execute the action."""
        try:
            logger.debug(f"Action '{self.name}' executing")
            status = self.action_func(context)
            logger.debug(f"Action '{self.name}' → {status.value}")
            
            # If action succeeded and has a decision type, store it in context
            if status == NodeStatus.SUCCESS and self.decision_type:
                context.last_decision_type = self.decision_type
                logger.debug(f"Action '{self.name}' set decision: {self.decision_type}")
            
            return status
        except Exception as e:
            logger.error(f"Action '{self.name}' raised exception: {e}")
            return NodeStatus.FAILURE


class Decorator(Node):
    """
    Decorator node base class.
    
    Wraps a single child node and modifies its behavior.
    """
    
    def __init__(self, name: str, child: Node) -> None:
        """
        Initialize decorator.
        
        Args:
            name: Human-readable name
            child: Child node to decorate
        """
        super().__init__(name)
        self._children = [child]
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Evaluate decorated child."""
        raise NotImplementedError


class Inverter(Decorator):
    """
    Inverts the result of the child node.
    
    SUCCESS → FAILURE
    FAILURE → SUCCESS
    RUNNING → RUNNING
    """
    
    def __init__(self, name: str, child: Node) -> None:
        super().__init__(name, child)
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Invert child's result."""
        child_status = self._children[0].evaluate(context)
        
        if child_status == NodeStatus.SUCCESS:
            logger.debug(f"Inverter '{self.name}' → FAILURE (inverted SUCCESS)")
            return NodeStatus.FAILURE
        elif child_status == NodeStatus.FAILURE:
            logger.debug(f"Inverter '{self.name}' → SUCCESS (inverted FAILURE)")
            return NodeStatus.SUCCESS
        else:
            logger.debug(f"Inverter '{self.name}' → RUNNING")
            return NodeStatus.RUNNING


class Repeater(Decorator):
    """
    Repeats the child node a specified number of times.
    
    If count is None, repeats indefinitely.
    """
    
    def __init__(self, name: str, child: Node, count: Optional[int] = None) -> None:
        """
        Initialize repeater.
        
        Args:
            name: Human-readable name
            child: Child node to repeat
            count: Number of times to repeat (None = infinite)
        """
        super().__init__(name, child)
        self.count = count
        self._current_count = 0
    
    def evaluate(self, context: ScreenContext) -> NodeStatus:
        """Repeat child node."""
        if self.count is not None and self._current_count >= self.count:
            logger.debug(f"Repeater '{self.name}' → SUCCESS (completed {self.count} iterations)")
            self._current_count = 0
            return NodeStatus.SUCCESS
        
        status = self._children[0].evaluate(context)
        
        if status == NodeStatus.SUCCESS:
            self._current_count += 1
            logger.debug(f"Repeater '{self.name}' iteration {self._current_count} complete")
            
            if self.count is None or self._current_count < self.count:
                # Continue repeating
                return NodeStatus.RUNNING
            else:
                # Done
                self._current_count = 0
                return NodeStatus.SUCCESS
        
        return status


class BehaviorTree:
    """
    Behavior tree for autonomous decision-making.
    
    The tree is evaluated periodically by the autonomy loop.
    Each evaluation returns a Decision that the UI can act upon.
    """
    
    def __init__(self, root: Node) -> None:
        """
        Initialize behavior tree.
        
        Args:
            root: Root node of the tree
        """
        self.root = root
        self._last_decision: Optional[Decision] = None
        self._evaluation_count = 0
        
        logger.info(f"BehaviorTree initialized with root: {root.name}")
    
    def evaluate(self, context: ScreenContext) -> Decision:
        """
        Evaluate the tree and return a decision.
        
        Args:
            context: Current screen and user context
            
        Returns:
            Decision object based on tree evaluation
        """
        self._evaluation_count += 1
        
        logger.debug(f"Tree evaluation #{self._evaluation_count}")
        logger.debug(f"Context: user_present={context.user_present}, "
                    f"idle_time={context.user_idle_time:.1f}s, "
                    f"window={context.active_window}")
        
        # Evaluate tree
        status = self.root.evaluate(context)
        
        # Map tree result to decision
        decision = self._status_to_decision(status, context)
        self._last_decision = decision
        
        logger.info(f"Tree decision: {decision}")
        return decision
    
    def _status_to_decision(self, status: NodeStatus, context: ScreenContext) -> Decision:
        """
        Convert tree evaluation status to a concrete decision.
        
        Args:
            status: NodeStatus from tree evaluation
            context: Current context
            
        Returns:
            Decision object
        """
        from src.brain.decisions import DecisionType, make_idle_decision
        
        # Check if an action set a specific decision type
        if hasattr(context, 'last_decision_type') and context.last_decision_type:
            decision_type = context.last_decision_type
            context.last_decision_type = None  # Reset for next evaluation
            
            # Map DecisionType to Decision
            if decision_type == DecisionType.GREET:
                from src.brain.decisions import make_greet_decision
                return make_greet_decision()
            elif decision_type == DecisionType.OBSERVE_SCREEN:
                from src.brain.decisions import make_observe_decision
                return make_observe_decision()
            elif decision_type == DecisionType.INTERACT:
                from src.brain.decisions import make_interact_decision
                return make_interact_decision()
            elif decision_type == DecisionType.WANDER:
                from src.brain.decisions import make_wander_decision
                return make_wander_decision()
            elif decision_type == DecisionType.PACE:
                from src.brain.decisions import make_pace_decision
                return make_pace_decision()
            elif decision_type == DecisionType.SIT:
                from src.brain.decisions import make_sit_decision
                return make_sit_decision()
            elif decision_type == DecisionType.SLEEP:
                from src.brain.decisions import make_sleep_decision
                return make_sleep_decision()
        
        # Default mapping based on status
        if status == NodeStatus.SUCCESS:
            # Default to idle if tree succeeds but doesn't specify action
            return make_idle_decision()
        elif status == NodeStatus.RUNNING:
            # Continue current action
            if self._last_decision:
                return self._last_decision
            return make_idle_decision()
        else:
            # Failure - default to idle
            return make_idle_decision()
    
    def get_stats(self) -> dict:
        """Get tree statistics."""
        return {
            "evaluations": self._evaluation_count,
            "last_decision": str(self._last_decision) if self._last_decision else None
        }


# Testing helper
if __name__ == "__main__":
    import sys
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=" * 60)
    logger.info("BEHAVIOR TREE TEST")
    logger.info("=" * 60)
    
    # Create a simple test tree
    # Root: Selector
    #   - Condition: User present?
    #   - Action: Idle
    
    def user_present_condition(context: ScreenContext) -> bool:
        """Check if user is present."""
        return context.user_present
    
    def idle_action(context: ScreenContext) -> NodeStatus:
        """Idle action."""
        logger.info("Executing idle action")
        return NodeStatus.SUCCESS
    
    def wander_action(context: ScreenContext) -> NodeStatus:
        """Wander action."""
        logger.info("Executing wander action")
        return NodeStatus.SUCCESS
    
    # Build tree
    user_present_cond = Condition("User Present?", user_present_condition)
    greet_action = Action("Greet", lambda ctx: NodeStatus.SUCCESS)
    idle_action_node = Action("Idle", idle_action)
    wander_action_node = Action("Wander", wander_action)
    
    # Priority 1: If user present, greet
    greet_sequence = Sequence("Greet Sequence")
    greet_sequence.add_child(user_present_cond)
    greet_sequence.add_child(greet_action)
    
    # Priority 2: Wander
    wander_action_node_eval = Action("Wander", wander_action)
    
    # Root selector
    root = Selector("Root")
    root.add_child(greet_sequence)
    root.add_child(wander_action_node_eval)
    root.add_child(idle_action_node)
    
    # Create tree
    tree = BehaviorTree(root)
    
    # Test with user present
    logger.info("\n--- Test 1: User Present ---")
    context1 = ScreenContext(user_present=True, user_idle_time=0.0)
    decision1 = tree.evaluate(context1)
    logger.info(f"Decision: {decision1}")
    
    # Test with user absent
    logger.info("\n--- Test 2: User Absent ---")
    context2 = ScreenContext(user_present=False, user_idle_time=5.0)
    decision2 = tree.evaluate(context2)
    logger.info(f"Decision: {decision2}")
    
    # Test stats
    stats = tree.get_stats()
    logger.info(f"\nTree stats: {stats}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Behavior tree test complete")
    logger.info("=" * 60)