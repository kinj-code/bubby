# Phase 2: The Brain & Autonomy - Architecture Plan

## Overview
Building the offline decision-making core for Bubby. The companion will autonomously decide between idling, wandering, observing the screen, or interacting with the user.

---

## 🏗️ Architecture Decision: Custom Minimal Behavior Tree

### Recommendation: **Custom Implementation** (No External Library)

**Rationale:**
1. **Minimal Footprint**: py_trees adds ~2MB+ and unnecessary complexity
2. **Full Control**: Custom implementation tailored to our specific needs
3. **Learning Curve**: Simpler to debug and extend
4. **No Bloat**: Only implement nodes we actually use
5. **Performance**: Lighter weight, faster execution

### Behavior Tree Structure

```
Root (Selector)
├── [Priority 1] Emergency Actions
│   ├── Condition: Is User Present?
│   └── Action: Greet User
│
├── [Priority 2] Contextual Awareness
│   ├── Condition: Screen State Changed?
│   ├── Action: Observe Screen
│   └── Action: React to Content
│
├── [Priority 3] Autonomous Behavior
│   ├── Sequence: Idle Timer Exceeded?
│   │   ├── Condition: Idle Time > Threshold
│   │   ├── Selector: Choose Activity
│   │   │   ├── Action: Wander
│   │   │   ├── Action: Pace
│   │   │   └── Action: Sit Down
│   │   └── Action: Play Idle Animation
│
└── [Priority 4] Default
    └── Action: Idle Animation
```

### Core Node Types (Minimal Set)

```python
class Node:
    """Base behavior tree node"""
    status: Status  # SUCCESS, FAILURE, RUNNING
    
class Selector(Node):
    """Try children in order until one succeeds"""
    
class Sequence(Node):
    """Execute children in order until one fails"""
    
class Condition(Node):
    """Check a state/condition"""
    
class Action(Node):
    """Execute a behavior"""
    
class Decorator(Node):
    """Modify child node behavior (e.g., Inverter, Repeater)"""
```

### Node Count Estimate
- **~15-20 nodes total** for full behavior tree
- **~5 node types** to implement
- **~200-300 lines of code** for the entire tree system

---

## 🔄 Autonomy Loop Architecture

### Design: **Background Thread with Qt Signals**

```
┌─────────────────────────────────────────┐
│  Main Thread (PySide6 UI)               │
│  ┌───────────────────────────────────┐  │
│  │  OverlayWindow                    │  │
│  │  - Display animations             │  │
│  │  - Handle user input              │  │
│  │  - Receive signals from brain     │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    ▲
                    │ Qt Signals (thread-safe)
                    │
┌─────────────────────────────────────────┐
│  Background Thread (Autonomy Loop)      │
│  ┌───────────────────────────────────┐  │
│  │  AutonomyLoop                     │  │
│  │  - Run behavior tree              │  │
│  │  - Check conditions               │  │
│  │  - Make decisions                 │  │
│  │  - Emit signals to UI             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Implementation Strategy

**Option A: QThread (Recommended)**
```python
class AutonomyLoop(QThread):
    """Background thread for autonomous decision-making"""
    
    decision_made = Signal(str)  # Emit decision to UI
    
    def run(self):
        while not self._stop_event.is_set():
            # Run behavior tree
            decision = self._evaluate_tree()
            
            # Emit decision to UI thread
            self.decision_made.emit(decision)
            
            # Sleep for interval (e.g., 1-5 seconds)
            time.sleep(self._decision_interval)
```

**Option B: threading.Thread + QTimer**
```python
class AutonomyLoop:
    """Background decision loop using threading"""
    
    def __init__(self):
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._stop_event = Event()
        
    def start(self):
        self._thread.start()
        
    def _run_loop(self):
        while not self._stop_event.is_set():
            decision = self._make_decision()
            # Use QMetaObject.invokeMethod to safely call UI
            QMetaObject.invokeMethod(
                self._callback,
                "handle_decision",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, decision)
            )
            time.sleep(1.0)
```

**Recommendation: Option A (QThread)** - Cleaner signal/slot integration with PySide6

### Loop Frequency
- **Decision interval**: 1-2 seconds (configurable)
- **Low priority**: Runs only when UI is idle
- **Non-blocking**: Never interferes with UI responsiveness
- **Resource-aware**: Can slow down if system is busy

---

## 📦 Dependencies for Phase 2

### New Dependencies Required
```bash
# No new external dependencies needed!
# We'll use only standard library + existing deps:
# - threading/QThread (built-in)
# - time (built-in)
# - enum (built-in)
# - dataclasses (built-in)
# - logging (built-in)
```

### Why No New Dependencies?
1. **Behavior tree**: Custom implementation (~300 lines)
2. **Async loop**: QThread or threading.Thread (built-in)
3. **State management**: Simple dataclasses
4. **Decision logic**: Pure Python

**Total new code**: ~500-700 lines
**Total new dependencies**: 0

---

## 🗂️ File Structure for Phase 2

```
src/
├── brain/
│   ├── __init__.py
│   ├── behavior_tree.py          # Tree nodes and evaluation
│   ├── autonomy_loop.py          # Background decision loop
│   ├── context_manager.py        # Screen state & idle detection
│   └── decisions.py              # Decision enums and data classes
```

### Component Breakdown

**1. `decisions.py`** (~100 lines)
```python
class Decision:
    """Represents a decision made by the brain"""
    type: str  # "idle", "wander", "observe", "interact"
    params: dict
    priority: int

class ScreenContext:
    """Current screen state"""
    active_window: str
    idle_time: float
    user_present: bool
    content_type: str  # "browser", "code", "video", etc.
```

**2. `behavior_tree.py`** (~300 lines)
```python
class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

class Node:
    def evaluate(self, context: dict) -> NodeStatus:
        raise NotImplementedError

class Selector(Node):
    # Try children until one succeeds

class Sequence(Node):
    # Execute children until one fails

class Condition(Node):
    # Check a condition

class Action(Node):
    # Execute an action
```

**3. `context_manager.py`** (~200 lines)
```python
class ContextManager:
    """Track screen state and user presence"""
    
    def get_idle_time(self) -> float:
        """Time since last user input"""
        
    def get_active_window(self) -> str:
        """Currently focused window"""
        
    def is_user_present(self) -> bool:
        """Detect if user is at computer"""
        
    def get_screen_content_type(self) -> str:
        """Classify what's on screen"""
```

**4. `autonomy_loop.py`** (~200 lines)
```python
class AutonomyLoop(QThread):
    """Background decision-making loop"""
    
    decision_made = Signal(Decision)
    
    def __init__(self, behavior_tree: BehaviorTree):
        self._tree = behavior_tree
        self._context = ContextManager()
        
    def run(self):
        while not self._stop_event.is_set():
            # Evaluate tree
            decision = self._tree.evaluate(self._context)
            
            # Emit to UI
            self.decision_made.emit(decision)
            
            # Wait
            time.sleep(self._interval)
```

---

## 🧪 Testing Strategy

### Test Structure
```python
# test_behavior_tree.py
- Test node evaluation
- Test tree traversal
- Test decision making

# test_autonomy_loop.py
- Test loop startup/shutdown
- Test decision emission
- Test thread safety
- Test signal delivery

# test_context_manager.py
- Test idle time tracking
- Test window detection
- Test user presence detection
```

### Test Approach
1. **Unit tests**: Test each node type in isolation
2. **Integration tests**: Test full tree evaluation
3. **Signal tests**: Verify decisions reach UI thread
4. **Performance tests**: Ensure loop doesn't block UI

---

## 🎯 Phase 2, Part 1 Scope

### What We'll Build:
1. ✅ Decision dataclasses and enums
2. ✅ Minimal behavior tree nodes (Selector, Sequence, Condition, Action)
3. ✅ Context manager stub (idle time, window tracking)
4. ✅ Autonomy loop with QThread
5. ✅ Signal-based communication to UI
6. ✅ Test suite for all components

### What We'll NOT Build Yet:
- ❌ Actual LLM/VLM integration (Phase 3)
- ❌ Real screen content analysis (Phase 3)
- ❌ Complex behavior trees (simple version first)
- ❌ Learning/adaptation (future phase)

---

## 📋 Implementation Order

### Step 1: Core Data Structures
- `decisions.py`: Decision, ScreenContext dataclasses
- Enums for decision types and node status

### Step 2: Behavior Tree Engine
- `behavior_tree.py`: Node base class
- Selector, Sequence, Condition, Action nodes
- Tree evaluation logic

### Step 3: Context Manager
- `context_manager.py`: Screen state tracking
- Idle time detection
- Window tracking (stub for now)

### Step 4: Autonomy Loop
- `autonomy_loop.py`: QThread implementation
- Decision loop with configurable interval
- Signal emission to UI

### Step 5: Integration Tests
- Test each component independently
- Test signal/slot communication
- Verify non-blocking behavior

---

## 🔧 Installation Commands

### No New Dependencies Required!
```bash
# All dependencies already installed from Phase 1:
# - PySide6 (for QThread, signals, slots)
# - psutil (for system monitoring, already installed)
# - Standard library: threading, time, enum, dataclasses
```

### Verify Installation:
```bash
cd /home/kinj/.continue/bubby
source venv/bin/activate
python3 -c "from PySide6.QtCore import QThread; print('QThread OK')"
```

---

## ✅ Success Criteria

### Phase 2, Part 1 Complete When:
1. ✅ Behavior tree evaluates correctly
2. ✅ Autonomy loop runs in background without blocking UI
3. ✅ Decisions emit via Qt signals
4. ✅ Context manager tracks basic state
5. ✅ All tests pass
6. ✅ No new external dependencies added

### Performance Targets:
- Decision loop: 1-2 decisions per second
- CPU usage: <5% during idle
- Memory: <10MB for brain module
- Signal latency: <50ms from decision to UI update

---

## 🚀 Ready to Proceed?

**Phase 2, Part 1: The Brain & Autonomy Foundation**

This will give us:
- Structured decision-making capability
- Background autonomy without UI blocking
- Foundation for future AI integration
- Clean separation of concerns

**Awaiting your confirmation to begin implementation.**