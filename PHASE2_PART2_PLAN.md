# Phase 2, Part 2: UI & Brain Integration - Architecture Plan

## Overview
Connect the autonomous brain to the transparent overlay UI, enabling visual feedback and movement based on decisions.

---

## 🏗️ Integration Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Main Thread (PySide6 UI)                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  App (src/app.py)                                  │  │
│  │  - Initializes all components                      │  │
│  │  - Wires brain → UI signals                        │  │
│  └────────────────────────────────────────────────────┘  │
│                          ▲                               │
│                          │ Qt Signals                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  OverlayWindow (src/ui/overlay.py)                 │  │
│  │  - Receives decision_made signals                  │  │
│  │  - Updates visual state (text/color)               │  │
│  │  - Moves window for WANDER decisions               │  │
│  │  - Shows behavior state on screen                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                    ▲
                    │ Qt Signals (thread-safe)
                    │
┌──────────────────────────────────────────────────────────┐
│  Background Thread (AutonomyLoop)                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  AutonomyLoop                                      │  │
│  │  - Runs behavior tree                             │  │
│  │  - Emits decision_made(Decision)                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 📡 Signal Flow

```
AutonomyLoop.decision_made(Decision)
    ↓
App.on_decision(Decision)
    ↓
OverlayWindow.update_behavior_state(Decision)
    ↓
Visual Update:
  - Update state text
  - Change window color/hint
  - Move window (if WANDER)
```

---

## 🎨 Visual State Design

### State Display (Placeholder Text)
```
┌─────────────────────┐
│                     │
│   [WANDER]          │  ← State text in center
│                     │
│                     │
│              [X]    │  ← Close zone (bottom-right)
└─────────────────────┘
```

### State Colors (Window Hints)
- **IDLE**: No color change (transparent)
- **WANDER**: Light blue tint
- **PACE**: Light green tint
- **SIT**: Light orange tint
- **OBSERVE**: Light purple tint
- **INTERACT**: Light yellow tint
- **GREET**: Light pink tint
- **SLEEP**: Dark blue tint

---

## 🚶 Movement System (WANDER)

### Safe Movement Algorithm
```python
def wander_to_new_position(current_pos, screen_rect):
    """Generate safe target position within screen bounds."""
    margin = 100  # Keep away from edges
    
    # Random target within safe area
    target_x = random.randint(screen_rect.left() + margin,
                             screen_rect.right() - margin - window_width)
    target_y = random.randint(screen_rect.top() + margin,
                             screen_rect.bottom() - margin - window_height)
    
    return QPoint(target_x, target_y)

def animate_movement(start_pos, end_pos, duration=2.0):
    """Smoothly animate window movement."""
    # Use QPropertyAnimation for smooth movement
    pass
```

### Boundary Checking
- Get screen geometry via QApplication.primaryScreen().availableGeometry()
- Clamp all positions to safe area
- Account for window size
- Prevent movement off-screen

---

## 🔧 Implementation Plan

### Step 1: Update OverlayWindow
- Add `update_behavior_state(decision)` method
- Add state text display widget
- Add `wander_to(target_pos)` method with boundary checking
- Add smooth movement animation

### Step 2: Update App
- Create AutonomyLoop instance
- Wire decision_made signal to OverlayWindow
- Start loop on app startup
- Handle cleanup on exit

### Step 3: Create Test Script
- Boot integrated system
- Run for 15-30 seconds
- Log all decisions and movements
- Show live window behavior

---

## 📋 Code Structure

### src/ui/overlay.py additions:
```python
class OverlayWindow(QWidget):
    def update_behavior_state(self, decision: Decision) -> None:
        """Update visual state based on decision."""
        
    def wander_to(self, target: QPoint) -> None:
        """Safely move window to new position."""
        
    def _get_safe_bounds(self) -> QRect:
        """Get screen boundaries for safe movement."""
```

### src/app.py updates:
```python
def main():
    app = QApplication(sys.argv)
    
    # Create components
    overlay = OverlayWindow()
    tree = create_behavior_tree()
    context = ContextManager()
    loop = AutonomyLoop(tree, context)
    
    # Wire signals
    loop.decision_made.connect(overlay.update_behavior_state)
    
    # Start
    loop.start()
    overlay.show()
    
    sys.exit(app.exec())
```

---

## ✅ Success Criteria

1. OverlayWindow receives and displays decisions
2. Window moves smoothly when WANDER decision made
3. All movement stays within screen bounds
4. State text updates correctly
5. Test runs for 15-30 seconds autonomously
6. No crashes or boundary violations

---

## 🚀 Ready to Implement

This will create a fully integrated, autonomous desktop companion that:
- Makes its own decisions
- Visually communicates its state
- Moves around the screen safely
- Runs entirely locally

**Awaiting confirmation to begin implementation.**