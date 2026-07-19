# Phase 3: Vision Pipeline & Short-Term Memory Buffer

## Overview
Build the infrastructure for the companion to "see" the screen, process visual information, and maintain short-term memory of what it observes.

---

## 🏗️ Architecture Design

```
┌──────────────────────────────────────────────────────────┐
│  Vision Pipeline (src/vision/pipeline.py)                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Input: Raw frame (numpy array / PIL Image)        │  │
│  │  ↓                                                 │  │
│  │  Preprocessing:                                    │  │
│  │  - Downsample to VLM input size (224x224)          │  │
│  │  - Color space conversion (RGB)                    │  │
│  │  - Normalization (0-255 → 0-1)                     │  │
│  │  ↓                                                 │  │
│  │  Output: Model-ready tensor                        │  │
│  │  [1, 3, 224, 224] float32                         │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                │
│                          │ Processed frame                │
│                          ▼                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Memory Buffer (src/vision/memory_buffer.py)       │  │
│  │  - Stores text descriptions (not raw images)       │  │
│  │  - Rolling window: last N observations             │  │
│  │  - Token-limited to prevent RAM overflow           │  │
│  │  - Chronological log with timestamps               │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
src/vision/
├── __init__.py              # Module exports
├── pipeline.py              # Frame preprocessing
└── memory_buffer.py         # Rolling memory queue

tests/
├── test_vision_pipeline.py  # Frame downsampling tests
└── test_memory_buffer.py    # Buffer retention tests
```

---

## 🎯 Vision Pipeline (`src/vision/pipeline.py`)

### Responsibilities:
1. Accept raw frames from capture system
2. Downsample to VLM input size (224x224 for Moondream2)
3. Normalize and format for model input
4. **Memory-efficient**: Discard raw frame after processing

### Key Methods:
```python
class VisionPipeline:
    def preprocess_frame(self, raw_frame) -> np.ndarray:
        """Downsample and normalize frame for VLM."""
        
    def to_model_input(self, processed_frame) -> torch.Tensor:
        """Convert to model-ready tensor format."""
        
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return expected input shape (C, H, W)."""
```

### Memory Optimization:
- Raw frame discarded immediately after preprocessing
- Only store processed tensor (224x224x3 = ~600KB vs 1920x1080x3 = ~6MB)
- Use numpy arrays (not PIL) for efficiency
- Batch processing support for multiple frames

---

## 🧠 Memory Buffer (`src/vision/memory_buffer.py`)

### Responsibilities:
1. Store text descriptions of screen states
2. Maintain rolling window of last N observations
3. Token-limited to prevent RAM overflow
4. Provide temporal context for decision-making

### Key Methods:
```python
class MemoryBuffer:
    def add_observation(self, description: str, metadata: dict) -> None:
        """Add new observation to buffer."""
        
    def get_recent(self, n: int = 5) -> List[Observation]:
        """Get last N observations."""
        
    def get_context_window(self, max_tokens: int = 512) -> str:
        """Get recent observations within token limit."""
        
    def get_timeline(self, seconds: int = 60) -> List[Observation]:
        """Get observations from last N seconds."""
```

### Data Structure:
```python
@dataclass
class Observation:
    timestamp: float
    description: str  # Text description from VLM (future)
    metadata: dict    # Window, app, confidence
    tokens: int       # Token count for budget management
```

### Memory Budget:
- Max observations: 50 items
- Max tokens: 2048 tokens (~1.5KB)
- Max age: 300 seconds (5 minutes)
- Auto-prune oldest when limits exceeded

---

## 🔧 Technical Specifications

### Frame Preprocessing:
- **Input**: Raw screen capture (1920x1080x3 or variable)
- **Output**: VLM-ready tensor (1x3x224x224)
- **Downsampling**: Bilinear interpolation via PIL/numpy
- **Normalization**: 0-255 → 0-1 float32
- **Memory**: ~600KB per processed frame

### Memory Buffer:
- **Storage**: Text descriptions only (not images)
- **Format**: JSON-serializable observations
- **Retention**: Rolling window with time/size limits
- **Token counting**: Rough estimate (chars/4)
- **Memory**: ~10-50KB for 50 observations

---

## ✅ Success Criteria

1. Vision pipeline downsamples frames correctly
2. Output format matches VLM expectations (224x224x3)
3. Memory buffer stores observations chronologically
4. Buffer respects token/size/time limits
5. Old observations pruned automatically
6. No memory leaks (raw frames discarded)
7. Standalone tests verify functionality

---

## 🧪 Test Strategy

### Test 1: Frame Downsampling
- Create dummy 1920x1080 frame
- Process through pipeline
- Verify output shape (224x224x3)
- Verify normalization (0-1 range)
- Verify memory cleanup

### Test 2: Buffer Retention
- Add 100 observations
- Verify buffer limits to 50
- Verify token counting
- Verify time-based pruning
- Verify chronological order

### Test 3: Integration
- Simulate frame capture → pipeline → buffer
- Verify end-to-end flow
- Verify memory usage stays <100MB

---

## 🚀 Implementation Order

1. Create `src/vision/__init__.py`
2. Implement `src/vision/pipeline.py` with tests
3. Implement `src/vision/memory_buffer.py` with tests
4. Create standalone test scripts
5. Verify memory efficiency

---

## 📋 Dependencies

**No new dependencies required!**
- Use existing numpy (from capture module)
- Use standard library dataclasses
- Use typing for type hints

**Future (Phase 4):**
- torch or llama-cpp-python for VLM inference
- PIL for advanced image processing

---

## 🎯 Ready to Implement

This creates the visual perception foundation:
- ✅ Frame preprocessing for VLM input
- ✅ Memory-efficient processing
- ✅ Short-term temporal awareness
- ✅ Token-aware buffer management
- ✅ No RAM overflow risk

**Awaiting confirmation to begin implementation.**