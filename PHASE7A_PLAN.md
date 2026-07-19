# Phase 7A: Local LLM Integration (Qwen2.5-1.5B / Llama-3.2-1B)

## Objective
Upgrade the companion's "Voice" from template-based to dynamic conversational LLM while staying within 16GB RAM budget.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAM Budget (16GB)                        │
├─────────────────────────────────────────────────────────────┤
│  OS / DE / Browser / IDE     │  ~6-8 GB                    │
│  Moondream2 (VLM)            │  ~1.8 GB (Q4_K_M GGUF)      │
│  Local LLM (Qwen2.5-1.5B)    │  ~1.2 GB (Q4_K_M GGUF)      │
│  Python overhead             │  ~0.5 GB                    │
│  ───────────────────────────────────────────────────────── │
│  Total AI Stack              │  ~3.5 GB                    │
│  Headroom                    │  ~4-6 GB                    │
└─────────────────────────────────────────────────────────────┘
```

## Components to Implement

### 1. `src/llm/__init__.py`
Module exports.

### 2. `src/llm/inference.py` - Core LLM Wrapper
- `llama-cpp-python` wrapper with context management
- Streaming token generation support
- Quantization config (Q4_K_M default)
- Thread-safe singleton pattern
- Graceful fallback if model not found

### 3. `src/llm/model_manager.py` - Model Download & Management
- Download GGUF models from HuggingFace Hub
- Verify checksums
- Model registry (available models, sizes, URLs)
- Auto-detect best model for hardware

### 4. `src/persona/llm_synthesis.py` - LLM-Backed Synthesis
- Inherits from `SynthesisEngine` interface
- Builds system prompt from `PersonaConfig`
- Injects LTM context + current reasoning
- Streams response for real-time feel
- Falls back to template engine on error

### 5. `scripts/download_llm.py` - CLI Model Downloader
- Progress bars
- Resume support
- Model selection menu

### 6. Integration
- Update `src/app.py` to initialize LLM synthesis when available
- Environment variable to toggle: `BUBBY_USE_LLM=1`

## Model Selection

| Model | Size (Q4_K_M) | Context | Strengths |
|-------|--------------|---------|-----------|
| **Qwen2.5-1.5B-Instruct** | ~1.2 GB | 32K | Best coding, multilingual, instruction following |
| Llama-3.2-1B-Instruct | ~1.1 GB | 128K | Strong reasoning, larger context |
| Phi-3-mini-4k | ~2.3 GB | 4K | Good quality but larger |

**Recommendation: Qwen2.5-1.5B-Instruct** - Best balance of coding ability, size, and instruction following.

## Prompt Template Structure

```python
SYSTEM_PROMPT = """{persona_backstory}

## Personality
{persona_traits}

## Response Guidelines
{response_rules}

## Context
- Current screen: {content_type} (confidence: {confidence:.0%})
- User state: {user_state}
- Relevant memories: {memory_context}

## Response
Generate a natural, in-character response as {name}. Max {max_length} chars. No internal state leakage."""
```

## Implementation Order

1. ✅ Add `llama-cpp-python` to requirements
2. ✅ Create `src/llm/inference.py`
3. ✅ Create `src/llm/model_manager.py`
4. ✅ Create `src/persona/llm_synthesis.py`
5. ✅ Create `scripts/download_llm.py`
6. ✅ Update `src/app.py` for LLM integration
7. ✅ Test with model download
8. ✅ Run integration tests