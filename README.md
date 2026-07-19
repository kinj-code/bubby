<p align="center">
  <img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f916.svg" width="80" alt="Bubby">
</p>

<h1 align="center">🐞 Bubby</h1>
<h3 align="center">Autonomous Desktop AI Companion — 100% Offline, 100% Private</h3>

<p align="center">
  <strong>5.4 GB RAM · i5 CPU · Linux (Zorin OS) · PySide6 + llama.cpp + Moondream2 + Piper TTS</strong>
</p>

---

## What is Bubby?

Bubby is a **fully offline desktop AI companion** that lives on your Linux desktop. It has:

- 🧠 **A Brain** — Local LLM (Qwen/Llama 1-1.5B quantized) with grammar-constrained JSON generation
- 👀 **Eyes** — Moondream2 VLM for screen understanding
- 🗣️ **A Voice** — Piper TTS for offline speech synthesis
- 🎭 **An Avatar** — Floating transparent Qt overlay with 7 animated states
- 🤲 **Hands** — 22 whitelisted system commands (battery, disk, brightness, etc.)
- 📚 **A Library** — FAISS RAG pipeline with Knowledge Graph for document reasoning
- 🔄 **Self-Refining Memory** — Feedback engine that learns which retrievals are helpful
- 🛡️ **Cognitive Critic** — Safety layer that blocks hallucinations before output
- 📡 **Sensors** — Terminal error detection, calendar monitoring, mobile bridge

All running on **~5.4 GB of RAM** on an Intel i5 processor. No cloud. No API keys. No internet.

---

## Architecture

```
┌────────── SENSORS ───────────────────────────────────────┐
│ Vision(PySide6) · Terminal(/tmp/bubby_*) · Mobile(TCP)   │
│ Calendar(~/Documents/*.ics)                              │
├──────────────────────────────────────────────────────────┤
│             ↓                                            │
├────────── COGNITION ────────────────────────────────────┤
│ VLM(Moondream2) → LLM(Qwen/Llama with JSON grammar)      │
│ RAG(FAISS) → KnowledgeGraph(NetworkX) → Feedback(sqlite3)│
│ Critic(Utility+Safety+Redundancy) ←→ Proactivity(5-factor)│
├──────────────────────────────────────────────────────────┤
│             ↓                                            │
├────────── OUTPUT ───────────────────────────────────────┤
│ Avatar(Qt+7 states) · PiperTTS · SystemActions(22 cmds)  │
├──────────────────────────────────────────────────────────┤
│             ↓                                            │
├────────── HARDENING ────────────────────────────────────┤
│ Profiler · Watchdog · CheckpointManager                  │
│ 5.4 GB AI / 10.6 GB free · 100% Offline · Linux         │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/bubby.git
cd bubby

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (first time only)
python scripts/download_llm.py --best
python scripts/download_voice.py --best

# 4. Index your study materials
python -c "
from src.memory.parser import DocumentParser
from src.memory.ingestion import DocumentIngestion
from src.memory.embedding import EmbeddingEngine
from src.memory.vector_db import VectorStore
e = EmbeddingEngine()
v = VectorStore()
d = DocumentIngestion(e, v)
d.ingest_directory('/path/to/your/documents')
"

# 5. Launch the companion
BUBBY_USE_LLM=1 BUBBY_USE_TTS=1 python src/app.py
```

---

## Project Structure

```
bubby/
├── src/
│   ├── brain/          # Behavior tree, autonomy loop, proactivity, critic, graph builder
│   ├── vision/         # VLM engine, screen capture, change detection
│   ├── memory/         # Embedding, FAISS vector DB, long-term memory, RAG parser
│   │                   #   ingestion, feedback engine, knowledge graph, synthetic gen
│   ├── persona/        # Synthesis engine, LLM synthesis, prompts, RAG bridge
│   ├── interaction/    # Interaction handler (UI + TTS + action routing)
│   ├── llm/            # llama-cpp-python inference with JSON grammar constraints
│   ├── voice/          # Piper TTS engine
│   ├── ui/             # Transparent Qt overlay, avatar animation engine
│   ├── actions/        # Whitelisted system command executor (22 commands)
│   ├── sensors/        # Terminal sensor, mobile bridge sensor
│   ├── network/        # Local TCP bridge server for Android client
│   ├── integrations/   # Calendar sensor (.ics parser)
│   ├── perf/           # Pipeline profiler, process watchdog
│   ├── data/           # State persistence & checkpoint manager
│   └── app.py          # Main application entry point
├── scripts/            # Model downloaders (LLM, voice, VLM)
├── mobile/             # Android Kotlin bridge service
├── test_*.py           # Integration test suites (15+ test files)
├── run_autonomous.py   # Mission Control — autonomous daemon loop
└── requirements.txt
```

---

## Modules by Phase

| Phase | System | Key Files |
|---|---|---|
| 1-4 | **Vision + Brain** | `vision/pipeline.py`, `brain/autonomy_loop.py`, `brain/reasoning.py`, `ui/overlay.py` |
| 5 | **Memory** | `memory/embedding.py`, `memory/vector_db.py`, `memory/long_term_memory.py` |
| 6 | **Interaction** | `interaction/handler.py`, `persona/synthesis.py` |
| 7 | **LLM Integration** | `llm/inference.py`, `persona/prompts.py`, `persona/response_parser.py` |
| 8 | **Omni-Integration** | `ui/avatar.py`, `voice/tts_engine.py`, `actions/executor.py` |
| 9 | **Proactive Autonomy** | `brain/proactivity.py`, `brain/critic.py`, `sensors/terminal.py` |
| 10 | **Mobile Bridge** | `network/local_server.py`, `sensors/mobile.py`, `mobile/OfflineBridgeService.kt` |
| 11 | **RAG Pipeline** | `memory/parser.py`, `memory/ingestion.py`, `persona/rag_bridge.py` |
| 12 | **Self-Refining Memory** | `memory/feedback.py`, `memory/knowledge_graph.py`, `brain/graph_builder.py` |
| 13 | **Hardening + Workflow** | `perf/profiler.py`, `perf/watchdog.py`, `integrations/calendar.py` |

---

## RAM Budget

| Component | RAM |
|---|---|
| Moondream2 VLM | 1,800 MB |
| Sub-3B LLM (Q4_K_M) | 2,704 MB |
| Embedding model | 300 MB |
| Qt/Python overhead | 500 MB |
| Piper TTS | 50 MB |
| Avatar UI | 100 MB |
| Feedback + Knowledge Graph | 17 MB |
| Profiler + Watchdog + Calendar | 7 MB |
| **TOTAL AI Stack** | **5,478 MB (5.4 GB)** |
| **Free for OS/apps** | **10,522 MB (10.3 GB)** |

---

## System Actions (Whitelisted)

The companion can execute these 22 commands after safety validation:

| Category | Commands |
|---|---|
| **System Info** | `check_battery`, `check_disk`, `check_memory`, `check_uptime`, `check_cpu`, `check_date`, `check_weather` |
| **Utility** | `open_terminal`, `open_calculator`, `open_files`, `open_browser`, `open_settings`, `open_vscode`, `take_screenshot` |
| **Power** | `lock_screen`*, `sleep_system`* |
| **Display** | `brightness_up`, `brightness_down`, `volume_up`, `volume_down`, `volume_mute` |

*\*requires user approval*

---

## Running Autonomous (168-Hour Stress Test)

```bash
# Pre-flight
sudo systemctl mask sleep.target suspend.target hibernate.target
python test_phase_12_unified.py

# Launch mission
nohup python run_autonomous.py > logs/session_output.log 2>&1 &

# Monitor
tail -f logs/mission_control.log
```

---

## Requirements

- **OS:** Linux (Ubuntu 22.04+, Zorin OS, or compatible)
- **Python:** 3.10+
- **RAM:** 16 GB recommended (works on 8 GB with lighter models)
- **CPU:** Intel i5 or equivalent (4+ cores)
- **GPU:** None required (100% CPU inference)
- **Storage:** ~5 GB for models + documents

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ on Zorin OS. No cloud. No API keys. Just local inference.</sub>
</p>