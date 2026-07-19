# Bubby — Desktop AI Companion

Bubby is a local, offline AI companion that runs on your Linux desktop. It watches your screen, monitors your terminal for errors, checks your calendar, and proactively offers help — all running on your own hardware with no internet dependency.

## Quick Start

```bash
git clone git@github.com:kinj-code/bubby.git
cd bubby
bash setup.sh
source .venv/bin/activate
python3 run_autonomous.py
```

Requires Python 3.11+, a Wayland desktop, and ~8GB free RAM. A local GGUF language model (3B parameters recommended) is auto-detected if you have GPT4All installed.

## How It Works

Bubby runs as a headless daemon with four pipelines:

1. **Vision** — captures screen state and reasons about what you're doing
2. **Sensors** — polls your terminal exit codes and iCalendar files for deadlines
3. **Cognition** — routes observations through a behavior tree into persona-aware synthesis
4. **Critic+Policy** — validates LLM output for groundedness, safety, and provenance before it reaches you

All inference runs locally via `llama-cpp-python`. No API keys, no cloud, no telemetry.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `BUBBY_USE_LLM` | `1` | Set to `0` for template-only responses (no model needed) |
| `BUBBY_USE_TTS` | `0` | Set to `1` to enable Piper text-to-speech |
| `QT_QPA_PLATFORM` | `offscreen` | Use `wayland` for GUI mode with avatar overlay |

## Running

```bash
# Foreground (logs to terminal + file)
python3 run_autonomous.py

# Background (168-hour stress test)
nohup .venv/bin/python3 run_autonomous.py > logs/session_output.log 2>&1 &

# View live logs
tail -f logs/session_output.log
```

## Architecture

```
src/
├── actions/       Action whitelist + provenance policy
├── brain/         Autonomy loop, behavior tree, reasoning
├── integrations/  Calendar sensor (.ics parser)
├── interaction/   Output pipeline (critic → policy → display)
├── llm/           llama-cpp inference wrapper
├── memory/        Vector DB (FAISS), embeddings, knowledge graph
├── network/       Event bus, local TCP bridge (for mobile pairing)
├── persona/       Template + LLM synthesis, prompts
├── sensors/       Terminal error monitor
├── vision/        Screen capture pipeline
└── voice/         Piper TTS integration
```

## Requirements

- Linux (Wayland)
- Python 3.11+
- ~8GB free RAM (16GB recommended)
- ~2GB disk for a 3B GGUF model