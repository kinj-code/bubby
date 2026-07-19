#!/bin/bash
# Bubby — One-command setup for Linux (Wayland)
# Run: bash setup.sh
set -e

BUBBY_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "============================================"
echo "  Bubby — Desktop AI Companion Setup"
echo "  Target: $BUBBY_DIR"
echo "============================================"

# ── System dependencies ──
echo ""
echo "[1/5] Checking system dependencies..."
MISSING=""
for pkg in python3 python3-venv python3-dev cmake gcc g++; do
    if ! dpkg -s "$pkg" &>/dev/null && ! rpm -q "$pkg" &>/dev/null; then
        echo "  Missing: $pkg"
        MISSING="$MISSING $pkg"
    fi
done
if [ -n "$MISSING" ]; then
    echo "  Run: sudo apt install $MISSING"
    echo "  (or equivalent for your distro)"
    exit 1
fi
echo "  ✓ All system dependencies found"

# ── Python venv ──
echo ""
echo "[2/5] Creating Python virtual environment..."
python3 -m venv "$BUBBY_DIR/.venv"
"$BUBBY_DIR/.venv/bin/python3" -m ensurepip --upgrade 2>/dev/null || true
"$BUBBY_DIR/.venv/bin/pip" install --upgrade pip --quiet
echo "  ✓ Virtual environment created"

# ── Python dependencies ──
echo ""
echo "[3/5] Installing Python packages (this may take a few minutes)..."
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"
"$BUBBY_DIR/.venv/bin/pip" install --no-cache-dir -r "$BUBBY_DIR/requirements.txt" --quiet
echo "  ✓ Python packages installed"

# ── Model setup ──
echo ""
echo "[4/5] Setting up models directory..."
mkdir -p "$BUBBY_DIR/models/llm"
mkdir -p "$BUBBY_DIR/data/knowledge"
mkdir -p "$BUBBY_DIR/data/memory"
mkdir -p "$BUBBY_DIR/data/backups"
mkdir -p "$BUBBY_DIR/logs"

# Check for existing GPT4All models
FOUND_MODEL=""
for candidate in \
    "$HOME/.local/share/nomic.ai/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf" \
    "$HOME/.local/share/nomic.ai/GPT4All/Llama-3.2-1B-Instruct-Q4_0.gguf" \
    "$HOME/.local/share/nomic.ai/GPT4All/qwen2-1_5b-instruct-q4_0.gguf"; do
    if [ -f "$candidate" ]; then
        FOUND_MODEL="$candidate"
        break
    fi
done

if [ -n "$FOUND_MODEL" ]; then
    ln -sf "$FOUND_MODEL" "$BUBBY_DIR/models/llm/"
    echo "  ✓ Model found and linked: $(basename "$FOUND_MODEL")"
else
    echo "  ⚠ No local GGUF model found."
    echo "  Download one with:"
    echo "    mkdir -p $BUBBY_DIR/models/llm"
    echo "    wget -P $BUBBY_DIR/models/llm https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
fi

# ── Create .env ──
if [ ! -f "$BUBBY_DIR/.env" ]; then
    cat > "$BUBBY_DIR/.env" <<EOF
# Bubby configuration
BUBBY_USE_LLM=1      # Set to 0 to use template-only mode
BUBBY_USE_TTS=0      # Set to 1 if Piper TTS is installed
QT_QPA_PLATFORM=offscreen  # Headless mode (use 'wayland' for GUI)
EOF
    echo "  ✓ Created .env configuration file"
fi

# ── Done ──
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "============================================"
echo "  To start Bubby:"
echo "    cd $BUBBY_DIR"
echo "    source .venv/bin/activate"
echo "    python3 run_autonomous.py"
echo ""
echo "  To run in background (168-hour test):"
echo "    nohup .venv/bin/python3 run_autonomous.py > logs/session_output.log 2>&1 &"
echo ""
echo "  To view logs:"
echo "    tail -f logs/session_output.log"
echo "============================================"