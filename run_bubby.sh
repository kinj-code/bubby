#!/bin/bash
# ── Bubby Desktop Companion — One-Click Launcher ─────────────────
# Starts the Rust core (optional) and the Python UI.
# Usage: bash run_bubby.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Paths ────────────────────────────────────────────────────────
RUST_BIN="$SCRIPT_DIR/bubby_core/target/release/bubby-core"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
SYSTEM_PYTHON="python3"
PID_FILE="/tmp/bubby_rust_core.pid"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

# ── Cleanup on exit ──────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Bubby: Shutting down..."
    if [ -f "$PID_FILE" ]; then
        RUST_PID=$(cat "$PID_FILE")
        if kill -0 "$RUST_PID" 2>/dev/null; then
            echo "  Stopping Rust core (PID $RUST_PID)..."
            kill "$RUST_PID" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$PID_FILE"
    fi
    echo "  Goodbye!"
}
trap cleanup EXIT INT TERM

# ── Start Rust core (optional — skip if not built) ──────────────
if [ -f "$RUST_BIN" ]; then
    echo "Bubby: Starting Rust core..."
    "$RUST_BIN" > "$LOG_DIR/rust_core.log" 2>&1 &
    RUST_PID=$!
    echo "$RUST_PID" > "$PID_FILE"
    echo "  Rust core PID: $RUST_PID"
    sleep 1
    if ! kill -0 "$RUST_PID" 2>/dev/null; then
        echo "  WARNING: Rust core failed to start. Check $LOG_DIR/rust_core.log"
        rm -f "$PID_FILE"
    fi
else
    echo "Bubby: Rust core not built — running Python UI only."
    echo "  (Build with: cd bubby_core && cargo build --release --bin bubby-core)"
fi

# ── Select Python interpreter ────────────────────────────────────
if [ -f "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
    echo "  Using venv Python: $PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="$SYSTEM_PYTHON"
    echo "  Using system Python: $PYTHON"
else
    echo "ERROR: No Python3 found. Install Python 3.10+."
    exit 1
fi

# ── Launch Bubby UI ──────────────────────────────────────────────
echo "Bubby: Launching UI..."
export PYTHONPATH="$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/src/app.py"