#!/bin/bash
# Bubby daemon launcher with auto-restart
# Run: bash scripts/daemon.sh
set -e

BUBBY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$BUBBY_DIR/logs/daemon.log"
PID_FILE="$BUBBY_DIR/logs/bubby.pid"

echo "Bubby Daemon — starting..."

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing daemon (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Start the daemon
cd "$BUBBY_DIR"
nohup "$BUBBY_DIR/.venv/bin/python3" run_autonomous.py > logs/session_output.log 2>&1 &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"
echo "Daemon started with PID $DAEMON_PID"
echo ""
echo "  View logs:   tail -f $BUBBY_DIR/logs/session_output.log"
echo "  Stop:        kill \$(cat $BUBBY_DIR/logs/bubby.pid)"
echo "  Status:      kill -0 \$(cat $BUBBY_DIR/logs/bubby.pid) && echo 'running' || echo 'stopped'"