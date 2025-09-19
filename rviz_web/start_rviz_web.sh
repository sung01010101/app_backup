#!/bin/bash

# Initialize ros
source /opt/ros/jazzy/setup.bash

# Function to cleanup background processes
cleanup() {
    echo "$(date): Shutting down services..."
    if [ -n "${RVIZ_PID:-}" ]; then
        kill "$RVIZ_PID" 2>/dev/null || true
        echo "$(date): Stopped RVIZ service (PID: $RVIZ_PID)"
    fi
    # Remove PID file if present
    rm -f "$PIDFILE" 2>/dev/null || true
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Resolve script directory and run from there (no hardcoded paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Service settings
PORT=5003
PIDFILE="$SCRIPT_DIR/.rviz_web.pid"

# Helpers to stop existing service on the same port
kill_by_port() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN || true)
    else
        mapfile -t pids < <(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p && $1 == "LISTEN" {print $NF}' | grep -Po 'pid=\K[0-9]+' || true)
    fi
    for pid in "${pids[@]:-}"; do
        [ -n "$pid" ] || continue
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "$(date): Killing process on port $port (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            sleep 1
            if ps -p "$pid" >/dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
}

stop_existing() {
    if [ -f "$PIDFILE" ]; then
        oldpid="$(cat "$PIDFILE" 2>/dev/null || true)"
        if [ -n "$oldpid" ] && ps -p "$oldpid" >/dev/null 2>&1; then
            echo "$(date): Stopping previous RVIZ service (PID: $oldpid)"
            kill "$oldpid" 2>/dev/null || true
            sleep 1
            if ps -p "$oldpid" >/dev/null 2>&1; then
                kill -9 "$oldpid" 2>/dev/null || true
            fi
        fi
        rm -f "$PIDFILE"
    fi
    kill_by_port "$PORT"
}

# Ensure no previous instance is running
stop_existing

# Pick a Python interpreter (prefer project venvs)
if [ -x "$SCRIPT_DIR/.venv-py3.12/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv-py3.12/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
else
  echo "ERROR: No suitable Python interpreter found."
  exit 1
fi

"$PY" app.py &

RVIZ_PID=$!
# Persist PID for future restarts
echo "$RVIZ_PID" > "$PIDFILE"
echo "$(date): RVIZ Service PID: $RVIZ_PID"