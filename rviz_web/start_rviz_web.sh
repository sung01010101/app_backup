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
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Resolve script directory and run from there (no hardcoded paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
echo "$(date): RVIZ Service PID: $RVIZ_PID"