#!/bin/bash

# Function to cleanup background processes
cleanup() {
    echo "$(date): Shutting down services..."
    if [ ! -z "$YOLO_PID" ]; then
        kill $YOLO_PID 2>/dev/null || true
        echo "$(date): Stopped YOLO service (PID: $YOLO_PID)"
    fi
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Resolve script directory and run from there (no hardcoded paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick a Python interpreter (prefer project venvs)
if [ -x "$SCRIPT_DIR/.venv-py3.11/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv-py3.11/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python"
else
  echo "ERROR: No suitable Python interpreter found."
  exit 1
fi

"$PY" app.py &

YOLO_PID=$!
echo "$(date): YOLO Service PID: $YOLO_PID"