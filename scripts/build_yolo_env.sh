#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="yolo_web"
PYEXEC="python3.11"
PROJ_DIR="$ROOT_DIR/$PROJ"
REQ="$PROJ_DIR/requirements.txt"
VENV_DIR="$PROJ_DIR/.venv-py3.11"

echo "Creating project directory: $PROJ_DIR"
mkdir -p "$PROJ_DIR"

if [ ! -f "$REQ" ]; then
  cat > "$REQ" <<'EOF'
# requirements for yolo_web
flask
# add other dependencies here
EOF
  echo "Created $REQ"
else
  echo "requirements.txt already exists at $REQ; leaving unchanged."
fi

if command -v "$PYEXEC" >/dev/null 2>&1; then
  if [ -d "$VENV_DIR" ]; then
    echo "Existing virtualenv found: $VENV_DIR (will reuse)"
  else
    echo "Creating virtualenv with $PYEXEC -> $VENV_DIR"
    "$PYEXEC" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    if [ -s "$REQ" ]; then
      echo "Installing requirements into virtualenv..."
      "$VENV_DIR/bin/pip" install -r "$REQ"
    fi
  fi
  echo "Virtualenv ready: $VENV_DIR"
else
  echo "Warning: $PYEXEC not found in PATH. Virtualenv not created. Install Python 3.11 or adjust PATH to create it."
fi

echo "Done: $PROJ"
