#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -A PROJECTS=(
  ["rviz_web"]="3.12"
  ["yolo_web"]="3.11"
)

for proj in "${!PROJECTS[@]}"; do
  pyver="${PROJECTS[$proj]}"
  pyexec="python${pyver}"
  proj_dir="${ROOT_DIR}/${proj}"

  echo "=== Processing ${proj} (target python ${pyver}) ==="

  if [ ! -d "$proj_dir" ]; then
    echo "Skipping: directory not found: $proj_dir"
    continue
  fi

  if ! command -v "$pyexec" >/dev/null 2>&1; then
    echo "ERROR: ${pyexec} not found in PATH. Install Python ${pyver} or adjust PATH."
    echo "You can install with your distro package manager or use pyenv."
    continue
  fi

  cd "$proj_dir"

  # ensure requirements.txt exists (create placeholder if missing)
  if [ ! -f requirements.txt ]; then
    echo "No requirements.txt found in ${proj_dir} — creating an empty placeholder."
    printf "# Requirements for %s\n" "$proj" > requirements.txt
  fi

  VENV_DIR=".venv-py${pyver}"
  if [ -d "$VENV_DIR" ]; then
    echo "Existing virtualenv found: $proj_dir/$VENV_DIR (will reuse)"
  else
    echo "Creating virtualenv with ${pyexec} -> $VENV_DIR"
    "$pyexec" -m venv "$VENV_DIR"
  fi

  # Activate and install
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel

  if [ -s requirements.txt ]; then
    echo "Installing requirements for ${proj}..."
    pip install -r requirements.txt
  else
    echo "requirements.txt is empty for ${proj}; nothing to install."
  fi

  deactivate

  echo "Done: ${proj}. Activate with: source ${proj_dir}/${VENV_DIR}/bin/activate"
  echo

done

echo "All done."
