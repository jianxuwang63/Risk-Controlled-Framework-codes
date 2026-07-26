#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] Python 3 was not found. Install 64-bit Python 3.10 or 3.11, then retry."
  read "?Press Return to close..."
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
  echo "[ERROR] This package requires Python 3.10-3.13. Python 3.11 is recommended."
  echo "Install it from https://www.python.org/downloads/macos/ and retry."
  read "?Press Return to close..."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating the Python virtual environment..."
  python3 -m venv .venv
fi

echo "Installing application dependencies. This can take several minutes..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements-app.txt

echo
echo "MACOS SETUP COMPLETED"
echo "Next, double-click self_test_macos.command."
read "?Press Return to close..."
