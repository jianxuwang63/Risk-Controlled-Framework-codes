#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

PYTHON_CMD=""
for CANDIDATE in python3.11 python3.12 python3.13 python3.10 python3; do
  CANDIDATE_PATH="$(command -v "$CANDIDATE" 2>/dev/null || true)"
  if [[ -n "$CANDIDATE_PATH" ]] &&
    "$CANDIDATE_PATH" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
    PYTHON_CMD="$CANDIDATE_PATH"
    break
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  echo "[ERROR] A supported Python installation was not found."
  echo "Install 64-bit Python 3.10-3.13 from https://www.python.org/downloads/macos/ and retry."
  read "?Press Return to close..."
  exit 1
fi

echo "Using $("$PYTHON_CMD" --version) at $PYTHON_CMD"
"$PYTHON_CMD" verify_full_reviewer_integrity.py

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating the local Python environment..."
  "$PYTHON_CMD" -m venv .venv
fi

echo "Installing reviewer application dependencies."
echo "Internet access is required for this one-time setup."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements-app.txt

echo
echo "FULL REVIEWER SETUP COMPLETED"
echo "Next, double-click start_full_reviewer_macos.command."
read "?Press Return to close..."
