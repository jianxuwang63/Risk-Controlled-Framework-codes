#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_CMD=""
for candidate in python3.11 python3.12 python3.13 python3.10 python3; do
  candidate_path="$(command -v "$candidate" 2>/dev/null || true)"
  if [[ -n "$candidate_path" ]] &&
    "$candidate_path" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
    PYTHON_CMD="$candidate_path"
    break
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  echo "[ERROR] Python 3.10-3.13 is required; Python 3.11 is recommended." >&2
  exit 1
fi

echo "Using $("$PYTHON_CMD" --version) at $PYTHON_CMD"
"$PYTHON_CMD" verify_full_reviewer_integrity.py

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_CMD" -m venv .venv
fi

echo "Installing reviewer application dependencies."
echo "Internet access is required for this one-time setup."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements-app.txt
echo "FULL REVIEWER SETUP COMPLETED"
echo "Next, run ./start_full_reviewer_linux.sh"
