#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] Create the environment and install requirements-app.txt first."
  echo "See REVIEWER_QUICKSTART.md."
  exit 1
fi

unset APP_API_KEY APP_SESSION_SECRET MODEL_CHECKPOINTS POLICY_PATH || true
export APP_MODE=demo
export PILOT_PHASE=assisted
export PUBLIC_INTERNET_MODE=false
export OFFLINE_ONLY=true
export ALLOW_PRIVATE_LAN=false
export PERSIST_IMAGES=false
export DATA_DIR="$PROJECT_DIR/runtime/reviewer_demo"
export DATABASE_PATH="$DATA_DIR/demo.db"
export HOST=127.0.0.1
export PORT=8000

echo "Starting the password-free reviewer demonstration."
echo "Open http://127.0.0.1:8000/"
echo "Scores are deterministic placeholders, not clinical model outputs."
echo "Press Control+C to stop."
exec ".venv/bin/python" -m clinical_app
