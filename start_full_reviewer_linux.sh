#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] Run ./setup_full_reviewer_linux.sh first." >&2
  exit 1
fi

POLICY_PATH="$PROJECT_DIR/weights/deployment_policy.cost5.validation.json"
CHECKPOINTS=()
for fold in 1 2 3 4 5; do
  checkpoint="$PROJECT_DIR/weights/cost_5.0_fold_${fold}_LAST.pth"
  [[ -f "$checkpoint" ]] || {
    echo "[ERROR] Missing model checkpoint: $checkpoint" >&2
    exit 1
  }
  CHECKPOINTS+=("$checkpoint")
done
[[ -f "$POLICY_PATH" ]] || {
  echo "[ERROR] Missing deployment policy: $POLICY_PATH" >&2
  exit 1
}

export APP_MODE=validation
export PILOT_PHASE=silent
export OFFLINE_ONLY=true
export ALLOW_PRIVATE_LAN=false
export PUBLIC_INTERNET_MODE=false
unset APP_API_KEY APP_SESSION_SECRET ALLOWED_ORIGINS || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODEL_VERSION=selectivenet-cost5-ensemble-v1
export MODEL_CHECKPOINTS
MODEL_CHECKPOINTS="$(IFS=,; echo "${CHECKPOINTS[*]}")"
export POLICY_PATH
export MODEL_DEVICE=cpu
export TILE_BATCH_SIZE=16
export MAX_TILES=1024
export MAX_IMAGES_PER_CASE=300
export MAX_CASE_UPLOAD_MB=1000
export DATA_DIR="$PROJECT_DIR/runtime/full_reviewer"
export DATABASE_PATH="$DATA_DIR/pilot.db"
export PERSIST_IMAGES=false
export IMAGE_STORE_DIR="$DATA_DIR/images"
export HOST=127.0.0.1
export PORT="${PORT:-8000}"

mkdir -p "$DATA_DIR"
PREFLIGHT_ARGS=(--policy "$POLICY_PATH" --data-dir "$DATA_DIR")
for checkpoint in "${CHECKPOINTS[@]}"; do
  PREFLIGHT_ARGS+=(--checkpoint "$checkpoint")
done

echo "Verifying the five Cost-5 checkpoints and the frozen FNR-5% policy..."
".venv/bin/python" -m clinical_app.offline_preflight "${PREFLIGHT_ARGS[@]}"

echo "Starting the password-free, localhost-only full reviewer application."
echo "Open http://127.0.0.1:${PORT}/ and press Ctrl+C here to stop."
if command -v xdg-open >/dev/null 2>&1; then
  (sleep 3; xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true) &
fi
".venv/bin/python" -m clinical_app
