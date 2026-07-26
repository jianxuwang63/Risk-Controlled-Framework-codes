#!/bin/zsh
set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] The Python environment is missing."
  echo "Double-click setup_full_reviewer_macos.command first."
  read "?Press Return to close..."
  exit 1
fi

POLICY_PATH="$PROJECT_DIR/weights/deployment_policy.cost5.validation.json"
CHECKPOINTS=()
for FOLD_NUMBER in 1 2 3 4 5; do
  CHECKPOINT="$PROJECT_DIR/weights/cost_5.0_fold_${FOLD_NUMBER}_LAST.pth"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Missing model checkpoint: $CHECKPOINT"
    read "?Press Return to close..."
    exit 1
  fi
  CHECKPOINTS+=("$CHECKPOINT")
done

if [[ ! -f "$POLICY_PATH" ]]; then
  echo "[ERROR] Missing deployment policy: $POLICY_PATH"
  read "?Press Return to close..."
  exit 1
fi

export APP_MODE=validation
export PILOT_PHASE=assisted
export OFFLINE_ONLY=true
export ALLOW_PRIVATE_LAN=false
export PUBLIC_INTERNET_MODE=false
unset APP_API_KEY APP_SESSION_SECRET ALLOWED_ORIGINS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODEL_VERSION=selectivenet-cost5-ensemble-v1
export MODEL_CHECKPOINTS="${(j:,:)CHECKPOINTS}"
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

echo "Verifying the five Cost-5 checkpoints and the frozen FNR-5% policy..."
PREFLIGHT_ARGS=(
  --policy "$POLICY_PATH"
  --data-dir "$DATA_DIR"
)
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  PREFLIGHT_ARGS+=(--checkpoint "$CHECKPOINT")
done
".venv/bin/python" -m clinical_app.offline_preflight "${PREFLIGHT_ARGS[@]}"

echo
echo "Starting the password-free, localhost-only full reviewer application."
echo "The browser will open at http://127.0.0.1:${PORT}/"
echo "Uploaded image bytes are not retained. Local review records stay under runtime/full_reviewer."
echo "Keep this window open. Press Control+C to stop."
(sleep 3; open "http://127.0.0.1:${PORT}/") &
".venv/bin/python" -m clinical_app
