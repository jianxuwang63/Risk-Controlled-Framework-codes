@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The Python environment is missing.
  echo Run setup_full_reviewer_windows.cmd first.
  pause
  exit /b 1
)

if not exist "weights\deployment_policy.cost5.validation.json" (
  echo [ERROR] Missing weights\deployment_policy.cost5.validation.json.
  pause
  exit /b 1
)

for /L %%G in (1,1,5) do (
  if not exist "weights\cost_5.0_fold_%%G_LAST.pth" (
    echo [ERROR] Missing weights\cost_5.0_fold_%%G_LAST.pth.
    pause
    exit /b 1
  )
)

set "APP_MODE=validation"
set "PILOT_PHASE=assisted"
set "OFFLINE_ONLY=true"
set "ALLOW_PRIVATE_LAN=false"
set "PUBLIC_INTERNET_MODE=false"
set "APP_API_KEY="
set "APP_SESSION_SECRET="
set "ALLOWED_ORIGINS="
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "MODEL_VERSION=selectivenet-cost5-ensemble-v1"
set "MODEL_CHECKPOINTS=%~dp0weights\cost_5.0_fold_1_LAST.pth,%~dp0weights\cost_5.0_fold_2_LAST.pth,%~dp0weights\cost_5.0_fold_3_LAST.pth,%~dp0weights\cost_5.0_fold_4_LAST.pth,%~dp0weights\cost_5.0_fold_5_LAST.pth"
set "POLICY_PATH=%~dp0weights\deployment_policy.cost5.validation.json"
set "MODEL_DEVICE=cpu"
set "TILE_BATCH_SIZE=16"
set "MAX_TILES=1024"
set "MAX_IMAGES_PER_CASE=300"
set "MAX_CASE_UPLOAD_MB=1000"
set "DATA_DIR=%~dp0runtime\full_reviewer"
set "DATABASE_PATH=%~dp0runtime\full_reviewer\pilot.db"
set "PERSIST_IMAGES=false"
set "IMAGE_STORE_DIR=%~dp0runtime\full_reviewer\images"
set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo Verifying the five Cost-5 checkpoints and the frozen FNR-5%% policy...
".venv\Scripts\python.exe" -m clinical_app.offline_preflight ^
  --policy "%POLICY_PATH%" ^
  --checkpoint "weights\cost_5.0_fold_1_LAST.pth" ^
  --checkpoint "weights\cost_5.0_fold_2_LAST.pth" ^
  --checkpoint "weights\cost_5.0_fold_3_LAST.pth" ^
  --checkpoint "weights\cost_5.0_fold_4_LAST.pth" ^
  --checkpoint "weights\cost_5.0_fold_5_LAST.pth" ^
  --data-dir "%DATA_DIR%"
if errorlevel 1 (
  echo.
  echo [ERROR] Model or policy verification failed. Do not start this package.
  pause
  exit /b 1
)

echo.
echo Starting the password-free, localhost-only full reviewer application.
echo The browser will open at http://127.0.0.1:%PORT%/
echo Uploaded image bytes are not retained. Local review records stay under runtime\full_reviewer.
echo Keep this window open. Press Ctrl+C to stop.
start "" powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%PORT%/'"
".venv\Scripts\python.exe" -m clinical_app
if errorlevel 1 (
  echo.
  echo [ERROR] The application stopped unexpectedly.
  pause
  exit /b 1
)
