@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run setup_windows.cmd first.
  pause
  exit /b 1
)

set "APP_API_KEY="
set "APP_SESSION_SECRET="
set "MODEL_CHECKPOINTS="
set "POLICY_PATH="
set "APP_MODE=demo"
set "PILOT_PHASE=assisted"
set "PUBLIC_INTERNET_MODE=false"
set "OFFLINE_ONLY=true"
set "ALLOW_PRIVATE_LAN=false"
set "PERSIST_IMAGES=false"
set "DATA_DIR=%CD%\runtime\reviewer_demo"
set "DATABASE_PATH=%DATA_DIR%\demo.db"
set "HOST=127.0.0.1"
set "PORT=8000"

echo Starting the password-free reviewer demonstration.
echo Open http://127.0.0.1:8000/
echo Scores are deterministic placeholders, not clinical model outputs.
echo Press Control+C to stop.
".venv\Scripts\python.exe" -m clinical_app
