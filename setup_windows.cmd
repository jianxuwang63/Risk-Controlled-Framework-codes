@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
  if not defined PYTHON_CMD (
    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.10"
  )
  if not defined PYTHON_CMD set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo [ERROR] Python 3 was not found.
  echo Install 64-bit Python 3.10 or 3.11, select "Add Python to PATH", then retry.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing application dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if errorlevel 1 goto :failed

echo.
echo Setup completed. Run self_test_windows.cmd before any hospital collection.
echo For offline use, disconnect the network first. For the approved public HTTPS
echo workflow, keep networking available and follow PUBLIC_HTTPS_DEPLOYMENT.md.
pause
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Copy the full message above to the technical team.
pause
exit /b 1
