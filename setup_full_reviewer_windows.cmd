@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.13"
  if not defined PYTHON_CMD (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"
  )
  if not defined PYTHON_CMD (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.11"
  )
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
  echo Install 64-bit Python 3.10-3.13 and select "Add Python to PATH".
  pause
  exit /b 1
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This package requires 64-bit Python 3.10-3.13.
  echo Python 3.11 is recommended.
  pause
  exit /b 1
)

%PYTHON_CMD% --version
%PYTHON_CMD% verify_full_reviewer_integrity.py
if errorlevel 1 goto :failed

if not exist ".venv\Scripts\python.exe" (
  echo Creating the local Python environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing reviewer application dependencies.
echo Internet access is required for this one-time setup.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if errorlevel 1 goto :failed

echo.
echo FULL REVIEWER SETUP COMPLETED
echo Next, double-click start_full_reviewer_windows.cmd.
pause
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Copy the complete message above when requesting support.
pause
exit /b 1
