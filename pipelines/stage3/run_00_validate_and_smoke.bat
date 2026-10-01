@echo off
setlocal
cd /d "%~dp0"
python stage3_major_revision.py --project-root "." --mode smoke
if errorlevel 1 (
  echo.
  echo ERROR: Stage-3 validation/smoke failed. Do not continue.
  pause
  exit /b 1
)
echo.
echo Stage-3 validation/smoke PASSED.
pause
