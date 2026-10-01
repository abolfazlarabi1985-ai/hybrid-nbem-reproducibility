@echo off
setlocal
cd /d "%~dp0"
python stage3_major_revision.py --project-root "." --mode analyze
if errorlevel 1 (
  echo ERROR: Re-analysis failed.
  pause
  exit /b 1
)
echo Re-analysis completed.
pause
