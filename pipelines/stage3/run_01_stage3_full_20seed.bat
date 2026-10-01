@echo off
setlocal
cd /d "%~dp0"
python stage3_major_revision.py --project-root "." --mode full
if errorlevel 1 (
  echo.
  echo ERROR: Stage-3 full run failed or is incomplete. Keep the folder; the run is resume-safe.
  pause
  exit /b 1
)
echo.
echo Stage-3 FULL 20-SEED RUN COMPLETED SUCCESSFULLY.
echo Send the entire results\major_revision_stage3 folder for review.
pause
