@echo off
setlocal
cd /d "%~dp0"
python stage1_major_revision.py --project-root "%~dp0" --mode audit
if errorlevel 1 (
  echo.
  echo ERROR: Stage-1 audit failed. Do not continue.
  pause
  exit /b 1
)
echo.
echo Audit completed successfully. Check results\major_revision_stage1\00_exact_snapshot_verification.csv
pause
