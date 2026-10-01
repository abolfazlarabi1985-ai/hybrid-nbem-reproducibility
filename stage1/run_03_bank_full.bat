@echo off
setlocal
cd /d "%~dp0"
python stage1_major_revision.py --project-root "%~dp0" --mode bank --seeds 13 21 42 87 123 --folds 10 --output-dir results/major_revision_stage1
if errorlevel 1 (
  echo.
  echo ERROR: full Bank sensitivity run failed.
  pause
  exit /b 1
)
echo.
echo Full Bank sensitivity completed successfully.
echo Please ZIP the entire results\major_revision_stage1 folder and send it for review.
pause
