@echo off
setlocal
pushd "%~dp0"
echo Running FULL Stage 1B duplicate-group sensitivity for the core claim models...
echo This is NOT a quick test and may take a long time.
python stage1b_duplicate_leakage_sensitivity.py --project-root "." --mode sensitivity
if errorlevel 1 (
  echo.
  echo ERROR: Stage 1B sensitivity failed.
  pause
  popd
  exit /b 1
)
echo.
echo SUCCESS: Stage 1B sensitivity completed.
echo Zip the entire results\major_revision_stage1b folder and send it for review.
pause
popd
