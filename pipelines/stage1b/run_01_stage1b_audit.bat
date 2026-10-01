@echo off
setlocal
pushd "%~dp0"
echo Running Stage 1B audit only...
python stage1b_duplicate_leakage_sensitivity.py --project-root "." --mode audit
if errorlevel 1 (
  echo.
  echo ERROR: Stage 1B audit failed. Do not continue.
  pause
  popd
  exit /b 1
)
echo.
echo SUCCESS: Stage 1B audit completed.
echo Inspect results\major_revision_stage1b\25_selected_duplicate_heavy_datasets.csv
pause
popd
