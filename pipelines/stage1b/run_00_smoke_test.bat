@echo off
setlocal
pushd "%~dp0"
echo Smoke test only - outputs must NEVER be used in the manuscript.
python stage1b_duplicate_leakage_sensitivity.py --project-root "." --mode sensitivity --quick --rerun-record --seeds 42 --folds 2 --sensitivity-datasets haberman_s_survival
if errorlevel 1 (
  echo.
  echo ERROR: smoke test failed.
  pause
  popd
  exit /b 1
)
echo.
echo SUCCESS: smoke test completed.
pause
popd
