@echo off
setlocal
cd /d "%~dp0"
python stage1_major_revision.py --project-root "%~dp0" --mode bank --seeds 42 --folds 2 --quick --output-dir results/major_revision_stage1_smoke_test
if errorlevel 1 (
  echo.
  echo ERROR: smoke test failed.
  pause
  exit /b 1
)
echo.
echo Smoke test completed. DO NOT use smoke-test results in the manuscript.
pause
