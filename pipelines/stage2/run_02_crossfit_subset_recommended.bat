@echo off
setlocal
pushd "%~dp0"
python stage2_major_revision.py --mode crossfit_subset
if errorlevel 1 (
  echo.
  echo ERROR: Cross-fitted subset run failed. Inspect results and send them before proceeding.
  pause
  popd
  exit /b 1
)
echo.
echo PASS: Cross-fitted representative-subset run completed.
pause
popd
