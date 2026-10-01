@echo off
setlocal
pushd "%~dp0"
python stage2_major_revision.py --mode baselines
if errorlevel 1 (
  echo.
  echo ERROR: External baseline run failed. Do not continue to manuscript editing.
  pause
  popd
  exit /b 1
)
echo.
echo PASS: RF and Logistic Regression full run completed.
pause
popd
