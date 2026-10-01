@echo off
setlocal
pushd "%~dp0"
python stage2_major_revision.py --mode smoke
if errorlevel 1 (
  echo.
  echo ERROR: Stage-2 smoke test failed. Do not continue.
  pause
  popd
  exit /b 1
)
echo.
echo PASS: Stage-2 smoke test completed.
pause
popd
