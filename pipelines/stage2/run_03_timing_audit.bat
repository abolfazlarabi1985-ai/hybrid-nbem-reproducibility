@echo off
setlocal
pushd "%~dp0"
python stage2_major_revision.py --mode timing
if errorlevel 1 (
  echo.
  echo ERROR: Timing audit failed.
  pause
  popd
  exit /b 1
)
echo.
echo PASS: Table-6 timing audit completed.
pause
popd
