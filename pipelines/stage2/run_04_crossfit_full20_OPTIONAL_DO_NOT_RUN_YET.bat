@echo off
setlocal
pushd "%~dp0"
echo WARNING: This is the expensive full-20-dataset nested cross-fitting run.
echo Run it only after reviewing the representative-subset results.
pause
python stage2_major_revision.py --mode crossfit_full
if errorlevel 1 (
  echo.
  echo ERROR: Full cross-fitted run failed or was interrupted. It is resumable; rerun this file to resume.
  pause
  popd
  exit /b 1
)
echo.
echo PASS: Full-20 cross-fitted run completed.
pause
popd
