@echo off
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
python stage4_rf20_major_revision.py --project-root "." --mode full --rf-jobs -1
pause
