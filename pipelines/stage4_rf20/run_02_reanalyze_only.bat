@echo off
cd /d "%~dp0"
python stage4_rf20_major_revision.py --project-root "." --mode analyze --rf-jobs -1
pause
