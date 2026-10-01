#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 stage4_rf20_major_revision.py --project-root "." --mode full --rf-jobs -1 2>&1 | tee stage4_rf20_console.log
