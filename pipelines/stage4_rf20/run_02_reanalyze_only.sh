#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 stage4_rf20_major_revision.py --project-root "." --mode analyze --rf-jobs -1
