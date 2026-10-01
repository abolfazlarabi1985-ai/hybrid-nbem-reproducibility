#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 stage1_major_revision.py --project-root "$PWD" --mode bank --seeds 13 21 42 87 123 --folds 10 --output-dir results/major_revision_stage1
