#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 stage1_major_revision.py --project-root "$PWD" --mode audit
