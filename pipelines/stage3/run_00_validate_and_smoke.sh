#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python stage3_major_revision.py --project-root . --mode smoke
