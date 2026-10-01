#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python stage2_major_revision.py --mode crossfit_subset
