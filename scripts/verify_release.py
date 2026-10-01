#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
scripts=[
 'check_required_materials.py',
 'verify_stage_source_hashes.py',
 'recompute_table5.py',
 'recompute_focal_statistics.py',
 'verify_crossfit_outputs.py',
 'verify_duplicate_sensitivity.py',
 'verify_supplementary_tables.py',
 'check_no_primary_friedman.py',
]
for s in scripts:
    print(f'== {s} ==')
    subprocess.run([sys.executable, str(ROOT/'scripts'/s)], cwd=ROOT, check=True)
print('PASS: release verification suite completed.')
