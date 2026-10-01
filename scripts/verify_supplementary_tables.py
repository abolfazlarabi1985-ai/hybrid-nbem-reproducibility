#!/usr/bin/env python3
from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'supplementary/tables_S1_S13'
files=[D/f'S{i:02d}.csv' for i in range(1,14)]
missing=[str(p.relative_to(ROOT)) for p in files if not p.exists()]
if missing: raise RuntimeError('Missing supplementary tables: '+', '.join(missing))
index=json.loads((D/'TABLE_INDEX.json').read_text(encoding='utf-8'))
if [x['table'] for x in index] != [f'S{i}' for i in range(1,14)]: raise RuntimeError('Supplementary table index is not S1-S13')
for p in files:
    with p.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.reader(f))
    if len(rows)<2: raise RuntimeError(f'{p.name} has no data rows')
print('PASS: machine-readable Supplementary Tables S1-S13 are complete and sequential.')
