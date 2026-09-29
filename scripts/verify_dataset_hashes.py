#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
root=Path(__file__).resolve().parent
manifest=root/'DATASET_MANIFEST_PORTABLE.csv'
missing=[]; bad=[]; ok=0
with manifest.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        p=root/r['expected_relative_path']
        if not p.exists(): missing.append(str(p)); continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h.lower()!=r['sha256'].lower(): bad.append((str(p),h,r['sha256']))
        else: ok+=1
print(f'OK: {ok}/20')
if missing:
    print('MISSING:'); [print(' -',x) for x in missing]
if bad:
    print('HASH MISMATCH:'); [print(' -',*x,sep='\n   ') for x in bad]
sys.exit(1 if missing or bad else 0)
