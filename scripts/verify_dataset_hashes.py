#!/usr/bin/env python3
import argparse, csv, hashlib, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--manifest', default='data/DATASET_MANIFEST_PORTABLE.csv')
    ap.add_argument('--data-root', default=None, help='Optional replacement root for the manifest paths, e.g. /path/to/project')
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    manifest=root/args.manifest
    missing=[]; bad=[]; ok=0
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            rel=Path(r['expected_relative_path'])
            p=(Path(args.data_root).resolve()/rel if args.data_root else root/rel)
            if not p.exists(): missing.append(str(p)); continue
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            if h.lower()!=r['sha256'].lower(): bad.append((str(p),h,r['sha256']))
            else: ok+=1
    print(f'OK: {ok}/20')
    if missing:
        print('MISSING:'); [print(' -',x) for x in missing]
    if bad:
        print('HASH MISMATCH:'); [print(' -',*x,sep='\n   ') for x in bad]
    return 1 if missing or bad else 0
if __name__=='__main__': sys.exit(main())
