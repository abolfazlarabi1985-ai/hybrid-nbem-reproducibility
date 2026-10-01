#!/usr/bin/env python3
from pathlib import Path
import zipfile, re
ROOT=Path(__file__).resolve().parents[1]
patterns=[re.compile(r'friedman',re.I), re.compile(r'twelve[- ]model',re.I)]
# Current result text/CSV and final Supplementary XML must not present these analyses.
paths=list((ROOT/'results/current').rglob('*'))
for p in paths:
    if p.is_file() and p.suffix.lower() in {'.csv','.txt','.md','.json'}:
        txt=p.read_text(encoding='utf-8-sig',errors='ignore')
        for pat in patterns:
            if pat.search(txt): raise RuntimeError(f'Historical analysis term found in current evidence: {p.relative_to(ROOT)}')
for docx in [ROOT/'supplementary/Supplementary_Information_S1-S13_FINAL.docx']:
    with zipfile.ZipFile(docx) as z:
        txt=z.read('word/document.xml').decode('utf-8',errors='ignore')
        for pat in patterns:
            if pat.search(txt): raise RuntimeError(f'Historical analysis term found in final Supplementary: {docx.name}')
print('PASS: no Friedman/twelve-model ranking appears in current evidence or final Supplementary.')
