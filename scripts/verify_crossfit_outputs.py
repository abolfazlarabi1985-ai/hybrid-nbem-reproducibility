#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/current/crossfit/table_crossfit_subset_delta.csv'
B=ROOT/'results/current/crossfit/20_crossfit_subset_vs_revised_noncross_delta_summary.csv'
a=pd.read_csv(A); b=pd.read_csv(B)
if list(a.columns)!=list(b.columns): raise RuntimeError('Column mismatch')
if a.shape!=b.shape: raise RuntimeError('Shape mismatch')
for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]):
        if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-11, atol=1e-12, equal_nan=True): raise RuntimeError(f'Numeric mismatch: {c}')
    else:
        if not (a[c].astype(str).to_numpy()==b[c].astype(str).to_numpy()).all(): raise RuntimeError(f'Text mismatch: {c}')
pre=pd.read_csv(ROOT/'results/current/crossfit/19_crossfit_subset_predeclared.csv')
if len(pre)!=6: raise RuntimeError('Expected 6 predeclared cross-fit datasets')
if set(a['dataset'])!=set(pre['dataset']): raise RuntimeError('Cross-fit dataset set mismatch')
print('PASS: cross-fitted subset outputs match retained Stage-2 summary for all 6 predeclared datasets.')
