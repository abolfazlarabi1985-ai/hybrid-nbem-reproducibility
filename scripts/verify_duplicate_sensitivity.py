#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/current/duplicate_sensitivity/table_duplicate_sensitivity.csv'
B=ROOT/'results/current/duplicate_sensitivity/36_duplicate_sensitivity_delta_summary.csv'
a=pd.read_csv(A); b=pd.read_csv(B)
if list(a.columns)!=list(b.columns): raise RuntimeError('Column mismatch')
if a.shape!=b.shape: raise RuntimeError('Shape mismatch')
for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]):
        if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-11, atol=1e-12, equal_nan=True): raise RuntimeError(f'Numeric mismatch: {c}')
    else:
        if not (a[c].astype(str).to_numpy()==b[c].astype(str).to_numpy()).all(): raise RuntimeError(f'Text mismatch: {c}')
sel=pd.read_csv(ROOT/'results/current/duplicate_sensitivity/25_selected_duplicate_heavy_datasets.csv')
sel=sel[sel['selected_for_grouped_model_sensitivity'].astype(str).str.lower().isin(['true','1'])]
if set(sel['dataset']) != {'haberman_s_survival','internet_advertisements'}: raise RuntimeError('Unexpected duplicate-sensitivity dataset set')
print('PASS: duplicate-grouped sensitivity outputs match retained Stage-1b summary.')
