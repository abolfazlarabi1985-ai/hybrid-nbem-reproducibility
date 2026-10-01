#!/usr/bin/env python3
"""Independent reproducibility utility for the final 20-seed focal statistics.

This script works from the retained dataset-level 20-seed means. It recomputes
means, paired differences, wins/ties/losses, Wilcoxon signed-rank tests,
rank-biserial correlations, Holm adjustment across the three planned contrasts
within each metric, and an independent deterministic bootstrap CI over datasets.

The original Stage-3 and Stage-4 orchestration sources are restored under
`pipeline/` and are independently hash-verified against the retained run
manifests. This utility remains a separate cross-check of the consolidated final
three-comparator manuscript table: it recomputes deterministic quantities
exactly and performs an independent deterministic 50,000-replicate bootstrap.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/current/focal_20seed/table_focal_20seed_dataset.csv"
ARCHIVED = ROOT / "results/current/focal_20seed/table_focal_20seed_pairwise.csv"
OUT = ROOT / "results/current/focal_20seed/recomputed_focal_pairwise_verification.csv"
COMPARATORS = ["NBEM", "Logistic Regression", "Random Forest"]
METRICS = ["f1_weighted", "f1_macro"]
BOOT_REPS = 50000
BOOT_BASE_SEED = 20260927

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * p[idx])
        out[idx] = min(1.0, running)
    return out

def rank_biserial(d):
    d = np.asarray(d, dtype=float)
    nz = d != 0
    d = d[nz]
    r = rankdata(np.abs(d))
    total = r.sum()
    return float((r[d > 0].sum() - r[d < 0].sum()) / total) if total else 0.0

df = pd.read_csv(SRC)
if df["dataset"].nunique() != 20:
    raise RuntimeError("Expected exactly 20 datasets.")

rows = []
contrast_index = 0
for metric in METRICS:
    wide = df.pivot(index="dataset", columns="model", values=metric)
    tmp = []
    pvals = []
    for comparator in COMPARATORS:
        d = (wide["Hybrid NBEM"] - wide[comparator]).to_numpy(float)
        stat, p = wilcoxon(d, alternative="two-sided")
        rng = np.random.default_rng(BOOT_BASE_SEED + contrast_index)
        boot = d[rng.integers(0, len(d), size=(BOOT_REPS, len(d)))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        tmp.append((comparator, d, stat, p, lo, hi, rank_biserial(d)))
        pvals.append(p)
        contrast_index += 1
    adj = holm(pvals)
    for (comparator, d, stat, p, lo, hi, rb), ph in zip(tmp, adj):
        rows.append({
            "metric": metric,
            "comparator": comparator,
            "n_datasets": len(d),
            "hybrid_mean": wide["Hybrid NBEM"].mean(),
            "comparator_mean": wide[comparator].mean(),
            "mean_difference": d.mean(),
            "median_difference": np.median(d),
            "ci95_low_independent_recompute": lo,
            "ci95_high_independent_recompute": hi,
            "wins": int((d > 0).sum()),
            "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum()),
            "wilcoxon_statistic": stat,
            "p_raw": p,
            "rank_biserial": rb,
            "p_holm_three_contrasts": ph,
        })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

# Exact/near-exact validation against archived final table for deterministic quantities.
arch = pd.read_csv(ARCHIVED)
keys = ["metric", "comparator"]
merged = arch.merge(out, on=keys, suffixes=("_archived", "_recomputed"), validate="one_to_one")
exact_numeric = [
    "n_datasets", "hybrid_mean", "comparator_mean", "mean_difference",
    "median_difference", "wins", "ties", "losses", "wilcoxon_statistic",
    "p_raw", "rank_biserial", "p_holm_three_contrasts"
]
for col in exact_numeric:
    a = merged[f"{col}_archived"].astype(float).to_numpy()
    b = merged[f"{col}_recomputed"].astype(float).to_numpy()
    if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise RuntimeError(f"Mismatch in {col}")

# Independent 50,000-replicate bootstrap cross-check. The consolidated final
# table is retained as the manuscript source of record; this verifier uses a
# separate documented bootstrap stream and therefore checks agreement within a
# tight numerical tolerance rather than byte-for-byte endpoint identity.
for side in ["low", "high"]:
    a = merged[f"ci95_{side}"].astype(float).to_numpy()
    b = merged[f"ci95_{side}_independent_recompute"].astype(float).to_numpy()
    if np.max(np.abs(a-b)) > 0.001:
        raise RuntimeError(f"Independent bootstrap CI differs by >0.001 on {side} endpoint")

print(f"PASS: focal 20-seed statistics verified. Wrote {OUT.relative_to(ROOT)}")
print("PASS: independent 50,000-replicate bootstrap agrees with retained final CI endpoints within 0.001; original Stage-3/4 source is separately hash-verified.")
