#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "results" / "current" / "bank_seed42" / "ablation_dataset_seed42_corrected.csv"
OUTDIR = ROOT / "results" / "current" / "bank_seed42" / "recomputed"
OUTDIR.mkdir(parents=True, exist_ok=True)

FULL = "Hybrid NBEM"
REDUCED = ["Hybrid w/o Dependency", "Hybrid w/o Adaptive", "Hybrid w/o Deep"]

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

df = pd.read_csv(INFILE)
if df["dataset"].nunique() != 20:
    raise RuntimeError("Expected exactly 20 datasets in corrected ablation source.")

means = df.groupby("model")[["f1_weighted", "f1_macro"]].mean()
rows = []
for model in [FULL] + REDUCED:
    rows.append({
        "Model": model,
        "Weighted F1": means.loc[model, "f1_weighted"],
        "Macro-F1": means.loc[model, "f1_macro"],
        "Weighted F1 vs. Hybrid": means.loc[model, "f1_weighted"] - means.loc[FULL, "f1_weighted"],
        "Macro-F1 vs. Hybrid": means.loc[model, "f1_macro"] - means.loc[FULL, "f1_macro"],
    })
table5 = pd.DataFrame(rows)
table5.to_csv(OUTDIR / "table5_ablation_corrected_seed42.csv", index=False)

stats_rows = []
for metric in ["f1_weighted", "f1_macro"]:
    wide = df.pivot(index="dataset", columns="model", values=metric)
    temp = []
    pvals = []
    for model in REDUCED:
        delta = wide[FULL] - wide[model]
        stat, p = wilcoxon(delta, alternative="two-sided")
        temp.append((model, delta, stat, p))
        pvals.append(p)
    adj = holm(pvals)
    for (model, delta, stat, p), p_holm in zip(temp, adj):
        stats_rows.append({
            "metric": metric,
            "comparison": f"{FULL} vs {model}",
            "n_datasets": len(delta),
            "mean_Hybrid_minus_reduced": delta.mean(),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "wilcoxon_statistic": stat,
            "p_raw": p,
            "p_holm": p_holm,
        })
pd.DataFrame(stats_rows).to_csv(OUTDIR / "ablation_wilcoxon_holm_corrected_seed42.csv", index=False)
print(table5.to_string(index=False))

# Validate the recomputation against the retained final manuscript files.
arch_table = ROOT / "results" / "current" / "bank_seed42" / "table5_ablation_corrected_seed42.csv"
arch_stats = ROOT / "results" / "current" / "bank_seed42" / "ablation_wilcoxon_holm_corrected_seed42.csv"
if arch_table.exists():
    a = pd.read_csv(arch_table)
    b = pd.read_csv(OUTDIR / "table5_ablation_corrected_seed42.csv")
    # Normalize the delta-column header used by the final manuscript export.
    a = a.rename(columns={"Δ Weighted F1 vs. Hybrid": "Weighted F1 vs. Hybrid"})
    common = [c for c in b.columns if c in a.columns]
    if list(a[common].columns) != list(b[common].columns) or not np.allclose(a[common].select_dtypes(include=[np.number]), b[common].select_dtypes(include=[np.number]), rtol=1e-11, atol=1e-12, equal_nan=True):
        raise RuntimeError("Recomputed Table 5 does not match retained final Table 5 values.")
if arch_stats.exists():
    a = pd.read_csv(arch_stats)
    b = pd.read_csv(OUTDIR / "ablation_wilcoxon_holm_corrected_seed42.csv")
    num = [c for c in b.columns if c in a.columns and pd.api.types.is_numeric_dtype(b[c])]
    if not np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=1e-11, atol=1e-12, equal_nan=True):
        raise RuntimeError("Recomputed ablation Wilcoxon/Holm values do not match retained final values.")
print("PASS: corrected Table 5 and ablation Wilcoxon/Holm match retained final values.")
