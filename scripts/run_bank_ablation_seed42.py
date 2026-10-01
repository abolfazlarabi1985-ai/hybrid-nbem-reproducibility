#!/usr/bin/env python3
"""Regenerate the corrected Bank Marketing reference-seed ablation.

This is a targeted reproducibility utility. It imports the retained exact v4
base experiment implementation, applies the final Bank config that drops
`duration`, and evaluates the four Table-5 Hybrid variants with seed 42.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd

MODELS = [
    "Hybrid NBEM",
    "Hybrid w/o Dependency",
    "Hybrid w/o Adaptive",
    "Hybrid w/o Deep",
]

def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("nbem_v4_exact", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--data-root", default="datasets/processed")
    ap.add_argument("--output-dir", default="results/current/bank_seed42/rerun")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    base_path = root / "src" / "nbem_article_experiments_v4_EXACT_USED.py"
    config_path = root / "config" / "dataset_config_FINAL_bank_duration_removed.json"
    bank_path = root / args.data_root / "bank_marketing" / "bank_marketing.csv"
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(base_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bank = base.prepare_dataset(bank_path, config)

    # Guardrail: corrected protocol must remove duration.
    if "duration" not in [c.lower() for c in bank.dropped_columns]:
        raise RuntimeError("Corrected Bank protocol did not record 'duration' as a dropped feature.")

    frames = []
    for model in MODELS:
        print(f"Running {model} ...", flush=True)
        frame = base.evaluate_prepared_dataset(
            prepared=bank,
            model_names=[model],
            seed=42,
            requested_folds=10,
            n_bins=5,
            quick=False,
            prefer_grouped=False,
            analysis_name="bank_corrected_ablation_seed42",
            mlp_max_iter=300,
        )
        frame.to_csv(out / (model.lower().replace(" ", "_").replace("/", "_") + "_fold_results.csv"), index=False)
        frames.append(frame)

    all_folds = pd.concat(frames, ignore_index=True)
    all_folds.to_csv(out / "bank_seed42_all_ablation_fold_results.csv", index=False)
    summary = all_folds.groupby("model", as_index=False)[["f1_weighted", "f1_macro"]].mean()
    summary.to_csv(out / "bank_seed42_ablation_dataset_level.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
