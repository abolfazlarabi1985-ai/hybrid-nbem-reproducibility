#!/usr/bin/env python3
"""
NBEM Major Revision - Stage 1
=============================
Purpose
-------
1) Verify that the exact 20-dataset snapshot used in v4 is being used.
2) Perform a reproducible automated leakage/duplicate audit for all 20 datasets.
3) Run a controlled Bank Marketing sensitivity experiment with and without `duration`
   using the exact v4 preprocessing/model code, folds, seeds and metrics.

This script NEVER edits the source CSV files and NEVER overwrites the original v4 code/config.
All revision outputs are written under results/major_revision_stage1/.

Important scientific boundary
-----------------------------
Automated code can detect identifier-like, duplicate, target-deterministic and suspiciously
named variables, but it cannot by itself prove whether every variable is semantically
post-outcome. The generated manual review inventory is therefore part of the audit trail.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "base" / "nbem_article_experiments_v4_EXACT_USED.py"
DEFAULT_CONFIG = ROOT / "config" / "dataset_config_v4_exact_used.json"
DEFAULT_MANIFEST = ROOT / "manifest" / "DATASET_MANIFEST_PORTABLE.csv"


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("nbem_v4_exact", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base v4 module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module(BASE_SCRIPT)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def verify_snapshot(project_root: Path, manifest_path: Path, output_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        rel = Path(str(row["expected_relative_path"]))
        path = project_root / rel
        expected = str(row["sha256"]).strip().lower()
        actual = sha256_file(path) if path.exists() else ""
        rows.append({
            "dataset": row["dataset"],
            "relative_path": rel.as_posix(),
            "exists": path.exists(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_match": bool(path.exists() and actual == expected),
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "00_exact_snapshot_verification.csv", index=False, encoding="utf-8-sig")
    if not out["hash_match"].all():
        bad = out.loc[~out["hash_match"], ["dataset", "relative_path", "exists", "hash_match"]]
        raise RuntimeError(
            "Exact-v4 dataset verification FAILED. Do not run revision experiments.\n"
            + bad.to_string(index=False)
        )
    return out


def duplicate_stats(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    # Exact duplicates including target.
    exact_dup_mask = df.duplicated(keep=False)
    exact_excess = int(df.duplicated(keep="first").sum())

    feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        return {
            "exact_duplicate_excess": exact_excess,
            "rows_in_exact_duplicate_groups": int(exact_dup_mask.sum()),
            "feature_duplicate_excess": 0,
            "rows_in_feature_duplicate_groups": 0,
            "conflicting_feature_groups": 0,
            "rows_in_conflicting_feature_groups": 0,
        }

    feature_dup_mask = df.duplicated(subset=feature_cols, keep=False)
    feature_excess = int(df.duplicated(subset=feature_cols, keep="first").sum())

    # Duplicate feature vectors mapping to >1 target value are especially important.
    dup_only = df.loc[feature_dup_mask, feature_cols + [target]].copy()
    conflicting_groups = 0
    conflicting_rows = 0
    if not dup_only.empty:
        grouped = dup_only.groupby(feature_cols, dropna=False, sort=False)[target].agg(lambda s: s.astype(str).nunique())
        conflict_keys = grouped[grouped > 1]
        conflicting_groups = int(len(conflict_keys))
        if conflicting_groups:
            # Count rows belonging to conflicting feature groups without materializing huge Python tuples.
            counts = dup_only.groupby(feature_cols, dropna=False, sort=False).size()
            conflicting_rows = int(counts.loc[conflict_keys.index].sum())

    return {
        "exact_duplicate_excess": exact_excess,
        "rows_in_exact_duplicate_groups": int(exact_dup_mask.sum()),
        "feature_duplicate_excess": feature_excess,
        "rows_in_feature_duplicate_groups": int(feature_dup_mask.sum()),
        "conflicting_feature_groups": conflicting_groups,
        "rows_in_conflicting_feature_groups": conflicting_rows,
    }


def suspicious_name_reason(column: str) -> str:
    n = normalize_name(column)
    reasons: List[str] = []
    id_tokens = ("id", "identifier", "index", "record", "encounter", "patientnbr", "patientnumber")
    outcome_tokens = ("outcome", "target", "label", "classlabel")
    if n.endswith("id") or n in id_tokens or any(tok in n for tok in ("identifier", "recordid", "encounterid")):
        reasons.append("identifier-like name")
    if n in outcome_tokens:
        reasons.append("target-like name")
    if n in {"duration"}:
        reasons.append("known Bank Marketing post-contact variable requiring sensitivity analysis")
    return "; ".join(reasons)


def prepare_original_datasets(project_root: Path, config: Dict[str, Any]):
    paths = BASE.find_datasets(project_root, use_canonical_20=bool(config.get("use_canonical_20", True)))
    prepared = [BASE.prepare_dataset(path, config) for path in paths]
    return prepared


def audit_all(project_root: Path, config: Dict[str, Any], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_original_datasets(project_root, config)
    if len(prepared) != 20:
        raise RuntimeError(f"Expected 20 prepared datasets, found {len(prepared)}")

    summary_rows: List[Dict[str, Any]] = []
    column_rows: List[Dict[str, Any]] = []
    auto_flag_rows: List[Dict[str, Any]] = []

    for p in prepared:
        df = p.df.copy()
        ds = duplicate_stats(df, p.target)
        summary_rows.append({
            "dataset": p.key,
            "display_name": p.display_name,
            "rows_used": len(df),
            "features_used": df.shape[1] - 1,
            "target": p.target,
            "group_column": p.group_column or "",
            "configured_dropped_columns": ", ".join(p.dropped_columns),
            "automatic_leakage_flags": len(p.leakage_rows),
            "critical_audit_issues": " | ".join(p.critical_issues),
            **ds,
        })

        leakage_by_feature = {str(r["feature"]): r for r in p.leakage_rows}
        for col in df.columns:
            if col == p.target:
                continue
            s = df[col]
            auto = leakage_by_feature.get(str(col), {})
            name_reason = suspicious_name_reason(str(col))
            row = {
                "dataset": p.key,
                "feature": str(col),
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "missing_n": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100.0),
                "automatic_leakage_flag": bool(auto),
                "automatic_flag_reason": auto.get("flag_reason", ""),
                "name_based_review_reason": name_reason,
                "manual_source_review_required": bool(name_reason or auto),
                "manual_decision": "",
                "manual_rationale_or_source": "",
                "action_for_revision": "",
            }
            column_rows.append(row)
            if auto or name_reason:
                auto_flag_rows.append(row.copy())

    summary = pd.DataFrame(summary_rows)
    columns = pd.DataFrame(column_rows)
    flags = pd.DataFrame(auto_flag_rows)

    summary.to_csv(output_dir / "01_dataset_leakage_duplicate_audit_summary.csv", index=False, encoding="utf-8-sig")
    columns.to_csv(output_dir / "02_all_dataset_feature_review_inventory.csv", index=False, encoding="utf-8-sig")
    flags.to_csv(output_dir / "03_automatic_and_name_based_flags_for_manual_review.csv", index=False, encoding="utf-8-sig")

    # Bank-specific traceability file.
    bank = next(p for p in prepared if p.key == "bank_marketing")
    duration_col = BASE.resolve_column(bank.df.columns, ["duration"])
    bank_record = pd.DataFrame([{
        "dataset": bank.key,
        "target": bank.target,
        "rows": len(bank.df),
        "features_in_original_v4": bank.df.shape[1] - 1,
        "duration_present_in_original_v4_predictors": bool(duration_col is not None and duration_col != bank.target),
        "resolved_duration_column": duration_col or "",
        "required_revision_action": "Run controlled sensitivity with duration removed" if duration_col else "No duration column found",
    }])
    bank_record.to_csv(output_dir / "04_bank_duration_status.csv", index=False, encoding="utf-8-sig")

    return summary, columns


def add_bank_duration_drop(config: Dict[str, Any]) -> Dict[str, Any]:
    revised = copy.deepcopy(config)
    overrides = revised.setdefault("dataset_overrides", {})
    bank = overrides.setdefault("bank_marketing", {})
    drops = list(bank.get("drop_feature_candidates", []))
    if "duration" not in [str(x).lower() for x in drops]:
        drops.append("duration")
    bank["drop_feature_candidates"] = drops
    revised["experiment_revision"] = "major_revision_stage1_bank_duration_removed"
    return revised


def summarize_bank_results(df: pd.DataFrame, condition: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "OK"].copy()
    metric_cols = [
        "accuracy", "balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_macro", "recall_macro", "f1_macro", "roc_auc_weighted",
        "train_time_s", "inference_time_s", "peak_memory_mb",
    ]
    per_seed = (
        ok.groupby(["dataset", "display_name", "model", "seed"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    per_seed.insert(0, "condition", condition)

    agg_rows: List[Dict[str, Any]] = []
    for model, g in per_seed.groupby("model", sort=False):
        row: Dict[str, Any] = {
            "condition": condition,
            "dataset": g["dataset"].iloc[0],
            "display_name": g["display_name"].iloc[0],
            "model": model,
            "n_seeds": int(g["seed"].nunique()),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_sd_across_seed_means"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(row)
    return per_seed, pd.DataFrame(agg_rows)


def bank_sensitivity(
    project_root: Path,
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Sequence[int],
    folds: int,
    n_bins: int,
    mlp_max_iter: int,
    quick: bool,
) -> Dict[str, pd.DataFrame]:
    paths = BASE.find_datasets(project_root, use_canonical_20=True)
    bank_path = next(p for p in paths if BASE.canonical_dataset_key(p) == "bank_marketing")

    original = BASE.prepare_dataset(bank_path, config)
    revised_config = add_bank_duration_drop(config)
    without = BASE.prepare_dataset(bank_path, revised_config)

    if BASE.resolve_column(original.df.columns, ["duration"]) is None:
        raise RuntimeError("The exact v4 Bank Marketing dataset does not contain duration; sensitivity run aborted.")
    if BASE.resolve_column(without.df.columns, ["duration"]) is not None:
        raise RuntimeError("duration was not removed in the revised Bank configuration; sensitivity run aborted.")
    if len(original.df) != len(without.df):
        raise RuntimeError("Row count changed when dropping duration; this should not happen.")

    # Preserve the exact v4 primary key-model comparison.
    models = list(BASE.KEY_MODELS)
    frames: Dict[str, List[pd.DataFrame]] = {"with_duration": [], "without_duration": []}
    for seed in seeds:
        print(f"[Bank sensitivity] seed={seed}: WITH duration")
        a = BASE.evaluate_prepared_dataset(
            prepared=original,
            model_names=models,
            seed=int(seed),
            requested_folds=int(folds),
            n_bins=int(n_bins),
            quick=bool(quick),
            prefer_grouped=False,
            forced_protocol=None,
            analysis_name="bank_duration_sensitivity:with_duration",
            mlp_max_iter=int(mlp_max_iter),
        )
        a.insert(0, "condition", "with_duration")
        frames["with_duration"].append(a)

        print(f"[Bank sensitivity] seed={seed}: WITHOUT duration")
        b = BASE.evaluate_prepared_dataset(
            prepared=without,
            model_names=models,
            seed=int(seed),
            requested_folds=int(folds),
            n_bins=int(n_bins),
            quick=bool(quick),
            prefer_grouped=False,
            forced_protocol=None,
            analysis_name="bank_duration_sensitivity:without_duration",
            mlp_max_iter=int(mlp_max_iter),
        )
        b.insert(0, "condition", "without_duration")
        frames["without_duration"].append(b)

    with_df = pd.concat(frames["with_duration"], ignore_index=True)
    without_df = pd.concat(frames["without_duration"], ignore_index=True)
    with_df.to_csv(output_dir / "10_bank_with_duration_all_fold_results.csv", index=False, encoding="utf-8-sig")
    without_df.to_csv(output_dir / "11_bank_without_duration_all_fold_results.csv", index=False, encoding="utf-8-sig")

    with_seed, with_summary = summarize_bank_results(with_df, "with_duration")
    without_seed, without_summary = summarize_bank_results(without_df, "without_duration")
    per_seed = pd.concat([with_seed, without_seed], ignore_index=True)
    summary = pd.concat([with_summary, without_summary], ignore_index=True)
    per_seed.to_csv(output_dir / "12_bank_duration_per_seed_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "13_bank_duration_overall_summary.csv", index=False, encoding="utf-8-sig")

    # Paired deltas based on seed-level fold means; negative means performance fell after removal.
    metrics = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc_weighted", "train_time_s", "inference_time_s"]
    left = with_seed.rename(columns={m: f"{m}_with_duration" for m in metrics})
    right = without_seed.rename(columns={m: f"{m}_without_duration" for m in metrics})
    keep = ["dataset", "display_name", "model", "seed"]
    paired = left[keep + [f"{m}_with_duration" for m in metrics]].merge(
        right[keep + [f"{m}_without_duration" for m in metrics]], on=keep, how="inner", validate="one_to_one"
    )
    for m in metrics:
        paired[f"delta_without_minus_with_{m}"] = paired[f"{m}_without_duration"] - paired[f"{m}_with_duration"]
    paired.to_csv(output_dir / "14_bank_duration_paired_seed_deltas.csv", index=False, encoding="utf-8-sig")

    delta_agg_rows: List[Dict[str, Any]] = []
    for model, g in paired.groupby("model", sort=False):
        r: Dict[str, Any] = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for m in metrics:
            c = f"delta_without_minus_with_{m}"
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        delta_agg_rows.append(r)
    delta_agg = pd.DataFrame(delta_agg_rows)
    delta_agg.to_csv(output_dir / "15_bank_duration_delta_summary.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "16_bank_revised_config_duration_removed.json").open("w", encoding="utf-8") as f:
        json.dump(revised_config, f, indent=2, ensure_ascii=False)

    return {
        "with_duration": with_df,
        "without_duration": without_df,
        "per_seed": per_seed,
        "summary": summary,
        "paired": paired,
        "delta_summary": delta_agg,
    }


def write_run_manifest(args: argparse.Namespace, config: Dict[str, Any], output_dir: Path) -> None:
    payload = {
        "stage": "major_revision_stage1",
        "base_v4_script_sha256": sha256_file(BASE_SCRIPT),
        "base_v4_config_sha256": sha256_file(Path(args.config)),
        "dataset_manifest_sha256": sha256_file(Path(args.manifest)),
        "mode": args.mode,
        "seeds": args.seeds if args.seeds is not None else config.get("seeds"),
        "folds": args.folds if args.folds is not None else config.get("requested_folds"),
        "n_bins": config.get("n_bins", 5),
        "mlp_max_iter": config.get("mlp_max_iter", 300),
        "quick": bool(args.quick),
        "notes": [
            "Original v4 source/config are not modified.",
            "Bank sensitivity compares the same four primary in-family models under identical seeds and CV except for removal of duration.",
            "Automated audit does not replace semantic/manual source review of post-outcome variables.",
        ],
    }
    with (output_dir / "RUN_MANIFEST.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBEM Major Revision Stage 1: exact-data audit + Bank duration sensitivity")
    parser.add_argument("--project-root", type=str, default=str(ROOT), help="Folder containing datasets/processed")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", type=str, default="results/major_revision_stage1")
    parser.add_argument("--mode", choices=["audit", "bank", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Smoke-test only; NEVER use quick results in the paper.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    seeds = list(args.seeds) if args.seeds is not None and len(args.seeds) else list(config.get("seeds", [13, 21, 42, 87, 123]))
    folds = int(args.folds if args.folds is not None else config.get("requested_folds", 10))
    n_bins = int(config.get("n_bins", 5))
    mlp_max_iter = int(config.get("mlp_max_iter", 300))

    print("Step 0/3: verifying exact v4 dataset snapshot...")
    verify = verify_snapshot(project_root, manifest_path, output_dir)
    print(f"  exact hash matches: {int(verify['hash_match'].sum())}/{len(verify)}")

    if args.mode in {"audit", "all"}:
        print("Step 1/3: running all-dataset leakage/duplicate audit...")
        summary, columns = audit_all(project_root, config, output_dir)
        print(f"  audited datasets: {len(summary)}")
        print(f"  feature inventory rows: {len(columns)}")

    if args.mode in {"bank", "all"}:
        print("Step 2/3: running Bank Marketing duration sensitivity...")
        if args.quick:
            print("  WARNING: --quick is a smoke test only. Do not use these outputs in the manuscript.")
        bank_sensitivity(
            project_root=project_root,
            config=config,
            output_dir=output_dir,
            seeds=seeds,
            folds=folds,
            n_bins=n_bins,
            mlp_max_iter=mlp_max_iter,
            quick=args.quick,
        )

    write_run_manifest(args, config, output_dir)
    print("Step 3/3: done.")
    print(f"Outputs: {output_dir}")
    print("Do not edit the manuscript yet. Send the entire output folder for review before Stage 2.")


if __name__ == "__main__":
    main()
