#!/usr/bin/env python3
"""
NBEM Major Revision - Stage 1B
==============================

Goals
-----
1) Re-verify the exact 20-file v4 raw-data snapshot (SHA-256).
2) Preserve the Stage-1 Bank Marketing correction (`duration` removed in config,
   raw CSV unchanged).
3) Convert the five Stage-1 name-based flags into explicit semantic decisions.
4) Quantify whether exact predictor duplicates can cross train/test folds under the
   original article CV protocol.
5) For duplicate-heavy datasets, run a duplicate-group-aware CV sensitivity where
   identical predictor vectors are forced to stay in one fold.

No duplicate row is automatically deleted. This is a sensitivity analysis.
The exact v4 implementation is imported rather than rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "base" / "nbem_article_experiments_v4_EXACT_USED.py"
DEFAULT_CONFIG = ROOT / "config" / "dataset_config_revision_stage1_bank_duration_removed.json"
DEFAULT_MANIFEST = ROOT / "manifest" / "DATASET_MANIFEST_PORTABLE.csv"
STAGE1_AUDIT = ROOT / "previous_stage1_results" / "01_dataset_leakage_duplicate_audit_summary.csv"
REFERENCE_V4_RESULTS = ROOT / "reference_v4_record_results" / "all_fold_results_v4.csv"
DEFAULT_SEEDS = [13, 21, 42, 87, 123]
DEFAULT_CORE_MODELS = ["NBEM", "Hybrid NBEM"]
REPORT_MODEL_NAMES = {
    "NBEM": "NBEM-prop (implementation-level NBEM comparator)",
    "Hybrid NBEM": "Hybrid NBEM",
}


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("nbem_v4_exact_stage1b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import exact v4 module: {path}")
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
            "exists": bool(path.exists()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_match": bool(path.exists() and actual == expected),
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "00_exact_snapshot_verification.csv", index=False, encoding="utf-8-sig")
    if len(out) != 20 or not out["hash_match"].all():
        bad = out.loc[~out["hash_match"], ["dataset", "relative_path", "exists", "hash_match"]]
        raise RuntimeError("Exact-v4 dataset verification FAILED.\n" + bad.to_string(index=False))
    return out


def prepare_all(project_root: Path, config: Dict[str, Any]):
    paths = BASE.find_datasets(project_root, use_canonical_20=bool(config.get("use_canonical_20", True)))
    prepared = [BASE.prepare_dataset(path, config) for path in paths]
    if len(prepared) != 20:
        raise RuntimeError(f"Expected 20 prepared datasets, found {len(prepared)}")
    return prepared


def predictor_signature_ids(df: pd.DataFrame, target: str) -> np.ndarray:
    features = df.drop(columns=[target])
    hashes = pd.util.hash_pandas_object(features, index=False, categorize=True).to_numpy(dtype=np.uint64)
    codes, _ = pd.factorize(hashes, sort=False)
    return codes.astype(np.int64)


def semantic_review_table(prepared_by_key: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "dataset": "bank_marketing",
            "feature": "duration",
            "decision": "REMOVE_FROM_REVISED_PRIMARY_PROTOCOL",
            "reason": "Last-contact duration is known only after the call; Stage 1 quantified its effect. Raw CSV is unchanged.",
            "source_note": "UCI: last contact duration, in seconds.",
            "source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        },
        {
            "dataset": "diabetes_130_us_hospitals_for_years_1999_2008",
            "feature": "admission_type_id",
            "decision": "KEEP_AS_CATEGORICAL_FEATURE",
            "reason": "UCI marks it as a categorical Feature. The integer is a category code, not a unique encounter identifier.",
            "source_note": "UCI Diabetes 130-US Hospitals variable table.",
            "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        },
        {
            "dataset": "diabetes_130_us_hospitals_for_years_1999_2008",
            "feature": "admission_source_id",
            "decision": "KEEP_AS_CATEGORICAL_FEATURE",
            "reason": "UCI marks it as a categorical Feature. The integer is a source-category code, not a patient/encounter identifier.",
            "source_note": "UCI Diabetes 130-US Hospitals variable table.",
            "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        },
        {
            "dataset": "diabetes_130_us_hospitals_for_years_1999_2008",
            "feature": "discharge_disposition_id",
            "decision": "KEEP_WITH_PREDICTION_TIME_CLARIFICATION",
            "reason": "UCI marks it as a categorical Feature. The benchmark target is readmission within 30 days of discharge; the revised manuscript must define prediction at discharge. It is not the unique encounter ID.",
            "source_note": "UCI Diabetes 130-US Hospitals variable table and target definition.",
            "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        },
        {
            "dataset": "hepatitis",
            "feature": "STEROID",
            "decision": "KEEP_AS_CATEGORICAL_FEATURE",
            "reason": "The name heuristic falsely matched the letters 'id' at the end of 'steroid'; UCI defines Steroid as a categorical no/yes feature.",
            "source_note": "UCI Hepatitis variable table.",
            "source_url": "https://archive.ics.uci.edu/dataset/46/hepatitis",
        },
    ]
    out = pd.DataFrame(rows)
    ok_list: List[bool] = []
    notes: List[str] = []
    for _, r in out.iterrows():
        p = prepared_by_key[str(r["dataset"])]
        exists = str(r["feature"]) in [str(c) for c in p.df.columns]
        if r["dataset"] == "bank_marketing" and r["feature"] == "duration":
            ok = not exists
            note = "absent after Stage-1 revision config" if ok else "ERROR: duration still present"
        else:
            ok = exists
            note = "present as documented" if ok else "ERROR: documented feature missing"
        ok_list.append(bool(ok))
        notes.append(note)
    out["revision_config_integrity_ok"] = ok_list
    out["revision_config_integrity_note"] = notes
    return out


def load_stage1_duplicate_profile() -> pd.DataFrame:
    if not STAGE1_AUDIT.exists():
        raise FileNotFoundError(f"Stage-1 audit not found: {STAGE1_AUDIT}")
    a = pd.read_csv(STAGE1_AUDIT, encoding="utf-8-sig")
    if len(a) != 20:
        raise RuntimeError(f"Expected 20 rows in Stage-1 audit, found {len(a)}")
    a["rows_in_feature_duplicate_groups_pct"] = (
        100.0 * a["rows_in_feature_duplicate_groups"] / a["rows_used"].clip(lower=1)
    )
    a["rows_in_conflicting_feature_groups_pct"] = (
        100.0 * a["rows_in_conflicting_feature_groups"] / a["rows_used"].clip(lower=1)
    )
    a["duplicate_profile_provenance"] = "Stage-1 exact-data audit"
    return a


def original_main_cv_plan(prepared, seed: int, requested_folds: int):
    df = prepared.df.copy()
    encoder = BASE.LabelEncoder()
    y = encoder.fit_transform(df[prepared.target].astype(str))
    prefer_grouped = bool(
        prepared.groups is not None
        and prepared.key == "diabetes_130_us_hospitals_for_years_1999_2008"
    )
    plan = BASE.make_cv_plan(
        y=y,
        seed=int(seed),
        requested_folds=int(requested_folds),
        X=df,
        groups=prepared.groups,
        prefer_grouped=prefer_grouped,
        forced_protocol=None,
    )
    return plan


def split_exposure_for_dataset(prepared, seeds: Sequence[int], requested_folds: int) -> pd.DataFrame:
    df = prepared.df.copy().reset_index(drop=True)
    signature = predictor_signature_ids(df, prepared.target)
    y_text = df[prepared.target].astype(str).to_numpy()
    tmp = pd.DataFrame({"g": signature, "y": y_text})
    conflict_ids = set(tmp.groupby("g", sort=False)["y"].nunique().loc[lambda s: s > 1].index.tolist())

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        plan = original_main_cv_plan(prepared, int(seed), int(requested_folds))
        total_test = total_exposed = total_conflict_exposed = 0
        for fold, (tr, te) in enumerate(plan.splits, 1):
            train_groups = set(signature[tr].tolist())
            exposed = np.fromiter((g in train_groups for g in signature[te]), dtype=bool, count=len(te))
            conflict = np.fromiter((g in conflict_ids for g in signature[te]), dtype=bool, count=len(te))
            n_te = int(len(te))
            n_exp = int(exposed.sum())
            n_conf = int((exposed & conflict).sum())
            total_test += n_te
            total_exposed += n_exp
            total_conflict_exposed += n_conf
            rows.append({
                "dataset": prepared.key,
                "display_name": prepared.display_name,
                "seed": int(seed),
                "fold": int(fold),
                "cv_protocol": plan.protocol,
                "test_rows": n_te,
                "test_rows_with_identical_predictor_in_training": n_exp,
                "exposure_pct": float(100.0 * n_exp / max(n_te, 1)),
                "conflicting_duplicate_test_rows_exposed": n_conf,
            })
        rows.append({
            "dataset": prepared.key,
            "display_name": prepared.display_name,
            "seed": int(seed),
            "fold": 0,
            "cv_protocol": plan.protocol,
            "test_rows": int(total_test),
            "test_rows_with_identical_predictor_in_training": int(total_exposed),
            "exposure_pct": float(100.0 * total_exposed / max(total_test, 1)),
            "conflicting_duplicate_test_rows_exposed": int(total_conflict_exposed),
        })
    return pd.DataFrame(rows)


def run_audit(project_root: Path, config: Dict[str, Any], output_dir: Path, seeds: Sequence[int], folds: int, heavy_threshold_pct: float) -> pd.DataFrame:
    prepared = prepare_all(project_root, config)
    by_key = {p.key: p for p in prepared}

    semantic = semantic_review_table(by_key)
    semantic.to_csv(output_dir / "20_semantic_review_decisions.csv", index=False, encoding="utf-8-sig")
    if not semantic["revision_config_integrity_ok"].all():
        raise RuntimeError("Semantic-review config integrity failed. Inspect 20_semantic_review_decisions.csv")

    profile = load_stage1_duplicate_profile()
    profile.to_csv(output_dir / "21_duplicate_profile_all_datasets.csv", index=False, encoding="utf-8-sig")

    # Exposure is mathematically zero when the full dataset has no duplicate predictor vectors.
    # Therefore only the six Stage-1 datasets with feature duplicates need hashing/split analysis.
    duplicate_keys = profile.loc[profile["rows_in_feature_duplicate_groups"] > 0, "dataset"].tolist()
    exposure_frames: List[pd.DataFrame] = []
    for i, key in enumerate(duplicate_keys, 1):
        p = by_key[key]
        print(f"[exposure {i}/{len(duplicate_keys)}] {key}", flush=True)
        exposure_frames.append(split_exposure_for_dataset(p, seeds=seeds, requested_folds=folds))
    exposure = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    exposure.to_csv(output_dir / "22_duplicate_split_exposure_all_folds.csv", index=False, encoding="utf-8-sig")

    seed_rows: List[Dict[str, Any]] = []
    if not exposure.empty:
        seed_rows.extend(exposure.loc[exposure["fold"] == 0].to_dict("records"))
    no_dup = profile.loc[profile["rows_in_feature_duplicate_groups"] == 0]
    for _, r in no_dup.iterrows():
        for seed in seeds:
            seed_rows.append({
                "dataset": r["dataset"],
                "display_name": r["display_name"],
                "seed": int(seed),
                "fold": 0,
                "cv_protocol": "not_run_no_predictor_duplicates",
                "test_rows": int(r["rows_used"]),
                "test_rows_with_identical_predictor_in_training": 0,
                "exposure_pct": 0.0,
                "conflicting_duplicate_test_rows_exposed": 0,
            })
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(output_dir / "23_duplicate_split_exposure_per_seed.csv", index=False, encoding="utf-8-sig")

    agg = per_seed.groupby(["dataset", "display_name"], as_index=False).agg(
        exposure_pct_mean=("exposure_pct", "mean"),
        exposure_pct_sd=("exposure_pct", "std"),
        exposure_pct_max=("exposure_pct", "max"),
        conflicting_duplicate_test_rows_exposed_total=("conflicting_duplicate_test_rows_exposed", "sum"),
    )
    agg["exposure_pct_sd"] = agg["exposure_pct_sd"].fillna(0.0)
    keep_cols = [
        "dataset", "display_name", "rows_used", "features_used", "target",
        "feature_duplicate_excess", "rows_in_feature_duplicate_groups",
        "rows_in_feature_duplicate_groups_pct", "conflicting_feature_groups",
        "rows_in_conflicting_feature_groups", "rows_in_conflicting_feature_groups_pct",
    ]
    agg = agg.merge(profile[keep_cols], on=["dataset", "display_name"], how="left", validate="one_to_one")
    agg["selected_for_grouped_model_sensitivity"] = (
        agg["rows_in_feature_duplicate_groups_pct"] >= float(heavy_threshold_pct)
    )
    agg.to_csv(output_dir / "24_duplicate_split_exposure_summary.csv", index=False, encoding="utf-8-sig")
    selected = agg.loc[agg["selected_for_grouped_model_sensitivity"]].copy()
    selected.to_csv(output_dir / "25_selected_duplicate_heavy_datasets.csv", index=False, encoding="utf-8-sig")
    return selected


def summarize_model_results(df: pd.DataFrame, protocol_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [
        "accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc_weighted",
        "train_time_s", "inference_time_s", "peak_memory_mb",
    ]
    per_seed = ok.groupby(["dataset", "display_name", "model", "seed"], as_index=False)[metric_cols].mean(numeric_only=True)
    per_seed.insert(0, "duplicate_cv_protocol", protocol_label)
    per_seed["report_model"] = per_seed["model"].map(REPORT_MODEL_NAMES).fillna(per_seed["model"])

    rows: List[Dict[str, Any]] = []
    for (ds, disp, model), g in per_seed.groupby(["dataset", "display_name", "model"], sort=False):
        r: Dict[str, Any] = {
            "duplicate_cv_protocol": protocol_label,
            "dataset": ds,
            "display_name": disp,
            "model": model,
            "report_model": REPORT_MODEL_NAMES.get(model, model),
            "n_seeds": int(g["seed"].nunique()),
        }
        for m in metric_cols:
            r[f"{m}_mean"] = float(g[m].mean())
            r[f"{m}_sd_across_seed_means"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(r)
    return per_seed, pd.DataFrame(rows)


def selected_duplicate_keys(heavy_threshold_pct: float, explicit_datasets: Optional[Sequence[str]]) -> List[str]:
    if explicit_datasets:
        return list(explicit_datasets)
    profile = load_stage1_duplicate_profile()
    return profile.loc[
        profile["rows_in_feature_duplicate_groups_pct"] >= float(heavy_threshold_pct), "dataset"
    ].tolist()


def load_record_reference(selected_keys: Sequence[str], seeds: Sequence[int], model_names: Sequence[str]) -> pd.DataFrame:
    if not REFERENCE_V4_RESULTS.exists():
        raise FileNotFoundError(f"v4 reference results not found: {REFERENCE_V4_RESULTS}")
    df = pd.read_csv(REFERENCE_V4_RESULTS, encoding="utf-8-sig")
    out = df[
        df["dataset"].isin(selected_keys)
        & df["seed"].isin(list(map(int, seeds)))
        & df["model"].isin(list(model_names))
    ].copy()
    expected = {(d, int(s), m) for d in selected_keys for s in seeds for m in model_names}
    observed = set(zip(out["dataset"], out["seed"].astype(int), out["model"]))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError("Missing record-level v4 reference combinations: " + repr(missing[:10]))
    bad = out[out["status"] != "OK"]
    if not bad.empty:
        raise RuntimeError("Non-OK rows exist in record-level v4 reference for selected models/datasets.")
    return out


def run_duplicate_group_sensitivity(
    project_root: Path,
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Sequence[int],
    folds: int,
    n_bins: int,
    mlp_max_iter: int,
    heavy_threshold_pct: float,
    model_names: Sequence[str],
    explicit_datasets: Optional[Sequence[str]],
    quick: bool,
    rerun_record: bool,
) -> None:
    prepared = prepare_all(project_root, config)
    by_key = {p.key: p for p in prepared}
    selected_keys = selected_duplicate_keys(heavy_threshold_pct, explicit_datasets)
    if not selected_keys:
        raise RuntimeError("No duplicate-heavy datasets selected.")
    missing_keys = [k for k in selected_keys if k not in by_key]
    if missing_keys:
        raise RuntimeError("Unknown selected dataset key(s): " + ", ".join(missing_keys))
    print("Selected: " + ", ".join(selected_keys), flush=True)

    if rerun_record:
        record_frames: List[pd.DataFrame] = []
    else:
        record_df = load_record_reference(selected_keys, seeds, model_names)
        record_df.insert(0, "duplicate_cv_protocol", "record_level_v4_reference")
        record_df.to_csv(output_dir / "31_record_level_all_fold_results.csv", index=False, encoding="utf-8-sig")

    grouped_frames: List[pd.DataFrame] = []
    integrity_rows: List[Dict[str, Any]] = []

    for ds_i, key in enumerate(selected_keys, 1):
        p = by_key[key]
        signature = predictor_signature_ids(p.df.reset_index(drop=True), p.target)
        grouped_p = replace(p, groups=signature, group_column="__exact_predictor_signature__")

        encoder = BASE.LabelEncoder()
        y = encoder.fit_transform(p.df[p.target].astype(str))
        for seed in seeds:
            plan = BASE.make_cv_plan(
                y=y, seed=int(seed), requested_folds=int(folds), X=p.df,
                groups=signature, prefer_grouped=True, forced_protocol="grouped",
            )
            for fold, (tr, te) in enumerate(plan.splits, 1):
                overlap = len(set(signature[tr].tolist()).intersection(set(signature[te].tolist())))
                integrity_rows.append({
                    "dataset": key, "seed": int(seed), "fold": int(fold),
                    "cv_protocol": plan.protocol,
                    "predictor_signature_group_overlap_count": int(overlap),
                    "integrity_ok": bool(overlap == 0),
                })

        for seed in seeds:
            if rerun_record:
                print(f"[{ds_i}/{len(selected_keys)}] {key}, seed={seed}: record-level rerun", flush=True)
                a = BASE.evaluate_prepared_dataset(
                    prepared=p, model_names=list(model_names), seed=int(seed),
                    requested_folds=int(folds), n_bins=int(n_bins), quick=bool(quick),
                    prefer_grouped=False, forced_protocol="record_stratified",
                    analysis_name="duplicate_sensitivity:record_level_rerun",
                    mlp_max_iter=int(mlp_max_iter),
                )
                a.insert(0, "duplicate_cv_protocol", "record_level_rerun")
                record_frames.append(a)

            print(f"[{ds_i}/{len(selected_keys)}] {key}, seed={seed}: duplicate-grouped", flush=True)
            b = BASE.evaluate_prepared_dataset(
                prepared=grouped_p, model_names=list(model_names), seed=int(seed),
                requested_folds=int(folds), n_bins=int(n_bins), quick=bool(quick),
                prefer_grouped=True, forced_protocol="grouped",
                analysis_name="duplicate_sensitivity:duplicate_grouped",
                mlp_max_iter=int(mlp_max_iter),
            )
            b.insert(0, "duplicate_cv_protocol", "duplicate_grouped")
            grouped_frames.append(b)

    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(output_dir / "30_grouped_split_integrity.csv", index=False, encoding="utf-8-sig")
    if not integrity["integrity_ok"].all():
        raise RuntimeError("Grouped split integrity failed: duplicate signatures cross train/test.")

    if rerun_record:
        record_df = pd.concat(record_frames, ignore_index=True)
        record_df.to_csv(output_dir / "31_record_level_all_fold_results.csv", index=False, encoding="utf-8-sig")
    grouped_df = pd.concat(grouped_frames, ignore_index=True)
    grouped_df.to_csv(output_dir / "32_duplicate_grouped_all_fold_results.csv", index=False, encoding="utf-8-sig")

    rec_label = "record_level_rerun" if rerun_record else "record_level_v4_reference"
    rec_seed, rec_overall = summarize_model_results(record_df, rec_label)
    grp_seed, grp_overall = summarize_model_results(grouped_df, "duplicate_grouped")
    per_seed = pd.concat([rec_seed, grp_seed], ignore_index=True)
    overall = pd.concat([rec_overall, grp_overall], ignore_index=True)
    per_seed.to_csv(output_dir / "33_duplicate_sensitivity_per_seed_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "34_duplicate_sensitivity_overall_summary.csv", index=False, encoding="utf-8-sig")

    metrics = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc_weighted", "train_time_s", "inference_time_s"]
    left = rec_seed.rename(columns={m: f"{m}_record" for m in metrics})
    right = grp_seed.rename(columns={m: f"{m}_grouped" for m in metrics})
    keys = ["dataset", "display_name", "model", "seed"]
    paired = left[keys + [f"{m}_record" for m in metrics]].merge(
        right[keys + [f"{m}_grouped" for m in metrics]],
        on=keys, how="inner", validate="one_to_one",
    )
    paired["report_model"] = paired["model"].map(REPORT_MODEL_NAMES).fillna(paired["model"])
    for m in metrics:
        paired[f"delta_grouped_minus_record_{m}"] = paired[f"{m}_grouped"] - paired[f"{m}_record"]
    paired.to_csv(output_dir / "35_duplicate_sensitivity_paired_seed_deltas.csv", index=False, encoding="utf-8-sig")

    rows: List[Dict[str, Any]] = []
    for (ds, disp, model), g in paired.groupby(["dataset", "display_name", "model"], sort=False):
        r: Dict[str, Any] = {
            "dataset": ds, "display_name": disp, "model": model,
            "report_model": REPORT_MODEL_NAMES.get(model, model),
            "n_seeds": int(g["seed"].nunique()),
        }
        for m in metrics:
            c = f"delta_grouped_minus_record_{m}"
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(r)
    pd.DataFrame(rows).to_csv(output_dir / "36_duplicate_sensitivity_delta_summary.csv", index=False, encoding="utf-8-sig")


def write_manifest(args: argparse.Namespace, config_path: Path, manifest_path: Path, output_dir: Path, seeds: Sequence[int], folds: int, models: Sequence[str]) -> None:
    payload = {
        "stage": "major_revision_stage1b_duplicate_leakage_sensitivity",
        "base_v4_script_sha256": sha256_file(BASE_SCRIPT),
        "revision_config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "stage1_audit_sha256": sha256_file(STAGE1_AUDIT),
        "record_level_v4_reference_sha256": sha256_file(REFERENCE_V4_RESULTS),
        "mode": args.mode,
        "seeds": list(map(int, seeds)),
        "requested_folds": int(folds),
        "duplicate_heavy_threshold_pct": float(args.heavy_threshold_pct),
        "models": list(models),
        "record_side": "rerun" if args.rerun_record else "exact v4 reference results",
        "quick": bool(args.quick),
        "notes": [
            "Raw CSV files are not edited.",
            "Stage-1 revision config removes Bank Marketing duration in memory.",
            "No duplicate rows are automatically deleted.",
            "Duplicate-grouped CV keeps exact predictor signatures within a single fold.",
            "Default grouped sensitivity evaluates the core comparison: NBEM-prop vs Hybrid NBEM.",
        ],
    }
    with (output_dir / "RUN_MANIFEST_STAGE1B.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBEM Major Revision Stage 1B")
    parser.add_argument("--project-root", type=str, default=str(ROOT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", type=str, default="results/major_revision_stage1b")
    parser.add_argument("--mode", choices=["audit", "sensitivity", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--heavy-threshold-pct", type=float, default=10.0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--sensitivity-datasets", nargs="*", default=None)
    parser.add_argument("--rerun-record", action="store_true",
                        help="Rerun the record-level side instead of reusing the exact v4 record results.")
    parser.add_argument("--quick", action="store_true", help="Smoke test only; NEVER report quick outputs.")
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
    seeds = list(args.seeds) if args.seeds else list(config.get("seeds", DEFAULT_SEEDS))
    folds = int(args.folds if args.folds is not None else config.get("requested_folds", 10))
    n_bins = int(config.get("n_bins", 5))
    mlp_max_iter = int(config.get("mlp_max_iter", 300))
    models = list(args.models) if args.models else list(DEFAULT_CORE_MODELS)

    print("Step 0/4: verifying exact v4 raw dataset snapshot...", flush=True)
    verify = verify_snapshot(project_root, manifest_path, output_dir)
    print(f"  exact hash matches: {int(verify['hash_match'].sum())}/{len(verify)}", flush=True)

    if args.mode in {"audit", "all"}:
        print("Step 1/4: semantic review + duplicate split-exposure audit...", flush=True)
        selected = run_audit(project_root, config, output_dir, seeds, folds, float(args.heavy_threshold_pct))
        print("  selected: " + ", ".join(selected["dataset"].tolist()), flush=True)

    if args.mode in {"sensitivity", "all"}:
        print("Step 2/4: duplicate-grouped model sensitivity...", flush=True)
        if args.quick:
            print("  WARNING: --quick is smoke-test only.", flush=True)
        run_duplicate_group_sensitivity(
            project_root=project_root, config=config, output_dir=output_dir,
            seeds=seeds, folds=folds, n_bins=n_bins, mlp_max_iter=mlp_max_iter,
            heavy_threshold_pct=float(args.heavy_threshold_pct), model_names=models,
            explicit_datasets=args.sensitivity_datasets, quick=bool(args.quick),
            rerun_record=bool(args.rerun_record),
        )

    print("Step 3/4: writing reproducibility manifest...", flush=True)
    write_manifest(args, config_path, manifest_path, output_dir, seeds, folds, models)
    print("Step 4/4: done.", flush=True)
    print(f"Outputs: {output_dir}", flush=True)
    print("Do not edit the manuscript yet. Send the complete Stage-1B output folder before Stage 2.", flush=True)


if __name__ == "__main__":
    main()
