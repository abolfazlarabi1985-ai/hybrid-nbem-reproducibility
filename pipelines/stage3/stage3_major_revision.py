#!/usr/bin/env python3
"""
NBEM Major Revision - Stage 3
=============================

Predeclared 20-seed robustness study for:
  1) Hybrid NBEM vs NBEM-prop (implementation-level NBEM comparator)
  2) Hybrid NBEM vs Logistic Regression

Research-integrity design
-------------------------
- The 20 seeds are locked in config/stage3_seed_plan.json before Stage-3 outcomes.
- The original five seeds are retained without exclusion.
- Fifteen additional seeds were deterministically generated before Stage-3 outcomes.
- All three models share the exact same outer train/test split for each dataset+seed.
- Primary statistical unit is the DATASET, not folds or seeds.
- Primary metrics are Weighted-F1 and Macro-F1.
- Wilcoxon signed-rank tests are paired across 20 dataset-level 20-seed means.
- Holm correction is applied across the two planned comparator contrasts within each metric.
- Effect size is paired rank-biserial correlation.
- 95% CIs for mean paired gains are nonparametric bootstrap CIs over datasets.
- Seed-level analyses are descriptive robustness summaries, not pseudo-replication.

Efficiency
----------
The already validated five-seed final-protocol results are reused. Only the 15
new predeclared seeds are fitted. The final outputs always contain all 20 seeds.

Frozen revision protocol
------------------------
- Exact 20-dataset snapshot verified by SHA-256.
- Bank Marketing `duration` removed by the frozen revision config.
- Diabetes outer CV patient-grouped as in the final protocol.
- Duplicate records are not blindly deleted; Stage-1B remains a separate robustness check.
- Exact v4 NBEM/Hybrid implementation is imported, not rewritten.
- Logistic Regression reproduces the Stage-2 baseline preprocessing/specification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr, t as student_t, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "base" / "nbem_article_experiments_v4_EXACT_USED.py"
DEFAULT_CONFIG = ROOT / "config" / "dataset_config_revision_frozen.json"
V4_CONFIG = ROOT / "config" / "dataset_config_v4_exact_used.json"
DEFAULT_MANIFEST = ROOT / "manifest" / "DATASET_MANIFEST_PORTABLE.csv"
SEED_PLAN = ROOT / "config" / "stage3_seed_plan.json"
PRELOADED = ROOT / "provenance" / "preloaded_first5_all_fold_results.csv"
DEFAULT_OUTPUT = ROOT / "results" / "major_revision_stage3"

METRICS = [
    "accuracy", "balanced_accuracy",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "precision_macro", "recall_macro", "f1_macro", "roc_auc_weighted",
]
PRIMARY_METRICS = ["f1_weighted", "f1_macro"]
MODELS = ["NBEM", "Hybrid NBEM", "Logistic Regression"]
REPORT_NAMES = {
    "NBEM": "NBEM-prop (implementation-level NBEM comparator)",
    "Hybrid NBEM": "Hybrid NBEM",
    "Logistic Regression": "Logistic Regression",
}
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260924
TIE_TOL = 1e-12


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("nbem_v4_exact_stage3", path)
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_seed_plan() -> Dict[str, Any]:
    plan = load_json(SEED_PLAN)
    original = list(map(int, plan["original_five_seeds"]))
    extra = list(map(int, plan["additional_fifteen_seeds"]))
    all20 = list(map(int, plan["all_twenty_seeds"]))
    if original != [13, 21, 42, 87, 123]:
        raise RuntimeError("Locked original seed set was altered.")
    # independently regenerate the fifteen additional seeds
    rng = np.random.default_rng(20260924)
    regen: List[int] = []
    blocked = set(original)
    while len(regen) < 15:
        x = int(rng.integers(1, 1000000))
        if x not in blocked and x not in regen:
            regen.append(x)
    if regen != extra or all20 != original + extra or len(set(all20)) != 20:
        raise RuntimeError("Locked Stage-3 seed plan failed deterministic verification.")
    return plan


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
        raise RuntimeError("Exact dataset verification FAILED.\n" + bad.to_string(index=False))
    return out


def verify_revision_config_scope() -> None:
    a = load_json(V4_CONFIG)
    b = load_json(DEFAULT_CONFIG)
    a_label = a.pop("experiment_revision", None)
    b_label = b.pop("experiment_revision", None)
    bank_a = a.get("dataset_overrides", {}).get("bank_marketing", {})
    bank_b = b.get("dataset_overrides", {}).get("bank_marketing", {})
    drop = bank_b.get("drop_feature_candidates", [])
    # Remove the one scientifically intended Stage-1 difference and compare the rest.
    bank_b2 = dict(bank_b)
    bank_b2.pop("drop_feature_candidates", None)
    a2 = json.loads(json.dumps(a))
    b2 = json.loads(json.dumps(b))
    a2.setdefault("dataset_overrides", {})["bank_marketing"] = bank_a
    b2.setdefault("dataset_overrides", {})["bank_marketing"] = bank_b2
    if a2 != b2 or drop != ["duration"]:
        raise RuntimeError("Frozen revision config differs from v4 beyond the intended Bank duration removal.")


def prepare_all(project_root: Path, config: Dict[str, Any]) -> List[Any]:
    paths = BASE.find_datasets(project_root, use_canonical_20=bool(config.get("use_canonical_20", True)))
    prepared = [BASE.prepare_dataset(path, config) for path in paths]
    if len(prepared) != 20:
        raise RuntimeError(f"Expected 20 prepared datasets, found {len(prepared)}")
    bad = [p for p in prepared if p.critical_issues]
    if bad:
        msg = "\n".join(f"- {p.key}: {' | '.join(p.critical_issues)}" for p in bad)
        raise RuntimeError("Critical dataset preparation issue(s):\n" + msg)
    bank = next(p for p in prepared if p.key == "bank_marketing")
    if "duration" in bank.df.columns:
        raise RuntimeError("Frozen revision protocol failed: Bank Marketing duration remains present.")
    return prepared


def prefer_grouped_main(prepared: Any) -> bool:
    return bool(prepared.groups is not None and prepared.key == "diabetes_130_us_hospitals_for_years_1999_2008")


class LogisticBaselinePreprocessor:
    """Exact Stage-2 Logistic Regression baseline preprocessing."""
    def fit(self, df: pd.DataFrame, target: str) -> "LogisticBaselinePreprocessor":
        self.target = target
        b, c, n = BASE.detect_feature_groups(df, target)
        self.cat_cols = list(b) + list(c)
        self.num_cols = list(n)
        self.cat_imp = SimpleImputer(strategy="most_frequent")
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        self.num_imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X = df.drop(columns=[target])
        if self.cat_cols:
            C = BASE.string_frame_preserve_missing(X[self.cat_cols])
            Cimp = self.cat_imp.fit_transform(C)
            self.ohe.fit(Cimp)
        if self.num_cols:
            N = self.num_imp.fit_transform(X[self.num_cols])
            self.scaler.fit(N)
        return self

    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target])
        parts: List[Any] = []
        if self.cat_cols:
            C = BASE.string_frame_preserve_missing(X[self.cat_cols])
            parts.append(self.ohe.transform(self.cat_imp.transform(C)))
        if self.num_cols:
            N = self.scaler.transform(self.num_imp.transform(X[self.num_cols]).astype(float))
            parts.append(sparse.csr_matrix(N))
        if not parts:
            return sparse.csr_matrix(np.zeros((len(df), 1), dtype=float))
        return sparse.hstack(parts, format="csr")


def lr_factory(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight=None,
        random_state=int(seed),
    )


def normalize_fold_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    keep = [
        "analysis", "dataset", "display_name", "file", "target", "target_transform",
        "model", "seed", "fold", "cv_protocol", "n_splits_or_total_folds",
        "stratified", "grouped", "n_rows", "n_features", "n_classes",
        "n_boolean", "n_categorical", "n_numerical",
    ] + METRICS + ["train_time_s", "inference_time_s", "peak_memory_mb", "status"]
    out = df.copy()
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[keep].copy()
    out["source_provenance"] = source
    out["report_model"] = out["model"].map(REPORT_NAMES).fillna(out["model"])
    return out


def validate_preloaded(plan: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    pre = pd.read_csv(PRELOADED, encoding="utf-8-sig")
    original = list(map(int, plan["original_five_seeds"]))
    checks: List[Dict[str, Any]] = []
    checks.append({"check": "models_exact", "pass": set(pre["model"].unique()) == set(MODELS)})
    checks.append({"check": "datasets_20", "pass": pre["dataset"].nunique() == 20})
    checks.append({"check": "seeds_exact_original_five", "pass": set(map(int, pre["seed"].unique())) == set(original)})
    checks.append({"check": "all_status_OK", "pass": bool(pre["status"].eq("OK").all())})
    combo = pre.groupby(["dataset", "seed", "model"]).size().reset_index(name="rows")
    checks.append({"check": "all_20x5x3_dataset_seed_model_combos_present", "pass": len(combo) == 20 * 5 * 3})
    sig = pre.groupby(["dataset", "seed", "model"]).agg(
        nfold=("fold", "nunique"), protocol=("cv_protocol", "first")
    ).reset_index()
    same = True
    for _, g in sig.groupby(["dataset", "seed"]):
        if g["nfold"].nunique() != 1 or g["protocol"].nunique() != 1 or set(g["model"]) != set(MODELS):
            same = False
            break
    checks.append({"check": "same_outer_fold_count_and_protocol_across_models", "pass": same})
    bank = pre[pre["dataset"].eq("bank_marketing")]
    checks.append({"check": "bank_present_all_models_all_original_seeds", "pass": bank["seed"].nunique() == 5 and bank["model"].nunique() == 3})
    ch = pd.DataFrame(checks)
    ch.to_csv(output_dir / "02_preloaded_first5_validation.csv", index=False, encoding="utf-8-sig")
    if not ch["pass"].all():
        raise RuntimeError("Preloaded first-five validation failed:\n" + ch.to_string(index=False))
    return normalize_fold_frame(pre, "validated preloaded first-five final-protocol result")


def evaluate_three_models(prepared: Any, seed: int, requested_folds: int, n_bins: int,
                          mlp_max_iter: int, quick: bool = False) -> pd.DataFrame:
    BASE.set_reproducibility(int(seed))
    df = prepared.df.copy().reset_index(drop=True)
    groups = prepared.groups.copy() if prepared.groups is not None else None
    if quick and len(df) > 900:
        idx = df.sample(n=900, random_state=int(seed)).index.to_numpy()
        df = df.iloc[idx].reset_index(drop=True)
        if groups is not None:
            groups = groups[idx]

    enc = LabelEncoder()
    y = enc.fit_transform(df[prepared.target].astype(str))
    global_classes = np.arange(len(enc.classes_))
    plan = BASE.make_cv_plan(
        y=y,
        seed=int(seed),
        requested_folds=int(requested_folds),
        X=df,
        groups=groups,
        prefer_grouped=prefer_grouped_main(prepared),
        forced_protocol=None,
    )

    rows: List[Dict[str, Any]] = []
    for fold, (tr, te) in enumerate(plan.splits, 1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        y_train, y_test = y[tr], y[te]
        if len(np.unique(y_train)) < 2:
            for name in MODELS:
                rows.append({
                    "analysis": "major_revision_stage3", "dataset": prepared.key,
                    "display_name": prepared.display_name, "model": name, "seed": int(seed),
                    "fold": int(fold), "cv_protocol": plan.protocol,
                    "n_splits_or_total_folds": int(plan.n_splits),
                    "status": "SKIPPED: fewer than two training classes",
                })
            continue

        # Exact v4 NBEM/Hybrid preprocessing, fitted on this training fold only.
        nbprep = BASE.NBEMPreprocessor(n_bins=n_bins).fit(train_df, prepared.target)
        nbtrain = nbprep.transform(train_df)
        nbtest = nbprep.transform(test_df)

        common = {
            "analysis": "major_revision_stage3",
            "dataset": prepared.key,
            "display_name": prepared.display_name,
            "file": str(prepared.path),
            "target": prepared.target,
            "target_transform": prepared.target_transform,
            "seed": int(seed),
            "fold": int(fold),
            "cv_protocol": plan.protocol,
            "n_splits_or_total_folds": int(plan.n_splits),
            "stratified": bool(plan.stratified),
            "grouped": bool(plan.grouped),
            "n_rows": int(len(df)),
            "n_features": int(df.shape[1] - 1),
            "n_classes": int(len(global_classes)),
            "n_boolean": int(len(nbprep.bool_cols)),
            "n_categorical": int(len(nbprep.cat_cols)),
            "n_numerical": int(len(nbprep.num_cols)),
        }

        for name in ["NBEM", "Hybrid NBEM"]:
            base = {**common, "model": name}
            try:
                spec = BASE.get_model_spec(name, seed=int(seed), mlp_max_iter=int(mlp_max_iter))
                model = spec.factory()
                X_train = BASE.get_input(nbtrain, spec.fit_input)
                X_test = BASE.get_input(nbtest, spec.pred_input)
                tracemalloc.start()
                t0 = time.perf_counter(); model.fit(X_train, y_train); fit_s = time.perf_counter() - t0
                t1 = time.perf_counter(); pred = model.predict(X_test); raw = model.predict_proba(X_test); infer_s = time.perf_counter() - t1
                _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                proba = BASE.align_proba(raw, getattr(model, "classes_", np.unique(y_train)), global_classes)
                metrics = BASE.compute_metrics(y_test, pred, proba, global_classes)
                rows.append({**base, **metrics, "train_time_s": float(fit_s),
                             "inference_time_s": float(infer_s), "peak_memory_mb": float(peak/(1024**2)),
                             "status": "OK"})
            except Exception as exc:
                try: tracemalloc.stop()
                except Exception: pass
                rows.append({**base, "status": f"ERROR: {type(exc).__name__}: {exc}"})

        # Stage-2 exact Logistic Regression representation, on the SAME tr/te split.
        base = {**common, "model": "Logistic Regression"}
        try:
            lrp = LogisticBaselinePreprocessor().fit(train_df, prepared.target)
            Xtr = lrp.transform(train_df)
            Xte = lrp.transform(test_df)
            model = lr_factory(int(seed))
            tracemalloc.start()
            t0 = time.perf_counter(); model.fit(Xtr, y_train); fit_s = time.perf_counter() - t0
            t1 = time.perf_counter(); pred = model.predict(Xte); raw = model.predict_proba(Xte); infer_s = time.perf_counter() - t1
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            proba = BASE.align_proba(raw, getattr(model, "classes_", np.unique(y_train)), global_classes)
            metrics = BASE.compute_metrics(y_test, pred, proba, global_classes)
            rows.append({**base, **metrics, "train_time_s": float(fit_s),
                         "inference_time_s": float(infer_s), "peak_memory_mb": float(peak/(1024**2)),
                         "status": "OK"})
        except Exception as exc:
            try: tracemalloc.stop()
            except Exception: pass
            rows.append({**base, "status": f"ERROR: {type(exc).__name__}: {exc}"})

    return normalize_fold_frame(pd.DataFrame(rows), "new Stage-3 fit from locked additional seed")


def run_additional(prepared_list: Sequence[Any], seeds: Sequence[int], output_dir: Path,
                   requested_folds: int, n_bins: int, mlp_max_iter: int) -> pd.DataFrame:
    cache = output_dir / "cache" / "additional_fifteen"
    cache.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    total = len(prepared_list) * len(seeds)
    k = 0
    for p in prepared_list:
        for seed in seeds:
            k += 1
            fn = cache / f"{p.key}__seed_{int(seed)}.csv"
            if fn.exists():
                print(f"[{k}/{total}] resume: {p.key}, seed={seed}")
                fr = pd.read_csv(fn, encoding="utf-8-sig")
            else:
                print(f"[{k}/{total}] run: {p.key}, seed={seed}")
                fr = evaluate_three_models(p, int(seed), requested_folds, n_bins, mlp_max_iter, quick=False)
                fr.to_csv(fn, index=False, encoding="utf-8-sig")
            frames.append(fr)
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                output_dir / "09_additional_fifteen_PARTIAL_DO_NOT_REPORT.csv", index=False, encoding="utf-8-sig"
            )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def per_seed_dataset_model(folds: pd.DataFrame) -> pd.DataFrame:
    ok = folds[folds["status"].eq("OK")].copy()
    cols = [c for c in METRICS if c in ok.columns]
    return ok.groupby(["dataset", "display_name", "model", "report_model", "seed"], as_index=False)[cols].mean()


def mean_ci_t(vals: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    m = float(np.mean(vals)) if len(vals) else math.nan
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if len(vals) <= 1:
        return m, sd, m, m
    se = sd / math.sqrt(len(vals))
    crit = float(student_t.ppf(1-alpha/2, df=len(vals)-1))
    return m, sd, m-crit*se, m+crit*se


def dataset_model_summary(ps: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, g in ps.groupby(["dataset", "display_name", "model", "report_model"], sort=False):
        row = dict(zip(["dataset", "display_name", "model", "report_model"], keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for m in METRICS:
            if m not in g: continue
            mean, sd, lo, hi = mean_ci_t(pd.to_numeric(g[m], errors="coerce").to_numpy())
            row[f"{m}_mean"] = mean; row[f"{m}_sd_across_seeds"] = sd
            row[f"{m}_ci95_seed_lo"] = lo; row[f"{m}_ci95_seed_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_model_summary(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, report), g in ds.groupby(["model", "report_model"], sort=False):
        row = {"model": model, "report_model": report, "n_datasets": int(g["dataset"].nunique())}
        for m in METRICS:
            c = f"{m}_mean"
            if c not in g: continue
            mean, sd, lo, hi = mean_ci_t(pd.to_numeric(g[c], errors="coerce").to_numpy())
            row[f"{m}_mean_across_datasets"] = mean
            row[f"{m}_sd_across_datasets"] = sd
            row[f"{m}_ci95_dataset_lo"] = lo
            row[f"{m}_ci95_dataset_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d) & (np.abs(d) > TIE_TOL)]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d), method="average")
    wp = float(r[d > 0].sum()); wn = float(r[d < 0].sum())
    return (wp - wn) / (wp + wn) if (wp + wn) else 0.0


def bootstrap_mean_ci(diff: np.ndarray, salt: int) -> Tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0: return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(salt))
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_REPS, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n-rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def pairwise_primary(ds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tests: List[Dict[str, Any]] = []
    diffs_all: List[pd.DataFrame] = []
    wtl: List[Dict[str, Any]] = []
    for mi, metric in enumerate(PRIMARY_METRICS):
        col = f"{metric}_mean"
        metric_tests: List[Dict[str, Any]] = []
        for ci, comp in enumerate(["NBEM", "Logistic Regression"]):
            pvt = ds[ds["model"].isin(["Hybrid NBEM", comp])].pivot(index=["dataset","display_name"], columns="model", values=col).dropna()
            pvt["delta_hybrid_minus_comparator"] = pvt["Hybrid NBEM"] - pvt[comp]
            d = pvt["delta_hybrid_minus_comparator"].to_numpy(float)
            wins = int((d > TIE_TOL).sum()); losses = int((d < -TIE_TOL).sum()); ties = int(len(d)-wins-losses)
            try:
                stat, p_raw = wilcoxon(d, zero_method="wilcox", alternative="two-sided") if np.any(np.abs(d) > TIE_TOL) else (0.0, 1.0)
            except ValueError:
                stat, p_raw = 0.0, 1.0
            lo, hi = bootstrap_mean_ci(d, salt=100*mi+ci)
            metric_tests.append({
                "metric": metric,
                "contrast": f"Hybrid NBEM - {REPORT_NAMES[comp]}",
                "comparator_model": comp,
                "n_datasets": int(len(d)),
                "hybrid_mean_across_datasets": float(pvt["Hybrid NBEM"].mean()),
                "comparator_mean_across_datasets": float(pvt[comp].mean()),
                "mean_paired_difference": float(np.mean(d)),
                "median_paired_difference": float(np.median(d)),
                "bootstrap95_mean_difference_lo": lo,
                "bootstrap95_mean_difference_hi": hi,
                "wins": wins, "ties": ties, "losses": losses,
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p_raw": float(p_raw),
                "rank_biserial_correlation_positive_favors_hybrid": float(paired_rank_biserial(d)),
            })
            dd = pvt.reset_index()[["dataset","display_name","Hybrid NBEM",comp,"delta_hybrid_minus_comparator"]].copy()
            dd.insert(2,"metric",metric); dd.insert(3,"comparator_model",comp)
            diffs_all.append(dd)
            wtl.append({"metric":metric,"comparator_model":comp,"wins":wins,"ties":ties,"losses":losses,"n_datasets":len(d)})
        adj = holm_adjust([x["wilcoxon_p_raw"] for x in metric_tests])
        for x, p_adj in zip(metric_tests, adj):
            x["wilcoxon_p_holm_within_metric"] = float(p_adj)
        tests.extend(metric_tests)
    return pd.DataFrame(tests), pd.concat(diffs_all, ignore_index=True), pd.DataFrame(wtl)


def seed_level_stability(ps: pd.DataFrame) -> pd.DataFrame:
    seed_model = ps.groupby(["seed","model","report_model"], as_index=False)[PRIMARY_METRICS].mean()
    rows = []
    for (model, report), g in seed_model.groupby(["model","report_model"], sort=False):
        row = {"model":model,"report_model":report,"n_seeds":int(g.seed.nunique())}
        for m in PRIMARY_METRICS:
            mean, sd, lo, hi = mean_ci_t(g[m].to_numpy(float))
            row[f"{m}_mean_across_seeds_of_dataset_means"] = mean
            row[f"{m}_sd_across_seeds"] = sd
            row[f"{m}_ci95_seed_lo"] = lo
            row[f"{m}_ci95_seed_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def hybrid_win_fraction_across_seeds(ps: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for metric in PRIMARY_METRICS:
        for comp in ["NBEM","Logistic Regression"]:
            p=ps[ps.model.isin(["Hybrid NBEM",comp])].pivot(index=["dataset","display_name","seed"],columns="model",values=metric).dropna().reset_index()
            p["delta"] = p["Hybrid NBEM"]-p[comp]
            for (ds,dn),g in p.groupby(["dataset","display_name"],sort=False):
                d=g.delta.to_numpy(float)
                rows.append({
                    "dataset":ds,"display_name":dn,"metric":metric,"comparator_model":comp,
                    "n_seeds":len(d),"hybrid_win_fraction":float((d>TIE_TOL).mean()),
                    "tie_fraction":float((np.abs(d)<=TIE_TOL).mean()),
                    "hybrid_loss_fraction":float((d<-TIE_TOL).mean()),
                    "mean_seed_paired_delta":float(d.mean()),"median_seed_paired_delta":float(np.median(d)),
                })
    return pd.DataFrame(rows)


def dataset_characteristics(prepared: Sequence[Any]) -> pd.DataFrame:
    rows=[]
    for p in prepared:
        b,c,n=BASE.detect_feature_groups(p.df,p.target)
        nfeat=len(b)+len(c)+len(n)
        nclasses=int(p.df[p.target].astype(str).nunique())
        discrete=len(b)+len(c)
        if len(n)>0 and discrete>0: mix="mixed"
        elif len(n)>0: mix="numerical_only"
        else: mix="discrete_only"
        rows.append({
            "dataset":p.key,"display_name":p.display_name,"n_rows":len(p.df),"n_features":nfeat,
            "n_boolean":len(b),"n_categorical":len(c),"n_numerical":len(n),"n_classes":nclasses,
            "task_type":"multiclass" if nclasses>2 else "binary",
            "feature_mix":mix,
            "discrete_fraction":float(discrete/nfeat) if nfeat else math.nan,
            "numerical_fraction":float(len(n)/nfeat) if nfeat else math.nan,
        })
    out=pd.DataFrame(rows)
    med_rows=float(out.n_rows.median()); med_feat=float(out.n_features.median())
    out["size_group_predeclared_median_rule"]=np.where(out.n_rows>=med_rows,"larger_half","smaller_half")
    out["dimension_group_predeclared_median_rule"]=np.where(out.n_features>=med_feat,"higher_dim_half","lower_dim_half")
    out["log10_n_rows"]=np.log10(out.n_rows.astype(float))
    return out


def exploratory_capability_profiles(ds: pd.DataFrame, chars: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    # Descriptive/exploratory only; never use as confirmatory evidence without explicit labeling.
    wide={}
    for metric in PRIMARY_METRICS:
        p=ds.pivot(index="dataset",columns="model",values=f"{metric}_mean")
        for comp in ["NBEM","Logistic Regression"]:
            wide[(metric,comp)] = (p["Hybrid NBEM"]-p[comp]).rename(f"gain_{metric}_vs_{comp}")
    x=chars.set_index("dataset").copy()
    for s in wide.values(): x=x.join(s)
    x=x.reset_index()
    rows=[]
    for group_col in ["feature_mix","task_type","size_group_predeclared_median_rule","dimension_group_predeclared_median_rule"]:
        for group_val,g in x.groupby(group_col,dropna=False):
            for metric in PRIMARY_METRICS:
                for comp in ["NBEM","Logistic Regression"]:
                    col=f"gain_{metric}_vs_{comp}"
                    vals=pd.to_numeric(g[col],errors="coerce").dropna()
                    rows.append({"grouping":group_col,"group":group_val,"metric":metric,"comparator_model":comp,
                                 "n_datasets":len(vals),"mean_hybrid_gain":float(vals.mean()) if len(vals) else math.nan,
                                 "median_hybrid_gain":float(vals.median()) if len(vals) else math.nan})
    cor=[]
    for predictor in ["log10_n_rows","n_features","n_classes","discrete_fraction","numerical_fraction"]:
        for metric in PRIMARY_METRICS:
            for comp in ["NBEM","Logistic Regression"]:
                col=f"gain_{metric}_vs_{comp}"
                z=x[[predictor,col]].dropna()
                if len(z)>=3 and z[predictor].nunique()>1 and z[col].nunique()>1:
                    rho,p=spearmanr(z[predictor],z[col])
                else: rho,p=math.nan,math.nan
                cor.append({"analysis_label":"EXPLORATORY_NOT_CONFIRMATORY","predictor":predictor,"metric":metric,
                            "comparator_model":comp,"n_datasets":len(z),"spearman_rho":float(rho),"p_raw":float(p)})
    return pd.DataFrame(rows),pd.DataFrame(cor)


def manuscript_ready_summary(agg: pd.DataFrame, tests: pd.DataFrame, seedstab: pd.DataFrame, path: Path) -> None:
    lines=[
        "STAGE 3 - MANUSCRIPT-READY ROBUSTNESS SUMMARY",
        "=============================================",
        "Primary unit of inference: dataset (20 datasets).",
        "Seeds: 20 predeclared seeds; no seed excluded after observing results.",
        "Primary metrics: Weighted-F1 and Macro-F1.",
        "",
        "IMPORTANT: Use only after the full run passes all validation checks.",
        "Exploratory dataset-characteristic analyses must be labeled exploratory.",
        "",
        "Aggregate 20-seed model performance (dataset-level means):",
    ]
    for _,r in agg.iterrows():
        lines.append(f"- {r['report_model']}: Weighted-F1={r.get('f1_weighted_mean_across_datasets',math.nan):.6f}; Macro-F1={r.get('f1_macro_mean_across_datasets',math.nan):.6f}")
    lines += ["", "Planned pairwise comparisons:"]
    for _,r in tests.iterrows():
        lines.append(
            f"- {r['metric']}: {r['contrast']}: mean delta={r['mean_paired_difference']:+.6f}, "
            f"95% bootstrap CI [{r['bootstrap95_mean_difference_lo']:+.6f}, {r['bootstrap95_mean_difference_hi']:+.6f}], "
            f"W/T/L={int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}, "
            f"Wilcoxon Holm p={r['wilcoxon_p_holm_within_metric']:.6g}, rank-biserial={r['rank_biserial_correlation_positive_favors_hybrid']:+.4f}"
        )
    lines += ["", "Seed-level descriptive stability (not an independent-sample significance analysis):"]
    for _,r in seedstab.iterrows():
        lines.append(f"- {r['report_model']}: Weighted-F1 mean over seeds={r['f1_weighted_mean_across_seeds_of_dataset_means']:.6f} +/- {r['f1_weighted_sd_across_seeds']:.6f}; Macro-F1={r['f1_macro_mean_across_seeds_of_dataset_means']:.6f} +/- {r['f1_macro_sd_across_seeds']:.6f}")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def analyze(allfolds: pd.DataFrame, prepared: Sequence[Any], output_dir: Path, plan: Dict[str, Any]) -> None:
    expected=set(map(int,plan["all_twenty_seeds"]))
    if not allfolds["status"].eq("OK").all():
        bad=allfolds[~allfolds["status"].eq("OK")]
        raise RuntimeError("Non-OK rows exist; analysis stopped.\n"+bad[["dataset","model","seed","fold","status"]].head(30).to_string(index=False))
    if set(map(int,allfolds.seed.unique()))!=expected or allfolds.dataset.nunique()!=20 or set(allfolds.model.unique())!=set(MODELS):
        raise RuntimeError("Final Stage-3 matrix is incomplete (expected 20 datasets x 20 seeds x 3 models).")
    combo=allfolds.groupby(["dataset","seed","model"]).size().reset_index(name="rows")
    if len(combo)!=20*20*3:
        raise RuntimeError(f"Missing dataset-seed-model combinations: got {len(combo)}, expected 1200.")
    # split signatures must be identical across the three models
    sig=allfolds.groupby(["dataset","seed","model"]).agg(nfold=("fold","nunique"),protocol=("cv_protocol","first")).reset_index()
    for (ds,seed),g in sig.groupby(["dataset","seed"]):
        if g.nfold.nunique()!=1 or g.protocol.nunique()!=1 or set(g.model)!=set(MODELS):
            raise RuntimeError(f"Outer split mismatch across models: {ds}, seed={seed}")

    ps=per_seed_dataset_model(allfolds)
    ds=dataset_model_summary(ps)
    agg=aggregate_model_summary(ds)
    tests,diffs,wtl=pairwise_primary(ds)
    seedstab=seed_level_stability(ps)
    winfrac=hybrid_win_fraction_across_seeds(ps)
    chars=dataset_characteristics(prepared)
    prof,corr=exploratory_capability_profiles(ds,chars)

    allfolds.to_csv(output_dir/"10_stage3_all_fold_results_20seeds.csv",index=False,encoding="utf-8-sig")
    ps.to_csv(output_dir/"11_stage3_per_seed_dataset_model.csv",index=False,encoding="utf-8-sig")
    ds.to_csv(output_dir/"12_stage3_dataset_model_20seed_summary.csv",index=False,encoding="utf-8-sig")
    agg.to_csv(output_dir/"13_stage3_aggregate_model_summary.csv",index=False,encoding="utf-8-sig")
    tests.to_csv(output_dir/"14_primary_pairwise_dataset_level_tests.csv",index=False,encoding="utf-8-sig")
    diffs.to_csv(output_dir/"15_primary_pairwise_dataset_level_differences.csv",index=False,encoding="utf-8-sig")
    wtl.to_csv(output_dir/"16_win_tie_loss_dataset_level.csv",index=False,encoding="utf-8-sig")
    seedstab.to_csv(output_dir/"17_seed_level_aggregate_stability_DESCRIPTIVE.csv",index=False,encoding="utf-8-sig")
    winfrac.to_csv(output_dir/"18_hybrid_win_fraction_across_seeds_DESCRIPTIVE.csv",index=False,encoding="utf-8-sig")
    chars.to_csv(output_dir/"19_dataset_characteristics.csv",index=False,encoding="utf-8-sig")
    prof.to_csv(output_dir/"20_exploratory_gain_by_dataset_characteristics.csv",index=False,encoding="utf-8-sig")
    corr.to_csv(output_dir/"21_exploratory_gain_correlations.csv",index=False,encoding="utf-8-sig")
    manuscript_ready_summary(agg,tests,seedstab,output_dir/"22_MANUSCRIPT_READY_SUMMARY.txt")


def write_manifest(args: argparse.Namespace, output_dir: Path, snapshot: pd.DataFrame, plan: Dict[str, Any]) -> None:
    payload={
        "stage":"Major Revision Stage 3",
        "timestamp_local":pd.Timestamp.now().isoformat(),
        "python":sys.version,
        "platform":platform.platform(),
        "base_script_sha256":sha256_file(BASE_SCRIPT),
        "stage3_script_sha256":sha256_file(Path(__file__)),
        "revision_config_sha256":sha256_file(DEFAULT_CONFIG),
        "dataset_manifest_sha256":sha256_file(DEFAULT_MANIFEST),
        "seed_plan_sha256":sha256_file(SEED_PLAN),
        "preloaded_first5_sha256":sha256_file(PRELOADED),
        "exact_dataset_hash_matches":int(snapshot.hash_match.sum()),
        "exact_dataset_count":int(len(snapshot)),
        "all_twenty_seeds":plan["all_twenty_seeds"],
        "original_five_reused":plan["original_five_seeds"],
        "additional_fifteen_fitted":plan["additional_fifteen_seeds"],
        "models":MODELS,
        "outer_requested_folds":int(args.folds),
        "n_bins":int(args.n_bins),
        "mlp_max_iter":int(args.mlp_max_iter),
        "primary_metrics":PRIMARY_METRICS,
        "primary_inference_unit":"dataset",
        "holm_scope":"two planned comparator contrasts within each primary metric",
        "bootstrap_reps_over_datasets":BOOTSTRAP_REPS,
        "bootstrap_seed":BOOTSTRAP_SEED,
        "bank_duration_removed":True,
        "diabetes_outer_cv":"patient-grouped",
        "logistic_regression":"Stage-2 exact untuned baseline: C=1.0, lbfgs, fold-local one-hot discrete features, standardized numeric features",
        "exploratory_profiles":"descriptive only; not confirmatory",
    }
    (output_dir/"RUN_MANIFEST_STAGE3.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    rows=[]
    for p in sorted(output_dir.glob("*")):
        if p.is_file() and p.name!="CHECKSUMS_OUTPUT_SHA256.txt":
            rows.append(f"{sha256_file(p)}  {p.name}")
    (output_dir/"CHECKSUMS_OUTPUT_SHA256.txt").write_text("\n".join(rows)+"\n",encoding="utf-8")


def mode_smoke(prepared: Sequence[Any], output_dir: Path, args: argparse.Namespace) -> None:
    smoke=output_dir/"SMOKE_DO_NOT_REPORT"; smoke.mkdir(parents=True,exist_ok=True)
    p=next(x for x in prepared if x.key=="heart_disease")
    fr=evaluate_three_models(p,seed=20260924,requested_folds=2,n_bins=args.n_bins,mlp_max_iter=min(args.mlp_max_iter,60),quick=True)
    fr.to_csv(smoke/"smoke_three_models.csv",index=False,encoding="utf-8-sig")
    if not fr.status.eq("OK").all() or set(fr.model)!=set(MODELS):
        raise RuntimeError("Stage-3 smoke test failed.")
    print("SMOKE PASS. Do not report smoke outputs.")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--mode",choices=["smoke","full","analyze"],required=True)
    p.add_argument("--folds",type=int,default=10)
    p.add_argument("--n-bins",type=int,default=5)
    p.add_argument("--mlp-max-iter",type=int,default=300)
    return p.parse_args()


def main() -> None:
    args=parse_args(); args.project_root=args.project_root.resolve(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    print("Step 0/5: verify locked seed plan, exact snapshot, and frozen revision protocol...")
    plan=verify_seed_plan(); verify_revision_config_scope()
    snapshot=verify_snapshot(args.project_root,DEFAULT_MANIFEST,args.output_dir)
    print(f"exact hash matches: {int(snapshot.hash_match.sum())}/{len(snapshot)}")
    config=load_json(DEFAULT_CONFIG); prepared=prepare_all(args.project_root,config)
    print("frozen protocol: PASS (20 datasets; Bank duration absent)")
    pre=validate_preloaded(plan,args.output_dir)
    print("validated preloaded first five: PASS (20 datasets x 5 seeds x 3 models; same outer split signatures)")

    seedrows=[]
    for i,s in enumerate(plan["all_twenty_seeds"],1):
        seedrows.append({"order":i,"seed":int(s),"source":"preloaded validated original" if s in plan["original_five_seeds"] else "new locked Stage-3 seed"})
    pd.DataFrame(seedrows).to_csv(args.output_dir/"01_stage3_seed_plan.csv",index=False,encoding="utf-8-sig")

    if args.mode=="smoke":
        mode_smoke(prepared,args.output_dir,args)
        write_manifest(args,args.output_dir,snapshot,plan); write_checksums(args.output_dir); return

    if args.mode=="full":
        print("Step 1/5: fit the 15 additional predeclared seeds (resume-safe cache)...")
        new=run_additional(prepared,plan["additional_fifteen_seeds"],args.output_dir,args.folds,args.n_bins,args.mlp_max_iter)
        if new.empty or not new.status.eq("OK").all():
            raise RuntimeError("Additional-seed fitting incomplete or contains errors.")
        allfolds=pd.concat([pre,new],ignore_index=True,sort=False)
        print("Step 2/5: combine all 20 seeds and validate paired split integrity...")
        analyze(allfolds,prepared,args.output_dir,plan)
    else:
        print("Analyze-only mode: loading cached additional-seed results...")
        cache=args.output_dir/"cache"/"additional_fifteen"
        files=sorted(cache.glob("*.csv"))
        if len(files)!=20*15:
            raise RuntimeError(f"Expected 300 cached dataset-seed files; found {len(files)}. Run --mode full first.")
        new=pd.concat([pd.read_csv(f,encoding="utf-8-sig") for f in files],ignore_index=True,sort=False)
        analyze(pd.concat([pre,new],ignore_index=True,sort=False),prepared,args.output_dir,plan)

    print("Step 3/5: write reproducibility manifest and checksums...")
    write_manifest(args,args.output_dir,snapshot,plan); write_checksums(args.output_dir)
    print("Step 4/5: final QA summary...")
    tests=pd.read_csv(args.output_dir/"14_primary_pairwise_dataset_level_tests.csv",encoding="utf-8-sig")
    print(tests[["metric","comparator_model","mean_paired_difference","wins","ties","losses","wilcoxon_p_holm_within_metric","rank_biserial_correlation_positive_favors_hybrid"]].to_string(index=False))
    print("Step 5/5: DONE")
    print(f"Outputs: {args.output_dir}")


if __name__=="__main__":
    main()
