#!/usr/bin/env python3
"""
NBEM Major Revision - Stage 2
=============================

Purpose
-------
This stage runs ONLY the new experiments requested after the Stage-1/1B data audit:

1) External baselines on the frozen revised protocol:
   - Logistic Regression (standard untuned baseline)
   - Random Forest (standard untuned baseline)

2) Actual cross-fitted Hybrid NBEM stacking check:
   - OOF component probabilities are generated inside each outer training fold.
   - The preprocessing is also refit inside each inner fold.
   - The meta Logistic Regression sees only OOF component probabilities.
   - Final base components are refit on the complete outer-training fold before
     outer-test prediction.

3) Timing audit for NBEM-prop and Adaptive Weighted NBEM, with preprocessing,
   model-fit, and inference time reported separately to resolve the Table-6 ambiguity.

Scientific protocol frozen from Stage 1 / Stage 1B
--------------------------------------------------
- Exact 20-dataset snapshot verified by SHA-256.
- Bank Marketing `duration` is REMOVED by the revision config.
- Diabetes uses patient-grouped outer CV as in the final v4 protocol.
- Duplicate rows are NOT blindly deleted. Stage-1B duplicate-grouped sensitivity
  remains a separate robustness analysis.
- Main outer CV: same requested folds and five predefined seeds as the final article.

IMPORTANT
---------
The exact v4 implementation is imported, not rewritten. This file adds only the
new external-baseline, cross-fitting, timing, and summarization logic needed for
major revision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "base" / "nbem_article_experiments_v4_EXACT_USED.py"
DEFAULT_CONFIG = ROOT / "config" / "dataset_config_revision_frozen.json"
DEFAULT_MANIFEST = ROOT / "manifest" / "DATASET_MANIFEST_PORTABLE.csv"
DEFAULT_OUTPUT = ROOT / "results" / "major_revision_stage2"
REFERENCE_V4 = ROOT / "reference_v4_results" / "all_fold_results_v4.csv"
BANK_REVISED = ROOT / "previous_stage_results" / "11_bank_without_duration_all_fold_results.csv"
STAGE1B_SEMANTIC = ROOT / "previous_stage_results" / "20_semantic_review_decisions.csv"
STAGE1B_DUPLICATE = ROOT / "previous_stage_results" / "36_duplicate_sensitivity_delta_summary.csv"

DEFAULT_SEEDS = [13, 21, 42, 87, 123]
DEFAULT_INNER_FOLDS = 5
DEFAULT_BASELINE_MODELS = ["Logistic Regression", "Random Forest"]
DEFAULT_TIMING_MODELS = ["NBEM", "Adaptive Weighted NBEM"]

# Predeclared representative subset for the optional/supplementary cross-fitted
# check. Selection is by data characteristics, BEFORE seeing Stage-2 results.
# It spans binary/multiclass, categorical/numerical/mixed, medium/large, and
# high-dimensional settings. A full-20 mode is also provided.
CROSSFIT_SUBSET = [
    "bank_marketing",                                  # revised post-outcome feature handling
    "car_evaluation",                                 # categorical multiclass
    "dry_bean_dataset",                               # numerical multiclass
    "heart_disease",                                  # small mixed-type benchmark
    "internet_advertisements",                        # high-dimensional + duplicate sensitivity
    "productivity_prediction_of_garment_employees",   # mixed-type 3-class task
]

REPORT_MODEL_NAMES = {
    "NBEM": "NBEM-prop (implementation-level NBEM comparator)",
    "Adaptive Weighted NBEM": "Adaptive Weighted NBEM",
    "Hybrid NBEM": "Hybrid NBEM (non-cross-fitted)",
    "Cross-Fitted Hybrid NBEM": "Hybrid NBEM (actual cross-fitted stacker)",
    "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest",
}

METRICS = [
    "accuracy", "balanced_accuracy",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "precision_macro", "recall_macro", "f1_macro", "roc_auc_weighted",
]


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("nbem_v4_exact_stage2", path)
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
        raise RuntimeError("Exact dataset verification FAILED.\n" + bad.to_string(index=False))
    return out


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_all(project_root: Path, config: Dict[str, Any]) -> List[Any]:
    paths = BASE.find_datasets(project_root, use_canonical_20=bool(config.get("use_canonical_20", True)))
    prepared = [BASE.prepare_dataset(path, config) for path in paths]
    if len(prepared) != 20:
        raise RuntimeError(f"Expected 20 prepared datasets, found {len(prepared)}")
    bad = [p for p in prepared if p.critical_issues]
    if bad:
        msg = "\n".join(f"- {p.key}: {' | '.join(p.critical_issues)}" for p in bad)
        raise RuntimeError("Critical dataset preparation issue(s) under frozen revision config:\n" + msg)
    return prepared


def prefer_grouped_main(prepared: Any) -> bool:
    return bool(
        prepared.groups is not None
        and prepared.key == "diabetes_130_us_hospitals_for_years_1999_2008"
    )


class BaselinePreprocessor:
    """Fold-local baseline preprocessing using the same feature typing as exact v4.

    Categorical/Boolean columns are one-hot encoded from the training fold only.
    Numerical columns are median-imputed. Logistic Regression additionally scales
    numerical features. This is a standard baseline representation and is kept
    separate from the NBEM-specific representations.
    """

    def __init__(self, scale_numeric: bool):
        self.scale_numeric = bool(scale_numeric)

    def fit(self, df: pd.DataFrame, target: str) -> "BaselinePreprocessor":
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
            if self.scale_numeric:
                self.scaler.fit(N)
        return self

    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target])
        n_rows = len(df)
        parts: List[Any] = []
        if self.cat_cols:
            C = BASE.string_frame_preserve_missing(X[self.cat_cols])
            Cimp = self.cat_imp.transform(C)
            parts.append(self.ohe.transform(Cimp))
        if self.num_cols:
            N = self.num_imp.transform(X[self.num_cols]).astype(float)
            if self.scale_numeric:
                N = self.scaler.transform(N)
            parts.append(sparse.csr_matrix(N))
        if not parts:
            return sparse.csr_matrix(np.zeros((n_rows, 1), dtype=float))
        return sparse.hstack(parts, format="csr")


@dataclass
class BaselineSpec:
    name: str
    scale_numeric: bool
    factory: Any


def baseline_spec(name: str, seed: int) -> BaselineSpec:
    if name == "Logistic Regression":
        return BaselineSpec(
            name=name,
            scale_numeric=True,
            factory=lambda: LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight=None,
                random_state=int(seed),
            ),
        )
    if name == "Random Forest":
        return BaselineSpec(
            name=name,
            scale_numeric=False,
            factory=lambda: RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
                n_jobs=-1,
                random_state=int(seed),
            ),
        )
    raise ValueError(name)


def common_fold_base(prepared: Any, df: pd.DataFrame, plan: Any, seed: int, fold: int, model: str,
                     n_features: int, n_classes: int) -> Dict[str, Any]:
    return {
        "analysis": "major_revision_stage2",
        "dataset": prepared.key,
        "display_name": prepared.display_name,
        "file": str(prepared.path),
        "target": prepared.target,
        "target_transform": prepared.target_transform,
        "model": model,
        "report_model": REPORT_MODEL_NAMES.get(model, model),
        "seed": int(seed),
        "fold": int(fold),
        "cv_protocol": plan.protocol,
        "n_splits_or_total_folds": int(plan.n_splits),
        "stratified": bool(plan.stratified),
        "grouped": bool(plan.grouped),
        "n_rows": int(len(df)),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
    }


def evaluate_external_baselines(
    prepared: Any,
    seed: int,
    requested_folds: int,
    model_names: Sequence[str],
    quick: bool = False,
) -> pd.DataFrame:
    BASE.set_reproducibility(int(seed))
    df = prepared.df.copy().reset_index(drop=True)
    groups = prepared.groups.copy() if prepared.groups is not None else None
    if quick and len(df) > 1200:
        idx = df.sample(n=1200, random_state=int(seed)).index.to_numpy()
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
            for name in model_names:
                base = common_fold_base(prepared, df, plan, seed, fold, name, df.shape[1] - 1, len(global_classes))
                rows.append({**base, "status": "SKIPPED: fewer than two training classes"})
            continue

        for name in model_names:
            base = common_fold_base(prepared, df, plan, seed, fold, name, df.shape[1] - 1, len(global_classes))
            try:
                spec = baseline_spec(name, int(seed))
                tracemalloc.start()

                t0 = time.perf_counter()
                prep = BaselinePreprocessor(scale_numeric=spec.scale_numeric).fit(train_df, prepared.target)
                X_train = prep.transform(train_df)
                prep_train_s = time.perf_counter() - t0

                t1 = time.perf_counter()
                model = spec.factory()
                model.fit(X_train, y_train)
                model_fit_s = time.perf_counter() - t1

                t2 = time.perf_counter()
                X_test = prep.transform(test_df)
                prep_test_s = time.perf_counter() - t2

                t3 = time.perf_counter()
                pred = model.predict(X_test)
                raw_proba = model.predict_proba(X_test)
                infer_s = time.perf_counter() - t3

                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                proba = BASE.align_proba(raw_proba, getattr(model, "classes_", np.unique(y_train)), global_classes)
                metrics = BASE.compute_metrics(y_test, pred, proba, global_classes)
                rows.append({
                    **base, **metrics,
                    "preprocess_train_s": float(prep_train_s),
                    "model_fit_s": float(model_fit_s),
                    "total_train_pipeline_s": float(prep_train_s + model_fit_s),
                    "preprocess_test_s": float(prep_test_s),
                    "model_inference_s": float(infer_s),
                    "total_test_pipeline_s": float(prep_test_s + infer_s),
                    "peak_memory_mb": float(peak / (1024.0 * 1024.0)),
                    "representation": "fold-local one-hot categorical/boolean + median-imputed numeric" + (" + standardized numeric" if spec.scale_numeric else ""),
                    "status": "OK",
                })
            except Exception as exc:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
                rows.append({
                    **base,
                    **{m: np.nan for m in METRICS},
                    "status": f"ERROR: {type(exc).__name__}: {exc}",
                })
    return pd.DataFrame(rows)


def fit_hybrid_components(data: Dict[str, Any], y: np.ndarray, seed: int, mlp_max_iter: int) -> List[Tuple[str, Any, str]]:
    components: List[Tuple[str, Any, str]] = []
    dep = BASE.DependencyAwareNBEM(max_interactions=10, random_state=int(seed)).fit(data["X_disc"], y)
    components.append(("Dependency", dep, "X_disc"))
    aw = BASE.AdaptiveWeightedNBEM(temperature=1.0, lambda_entropy=0.5).fit(data, y)
    components.append(("Adaptive", aw, "data"))
    deep = BASE.ProbabilisticDeepNBEM(
        hidden_layer_sizes=(128, 64), max_iter=int(mlp_max_iter), random_state=int(seed)
    ).fit(data["X_cont"], y)
    components.append(("Deep", deep, "X_cont"))
    return components


def stack_component_probas(components: Sequence[Tuple[str, Any, str]], data: Dict[str, Any], global_classes: np.ndarray) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for _, comp, mode in components:
        if mode == "data":
            p = comp.predict_proba(data)
        else:
            p = comp.predict_proba(data[mode])
        cls = getattr(comp, "classes_", global_classes)
        blocks.append(BASE.align_proba(p, cls, global_classes))
    return np.hstack(blocks)


def build_inner_plan(
    prepared: Any,
    outer_train_df: pd.DataFrame,
    y_outer_train: np.ndarray,
    outer_train_groups: Optional[np.ndarray],
    seed: int,
    inner_folds: int,
):
    return BASE.make_cv_plan(
        y=y_outer_train,
        seed=int(seed),
        requested_folds=int(inner_folds),
        X=outer_train_df,
        groups=outer_train_groups,
        prefer_grouped=prefer_grouped_main(prepared),
        forced_protocol=None,
    )


def evaluate_crossfitted_hybrid(
    prepared: Any,
    seed: int,
    requested_folds: int,
    inner_folds: int,
    n_bins: int,
    mlp_max_iter: int,
    quick: bool = False,
) -> pd.DataFrame:
    """Strict outer-evaluation + inner OOF stacking.

    Each OOF probability vector is produced by a component model AND preprocessor
    that did not see that OOF sample during fitting.
    """
    BASE.set_reproducibility(int(seed))
    df = prepared.df.copy().reset_index(drop=True)
    groups = prepared.groups.copy() if prepared.groups is not None else None
    if quick and len(df) > 800:
        idx = df.sample(n=800, random_state=int(seed)).index.to_numpy()
        df = df.iloc[idx].reset_index(drop=True)
        if groups is not None:
            groups = groups[idx]

    enc = LabelEncoder()
    y = enc.fit_transform(df[prepared.target].astype(str))
    global_classes = np.arange(len(enc.classes_))
    outer = BASE.make_cv_plan(
        y=y,
        seed=int(seed),
        requested_folds=int(requested_folds),
        X=df,
        groups=groups,
        prefer_grouped=prefer_grouped_main(prepared),
        forced_protocol=None,
    )

    rows: List[Dict[str, Any]] = []
    for outer_fold, (tr, te) in enumerate(outer.splits, 1):
        base = common_fold_base(
            prepared, df, outer, seed, outer_fold, "Cross-Fitted Hybrid NBEM",
            df.shape[1] - 1, len(global_classes)
        )
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        y_train, y_test = y[tr], y[te]
        train_groups = groups[tr] if groups is not None else None
        if len(np.unique(y_train)) < 2:
            rows.append({**base, "status": "SKIPPED: fewer than two outer-training classes"})
            continue

        try:
            tracemalloc.start()
            inner = build_inner_plan(
                prepared=prepared,
                outer_train_df=train_df,
                y_outer_train=y_train,
                outer_train_groups=train_groups,
                seed=int(seed),
                inner_folds=int(inner_folds),
            )
            oof = np.full((len(train_df), 3 * len(global_classes)), np.nan, dtype=float)
            inner_protocol = inner.protocol

            t_oof0 = time.perf_counter()
            for inner_fold, (itr, iva) in enumerate(inner.splits, 1):
                if len(np.unique(y_train[itr])) < 2:
                    raise RuntimeError(
                        f"Inner cross-fitting split has fewer than two training classes "
                        f"(outer fold {outer_fold}, inner fold {inner_fold})."
                    )
                inner_train_df = train_df.iloc[itr].reset_index(drop=True)
                inner_val_df = train_df.iloc[iva].reset_index(drop=True)
                prep_i = BASE.NBEMPreprocessor(n_bins=int(n_bins)).fit(inner_train_df, prepared.target)
                data_i_train = prep_i.transform(inner_train_df)
                data_i_val = prep_i.transform(inner_val_df)
                comps_i = fit_hybrid_components(data_i_train, y_train[itr], int(seed), int(mlp_max_iter))
                oof[iva, :] = stack_component_probas(comps_i, data_i_val, global_classes)

            if np.isnan(oof).any():
                missing = int(np.isnan(oof).any(axis=1).sum())
                raise RuntimeError(f"OOF matrix incomplete: {missing} rows missing cross-fitted probabilities.")

            meta = LogisticRegression(max_iter=1000, class_weight=None, random_state=int(seed))
            meta.fit(oof, y_train)
            oof_meta_train_s = time.perf_counter() - t_oof0

            # Refit preprocessing and components on the full outer-training fold.
            t_final0 = time.perf_counter()
            prep_full = BASE.NBEMPreprocessor(n_bins=int(n_bins)).fit(train_df, prepared.target)
            data_full_train = prep_full.transform(train_df)
            final_components = fit_hybrid_components(data_full_train, y_train, int(seed), int(mlp_max_iter))
            final_base_fit_s = time.perf_counter() - t_final0

            t_testprep0 = time.perf_counter()
            data_test = prep_full.transform(test_df)
            test_prep_s = time.perf_counter() - t_testprep0

            t_pred0 = time.perf_counter()
            P_test = stack_component_probas(final_components, data_test, global_classes)
            pred = meta.predict(P_test)
            raw_meta_proba = meta.predict_proba(P_test)
            infer_s = time.perf_counter() - t_pred0

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            proba = BASE.align_proba(raw_meta_proba, meta.classes_, global_classes)
            metrics = BASE.compute_metrics(y_test, pred, proba, global_classes)
            rows.append({
                **base, **metrics,
                "inner_cv_protocol": inner_protocol,
                "inner_total_folds": int(inner.n_splits),
                "oof_meta_training_s": float(oof_meta_train_s),
                "final_base_refit_s": float(final_base_fit_s),
                "total_train_pipeline_s": float(oof_meta_train_s + final_base_fit_s),
                "preprocess_test_s": float(test_prep_s),
                "model_inference_s": float(infer_s),
                "total_test_pipeline_s": float(test_prep_s + infer_s),
                "peak_memory_mb": float(peak / (1024.0 * 1024.0)),
                "crossfit_preprocessing": "inner-fold refit; OOF rows never used in component/preprocessor fit",
                "status": "OK",
            })
        except Exception as exc:
            try:
                tracemalloc.stop()
            except Exception:
                pass
            rows.append({
                **base,
                **{m: np.nan for m in METRICS},
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows)


def evaluate_timing_audit(
    prepared: Any,
    seed: int,
    requested_folds: int,
    n_bins: int,
    mlp_max_iter: int,
    quick: bool = False,
) -> pd.DataFrame:
    """Precise NBEM-prop vs Adaptive timing with preprocessing separated."""
    BASE.set_reproducibility(int(seed))
    df = prepared.df.copy().reset_index(drop=True)
    groups = prepared.groups.copy() if prepared.groups is not None else None
    if quick and len(df) > 1200:
        idx = df.sample(n=1200, random_state=int(seed)).index.to_numpy()
        df = df.iloc[idx].reset_index(drop=True)
        if groups is not None:
            groups = groups[idx]

    enc = LabelEncoder()
    y = enc.fit_transform(df[prepared.target].astype(str))
    global_classes = np.arange(len(enc.classes_))
    plan = BASE.make_cv_plan(
        y=y, seed=int(seed), requested_folds=int(requested_folds), X=df,
        groups=groups, prefer_grouped=prefer_grouped_main(prepared), forced_protocol=None,
    )
    rows: List[Dict[str, Any]] = []

    for fold, (tr, te) in enumerate(plan.splits, 1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        y_train, y_test = y[tr], y[te]
        if len(np.unique(y_train)) < 2:
            continue

        tracemalloc.start()
        tp0 = time.perf_counter()
        prep = BASE.NBEMPreprocessor(n_bins=int(n_bins)).fit(train_df, prepared.target)
        train = prep.transform(train_df)
        prep_train_s = time.perf_counter() - tp0
        tt0 = time.perf_counter()
        test = prep.transform(test_df)
        prep_test_s = time.perf_counter() - tt0
        _, prep_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        for name in DEFAULT_TIMING_MODELS:
            base = common_fold_base(prepared, df, plan, seed, fold, name, df.shape[1]-1, len(global_classes))
            try:
                if name == "NBEM":
                    model = BASE.NBEMClassifier()
                else:
                    model = BASE.AdaptiveWeightedNBEM(temperature=1.0, lambda_entropy=0.5)
                tracemalloc.start()
                t0 = time.perf_counter()
                model.fit(train, y_train)
                fit_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                pred = model.predict(test)
                raw = model.predict_proba(test)
                infer_s = time.perf_counter() - t1
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                proba = BASE.align_proba(raw, getattr(model, "classes_", np.unique(y_train)), global_classes)
                metrics = BASE.compute_metrics(y_test, pred, proba, global_classes)
                rows.append({
                    **base, **metrics,
                    "preprocess_train_s_shared": float(prep_train_s),
                    "model_fit_s": float(fit_s),
                    "total_train_pipeline_s": float(prep_train_s + fit_s),
                    "preprocess_test_s_shared": float(prep_test_s),
                    "model_inference_s": float(infer_s),
                    "total_test_pipeline_s": float(prep_test_s + infer_s),
                    "preprocess_peak_memory_mb": float(prep_peak / (1024.0*1024.0)),
                    "model_peak_memory_mb": float(peak / (1024.0*1024.0)),
                    "status": "OK",
                })
            except Exception as exc:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
                rows.append({**base, "status": f"ERROR: {type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def per_seed_summary(folds: pd.DataFrame) -> pd.DataFrame:
    ok = folds.loc[folds["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    cols = [c for c in METRICS if c in ok.columns]
    time_cols = [
        c for c in [
            "preprocess_train_s", "model_fit_s", "total_train_pipeline_s",
            "preprocess_test_s", "model_inference_s", "total_test_pipeline_s",
            "oof_meta_training_s", "final_base_refit_s",
            "preprocess_train_s_shared", "preprocess_test_s_shared",
        ] if c in ok.columns
    ]
    agg_cols = cols + time_cols
    out = ok.groupby(["dataset", "display_name", "model", "report_model", "seed"], as_index=False)[agg_cols].mean()
    return out


def overall_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    numeric = [c for c in per_seed.columns if c not in {"dataset","display_name","model","report_model","seed"}]
    rows: List[Dict[str, Any]] = []
    for keys, g in per_seed.groupby(["dataset", "display_name", "model", "report_model"], sort=False):
        row = dict(zip(["dataset","display_name","model","report_model"], keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in numeric:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def revised_noncross_reference() -> pd.DataFrame:
    """Build performance reference with Bank replaced by Stage-1 no-duration run."""
    ref = pd.read_csv(REFERENCE_V4, encoding="utf-8-sig")
    bank = pd.read_csv(BANK_REVISED, encoding="utf-8-sig")
    bank = bank.drop(columns=["condition"], errors="ignore")
    key_models = ["WNB", "NBEM", "Adaptive Weighted NBEM", "Hybrid NBEM"]
    ref_keep = ref.loc[~((ref["dataset"] == "bank_marketing") & (ref["model"].isin(key_models)))].copy()
    combined = pd.concat([ref_keep, bank], ignore_index=True, sort=False)
    return combined


def crossfit_vs_reference(cross_per_seed: pd.DataFrame) -> pd.DataFrame:
    if cross_per_seed.empty:
        return pd.DataFrame()
    ref_folds = revised_noncross_reference()
    ref_ok = ref_folds.loc[(ref_folds["status"] == "OK") & (ref_folds["model"] == "Hybrid NBEM")].copy()
    ref_seed = ref_ok.groupby(["dataset","display_name","seed"], as_index=False)[[c for c in METRICS if c in ref_ok.columns]].mean()
    cf = cross_per_seed.loc[cross_per_seed["model"] == "Cross-Fitted Hybrid NBEM"].copy()
    merged = cf.merge(ref_seed, on=["dataset","display_name","seed"], how="inner", suffixes=("_crossfit","_noncross"))
    for m in METRICS:
        a = f"{m}_crossfit"; b = f"{m}_noncross"
        if a in merged.columns and b in merged.columns:
            merged[f"delta_crossfit_minus_noncross_{m}"] = merged[a] - merged[b]
    return merged


def dataset_aggregate_model_summary(*per_seed_frames: pd.DataFrame) -> pd.DataFrame:
    frames = [x for x in per_seed_frames if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True, sort=False)
    cols = [c for c in METRICS if c in allp.columns]
    dset = allp.groupby(["dataset","display_name","model","report_model"], as_index=False)[cols].mean()
    rows: List[Dict[str, Any]] = []
    for (model, report), g in dset.groupby(["model","report_model"], sort=False):
        row = {"model": model, "report_model": report, "n_datasets": int(g["dataset"].nunique())}
        for c in cols:
            row[f"{c}_mean_across_datasets"] = float(g[c].mean())
            row[f"{c}_sd_across_datasets"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_cached(
    prepared_list: Sequence[Any], seeds: Sequence[int], output_dir: Path, cache_subdir: str,
    runner, runner_kwargs: Dict[str, Any], file_prefix: str,
) -> pd.DataFrame:
    cache = output_dir / "cache" / cache_subdir
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
                frames.append(pd.read_csv(fn, encoding="utf-8-sig"))
                continue
            print(f"[{k}/{total}] run: {p.key}, seed={seed}")
            frame = runner(p, int(seed), **runner_kwargs)
            frame.to_csv(fn, index=False, encoding="utf-8-sig")
            frames.append(frame)
            pd.concat(frames, ignore_index=True, sort=False).to_csv(
                output_dir / f"{file_prefix}_PARTIAL.csv", index=False, encoding="utf-8-sig"
            )
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out


def save_bundle(prefix: str, folds: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds.to_csv(output_dir / f"{prefix}_all_fold_results.csv", index=False, encoding="utf-8-sig")
    ps = per_seed_summary(folds)
    ov = overall_summary(ps)
    ps.to_csv(output_dir / f"{prefix}_per_seed_summary.csv", index=False, encoding="utf-8-sig")
    ov.to_csv(output_dir / f"{prefix}_overall_summary.csv", index=False, encoding="utf-8-sig")
    return ps, ov


def validate_previous_stage_integrity() -> None:
    sem = pd.read_csv(STAGE1B_SEMANTIC, encoding="utf-8-sig")
    if len(sem) != 5 or not sem["revision_config_integrity_ok"].astype(bool).all():
        raise RuntimeError("Stage-1B semantic-decision integrity file is missing or not fully valid.")
    dup = pd.read_csv(STAGE1B_DUPLICATE, encoding="utf-8-sig")
    if dup.empty:
        raise RuntimeError("Stage-1B duplicate sensitivity output is missing.")


def mode_baselines(prepared: Sequence[Any], args, output_dir: Path) -> None:
    folds = run_cached(
        prepared_list=prepared,
        seeds=args.seeds,
        output_dir=output_dir,
        cache_subdir="external_baselines",
        runner=evaluate_external_baselines,
        runner_kwargs={
            "requested_folds": args.folds,
            "model_names": DEFAULT_BASELINE_MODELS,
            "quick": False,
        },
        file_prefix="10_external_baselines",
    )
    ps, _ = save_bundle("10_external_baselines", folds, output_dir)
    dataset_aggregate_model_summary(ps).to_csv(
        output_dir / "13_external_baselines_aggregate_across_datasets.csv", index=False, encoding="utf-8-sig"
    )
    errors = folds.loc[~folds["status"].eq("OK")]
    if not errors.empty:
        raise RuntimeError("External baseline run contains non-OK rows. Inspect 10_external_baselines_all_fold_results.csv")


def mode_crossfit(prepared: Sequence[Any], args, output_dir: Path, full: bool) -> None:
    if full:
        selected = list(prepared)
        prefix = "20_crossfit_full20"
        cache_subdir = "crossfit_full20"
    else:
        by_key = {p.key: p for p in prepared}
        selected = [by_key[k] for k in CROSSFIT_SUBSET]
        prefix = "20_crossfit_subset"
        cache_subdir = "crossfit_subset"
        pd.DataFrame({"dataset": CROSSFIT_SUBSET, "selection_basis": [
            "Bank duration correction / mixed binary task",
            "categorical multiclass task",
            "numerical multiclass medium-scale task",
            "small mixed-type clinical benchmark",
            "high-dimensional task with duplicate sensitivity",
            "mixed-type three-class task",
        ]}).to_csv(output_dir / "19_crossfit_subset_predeclared.csv", index=False, encoding="utf-8-sig")

    folds = run_cached(
        prepared_list=selected,
        seeds=args.seeds,
        output_dir=output_dir,
        cache_subdir=cache_subdir,
        runner=evaluate_crossfitted_hybrid,
        runner_kwargs={
            "requested_folds": args.folds,
            "inner_folds": args.inner_folds,
            "n_bins": args.n_bins,
            "mlp_max_iter": args.mlp_max_iter,
            "quick": False,
        },
        file_prefix=prefix,
    )
    ps, _ = save_bundle(prefix, folds, output_dir)
    cmp = crossfit_vs_reference(ps)
    cmp.to_csv(output_dir / f"{prefix}_vs_revised_noncross_per_seed.csv", index=False, encoding="utf-8-sig")
    if not cmp.empty:
        delta_cols = [c for c in cmp.columns if c.startswith("delta_crossfit_minus_noncross_")]
        summary_rows: List[Dict[str, Any]] = []
        for dataset, g in cmp.groupby("dataset", sort=False):
            row = {"dataset": dataset, "n_seeds": int(g["seed"].nunique())}
            for c in delta_cols:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / f"{prefix}_vs_revised_noncross_delta_summary.csv", index=False, encoding="utf-8-sig"
        )
    errors = folds.loc[~folds["status"].eq("OK")]
    if not errors.empty:
        raise RuntimeError(f"Cross-fit run contains non-OK rows. Inspect {prefix}_all_fold_results.csv")


def mode_timing(prepared: Sequence[Any], args, output_dir: Path) -> None:
    # Reference seed 42 is enough for the presentation/timing ambiguity; use every dataset.
    frames: List[pd.DataFrame] = []
    cache = output_dir / "cache" / "timing_audit"
    cache.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(prepared, 1):
        fn = cache / f"{p.key}__seed_42.csv"
        if fn.exists():
            print(f"[{i}/{len(prepared)}] resume timing: {p.key}")
            frames.append(pd.read_csv(fn, encoding="utf-8-sig"))
            continue
        print(f"[{i}/{len(prepared)}] timing: {p.key}")
        fr = evaluate_timing_audit(
            p, seed=42, requested_folds=args.folds, n_bins=args.n_bins,
            mlp_max_iter=args.mlp_max_iter, quick=False,
        )
        fr.to_csv(fn, index=False, encoding="utf-8-sig")
        frames.append(fr)
    folds = pd.concat(frames, ignore_index=True, sort=False)
    ps, ov = save_bundle("30_timing_audit", folds, output_dir)

    # Article-style aggregate: first average folds within dataset/model, then datasets.
    cols = [
        c for c in ["preprocess_train_s_shared","model_fit_s","total_train_pipeline_s",
                    "preprocess_test_s_shared","model_inference_s","total_test_pipeline_s"]
        if c in ps.columns
    ]
    d = ps.groupby(["dataset","model","report_model"], as_index=False)[cols].mean()
    rows = []
    for (model, report), g in d.groupby(["model","report_model"], sort=False):
        row = {"model": model, "report_model": report, "n_datasets": int(g["dataset"].nunique())}
        for c in cols:
            row[f"{c}_mean_across_datasets"] = float(g[c].mean())
            row[f"{c}_sd_across_datasets"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "33_timing_table6_clarification.csv", index=False, encoding="utf-8-sig")


def mode_smoke(prepared: Sequence[Any], args, output_dir: Path) -> None:
    # Small, non-scientific test only. Results are explicitly isolated and must not be reported.
    smoke_dir = output_dir / "SMOKE_DO_NOT_REPORT"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    by_key = {p.key: p for p in prepared}
    p = by_key["heart_disease"]
    b = evaluate_external_baselines(p, 42, 2, DEFAULT_BASELINE_MODELS, quick=True)
    c = evaluate_crossfitted_hybrid(p, 42, 2, 2, args.n_bins, min(args.mlp_max_iter, 60), quick=True)
    t = evaluate_timing_audit(p, 42, 2, args.n_bins, min(args.mlp_max_iter, 60), quick=True)
    b.to_csv(smoke_dir / "smoke_baselines.csv", index=False, encoding="utf-8-sig")
    c.to_csv(smoke_dir / "smoke_crossfit.csv", index=False, encoding="utf-8-sig")
    t.to_csv(smoke_dir / "smoke_timing.csv", index=False, encoding="utf-8-sig")
    all_status = pd.concat([
        b[["status"]].assign(part="baselines"),
        c[["status"]].assign(part="crossfit"),
        t[["status"]].assign(part="timing"),
    ], ignore_index=True)
    if not all_status["status"].eq("OK").all():
        raise RuntimeError("Smoke test failed. Inspect results/major_revision_stage2/SMOKE_DO_NOT_REPORT")
    print("SMOKE PASS. These outputs are not scientific results and must not be reported.")


def write_manifest(args, output_dir: Path, snapshot: pd.DataFrame) -> None:
    payload = {
        "stage": "Major Revision Stage 2",
        "timestamp_local": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "base_script_sha256": sha256_file(BASE_SCRIPT),
        "stage2_script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "exact_dataset_hash_matches": int(snapshot["hash_match"].sum()),
        "exact_dataset_count": int(len(snapshot)),
        "mode": args.mode,
        "seeds": list(map(int, args.seeds)),
        "outer_requested_folds": int(args.folds),
        "inner_requested_folds_for_crossfit": int(args.inner_folds),
        "n_bins": int(args.n_bins),
        "mlp_max_iter": int(args.mlp_max_iter),
        "bank_duration_removed_by_config": True,
        "primary_duplicate_policy": "no blind duplicate deletion; Stage-1B grouped sensitivity retained separately",
        "diabetes_outer_grouping": "patient-grouped",
        "external_baselines": {
            "Logistic Regression": "untuned C=1.0, lbfgs, one-hot categorical/boolean, standardized numeric",
            "Random Forest": "untuned standard RF, 100 trees, sqrt features, one-hot categorical/boolean, raw imputed numeric",
        },
        "crossfit_subset_predeclared": CROSSFIT_SUBSET,
    }
    with (output_dir / "RUN_MANIFEST_STAGE2.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mode", choices=["smoke","baselines","crossfit_subset","crossfit_full","timing"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--mlp-max-iter", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 0: verify exact dataset snapshot and frozen Stage-1/1B decisions...")
    snapshot = verify_snapshot(args.project_root, args.manifest, args.output_dir)
    print(f"exact hash matches: {int(snapshot['hash_match'].sum())}/{len(snapshot)}")
    validate_previous_stage_integrity()

    config = load_config(args.config)
    prepared = prepare_all(args.project_root, config)
    bank = next(p for p in prepared if p.key == "bank_marketing")
    if "duration" in bank.df.columns:
        raise RuntimeError("Frozen revision config failed: Bank Marketing duration is still present.")
    print("frozen protocol check: PASS (Bank duration absent; 20 datasets prepared)")

    if args.mode == "smoke":
        mode_smoke(prepared, args, args.output_dir)
    elif args.mode == "baselines":
        mode_baselines(prepared, args, args.output_dir)
    elif args.mode == "crossfit_subset":
        mode_crossfit(prepared, args, args.output_dir, full=False)
    elif args.mode == "crossfit_full":
        mode_crossfit(prepared, args, args.output_dir, full=True)
    elif args.mode == "timing":
        mode_timing(prepared, args, args.output_dir)

    write_manifest(args, args.output_dir, snapshot)
    print("DONE")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
