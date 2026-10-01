#!/usr/bin/env python3
"""Major Revision Stage 4: Random Forest on the exact locked 20-seed Stage-3 protocol.

Purpose
-------
This script extends the *already locked* Stage-3 20-seed comparison with Random Forest (RF).
It is deliberately designed to avoid post-hoc seed selection, hyperparameter cherry-picking,
or changes to the preprocessing protocol after outcomes are observed.

What is reused
--------------
- The exact 20 datasets and frozen revised preprocessing protocol (Bank Marketing duration removed).
- The exact 20 seed list locked before Stage-3 outcomes.
- The exact Stage-2 RF specification for the original five seeds:
    RandomForestClassifier(
      n_estimators=100, criterion='gini', max_depth=None,
      min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
      bootstrap=True, class_weight=None, random_state=seed, n_jobs=RF_JOBS)
- The already validated RF results for the original five seeds {13,21,42,87,123}.
- The finalized Stage-3 20-seed results for Hybrid NBEM, NBEM-prop, and Logistic Regression.

What is newly fitted
--------------------
Only RF for the 15 additional Stage-3 seeds, on the same CV split generator and fold-local
preprocessing used in Stage 2/3. The run is resume-safe at dataset x seed granularity.

Primary inference
-----------------
The confirmatory external-baseline contrast is Hybrid NBEM vs Random Forest at the DATASET level
(n=20 datasets), using the 20-seed mean for each dataset before paired inference. We report:
- mean paired difference (Hybrid - RF)
- bootstrap 95% CI over datasets
- Wilcoxon signed-rank p-value
- paired rank-biserial effect size
- Win/Tie/Loss over datasets
Seed-level summaries are descriptive robustness checks, not independent inferential units.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, t as student_t, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "base" / "nbem_article_experiments_v4_EXACT_USED.py"
REV_CONFIG = ROOT / "config" / "dataset_config_revision_frozen.json"
V4_CONFIG = ROOT / "config" / "dataset_config_v4_exact_used.json"
SEED_PLAN = ROOT / "config" / "stage4_seed_plan_LOCKED_SAME_AS_STAGE3.json"
MANIFEST = ROOT / "manifest" / "DATASET_MANIFEST_PORTABLE.csv"
STAGE2_SOURCE = ROOT / "provenance" / "stage2_external_baselines_first5_source.csv"
STAGE3_ALLFOLDS = ROOT / "provenance" / "10_stage3_all_fold_results_20seeds.csv"
STAGE3_DATASET_SUMMARY = ROOT / "provenance" / "12_stage3_dataset_model_20seed_summary.csv"
STAGE3_CHARS = ROOT / "provenance" / "19_dataset_characteristics.csv"
DEFAULT_OUTPUT = ROOT / "results" / "major_revision_stage4_rf20"

PRIMARY_METRICS = ["f1_weighted", "f1_macro"]
ALL_METRICS = [
    "accuracy", "balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
    "precision_macro", "recall_macro", "f1_macro", "roc_auc_weighted"
]
ORIGINAL_FIVE = [13, 21, 42, 87, 123]
TIE_TOL = 1e-12
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260926

spec = importlib.util.spec_from_file_location("nbem_v4_exact", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load exact v4 base script")
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)


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
    p = load_json(SEED_PLAN)
    original = list(map(int, p["original_five_seeds"]))
    extra = list(map(int, p["additional_fifteen_seeds"]))
    all20 = list(map(int, p["all_twenty_seeds"]))
    if original != ORIGINAL_FIVE:
        raise RuntimeError("Original five seed set changed.")
    rng = np.random.default_rng(20260924)
    regen: List[int] = []
    while len(regen) < 15:
        x = int(rng.integers(1, 1000000))
        if x not in set(original) and x not in regen:
            regen.append(x)
    if regen != extra or all20 != original + extra or len(set(all20)) != 20:
        raise RuntimeError("Locked Stage-3/4 20-seed plan failed deterministic verification.")
    return p


def verify_revision_config_scope() -> None:
    a = load_json(V4_CONFIG)
    b = load_json(REV_CONFIG)
    a.pop("experiment_revision", None); b.pop("experiment_revision", None)
    bank_a = a.get("dataset_overrides", {}).get("bank_marketing", {})
    bank_b = b.get("dataset_overrides", {}).get("bank_marketing", {})
    drop = bank_b.get("drop_feature_candidates", [])
    bank_b2 = dict(bank_b); bank_b2.pop("drop_feature_candidates", None)
    a2 = json.loads(json.dumps(a)); b2 = json.loads(json.dumps(b))
    a2.setdefault("dataset_overrides", {})["bank_marketing"] = bank_a
    b2.setdefault("dataset_overrides", {})["bank_marketing"] = bank_b2
    if a2 != b2 or drop != ["duration"]:
        raise RuntimeError("Frozen revision config differs from exact v4 beyond Bank duration removal.")


def verify_snapshot(project_root: Path, output_dir: Path) -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    rows = []
    for _, r in m.iterrows():
        rel = Path(str(r["expected_relative_path"]))
        path = project_root / rel
        expected = str(r["sha256"]).strip().lower()
        actual = sha256_file(path) if path.exists() else ""
        rows.append({"dataset": r["dataset"], "relative_path": rel.as_posix(), "exists": path.exists(),
                     "expected_sha256": expected, "actual_sha256": actual,
                     "hash_match": bool(path.exists() and actual == expected)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "00_exact_snapshot_verification.csv", index=False, encoding="utf-8-sig")
    if len(out) != 20 or not out["hash_match"].all():
        raise RuntimeError("Exact dataset snapshot verification failed.")
    return out


def prepare_all(project_root: Path) -> List[Any]:
    cfg = load_json(REV_CONFIG)
    paths = BASE.find_datasets(project_root, use_canonical_20=bool(cfg.get("use_canonical_20", True)))
    prepared = [BASE.prepare_dataset(p, cfg) for p in paths]
    if len(prepared) != 20:
        raise RuntimeError(f"Expected 20 datasets, found {len(prepared)}")
    bad = [p for p in prepared if p.critical_issues]
    if bad:
        raise RuntimeError("Critical dataset issue(s): " + "; ".join(f"{p.key}:{p.critical_issues}" for p in bad))
    bank = next(p for p in prepared if p.key == "bank_marketing")
    if "duration" in bank.df.columns:
        raise RuntimeError("Frozen revision protocol failed: Bank Marketing duration remains present.")
    return prepared


def prefer_grouped_main(prepared: Any) -> bool:
    return bool(prepared.groups is not None and prepared.key == "diabetes_130_us_hospitals_for_years_1999_2008")


class RFBaselinePreprocessor:
    """Exact Stage-2 RF preprocessing: fold-local OHE categorical/boolean + median numeric, no scaling."""
    def fit(self, df: pd.DataFrame, target: str) -> "RFBaselinePreprocessor":
        self.target = target
        b, c, n = BASE.detect_feature_groups(df, target)
        self.cat_cols = list(b) + list(c)
        self.num_cols = list(n)
        self.cat_imp = SimpleImputer(strategy="most_frequent")
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        self.num_imp = SimpleImputer(strategy="median")
        X = df.drop(columns=[target])
        if self.cat_cols:
            C = BASE.string_frame_preserve_missing(X[self.cat_cols])
            Cimp = self.cat_imp.fit_transform(C)
            self.ohe.fit(Cimp)
        if self.num_cols:
            self.num_imp.fit(X[self.num_cols])
        return self

    def transform(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target])
        parts: List[Any] = []
        if self.cat_cols:
            C = BASE.string_frame_preserve_missing(X[self.cat_cols])
            parts.append(self.ohe.transform(self.cat_imp.transform(C)))
        if self.num_cols:
            N = self.num_imp.transform(X[self.num_cols]).astype(float)
            parts.append(sparse.csr_matrix(N))
        if not parts:
            return sparse.csr_matrix(np.zeros((len(df), 1), dtype=float))
        return sparse.hstack(parts, format="csr")


def rf_factory(seed: int, rf_jobs: int) -> RandomForestClassifier:
    # EXACT Stage-2 RF hyperparameters; only n_jobs is exposed as a compute setting, not a statistical parameter.
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        n_jobs=int(rf_jobs),
        random_state=int(seed),
    )


def validate_stage3(plan: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(STAGE3_ALLFOLDS, encoding="utf-8-sig")
    checks = []
    checks.append(("20_datasets", df.dataset.nunique() == 20))
    checks.append(("20_locked_seeds", set(map(int, df.seed.unique())) == set(map(int, plan["all_twenty_seeds"]))))
    checks.append(("models_exact", set(df.model.unique()) == {"NBEM", "Hybrid NBEM", "Logistic Regression"}))
    checks.append(("all_status_OK", bool(df.status.eq("OK").all())))
    combo = df.groupby(["dataset", "seed", "model"]).size().reset_index(name="n")
    checks.append(("20x20x3_combos", len(combo) == 20 * 20 * 3))
    out = pd.DataFrame(checks, columns=["check", "pass"])
    out.to_csv(output_dir / "02_stage3_20seed_validation.csv", index=False, encoding="utf-8-sig")
    if not out["pass"].all():
        raise RuntimeError("Stage-3 provenance validation failed.\n" + out.to_string(index=False))
    return df


def normalize_rf_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["source_provenance"] = source
    out["report_model"] = "Random Forest"
    return out


def validate_rf_first5(plan: Dict[str, Any], stage3: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    s2 = pd.read_csv(STAGE2_SOURCE, encoding="utf-8-sig")
    rf = s2[s2.model.eq("Random Forest")].copy()
    original = set(map(int, plan["original_five_seeds"]))
    checks = []
    checks.append(("rf_only", set(rf.model.unique()) == {"Random Forest"}))
    checks.append(("20_datasets", rf.dataset.nunique() == 20))
    checks.append(("five_locked_original_seeds", set(map(int, rf.seed.unique())) == original))
    checks.append(("all_status_OK", bool(rf.status.eq("OK").all())))
    combo = rf.groupby(["dataset", "seed"]).size().reset_index(name="n")
    checks.append(("20x5_dataset_seed_combos", len(combo) == 20 * 5))
    # Protocol/fold-count pairing against Stage3 for the same dataset+seed.
    rsig = rf.groupby(["dataset", "seed"]).agg(rf_folds=("fold", "nunique"), rf_protocol=("cv_protocol", "first")).reset_index()
    s3 = stage3[stage3.seed.isin(list(original))]
    hsig = s3[s3.model.eq("Logistic Regression")].groupby(["dataset", "seed"]).agg(s3_folds=("fold", "nunique"), s3_protocol=("cv_protocol", "first")).reset_index()
    z = rsig.merge(hsig, on=["dataset", "seed"], how="outer")
    paired = bool((z.rf_folds == z.s3_folds).all() and (z.rf_protocol == z.s3_protocol).all())
    checks.append(("same_protocol_and_fold_count_as_stage3_first5", paired))
    out = pd.DataFrame(checks, columns=["check", "pass"])
    out.to_csv(output_dir / "03_rf_first5_validation.csv", index=False, encoding="utf-8-sig")
    if not out["pass"].all():
        raise RuntimeError("Stage-2 RF first-five validation failed.\n" + out.to_string(index=False))
    return normalize_rf_frame(rf, "validated Stage-2 RF first-five final-protocol result")


def evaluate_rf(prepared: Any, seed: int, requested_folds: int, rf_jobs: int, quick: bool = False) -> pd.DataFrame:
    BASE.set_reproducibility(int(seed))
    df = prepared.df.copy().reset_index(drop=True)
    groups = prepared.groups.copy() if prepared.groups is not None else None
    if quick and len(df) > 1200:
        idx = df.sample(n=1200, random_state=int(seed)).index.to_numpy()
        df = df.iloc[idx].reset_index(drop=True)
        if groups is not None:
            groups = groups[idx]
    enc = LabelEncoder(); y = enc.fit_transform(df[prepared.target].astype(str)); global_classes = np.arange(len(enc.classes_))
    plan = BASE.make_cv_plan(y=y, seed=int(seed), requested_folds=int(requested_folds), X=df, groups=groups,
                            prefer_grouped=prefer_grouped_main(prepared), forced_protocol=None)
    rows: List[Dict[str, Any]] = []
    for fold, (tr, te) in enumerate(plan.splits, 1):
        train_df = df.iloc[tr].reset_index(drop=True); test_df = df.iloc[te].reset_index(drop=True)
        y_train, y_test = y[tr], y[te]
        base = {
            "analysis":"major_revision_stage4_rf20", "dataset":prepared.key, "display_name":prepared.display_name,
            "file":str(prepared.path), "target":prepared.target, "target_transform":prepared.target_transform,
            "model":"Random Forest", "report_model":"Random Forest", "seed":int(seed), "fold":int(fold),
            "cv_protocol":plan.protocol, "n_splits_or_total_folds":int(plan.n_splits), "stratified":bool(plan.stratified),
            "grouped":bool(plan.grouped), "n_rows":int(len(df)), "n_features":int(df.shape[1]-1),
            "n_classes":int(len(global_classes)),
        }
        if len(np.unique(y_train)) < 2:
            rows.append({**base, "status":"SKIPPED: fewer than two training classes"}); continue
        try:
            tracemalloc.start()
            t0=time.perf_counter(); prep=RFBaselinePreprocessor().fit(train_df, prepared.target); Xtr=prep.transform(train_df); prep_train=time.perf_counter()-t0
            t1=time.perf_counter(); model=rf_factory(int(seed), rf_jobs); model.fit(Xtr,y_train); fit_s=time.perf_counter()-t1
            t2=time.perf_counter(); Xte=prep.transform(test_df); prep_test=time.perf_counter()-t2
            t3=time.perf_counter(); pred=model.predict(Xte); raw=model.predict_proba(Xte); infer_s=time.perf_counter()-t3
            _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            proba=BASE.align_proba(raw,getattr(model,"classes_",np.unique(y_train)),global_classes)
            metrics=BASE.compute_metrics(y_test,pred,proba,global_classes)
            rows.append({**base, **metrics,
                         "preprocess_train_s":float(prep_train), "model_fit_s":float(fit_s), "total_train_pipeline_s":float(prep_train+fit_s),
                         "preprocess_test_s":float(prep_test), "model_inference_s":float(infer_s), "total_test_pipeline_s":float(prep_test+infer_s),
                         "peak_memory_mb":float(peak/(1024**2)),
                         "representation":"fold-local one-hot categorical/boolean + median-imputed numeric",
                         "status":"OK"})
        except Exception as exc:
            try: tracemalloc.stop()
            except Exception: pass
            rows.append({**base, **{m:np.nan for m in ALL_METRICS}, "status":f"ERROR: {type(exc).__name__}: {exc}"})
    return normalize_rf_frame(pd.DataFrame(rows), "new Stage-4 RF fit from locked Stage-3 additional seed")


def run_additional(prepared: Sequence[Any], seeds: Sequence[int], output_dir: Path, folds: int, rf_jobs: int) -> pd.DataFrame:
    cache = output_dir / "cache" / "rf_additional_fifteen"; cache.mkdir(parents=True, exist_ok=True)
    frames=[]; total=len(prepared)*len(seeds); k=0
    for p in prepared:
        for seed in seeds:
            k+=1; fn=cache/f"{p.key}__seed_{int(seed)}.csv"
            if fn.exists():
                print(f"[{k}/{total}] resume: {p.key}, seed={seed}", flush=True); fr=pd.read_csv(fn,encoding="utf-8-sig")
            else:
                print(f"[{k}/{total}] run: {p.key}, seed={seed}", flush=True); fr=evaluate_rf(p,int(seed),folds,rf_jobs,quick=False); fr.to_csv(fn,index=False,encoding="utf-8-sig")
            frames.append(fr)
            pd.concat(frames,ignore_index=True,sort=False).to_csv(output_dir/"09_rf_additional15_PARTIAL_DO_NOT_REPORT.csv",index=False,encoding="utf-8-sig")
    return pd.concat(frames,ignore_index=True,sort=False)


def per_seed_rf(folds: pd.DataFrame) -> pd.DataFrame:
    ok=folds[folds.status.eq("OK")].copy(); cols=[c for c in ALL_METRICS if c in ok.columns]
    return ok.groupby(["dataset","display_name","model","report_model","seed"],as_index=False)[cols].mean()


def dataset_summary(ps: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for keys,g in ps.groupby(["dataset","display_name","model","report_model"],sort=False):
        row=dict(zip(["dataset","display_name","model","report_model"],keys)); row["n_seeds"]=int(g.seed.nunique())
        for m in ALL_METRICS:
            if m in g:
                vals=g[m].dropna().to_numpy(float); row[m]=float(np.mean(vals)) if len(vals) else np.nan; row[m+"_sd_across_seeds"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(ds4: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for model,g in ds4.groupby("report_model",sort=False):
        row={"model":model,"n_datasets":int(g.dataset.nunique())}
        for m in ALL_METRICS:
            if m in g:
                vals=g[m].dropna().to_numpy(float); row[m]=float(np.mean(vals)) if len(vals) else np.nan; row[m+"_sd_across_datasets"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_weighted",ascending=False).reset_index(drop=True)


def rank_biserial(diff: np.ndarray) -> float:
    d=np.asarray(diff,float); d=d[np.isfinite(d) & (np.abs(d)>TIE_TOL)]
    if len(d)==0: return 0.0
    ranks=pd.Series(np.abs(d)).rank(method="average").to_numpy(float)
    rp=float(ranks[d>0].sum()); rn=float(ranks[d<0].sum()); den=rp+rn
    return (rp-rn)/den if den else 0.0


def bootstrap_ci(diff: np.ndarray, salt: int) -> Tuple[float,float]:
    d=np.asarray(diff,float); d=d[np.isfinite(d)]
    rng=np.random.default_rng(BOOTSTRAP_SEED+salt); n=len(d)
    idx=rng.integers(0,n,size=(BOOTSTRAP_REPS,n)); means=d[idx].mean(axis=1)
    lo,hi=np.quantile(means,[0.025,0.975]); return float(lo),float(hi)


def hybrid_vs_rf_tests(ds4: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows=[]; diffs=[]; wtl=[]
    for mi,m in enumerate(PRIMARY_METRICS):
        p=ds4.pivot(index="dataset",columns="report_model",values=m).dropna(subset=["Hybrid NBEM","Random Forest"])
        d=(p["Hybrid NBEM"]-p["Random Forest"]).to_numpy(float)
        stat,pv=wilcoxon(d,zero_method="wilcox",alternative="two-sided") if np.any(np.abs(d)>TIE_TOL) else (0.0,1.0)
        lo,hi=bootstrap_ci(d,mi*100)
        wins=int(np.sum(d>TIE_TOL)); ties=int(np.sum(np.abs(d)<=TIE_TOL)); losses=int(np.sum(d<-TIE_TOL))
        rows.append({"metric":m,"contrast":"Hybrid NBEM - Random Forest","n_datasets":len(d),"mean_paired_difference":float(np.mean(d)),"median_paired_difference":float(np.median(d)),"bootstrap95_ci_low":lo,"bootstrap95_ci_high":hi,"wins":wins,"ties":ties,"losses":losses,"wilcoxon_statistic":float(stat),"wilcoxon_p_two_sided":float(pv),"rank_biserial_positive_favors_hybrid":float(rank_biserial(d))})
        for ds,val in zip(p.index,d): diffs.append({"dataset":ds,"metric":m,"hybrid_minus_rf":float(val),"winner":"Hybrid NBEM" if val>TIE_TOL else ("Random Forest" if val<-TIE_TOL else "Tie")})
        wtl.append({"metric":m,"wins_hybrid":wins,"ties":ties,"losses_hybrid":losses})
    return pd.DataFrame(rows),pd.DataFrame(diffs),pd.DataFrame(wtl)


def seed_stability(stage3_ps: pd.DataFrame, rf_ps: pd.DataFrame) -> pd.DataFrame:
    h=stage3_ps[stage3_ps.report_model.eq("Hybrid NBEM")].groupby("seed")[PRIMARY_METRICS].mean().reset_index()
    r=rf_ps.groupby("seed")[PRIMARY_METRICS].mean().reset_index()
    z=h.merge(r,on="seed",suffixes=("_hybrid","_rf"))
    for m in PRIMARY_METRICS: z[m+"_hybrid_minus_rf"] = z[m+"_hybrid"]-z[m+"_rf"]
    return z


def capability_profiles(ds4: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    chars=pd.read_csv(STAGE3_CHARS,encoding="utf-8-sig")
    h=ds4.pivot(index="dataset",columns="report_model",values=PRIMARY_METRICS)
    rows=[]
    for m in PRIMARY_METRICS:
        if (m,"Hybrid NBEM") not in h.columns or (m,"Random Forest") not in h.columns: continue
        tmp=pd.DataFrame({"dataset":h.index,"gain":h[(m,"Hybrid NBEM")]-h[(m,"Random Forest")]}).reset_index(drop=True).merge(chars,on="dataset",how="left")
        for col in ["task_type","size_half","feature_width_half"]:
            if col in tmp.columns:
                for val,g in tmp.groupby(col,dropna=False):
                    rows.append({"metric":m,"group_variable":col,"group":str(val),"n_datasets":len(g),"mean_hybrid_minus_rf":float(g.gain.mean())})
    prof=pd.DataFrame(rows)
    corr=[]
    for m in PRIMARY_METRICS:
        if (m,"Hybrid NBEM") not in h.columns: continue
        tmp=pd.DataFrame({"dataset":h.index,"gain":h[(m,"Hybrid NBEM")]-h[(m,"Random Forest")]}).reset_index(drop=True).merge(chars,on="dataset",how="left")
        for col in ["n_rows","n_features","n_classes"]:
            if col in tmp and tmp[col].notna().sum()>=5:
                rho,p=spearmanr(tmp[col],tmp.gain,nan_policy="omit")
                corr.append({"metric":m,"characteristic":col,"spearman_rho":float(rho),"p_two_sided":float(p),"n_datasets":int(tmp[[col,"gain"]].dropna().shape[0])})
    return prof,pd.DataFrame(corr)


def analyze(rf20: pd.DataFrame, stage3: pd.DataFrame, output_dir: Path, plan: Dict[str,Any]) -> None:
    if not rf20.status.eq("OK").all(): raise RuntimeError("RF 20-seed results contain non-OK rows.")
    if rf20.dataset.nunique()!=20 or set(map(int,rf20.seed.unique()))!=set(map(int,plan["all_twenty_seeds"])):
        raise RuntimeError("RF final matrix incomplete.")
    combo=rf20.groupby(["dataset","seed"]).size().reset_index(name="n")
    if len(combo)!=400: raise RuntimeError(f"Expected 400 RF dataset-seed combinations, got {len(combo)}")
    # Pair protocol/fold count against Stage3 Logistic for every dataset+seed.
    rs=rf20.groupby(["dataset","seed"]).agg(rf_folds=("fold","nunique"),rf_protocol=("cv_protocol","first")).reset_index()
    ss=stage3[stage3.model.eq("Logistic Regression")].groupby(["dataset","seed"]).agg(s3_folds=("fold","nunique"),s3_protocol=("cv_protocol","first")).reset_index()
    z=rs.merge(ss,on=["dataset","seed"],how="outer")
    if not ((z.rf_folds==z.s3_folds).all() and (z.rf_protocol==z.s3_protocol).all()): raise RuntimeError("RF vs Stage3 split protocol/fold mismatch detected.")

    rf_ps=per_seed_rf(rf20); rf_ds=dataset_summary(rf_ps)
    s3_ps=pd.read_csv(ROOT/"provenance"/"11_stage3_per_seed_dataset_model.csv",encoding="utf-8-sig")
    s3_ds=pd.read_csv(STAGE3_DATASET_SUMMARY,encoding="utf-8-sig")
    # Standardize Stage3 report label for NBEM.
    if "report_model" not in s3_ds.columns: s3_ds["report_model"]=s3_ds["model"].replace({"NBEM":"NBEM-prop"})
    else: s3_ds["report_model"]=s3_ds["report_model"].replace({"NBEM":"NBEM-prop"})
    if "report_model" not in s3_ps.columns: s3_ps["report_model"]=s3_ps["model"].replace({"NBEM":"NBEM-prop"})
    else: s3_ps["report_model"]=s3_ps["report_model"].replace({"NBEM":"NBEM-prop"})
    rf_ds2=rf_ds.copy(); rf_ds2["report_model"]="Random Forest"
    ds4=pd.concat([s3_ds,rf_ds2],ignore_index=True,sort=False)
    agg=aggregate_summary(ds4)
    tests,diffs,wtl=hybrid_vs_rf_tests(ds4)
    stab=seed_stability(s3_ps,rf_ps)
    prof,corr=capability_profiles(ds4)

    rf20.to_csv(output_dir/"10_rf_all_fold_results_20seeds.csv",index=False,encoding="utf-8-sig")
    rf_ps.to_csv(output_dir/"11_rf_per_seed_dataset_model.csv",index=False,encoding="utf-8-sig")
    rf_ds.to_csv(output_dir/"12_rf_dataset_model_20seed_summary.csv",index=False,encoding="utf-8-sig")
    ds4.to_csv(output_dir/"13_four_model_dataset_summary_20seeds.csv",index=False,encoding="utf-8-sig")
    agg.to_csv(output_dir/"14_four_model_aggregate_summary_20seeds.csv",index=False,encoding="utf-8-sig")
    tests.to_csv(output_dir/"15_hybrid_vs_rf_primary_dataset_level_tests.csv",index=False,encoding="utf-8-sig")
    diffs.to_csv(output_dir/"16_hybrid_vs_rf_dataset_level_differences.csv",index=False,encoding="utf-8-sig")
    wtl.to_csv(output_dir/"17_hybrid_vs_rf_win_tie_loss.csv",index=False,encoding="utf-8-sig")
    stab.to_csv(output_dir/"18_hybrid_vs_rf_seed_level_stability_DESCRIPTIVE.csv",index=False,encoding="utf-8-sig")
    prof.to_csv(output_dir/"19_hybrid_vs_rf_exploratory_profiles.csv",index=False,encoding="utf-8-sig")
    corr.to_csv(output_dir/"20_hybrid_vs_rf_exploratory_correlations.csv",index=False,encoding="utf-8-sig")

    lines=["MAJOR REVISION STAGE 4 - 20-SEED RANDOM FOREST COMPARISON","", "Four-model aggregate means (equal weight per dataset):",agg.to_string(index=False),"","Hybrid vs Random Forest planned dataset-level contrasts:",tests.to_string(index=False),"", "Interpretation guardrails:","- The external-baseline comparison is descriptive/confirmatory for the revised benchmark, not a claim of universal superiority.","- Statistical inference uses datasets (n=20) as paired units; seeds are repeated robustness runs, not independent inferential samples.","- Report the actual direction of the RF comparison even if it is unfavorable to Hybrid NBEM.","- Do not select or exclude seeds after viewing outcomes."]
    (output_dir/"21_MANUSCRIPT_READY_RF20_SUMMARY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_manifest(args: argparse.Namespace, output_dir: Path, snapshot: pd.DataFrame, plan: Dict[str,Any]) -> None:
    payload={
        "stage":"Major Revision Stage 4 - RF20",
        "timestamp_local":pd.Timestamp.now().isoformat(), "python":sys.version, "platform":platform.platform(),
        "base_script_sha256":sha256_file(BASE_SCRIPT), "stage4_script_sha256":sha256_file(Path(__file__)),
        "revision_config_sha256":sha256_file(REV_CONFIG), "seed_plan_sha256":sha256_file(SEED_PLAN),
        "stage2_rf_source_sha256":sha256_file(STAGE2_SOURCE), "stage3_20seed_source_sha256":sha256_file(STAGE3_ALLFOLDS),
        "exact_dataset_hash_matches":int(snapshot.hash_match.sum()), "exact_dataset_count":int(len(snapshot)),
        "all_twenty_seeds":plan["all_twenty_seeds"], "original_five_rf_reused":plan["original_five_seeds"],
        "additional_fifteen_rf_fitted":plan["additional_fifteen_seeds"], "rf_jobs":int(args.rf_jobs),
        "rf_spec":{"n_estimators":100,"criterion":"gini","max_depth":None,"min_samples_split":2,"min_samples_leaf":1,"max_features":"sqrt","bootstrap":True,"class_weight":None,"random_state":"seed"},
        "primary_external_contrast":"Hybrid NBEM vs Random Forest", "primary_metrics":PRIMARY_METRICS,
        "primary_inference_unit":"dataset", "bootstrap_reps":BOOTSTRAP_REPS, "bootstrap_seed":BOOTSTRAP_SEED,
        "bank_duration_removed":True, "diabetes_outer_cv":"patient-grouped",
        "guardrail":"No post-outcome seed selection or RF hyperparameter tuning in Stage 4; exact Stage-2 RF specification retained for comparability."
    }
    (output_dir/"RUN_MANIFEST_STAGE4_RF20.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    lines=[]
    for p in sorted(output_dir.glob("*")):
        if p.is_file() and p.name!="CHECKSUMS_OUTPUT_SHA256.txt": lines.append(f"{sha256_file(p)}  {p.name}")
    (output_dir/"CHECKSUMS_OUTPUT_SHA256.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=ROOT)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--mode",choices=["smoke","full","analyze"],required=True)
    ap.add_argument("--folds",type=int,default=10)
    ap.add_argument("--rf-jobs",type=int,default=-1,help="RandomForest n_jobs. Use -1 to use all available CPU cores.")
    return ap.parse_args()


def main() -> None:
    args=parse_args(); args.project_root=args.project_root.resolve(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    print("Step 0/6: validate locked 20-seed plan, exact datasets, and frozen revision protocol...",flush=True)
    plan=verify_seed_plan(); verify_revision_config_scope(); snap=verify_snapshot(args.project_root,args.output_dir)
    print(f"exact hash matches: {int(snap.hash_match.sum())}/{len(snap)}",flush=True)
    prepared=prepare_all(args.project_root); print("frozen protocol: PASS (20 datasets; Bank duration removed)",flush=True)
    s3=validate_stage3(plan,args.output_dir); print("Stage-3 finalized 20-seed matrix: PASS",flush=True)
    rf5=validate_rf_first5(plan,s3,args.output_dir); print("Stage-2 RF original five seeds: PASS",flush=True)
    pd.DataFrame([{"order":i+1,"seed":int(s),"rf_source":"validated Stage-2 reuse" if s in ORIGINAL_FIVE else "new Stage-4 fit"} for i,s in enumerate(plan["all_twenty_seeds"])]).to_csv(args.output_dir/"01_stage4_rf20_seed_plan.csv",index=False,encoding="utf-8-sig")

    if args.mode=="smoke":
        p=next(x for x in prepared if x.key=="heart_disease")
        fr=evaluate_rf(p,20260926,2,args.rf_jobs,quick=True); sd=args.output_dir/"SMOKE_DO_NOT_REPORT"; sd.mkdir(exist_ok=True); fr.to_csv(sd/"smoke_rf.csv",index=False,encoding="utf-8-sig")
        if not fr.status.eq("OK").all(): raise RuntimeError("RF smoke test failed")
        print("SMOKE PASS. Do not report smoke outputs.",flush=True); write_manifest(args,args.output_dir,snap,plan); write_checksums(args.output_dir); return

    if args.mode=="full":
        print("Step 1/6: fit RF for the 15 additional LOCKED Stage-3 seeds (resume-safe)...",flush=True)
        new=run_additional(prepared,plan["additional_fifteen_seeds"],args.output_dir,args.folds,args.rf_jobs)
    else:
        print("Analyze-only: loading cached 15-seed RF files...",flush=True)
        files=sorted((args.output_dir/"cache"/"rf_additional_fifteen").glob("*.csv"))
        if len(files)!=300: raise RuntimeError(f"Expected 300 cached dataset-seed files; found {len(files)}")
        new=pd.concat([pd.read_csv(f,encoding="utf-8-sig") for f in files],ignore_index=True,sort=False)
    if new.empty or not new.status.eq("OK").all(): raise RuntimeError("Additional RF run incomplete or contains errors")
    new.to_csv(args.output_dir/"04_rf_additional15_all_fold_results.csv",index=False,encoding="utf-8-sig")
    rf20=pd.concat([rf5,new],ignore_index=True,sort=False)
    print("Step 2/6: combine five validated + fifteen new RF seeds...",flush=True)
    print("Step 3/6: validate paired protocol and analyze Hybrid vs RF...",flush=True)
    analyze(rf20,s3,args.output_dir,plan)
    print("Step 4/6: write manifest and checksums...",flush=True); write_manifest(args,args.output_dir,snap,plan); write_checksums(args.output_dir)
    print("Step 5/6: final primary contrast...",flush=True)
    tests=pd.read_csv(args.output_dir/"15_hybrid_vs_rf_primary_dataset_level_tests.csv",encoding="utf-8-sig"); print(tests.to_string(index=False),flush=True)
    print("Step 6/6: DONE",flush=True); print(f"Outputs: {args.output_dir}",flush=True)

if __name__=="__main__": main()
