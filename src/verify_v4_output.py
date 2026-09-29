#!/usr/bin/env python3
"""Post-run checks for the corrected NBEM v4 experiment output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def norm(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()

    required = [
        out / "FINAL_VALIDATION.json",
        out / "dataset_audit.csv",
        out / "dataset_class_distributions.csv",
        out / "all_fold_results.csv",
        out / "article_tables",
        out / "article_figures",
    ]
    missing = [str(p) for p in required if not p.exists()]
    failures: list[str] = []
    if missing:
        failures.append("Missing required outputs: " + ", ".join(missing))

    if not failures:
        validation = json.loads((out / "FINAL_VALIDATION.json").read_text(encoding="utf-8"))
        if not validation.get("passed", False):
            failures.append("FINAL_VALIDATION.json does not report passed=true.")

        audit = pd.read_csv(out / "dataset_audit.csv")
        classes = pd.read_csv(out / "dataset_class_distributions.csv")
        results = pd.read_csv(out / "all_fold_results.csv", low_memory=False)

        ckd = audit[audit["dataset"] == "chronic_kidney_disease"]
        if ckd.empty:
            failures.append("CKD audit row is missing.")
        else:
            row = ckd.iloc[0]
            if int(row["n_classes"]) != 2:
                failures.append(f"CKD must be binary after cleanup; found {int(row['n_classes'])} classes.")
            dropped = norm(row.get("dropped_columns", ""))
            if "id" not in dropped:
                failures.append("CKD identifier column was not reported as dropped.")

        ckd_classes = classes[classes["dataset"] == "chronic_kidney_disease"]
        labels = {str(x).strip() for x in ckd_classes["class_label"].tolist()}
        if any("\t" in str(x) or str(x) != str(x).strip() for x in ckd_classes["class_label"].tolist()):
            failures.append("CKD class labels still contain surrounding whitespace or tab characters.")
        if labels != {"ckd", "notckd"}:
            failures.append(f"Unexpected CKD labels after cleanup: {sorted(labels)}")

        expected_drops = {
            "cirrhosis_patient_survival_prediction_dataset_1": "id",
            "ecoli": "sequence_name",
            "glass_identification": "id",
        }
        for dataset, expected in expected_drops.items():
            rows = audit[audit["dataset"] == dataset]
            if rows.empty:
                failures.append(f"Audit row missing for {dataset}.")
                continue
            dropped = norm(rows.iloc[0].get("dropped_columns", ""))
            if norm(expected) not in dropped:
                failures.append(f"Expected identifier {expected!r} was not dropped for {dataset}.")

        if "status" in results.columns:
            non_ok = results[results["status"].astype(str) != "OK"]
            if not non_ok.empty:
                failures.append(f"Found {len(non_ok)} non-OK fold result rows.")

        forbidden_models = {"xgboost", "lightgbm", "catboost", "random forest", "tabpfn"}
        observed_models = {str(x).strip().lower() for x in results.get("model", pd.Series(dtype=str)).dropna()}
        present = sorted(forbidden_models & observed_models)
        if present:
            failures.append("Forbidden modern baselines found in results: " + ", ".join(present))

        figures = sorted((out / "article_figures").glob("*.png"))
        if len(figures) != 3:
            failures.append(f"Expected exactly 3 article figures; found {len(figures)}.")

    if failures:
        print("V4 OUTPUT VERIFICATION: FAILED")
        for item in failures:
            print("- " + item)
        return 1

    print("V4 OUTPUT VERIFICATION: PASSED")
    print(f"Output directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
