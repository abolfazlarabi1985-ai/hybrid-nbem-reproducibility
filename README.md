# Hybrid NBEM — Final Article Reproducibility Pipeline

This repository contains the final NB/NBEM-family experimental pipeline for the
article **Adaptive Weighted and Hybrid Extensions of Naive Bayes Enrichment
Method for Mixed-Type Data Classification**.

The pipeline is deliberately aligned with the actual scope of the manuscript:
it compares the proposed extensions within the Naive Bayes/NBEM family and does
not execute Random Forest, gradient boosting, XGBoost, LightGBM, CatBoost, or
TabPFN.

## What one full run produces

- critical target/leakage audits, especially for Diabetes;
- five predefined random seeds for WNB, NBEM, Adaptive Weighted NBEM, and Hybrid NBEM: `13, 21, 42, 87, 123`;
- one common reference seed (`42`) for the complete twelve-model ranking and ablation, so all ablation variants use identical splits;
- weighted-F1 and macro-F1 together with accuracy, balanced accuracy,
  Precision, Recall, and ROC-AUC;
- WNB in the principal comparison tables;
- complete Hybrid NBEM ablation;
- Friedman and paired Wilcoxon tests with Holm adjustment;
- CV-sensitivity analysis for grouped, adjusted-fold, and non-stratified cases;
- article-ready CSV, Excel, LaTeX tables, and Figures 1--3;
- computational cost as a table only;
- exact code/config/environment copies for reproducibility.

## Important corrections built into the pipeline

### Diabetes

The official task for the 130-US Hospitals dataset is early readmission. The
pipeline therefore requires the target `readmitted`, drops record/patient IDs
from the predictors, and uses patient-grouped cross-validation when
`patient_nbr` is available. Record-level stratified CV is produced only as a
sensitivity comparator.

Official source: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008

### Cardiotocography

The preferred target is `NSP` (or a processed equivalent such as
`fetal_health`). The alternative expert label `CLASS` is removed from the
predictors to prevent label leakage.

Official source: https://archive.ics.uci.edu/dataset/193/cardiotocography

### Garment productivity

`actual_productivity` is continuous. For the article's classification setting,
the pipeline applies the fixed, predeclared three-class thresholds in
`config/dataset_config.json`:

- Low: `< 0.50`
- Medium: `0.50–<0.75`
- High: `>=0.75`

The thresholds are fixed before model evaluation and do not depend on a test
fold.

Official source: https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees

## Windows setup

1. Copy the twenty dataset folders into `datasets/processed/`.
2. Double-click `setup_windows.bat` once.
3. Double-click `run_article_experiments.bat`.

The experiment is resume-capable. Closing the computer or interrupting the run
does not require starting completed dataset/seed combinations again.

## Command-line setup

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
python code\nbem_article_experiments_v3.py --project-root . --config config\dataset_config.json
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/nbem_article_experiments_v3.py --project-root . --config config/dataset_config.json
```

## Audit before a long run

A separate audit-only command is available:

```bash
python code/nbem_article_experiments_v3.py --project-root . --config config/dataset_config.json --audit-only
```

The full command already runs the same audit first. If a questionable target or
direct leakage is detected, it stops before the expensive experiment and writes
`results/final_article_run/CRITICAL_ACTION_REQUIRED.txt`.

Do not bypass a critical audit for article results. The flag
`--allow-critical-audit-issues` exists only for debugging.

## Quick pipeline test

```bash
python code/nbem_article_experiments_v3.py --project-root . --config config/dataset_config.json --quick --seeds 42 --folds 3
```

Quick-mode results are not suitable for the manuscript.

## Result location

Open:

```text
results/final_article_run/
```

The two most convenient files are:

- `NBEM_article_results.xlsx`
- `RUN_SUMMARY.txt`

See `docs/OUTPUT_FILES.md` and `docs/MANUSCRIPT_MAPPING.md` for the complete
mapping between outputs and the manuscript.

## Runtime

The final experiment is intentionally extensive. The four principal models are run on 20 datasets and 5 seeds; the complete twelve-model ablation/ranking is run on the common reference seed 42; sensitivity analyses use the same five seeds. Runtime depends
strongly on CPU, RAM, dataset size, and MLP convergence. It may take many hours.
The cache makes the run resumable and prevents mixing outputs from different
code/configuration versions.

## Theoretical disclosure

The current Hybrid NBEM implementation trains the stacking layer on component
probabilities from the same training sample; it is not a fully cross-fitted
stacker. Theorem 3.3 must therefore be described as an idealized theoretical
framework rather than a finite-sample guarantee for this implementation.

## GitHub and DOI

Read `docs/GITHUB_PUBLISHING_FA.md`. Do not put a fabricated repository URL or
DOI in the manuscript. Replace the availability statement only after the public
repository and archived release exist.
