# Reproducibility protocol

## Retained implementation

`src/nbem_article_experiments_v4_EXACT_USED.py` is the exact retained v4 base implementation from the article archive.

The original Major Revision orchestration sources are restored under `pipeline/`:

- `pipeline/stage1/stage1_major_revision.py`
- `pipeline/stage1b/stage1b_duplicate_leakage_sensitivity.py`
- `pipeline/stage2/stage2_major_revision.py`
- `pipeline/stage3/stage3_major_revision.py`
- `pipeline/stage4_rf20/stage4_rf20_major_revision.py`

The Stage-2/3/4 source hashes exactly match the corresponding retained run-manifest hashes. Verify with:

```bash
python scripts/verify_stage_source_hashes.py
```

## Final Bank protocol

Use `config/dataset_config_FINAL_bank_duration_removed.json`. The Bank Marketing feature `duration` is excluded before model fitting.

The original Stage-2/3/4 frozen revision configs under `pipeline/` carry the same corrected Bank rule.

## Seeds

- Original five: `13, 21, 42, 87, 123`
- Reference-seed ablation: `42`
- Locked twenty seeds: see `config/seed_plan_20.json`
- Original locked Stage-3 plan: `pipeline/stage3/config/stage3_seed_plan.json`
- Stage 4 uses the same locked twenty-seed plan.

## Main stacking protocol

The main empirical Hybrid NBEM results are non-cross-fitted. The cross-fitted subset experiment is a separate robustness analysis. Its original implementation is in `pipeline/stage2/stage2_major_revision.py`; retained outputs are under `results/current/crossfit/` and the complete Stage-2 archive.

## Primary statistical unit

The manuscript's focal paired tests use one twenty-seed mean per dataset as the inferential unit (20 datasets), not folds or seeds as independent observations.

## Original stage analyses

### Stage 1

Exact snapshot / leakage audit plus Bank-duration sensitivity.

```bash
python pipeline/stage1/stage1_major_revision.py --mode audit
```

### Stage 1B

Duplicate-exposure audit and duplicate-grouped sensitivity.

```bash
python pipeline/stage1b/stage1b_duplicate_leakage_sensitivity.py --mode audit
```

### Stage 2

External baselines, predeclared cross-fit subset and timing audit.

```bash
python pipeline/stage2/stage2_major_revision.py --mode baselines
python pipeline/stage2/stage2_major_revision.py --mode crossfit_subset
python pipeline/stage2/stage2_major_revision.py --mode timing
```

### Stage 3

Locked twenty-seed focal experiment and original statistical analysis.

```bash
python pipeline/stage3/stage3_major_revision.py --mode full
```

The original source contains the bootstrap-CI, Wilcoxon, Holm and rank-biserial implementations used in Stage 3.

### Stage 4

Locked twenty-seed Random Forest extension.

```bash
python pipeline/stage4_rf20/stage4_rf20_major_revision.py --mode full
```

The original source contains the Hybrid-vs-RF Wilcoxon, bootstrap-CI and rank-biserial analysis.

## Current manuscript statistical outputs

- Corrected Table 5 ablation: `results/current/bank_seed42/`
- Five-seed in-family dataset table: `results/current/five_seed/`
- Twenty-seed focal summaries / bootstrap CIs / Wilcoxon / Holm: `results/current/focal_20seed/`
- Cross-fit sensitivity: `results/current/crossfit/`
- Duplicate-grouped sensitivity: `results/current/duplicate_sensitivity/`

Independent verification utilities under `scripts/` operate on these retained final outputs. They complement, rather than replace, the restored original stage source.

## Supplementary

The final formatted Supplementary Information is under `supplementary/`, together with machine-readable Tables S1-S13.

The current Supplementary does not present the earlier Friedman/twelve-model seed-42 ranking as primary inferential evidence.

## Complete verification

Run:

```bash
python scripts/verify_release.py
```

This checks repository completeness, exact Stage-2/3/4 source identity, corrected Table-5 statistics, focal paired statistics, cross-fit outputs, duplicate sensitivity, Supplementary S1-S13 completeness and absence of Friedman/twelve-model material from current evidence.
