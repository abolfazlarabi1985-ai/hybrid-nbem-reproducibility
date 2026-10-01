# Repository completeness checklist

This document maps the requested reproducibility items to concrete release paths.

| Requested item | Release location | Status |
|---|---|---|
| preprocessing | `src/nbem_article_experiments_v4_EXACT_USED.py` | present |
| dataset configuration | `config/dataset_config_FINAL_bank_duration_removed.json`; frozen configs in `pipeline/stage1b/`, `pipeline/stage2/`, `pipeline/stage3/`, `pipeline/stage4_rf20/` | present |
| all 20 seeds | `config/seed_plan_20.json`, `config/ALL_TWENTY_SEEDS.txt`, original Stage-3/4 seed plans under `pipeline/` | present |
| 5 original seeds | `config/ORIGINAL_FIVE_SEEDS.txt`; original Stage-3 plan | present |
| seed 42 ablation | `results/current/bank_seed42/`; targeted rerun script in `scripts/run_bank_ablation_seed42.py` | present |
| original statistical scripts | Stage 3 and Stage 4 source under `pipeline/`; Stage 2 source for baseline/cross-fit/timing | present and hash-verified |
| bootstrap CI | original Stage-3/4 functions under `pipeline/`; final outputs under `results/current/focal_20seed/` | present |
| Wilcoxon/Holm | original Stage-3 source plus final consolidation verifier; ablation verifier in `scripts/recompute_table5.py` | present |
| cross-fitted experiment | original source `pipeline/stage2/stage2_major_revision.py`; outputs `results/current/crossfit/` | present |
| duplicate sensitivity | original source `pipeline/stage1b/stage1b_duplicate_leakage_sensitivity.py`; outputs `results/current/duplicate_sensitivity/` | present |
| environment | `environment/requirements.txt`, `environment/execution_environment_original_v4.json`, stage-level requirements | present |
| Supplementary S1-S13 | `supplementary/` and `supplementary/tables_S1_S13/` | present |
| README for reproduction | `README.md`, `docs/REPRODUCIBILITY.md`, original stage READMEs | present |
| source provenance | `provenance/STAGE_SCRIPT_PROVENANCE.md`, `provenance/STAGE_SOURCE_HASH_VERIFICATION.csv`, `scripts/verify_stage_source_hashes.py` | present |

Run:

```bash
python scripts/verify_release.py
```

to verify these categories automatically.
