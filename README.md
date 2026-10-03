# Hybrid NBEM reproducibility package — v1.0.2

This repository supports the manuscript:

**Adaptive Weighted and Hybrid Extensions of the Naive Bayes Enrichment Method for Mixed-Type Data: An Evaluation within the NB/NBEM Family**

Authors: **Abolfazl Arabi** and **Samaneh Ghods**
Repository: https://github.com/abolfazlarabi1985-ai/hybrid-nbem-reproducibility

## What v1.0.2 adds

Version 1.0.2 supersedes the prepared v1.0.1 package by restoring the original Major Revision Stage 1, Stage 1B, Stage 2, Stage 3 and Stage 4 READY_TO_RUN source packages.

Most importantly, the exact Stage-2, Stage-3 and Stage-4 orchestration scripts are now present and their SHA-256 values were verified against the hashes recorded in the retained run manifests that produced the archived results:
- Stage 2 `stage2_major_revision.py`: `0218fbec5ed5074af38ecf75ad975b92da3e8ad16b1c1cf962c5ce3a6173a30f`
- Stage 3 `stage3_major_revision.py`: `ca31ece3fb189e3a6c74e5f4975b240e04ee2f8ac621425791a53fd9af0c228d`
- Stage 4 `stage4_rf20_major_revision.py`: `21c4a8f8545f519b85126a3d16c0a51b628497da4cdcfcb27d96408ecd92c1be`

This closes the source-provenance gap documented in v1.0.1. The package now exposes the original orchestration code for external baselines, cross-fitted subset analysis, timing audit, the locked 20-seed focal experiment, bootstrap/Wilcoxon/Holm statistics, and the 20-seed Random Forest extension.

The main empirical Hybrid NBEM stacker reported in the manuscript is **non-cross-fitted**. The cross-fitted results are a separate supplementary robustness analysis on the predeclared Stage-2 subset.

## Release / DOI status

The current archival release corresponding to this repository and the revised manuscript is **v1.0.2**:

- **v1.0.2 Zenodo DOI:** https://doi.org/10.5281/zenodo.23091693
- **DOI:** `10.5281/zenodo.23091693`

Version **v1.0.2** is the complete archived reproducibility release for the current manuscript. It includes the restored source provenance, preprocessing and dataset configurations, the five original seeds and locked twenty-seed plan, the corrected seed-42 Bank ablation, statistical scripts and outputs, the cross-fitted subset experiment, duplicate-sensitivity analysis, environment metadata, and Supplementary Tables S1–S13.

The earlier **v1.0.0** Zenodo release (`10.5281/zenodo.23045513`) is retained only as a historical version and should not be used as the archival identifier for the current **v1.0.2** reproducibility package.

## Repository map
- `src/` — exact retained v4 base implementation used in the article workflow.
- `pipeline/` — original Major Revision READY_TO_RUN source trees:
  - `stage1/` — exact-data/leakage audit and Bank `duration` sensitivity.
  - `stage1b/` — duplicate-exposure audit and grouped duplicate sensitivity.
  - `stage2/` — external baselines, cross-fitted subset experiment, and timing audit.
  - `stage3/` — locked 20-seed Hybrid/NBEM-prop/Logistic Regression experiment and statistical analysis.
  - `stage4_rf20/` — locked 20-seed Random Forest extension and Hybrid-vs-RF analysis.
- `config/` — final Bank-corrected configuration, original exact-used configuration, five-seed and twenty-seed plans.
- `data/` — dataset manifest, hashes, semantic audit, and data-placement instructions. Third-party dataset payloads are not redistributed.
- `results/current/` — current evidence used by the revised manuscript.
- `results/stage_archives/` — complete retained Stage 1-4 result archives, including caches and run manifests where available.
- `supplementary/` — final Supplementary Information and machine-readable Tables S1-S13.
- `scripts/` — targeted rerun, recomputation and verification utilities.
- `environment/` — requirements and retained execution-environment metadata.
- `notebooks/` — one-click verification notebook.
- `provenance/original_ready_to_run_packages/` — byte-preserved READY_TO_RUN ZIPs supplied by the authors.
- `provenance/STAGE_SOURCE_HASH_VERIFICATION.csv` — direct hash comparison between restored source and retained run manifests.
- `docs/` — reproducibility details, repository checklist and limitations.

## Installation

The retained run manifests record Python 3.10.11. Create a clean environment and install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Each restored `pipeline/stage*` directory also retains its own `requirements.txt`, README and original run wrappers.

## Dataset placement and preprocessing

Processed datasets are expected under:

```text
datasets/processed/<dataset-key>/<dataset-key>.csv
```

The exact expected relative paths and SHA-256 hashes are listed in `data/DATASET_MANIFEST_PORTABLE.csv` and in each original stage package.

The exact retained base preprocessing / experiment implementation is:

```text
src/nbem_article_experiments_v4_EXACT_USED.py
```

The final corrected configuration is:
```text
config/dataset_config_FINAL_bank_duration_removed.json
```

For Bank Marketing, `duration` is removed before fitting. The archived original configuration is retained only for provenance and must not be used to regenerate the corrected final Bank results.

For the dataset displayed as **Fetal Health** in the manuscript, the internal dataset key is `cardiotocography`; `NSP` is the target and the alternative label column `CLASS` is excluded.

## Seeds

Original five seeds:

```text
13, 21, 42, 87, 123
```

Locked twenty-seed plan:

```text
13, 21, 42, 87, 123, 370882, 65932, 69887, 156304, 535886,
648362, 663649, 519965, 934243, 41762, 273103, 627018, 266458,
503966, 43529
```

Reference seed for the final ablation: `42`.

The original locked Stage-3 seed plan is also retained at `pipeline/stage3/config/stage3_seed_plan.json`; Stage 4 uses the same locked plan.

## Original stage runners

The exact original runners can be inspected or executed from `pipeline/` after placing the datasets at the expected paths.

Examples:

```bash
# Stage 1: audit / Bank duration sensitivity
python pipeline/stage1/stage1_major_revision.py --mode audit

# Stage 1B: duplicate sensitivity
python pipeline/stage1b/stage1b_duplicate_leakage_sensitivity.py --mode audit

# Stage 2: external baselines
python pipeline/stage2/stage2_major_revision.py --mode baselines
# Stage 2: predeclared cross-fitted subset
python pipeline/stage2/stage2_major_revision.py --mode crossfit_subset

# Stage 2: timing audit
python pipeline/stage2/stage2_major_revision.py --mode timing

# Stage 3: locked 20-seed focal experiment
python pipeline/stage3/stage3_major_revision.py --mode full

# Stage 3: analyze cached 20-seed results
python pipeline/stage3/stage3_major_revision.py --mode analyze
# Stage 4: locked 20-seed Random Forest extension
python pipeline/stage4_rf20/stage4_rf20_major_revision.py --mode full
```

The exact original `.bat` / `.sh` wrappers are retained inside each stage tree.

## Statistical analyses now backed by original source

The restored Stage-3 source includes the original functions for:

- dataset-level paired differences;
- two-sided Wilcoxon signed-rank tests;
- Holm correction within each primary metric;
- rank-biserial effect size;
- deterministic bootstrap confidence intervals;
- seed-level stability and exploratory capability profiles.

The restored Stage-4 source includes the original Hybrid-vs-Random-Forest paired statistics, bootstrap CI and rank-biserial calculations.

The restored Stage-2 source includes the original external-baseline evaluation, predeclared cross-fitted Hybrid subset experiment and timing audit.

Verification utilities under `scripts/` remain useful as independent checks on retained outputs; the restored original Stage-2/3/4 source is now published separately under `pipeline/`.

## Corrected Bank seed-42 ablation

The current corrected Table-5 inputs and outputs are under:

```text
results/current/bank_seed42/
```

The targeted verification runner is:

```bash
python scripts/run_bank_ablation_seed42.py --project-root . --data-root datasets/processed
```

It uses the final Bank configuration, removes `duration`, fixes seed 42, requests 10-fold stratified CV, and evaluates the four Table-5 Hybrid variants.

## Cross-fitted and duplicate-sensitivity analyses

The main manuscript results use a **non-cross-fitted** stacker.

The original Stage-2 cross-fitted implementation is now public at `pipeline/stage2/stage2_major_revision.py`; retained outputs are under `results/current/crossfit/` and the full Stage-2 archive.

The original Stage-1B duplicate-group sensitivity implementation is public at `pipeline/stage1b/stage1b_duplicate_leakage_sensitivity.py`; retained outputs are under `results/current/duplicate_sensitivity/` and the Stage-1B archive.

## Supplementary Tables S1-S13

The final formatted supplement is provided as DOCX and PDF under `supplementary/`. Machine-readable companions `S01.csv` through `S13.csv` are indexed by `supplementary/tables_S1_S13/TABLE_INDEX.json`.

No Friedman / twelve-model seed-42 ranking is presented under `results/current/` or in the final Supplementary Information as primary inferential evidence. Historical result archives are retained only for provenance.

## Verification

Run the complete release verification suite:

```bash
python scripts/verify_release.py
```

The suite includes:

```bash
python scripts/check_required_materials.py
python scripts/verify_stage_source_hashes.py
python scripts/recompute_table5.py
python scripts/recompute_focal_statistics.py
python scripts/verify_crossfit_outputs.py
python scripts/verify_duplicate_sensitivity.py
python scripts/verify_supplementary_tables.py
python scripts/check_no_primary_friedman.py
```

`verify_stage_source_hashes.py` checks that restored Stage-2/3/4 source bytes exactly match the SHA-256 identifiers recorded in the retained run manifests.

## Provenance statement

The source-provenance limitation documented in v1.0.1 is closed in v1.0.2: the original READY_TO_RUN Stage-2, Stage-3 and Stage-4 orchestration source files have been restored, and their hashes exactly match the corresponding retained run manifests. The byte-preserved original stage ZIPs are retained in `provenance/original_ready_to_run_packages/`.

See `provenance/STAGE_SCRIPT_PROVENANCE.md`, `provenance/STAGE_SOURCE_HASH_VERIFICATION.csv` and `docs/KNOWN_LIMITATIONS.md`.
