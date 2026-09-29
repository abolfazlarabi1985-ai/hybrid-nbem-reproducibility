# Hybrid NBEM reproducibility package

This repository supports the manuscript:

**Adaptive Weighted and Hybrid Extensions of the Naive Bayes Enrichment Method for Mixed-Type Data: An Evaluation within the NB/NBEM Family**

Authors: **Abolfazl Arabi** and **Samaneh Ghods**.

Public repository: https://github.com/abolfazlarabi1985-ai/hybrid-nbem-reproducibility

## Reproducibility status

The package preserves the exact v4 experiment source and archived outputs from the manuscript workflow. It also contains the corrected Bank Marketing protocol used for the final ablation update.

The important protocol correction is:

- Bank Marketing: the post-outcome variable `duration` is removed in memory before model fitting.
- Reference seed for the ablation analysis: `42`.
- Requested outer folds: `10`.
- The corrected Table 5 and Wilcoxon/Holm files are in `results/corrected_bank_ablation_seed42/`.

The main empirical Hybrid NBEM stacker is **non-cross-fitted**. A separate cross-fitted subset analysis is preserved in the Stage-2 results archive.

## Repository layout

- `src/` - exact v4 experiment source retained from the article package.
- `config/` - original exact-used configuration plus the corrected Bank configuration.
- `scripts/` - verification and targeted reproducibility scripts.
- `results/final2_archive/` - the complete results folder preserved from `final-2`.
- `results/corrected_bank_ablation_seed42/` - corrected Bank seed-42 ablation inputs/outputs used to update Table 5 and p-values.
- `data/` - dataset manifest, hashes, audit metadata, and data-placement instructions. Third-party dataset payloads are not redistributed here.
- `environment/` - pinned Python dependencies.
- `provenance/` - archived earlier repository snapshot retained for provenance only.
- `docs/` - reproducibility, GitHub, Zenodo, results-map, and upload notes.

## Installation

Python 3.10 was used for the major-revision stage runs recorded in the archived run manifests.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Pinned dependencies:

- numpy 2.2.6
- pandas 2.3.3
- scipy 1.15.3
- matplotlib 3.10.9
- openpyxl 3.1.5
- scikit-learn 1.7.2

## Data placement

The code expects processed datasets under:

```text
datasets/processed/<dataset-key>/<dataset-key>.csv
```

Use `data/DATASET_MANIFEST_PORTABLE.csv` for the expected relative paths and SHA-256 hashes. See `data/README_DATA.md` for the dataset citations and the final preprocessing rules.

## Corrected Bank seed-42 ablation

To regenerate the four Bank Marketing ablation variants under the corrected protocol:

```bash
python scripts/run_bank_ablation_seed42.py --project-root . --data-root datasets/processed
```

This runner uses `config/dataset_config_FINAL_bank_duration_removed.json`, seed 42, 10-fold stratified CV, and `mlp_max_iter=300`.

To recompute the manuscript Table 5 aggregate and the two-sided Wilcoxon tests with Holm correction from the validated dataset-level source file:

```bash
python scripts/recompute_table5.py
```

Expected corrected aggregate means:

| Model | Weighted F1 | Macro-F1 |
|---|---:|---:|
| Hybrid NBEM | 0.7747343405 | 0.6685768047 |
| Hybrid w/o Dependency | 0.7627154380 | 0.6560115421 |
| Hybrid w/o Adaptive | 0.7596419979 | 0.6338046719 |
| Hybrid w/o Deep | 0.7556702224 | 0.6375066188 |

## Important provenance note

The `final-2` archive contains the exact v4 base script and the output archives for Major Revision Stages 1-4. The stage manifests record SHA-256 hashes for stage-specific runner scripts, but the corresponding Stage-2/Stage-3/Stage-4 runner source files were not present in `final-2`. This package therefore does **not** mislabel reconstructed code as those missing original scripts. New targeted scripts in `scripts/` are explicitly reproducibility utilities created to rerun or verify the documented analyses from the retained base source and result files.

## Citation and Zenodo

`CITATION.cff` and `.zenodo.json` are included. The Zenodo DOI is intentionally not hard-coded before a real Zenodo record exists. After the GitHub release is archived by Zenodo, update the manuscript Code/Data Availability statements with the assigned DOI.

## License

No software license has been selected on the authors' behalf in this package. Before public release, the authors should choose and add an appropriate license if they wish others to have explicit reuse permissions.
