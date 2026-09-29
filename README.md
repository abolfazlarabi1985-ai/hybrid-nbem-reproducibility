# Hybrid NBEM Reproducibility Package

This repository contains the code, configurations, provenance records, archived experimental outputs, and targeted reproducibility utilities supporting the manuscript:

> **Adaptive Weighted and Hybrid Extensions of the Naive Bayes Enrichment Method for Mixed-Type Data: An Evaluation within the NB/NBEM Family**

**Authors:** Abolfazl Arabi and Samaneh Ghods  
**Repository:** https://github.com/abolfazlarabi1985-ai/hybrid-nbem-reproducibility  
**Release version:** `v1.0.0`

---

## Overview

The repository preserves the exact v4 base experiment source retained in the manuscript workflow and the archived outputs used during the revision analyses. It also contains the corrected Bank Marketing protocol used for the final reference-seed ablation update.

The key Bank Marketing correction is:

- the post-outcome variable `duration` is removed before model fitting;
- the reference seed for the ablation analysis is `42`;
- the requested outer cross-validation setting is 10-fold stratified CV;
- the corrected Table 5 aggregates and paired Wilcoxon/Holm results are stored under `results/corrected_bank_ablation_seed42/`.

The main empirical Hybrid NBEM stacker reported in the manuscript is **non-cross-fitted**. A separate cross-fitted subset analysis is preserved in the archived Stage-2 results and should not be interpreted as the protocol used for the main reported Hybrid NBEM results.

---

## Repository Structure

```text
.
├── CITATION.cff
├── .zenodo.json
├── VERSION
├── requirements.txt
├── FILE_MANIFEST.txt
├── config/
│   ├── dataset_config_v4_exact_used_ORIGINAL.json
│   └── dataset_config_FINAL_bank_duration_removed.json
├── data/
│   ├── DATASET_MANIFEST_PORTABLE.csv
│   ├── README_DATA.md
│   └── dataset_audit_exact_v4.csv
├── docs/
│   ├── GITHUB_RELEASE.md
│   ├── KNOWN_LIMITATIONS.md
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS_MAP.md
│   ├── UPLOAD_CHECKLIST_FA.md
│   └── ZENODO_UPLOAD.md
├── environment/
│   └── requirements.txt
├── notebooks/
│   └── NBEM_Final_Article_One_Click.ipynb
├── scripts/
│   ├── recompute_table5.py
│   ├── run_bank_ablation_seed42.py
│   ├── verify_dataset_hashes.py
│   └── verify_release.py
├── src/
│   ├── nbem_article_experiments_v4_EXACT_USED.py
│   └── verify_v4_output.py
├── results/
│   ├── corrected_bank_ablation_seed42/
│   └── final2_archive/
├── checksums/
│   ├── SHA256SUMS.txt
│   ├── recompute_table5_stdout.txt
│   └── verify_release_stdout.txt
└── provenance/
    └── hybrid-nbem-reproducibility_uploaded_snapshot.zip
```

---

## Environment

Python 3.10 was used for the major-revision stage runs recorded in the archived run manifests.

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned core dependencies in the supplied environment file include:

- `numpy==2.2.6`
- `pandas==2.3.3`
- `scipy==1.15.3`
- `matplotlib==3.10.9`
- `openpyxl==3.1.5`
- `scikit-learn==1.7.2`

---

## Data

Third-party benchmark dataset payloads are **not redistributed** in this repository.

Processed datasets should be placed under:

```text
datasets/processed/<dataset-key>/<dataset-key>.csv
```

The expected relative paths and SHA-256 hashes are listed in:

```text
data/DATASET_MANIFEST_PORTABLE.csv
```

Dataset provenance, placement instructions, and final preprocessing notes are documented in:

```text
data/README_DATA.md
```

To verify locally available processed datasets against the supplied hashes, use:

```bash
python scripts/verify_dataset_hashes.py
```

---

## Corrected Bank Marketing Protocol

For the final Bank Marketing analysis, use:

```text
config/dataset_config_FINAL_bank_duration_removed.json
```

This configuration adds `duration` to the Bank Marketing `drop_feature_candidates` list. The original exact-used v4 configuration is retained separately for provenance and should not be used for the corrected final Bank results.

### Reference-seed ablation settings

- Seed: `42`
- Requested CV: 10-fold stratified cross-validation
- MLP maximum iterations used by the targeted rerun utility: `300`

To rerun the four Bank Marketing ablation variants under the corrected protocol:

```bash
python scripts/run_bank_ablation_seed42.py \
  --project-root . \
  --data-root datasets/processed
```

On Windows PowerShell, the same command can be run on one line:

```powershell
python scripts/run_bank_ablation_seed42.py --project-root . --data-root datasets/processed
```

---

## Recomputing Table 5 and Statistical Tests

The corrected dataset-level ablation source file is retained under:

```text
results/corrected_bank_ablation_seed42/ablation_dataset_seed42_corrected.csv
```

To recompute the corrected aggregate Table 5 values and the paired two-sided Wilcoxon signed-rank tests with Holm adjustment:

```bash
python scripts/recompute_table5.py
```

The corrected aggregate means are:

| Model | Weighted F1 | Macro-F1 |
|---|---:|---:|
| Hybrid NBEM | 0.7747343405 | 0.6685768047 |
| Hybrid w/o Dependency | 0.7627154380 | 0.6560115421 |
| Hybrid w/o Adaptive | 0.7596419979 | 0.6338046719 |
| Hybrid w/o Deep | 0.7556702224 | 0.6375066188 |

The corresponding retained outputs are:

```text
results/corrected_bank_ablation_seed42/table5_ablation_corrected_seed42.csv
results/corrected_bank_ablation_seed42/ablation_wilcoxon_holm_corrected_seed42.csv
```

For these corrected Table 5 comparisons, the statistical unit is the **dataset**: 20 dataset-level reference-seed scores are compared using paired two-sided Wilcoxon signed-rank tests, with Holm adjustment over the three planned reduced-Hybrid contrasts within each metric.

---

## Experimental Seeds

### Five-seed in-family protocol

```text
13, 21, 42, 87, 123
```

### Reference-seed ablation

```text
42
```

### Locked 20-seed focal protocol

```text
13, 21, 42, 87, 123,
370882, 65932, 69887, 156304, 535886,
648362, 663649, 519965, 934243, 41762,
273103, 627018, 266458, 503966, 43529
```

---

## Main Archived Results

The complete `04_RESULTS` directory retained from `final-2` is preserved under:

```text
results/final2_archive/
```

Important archived components include:

- **Stage 1:** leakage/feature audit and Bank duration sensitivity;
- **Stage 1b:** duplicate-profile and duplicate-grouped sensitivity analyses;
- **Stage 2:** external baselines, cross-fitted subset analysis, and timing audit;
- **Stage 3:** 20-seed focal analysis for Hybrid NBEM, NBEM, and Logistic Regression;
- **Stage 4:** 20-seed Random Forest extension and four-model summaries;
- manuscript-ready tables retained from `final-2`;
- original submitted-v4 summaries, figures, validation files, workbook, and complete-run archives.

For a detailed mapping, see:

```text
docs/RESULTS_MAP.md
```

The corrected Bank seed-42 ablation under:

```text
results/corrected_bank_ablation_seed42/
```

**supersedes the old Bank contribution to Table 5 and its paired ablation p-values.**

---

## Verification

To verify the release structure and retained outputs:

```bash
python scripts/verify_release.py
```

Reference verification output and file checksums are preserved under:

```text
checksums/
```

The release-level SHA-256 manifest is:

```text
checksums/SHA256SUMS.txt
```

---

## Reproducibility and Provenance Notes

The file:

```text
src/nbem_article_experiments_v4_EXACT_USED.py
```

is the exact v4 base implementation preserved in the supplied manuscript package.

The archived `final-2` materials contain the output archives for Major Revision Stages 1-4. Their run manifests retain SHA-256 hashes for certain stage-specific orchestration scripts; however, the corresponding original Stage-2, Stage-3, and Stage-4 runner source files were not present in the supplied `final-2` archive.

For that reason, this repository does **not** present reconstructed scripts as those missing original sources. The scripts in `scripts/` are explicitly provided as targeted reproducibility and verification utilities based on the retained base implementation, configurations, and result files.

Additional known limitations are documented in:

```text
docs/KNOWN_LIMITATIONS.md
```

---

## Cross-Fitting Scope

The main empirical Hybrid NBEM implementation used for the manuscript's primary reported results is **non-cross-fitted**.

A separate cross-fitted subset analysis is retained in the archived Stage-2 results as a supplementary sensitivity check. It is not the protocol used for the primary Hybrid NBEM results and should not be interpreted as such.

---

## One-Click Notebook

A convenience notebook is included at:

```text
notebooks/NBEM_Final_Article_One_Click.ipynb
```

Users should still consult the configuration, data-provenance, and reproducibility documentation before interpreting regenerated outputs.

---

## Citation

Citation metadata is provided in:

```text
CITATION.cff
```

The GitHub repository can be cited using GitHub's **Cite this repository** function after the metadata is visible on the public repository.

A Zenodo DOI is **not hard-coded** in this release. After `v1.0.0` is archived by Zenodo and a DOI is assigned, the DOI should be added to the manuscript's Code Availability and Data Availability statements and, if desired, to the repository metadata.

### Associated manuscript

**Abolfazl Arabi and Samaneh Ghods.**  
*Adaptive Weighted and Hybrid Extensions of the Naive Bayes Enrichment Method for Mixed-Type Data: An Evaluation within the NB/NBEM Family.*

---

## Zenodo Archiving

Metadata for Zenodo is provided in:

```text
.zenodo.json
```

Recommended release sequence:

1. verify the repository contents;
2. create the GitHub tag/release `v1.0.0`;
3. archive that release through the GitHub-Zenodo integration;
4. obtain the assigned Zenodo DOI;
5. update the manuscript and Response to Reviewers with the real DOI.

Do not insert a placeholder or fabricated DOI into the manuscript.

---

## License

No software license is included in this release. The authors should select and add an appropriate license before relying on the repository for explicit software reuse permissions.

---

## Contact

For questions about the reproducibility package, please use the GitHub repository's **Issues** section or contact the corresponding author through the contact information provided in the associated manuscript.
