# Original Major Revision execution pipeline

This directory contains the original READY_TO_RUN execution trees supplied by the authors.

- `stage1/`: exact-data audit and Bank-duration sensitivity.
- `stage1b/`: duplicate-exposure audit and grouped sensitivity.
- `stage2/`: Logistic Regression / Random Forest external baselines, cross-fitted Hybrid subset and timing audit.
- `stage3/`: locked 20-seed focal Hybrid/NBEM-prop/Logistic Regression study and statistical analysis.
- `stage4_rf20/`: locked 20-seed Random Forest extension.

Each stage directory retains its original README, configuration, run wrappers and provenance inputs. Result outputs are kept separately in `results/stage_archives/` to avoid unnecessary duplication.

The Stage-2/3/4 Python-source hashes exactly match the hashes recorded in the retained run manifests. See `provenance/STAGE_SOURCE_HASH_VERIFICATION.csv`.
