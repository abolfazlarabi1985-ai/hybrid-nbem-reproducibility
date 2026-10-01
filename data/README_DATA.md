# Dataset placement and audit

Third-party dataset payloads are not redistributed in this public release.

Place processed files at:

`datasets/processed/<dataset-key>/<dataset-key>.csv`

Expected relative paths and SHA-256 hashes are in `DATASET_MANIFEST_PORTABLE.csv`.

The final configuration is `../config/dataset_config_FINAL_bank_duration_removed.json`.

Important final protocol details:

- Bank Marketing: remove `duration` before model fitting.
- Diabetes 130-US-hospitals: use the readmission target with patient-grouped outer CV and identifier exclusions specified in the configuration.
- Fetal Health: internal dataset key `cardiotocography`; use `NSP` as target and exclude alternative label column `CLASS`.

Use `scripts/verify_dataset_hashes.py` after placing the processed datasets locally.
