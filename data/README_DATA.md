# Data availability and placement

This release does not redistribute the third-party benchmark datasets. The dataset manifest records the exact processed-file hashes used by the experiment package and the manuscript citation keys.

Place each processed dataset at the path specified by `expected_relative_path` in `DATASET_MANIFEST_PORTABLE.csv`, then run:

```bash
python scripts/verify_dataset_hashes.py
```

## Final leakage/preprocessing rules relevant to the revision

- **Bank Marketing:** target `y`; `duration` is removed before fitting because it is post-outcome information.
- **Diabetes 130-US Hospitals:** target `readmitted`; patient-grouped CV uses `patient_nbr`; `encounter_id` and patient identifiers are excluded from predictors.
- **Chronic Kidney Disease:** identifier-like `id` is excluded.
- **Cirrhosis:** identifier-like `ID` is excluded.
- **E. coli:** sequence identifier is excluded.
- **Glass Identification:** identifier-like `id` is excluded.
- **Cardiotocography:** `NSP` is the target; `CLASS` is excluded from predictors in the audited definition.
- **Productivity Prediction:** `actual_productivity` is mapped to the fixed task bins `(-inf,0.5)`, `[0.5,0.75)`, `[0.75,inf)` labelled Low/Medium/High. Results are conditional on this task definition.

For exact row counts, feature counts, targets, hashes and dropped-column metadata, use `DATASET_MANIFEST_PORTABLE.csv` and `dataset_audit_exact_v4.csv`.
