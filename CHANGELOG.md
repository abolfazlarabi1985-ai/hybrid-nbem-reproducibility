# Changelog

## v1.0.2

- Restored the original READY_TO_RUN source packages for Major Revision Stages 1, 1B, 2, 3 and 4.
- Verified the exact Stage-2, Stage-3 and Stage-4 orchestration-script SHA-256 values against the retained run manifests used to generate the archived results.
- Added `pipeline/` with original stage scripts, configs, run wrappers, seed plans, READMEs and provenance inputs.
- Added byte-preserved original stage ZIPs under `provenance/original_ready_to_run_packages/`.
- Added independent `verify_stage_source_hashes.py` provenance verification.
- Updated repository-completeness checks so original Stage-2/3/4 source is now required.
- Updated documentation to remove the obsolete v1.0.1 statement that Stage-2/3/4 orchestration sources were unavailable.
- Preserved final Supplementary Tables S1-S13, corrected Bank seed-42 ablation, 20-seed focal statistics, cross-fitted subset outputs, duplicate sensitivity, environment metadata and verification utilities from v1.0.1.

## v1.0.1

- Added final Supplementary Information and machine-readable Tables S1-S13.
- Added explicit five-seed, twenty-seed and reference-seed configuration files.
- Added current cross-fitted subset and duplicate-sensitivity outputs.
- Added focal twenty-seed statistical verification and repository-completeness checks.
- Clarified that historical Friedman/twelve-model material is not part of current inferential evidence.
