# Known reproducibility/provenance limitations

1. The supplied `final-2` archive retains the exact v4 base experiment script, but not the original source files for the Stage-2, Stage-3 and Stage-4 orchestration scripts whose hashes are recorded in the corresponding run manifests.
2. The corrected Bank reference-seed working artifacts retain fold-level output for Hybrid NBEM and Hybrid w/o Deep. The validated corrected dataset-level values for Hybrid w/o Dependency and Hybrid w/o Adaptive are retained, while their fold-level rerun files were not preserved. The targeted rerun utility regenerates them.
3. Third-party benchmark dataset payloads are not redistributed in this public-release package. Exact processed-file SHA-256 hashes and placement metadata are supplied instead.
4. No Zenodo DOI is hard-coded until Zenodo actually assigns one.
5. No software license is selected on behalf of the authors.
