# Known limitations and scope notes

1. Third-party benchmark dataset payloads are not redistributed. The repository provides expected processed-file paths, SHA-256 hashes, dataset audit metadata and placement instructions.
2. The main empirical Hybrid NBEM results in the manuscript are non-cross-fitted. The cross-fitted Stage-2 subset analysis is a separate supplementary robustness check and does not retroactively make the main experiment cross-fitted.
3. Corrected Bank seed-42 Table-5 results use the final protocol with `duration` removed. The repository preserves the original configuration only for provenance.
4. Historical Friedman / twelve-model seed-42 results may remain inside immutable historical stage/result archives, but they are not part of the current primary inferential evidence and are not reproduced in the final Supplementary S1-S13.
5. The v1.0.2 version-specific Zenodo DOI is not known until Zenodo archives the release. The prior v1.0.0 DOI is documented only as provenance.
6. No software license is selected on the authors' behalf in this package.

## Source provenance status

The v1.0.1 limitation concerning missing Stage-2/3/4 orchestration sources is resolved. The original source files were restored from the authors' READY_TO_RUN packages and their SHA-256 values exactly match the corresponding hashes stored in the retained run manifests.
