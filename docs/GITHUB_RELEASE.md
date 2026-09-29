# GitHub release checklist

1. Copy the contents of this package to the repository root.
2. Do not upload third-party benchmark CSV payloads unless their redistribution terms have been checked. Keep `data/` manifest/audit files public.
3. Run `python scripts/verify_release.py`.
4. Optionally rerun `python scripts/recompute_table5.py` and compare its output with the retained corrected files.
5. Choose and add a software license if the authors want to grant explicit reuse permissions.
6. Commit the release files.
7. Create a GitHub release/tag such as `v1.0.0`.
8. After Zenodo archives the release, add the real DOI badge/link to the README and manuscript availability statements.

Repository: https://github.com/abolfazlarabi1985-ai/hybrid-nbem-reproducibility
