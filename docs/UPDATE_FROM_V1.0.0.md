# Updating the public repository from v1.0.0 to v1.0.2

For a clean update, replace the repository working tree with the contents of this v1.0.2 package rather than merely dragging new files on top of v1.0.0. This prevents stale v1.0.0 paths from remaining visible as current evidence.

In particular, the old `results/final2_archive/Original_Submitted_v4/` path should not remain under the current-results hierarchy. The new release keeps historical workflow archives only under `results/stage_archives/` with an explicit provenance note.

After committing the replacement:

1. run `python scripts/verify_release.py` locally;
2. commit and push;
3. create GitHub tag/release `v1.0.2`;
4. let the enabled Zenodo integration archive the release;
5. record the new version-specific DOI;
6. update the manuscript / Response Letter if the new DOI is cited rather than a concept DOI.
