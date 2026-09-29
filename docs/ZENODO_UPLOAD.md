# Zenodo release checklist

Recommended route: use Zenodo's GitHub integration so the archived object corresponds exactly to a GitHub release.

1. Link the GitHub account to Zenodo.
2. In Zenodo > GitHub, enable `abolfazlarabi1985-ai/hybrid-nbem-reproducibility`.
3. Commit `CITATION.cff` and `.zenodo.json` before the release.
4. Create the GitHub release/tag (recommended first release tag: `v1.0.0`).
5. Let Zenodo archive that release.
6. Open the Zenodo record and verify title, creators, version, resource type = Software, keywords, visibility and license metadata.
7. Publish/confirm the record and copy the **real assigned DOI**.
8. Only after that DOI exists, insert it into the manuscript Code Availability, Data Availability, Response to Reviewers and repository README.

Important: this package intentionally contains no invented Zenodo DOI.
