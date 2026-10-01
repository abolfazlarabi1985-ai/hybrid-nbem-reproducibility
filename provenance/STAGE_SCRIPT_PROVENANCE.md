# Stage-script provenance

The original READY_TO_RUN packages for Major Revision Stages 1, 1B, 2, 3 and 4 were supplied by the authors and are preserved byte-for-byte under `provenance/original_ready_to_run_packages/`.

The extracted source trees are published under `pipeline/`.

For Stages 2, 3 and 4, the restored orchestration scripts were cryptographically checked against the SHA-256 values stored in the retained run manifests that accompany the archived outputs:

| Stage | Source file | Restored SHA-256 | Run-manifest SHA-256 | Status |
|---|---|---|---|---|
| 2 | `pipeline/stage2/stage2_major_revision.py` | `0218fbec5ed5074af38ecf75ad975b92da3e8ad16b1c1cf962c5ce3a6173a30f` | same | exact match |
| 3 | `pipeline/stage3/stage3_major_revision.py` | `ca31ece3fb189e3a6c74e5f4975b240e04ee2f8ac621425791a53fd9af0c228d` | same | exact match |
| 4 | `pipeline/stage4_rf20/stage4_rf20_major_revision.py` | `21c4a8f8545f519b85126a3d16c0a51b628497da4cdcfcb27d96408ecd92c1be` | same | exact match |

The retained base v4 implementation hash referenced by these manifests is:

`1560fe819ca698867442525b65c8e34b45cd2e6c4809344a5e5e050e5fd2d886`

The frozen revision configuration hash used by Stages 2-4 is:

`1ad64004297ef2e53461ed937f3929cf6857da0ce3d0cf161fbfb2a2c65f48a2`

The locked Stage-3 / Stage-4 seed-plan hash is:

`315f8ba90a1759839429fbac4a151f77f5b1b00bb380470573cc92443389fe79`

Run `python scripts/verify_stage_source_hashes.py` to reproduce these checks directly from the release.
