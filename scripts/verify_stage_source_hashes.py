#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, zipfile, tempfile, csv

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    2: "0218fbec5ed5074af38ecf75ad975b92da3e8ad16b1c1cf962c5ce3a6173a30f",
    3: "ca31ece3fb189e3a6c74e5f4975b240e04ee2f8ac621425791a53fd9af0c228d",
    4: "21c4a8f8545f519b85126a3d16c0a51b628497da4cdcfcb27d96408ecd92c1be",
}
FILES = {
    2: ROOT / "pipeline/stage2/stage2_major_revision.py",
    3: ROOT / "pipeline/stage3/stage3_major_revision.py",
    4: ROOT / "pipeline/stage4_rf20/stage4_rf20_major_revision.py",
}
ARCHIVES = {
    2: ROOT / "results/stage_archives/major_revision_stage2.zip",
    3: ROOT / "results/stage_archives/major_revision_stage3.zip",
    4: ROOT / "results/stage_archives/major_revision_stage4_rf20.zip",
}
MANIFEST_NAMES = {
    2: "major_revision_stage2/RUN_MANIFEST_STAGE2.json",
    3: "major_revision_stage3/RUN_MANIFEST_STAGE3.json",
    4: "major_revision_stage4_rf20/RUN_MANIFEST_STAGE4_RF20.json",
}
MANIFEST_KEYS = {2:"stage2_script_sha256",3:"stage3_script_sha256",4:"stage4_script_sha256"}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows=[]
for stage in (2,3,4):
    p=FILES[stage]
    if not p.exists():
        raise RuntimeError(f"Missing restored Stage-{stage} source: {p.relative_to(ROOT)}")
    observed=sha256(p)
    if observed != EXPECTED[stage]:
        raise RuntimeError(f"Stage-{stage} source hash mismatch: {observed} != {EXPECTED[stage]}")
    with zipfile.ZipFile(ARCHIVES[stage]) as z:
        manifest=json.loads(z.read(MANIFEST_NAMES[stage]).decode("utf-8-sig"))
    recorded=manifest[MANIFEST_KEYS[stage]]
    if recorded != observed:
        raise RuntimeError(f"Stage-{stage} run-manifest hash mismatch: {recorded} != {observed}")
    rows.append((stage,str(p.relative_to(ROOT)),observed,recorded,"exact_match"))
    print(f"PASS: Stage-{stage} restored source exactly matches retained run-manifest SHA-256: {observed}")

base = ROOT / "pipeline/stage2/base/nbem_article_experiments_v4_EXACT_USED.py"
base_expected = "1560fe819ca698867442525b65c8e34b45cd2e6c4809344a5e5e050e5fd2d886"
if sha256(base) != base_expected:
    raise RuntimeError("Retained v4 base implementation hash mismatch")
print(f"PASS: retained v4 base implementation SHA-256: {base_expected}")

cfg = ROOT / "pipeline/stage2/config/dataset_config_revision_frozen.json"
cfg_expected = "1ad64004297ef2e53461ed937f3929cf6857da0ce3d0cf161fbfb2a2c65f48a2"
if sha256(cfg) != cfg_expected:
    raise RuntimeError("Frozen revision config hash mismatch")
print(f"PASS: frozen revision configuration SHA-256: {cfg_expected}")

seed = ROOT / "pipeline/stage3/config/stage3_seed_plan.json"
seed_expected = "315f8ba90a1759839429fbac4a151f77f5b1b00bb380470573cc92443389fe79"
if sha256(seed) != seed_expected:
    raise RuntimeError("Locked Stage-3 seed-plan hash mismatch")
print(f"PASS: locked Stage-3/4 seed-plan SHA-256: {seed_expected}")
