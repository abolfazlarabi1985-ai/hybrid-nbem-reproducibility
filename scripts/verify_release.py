#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors = []

cfg = json.loads((ROOT / "config" / "dataset_config_FINAL_bank_duration_removed.json").read_text(encoding="utf-8"))
bank = cfg["dataset_overrides"]["bank_marketing"]
drops = [x.lower() for x in bank.get("drop_feature_candidates", [])]
if "duration" not in drops:
    errors.append("Final Bank config does not drop duration.")
if cfg.get("reference_seed") != 42:
    errors.append("reference_seed is not 42.")

src = pd.read_csv(ROOT / "results" / "corrected_bank_ablation_seed42" / "ablation_dataset_seed42_corrected.csv")
if src["dataset"].nunique() != 20:
    errors.append("Corrected ablation dataset source does not contain 20 datasets.")
expected = {
    "Hybrid NBEM": (0.7747343405473106, 0.6685768047262120),
    "Hybrid w/o Dependency": (0.7627154379923863, 0.6560115421235018),
    "Hybrid w/o Adaptive": (0.7596419978708581, 0.6338046718896291),
    "Hybrid w/o Deep": (0.7556702223669627, 0.6375066188123950),
}
means = src.groupby("model")[["f1_weighted","f1_macro"]].mean()
for model, (w,m) in expected.items():
    if abs(means.loc[model,"f1_weighted"] - w) > 1e-12 or abs(means.loc[model,"f1_macro"] - m) > 1e-12:
        errors.append(f"Corrected aggregate mismatch for {model}.")

if errors:
    raise SystemExit("RELEASE VERIFICATION FAILED:\n- " + "\n- ".join(errors))
print("Release verification passed: corrected Bank config, seed 42, dataset count, and Table 5 aggregate values are consistent.")
