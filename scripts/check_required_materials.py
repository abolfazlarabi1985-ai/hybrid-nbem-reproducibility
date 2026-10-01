#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
required={
'preprocessing implementation':'src/nbem_article_experiments_v4_EXACT_USED.py',
'final dataset configuration':'config/dataset_config_FINAL_bank_duration_removed.json',
'20-seed plan':'config/seed_plan_20.json',
'five original seeds':'config/ORIGINAL_FIVE_SEEDS.txt',
'seed-42 ablation':'results/current/bank_seed42/ablation_dataset_seed42_corrected.csv',
'bootstrap CI + Wilcoxon/Holm output':'results/current/focal_20seed/table_focal_20seed_pairwise.csv',
'focal statistical verification script':'scripts/recompute_focal_statistics.py',
'cross-fitted experiment':'results/current/crossfit/table_crossfit_subset_delta.csv',
'duplicate sensitivity':'results/current/duplicate_sensitivity/table_duplicate_sensitivity.csv',
'environment requirements':'environment/requirements.txt',
'execution environment snapshot':'environment/execution_environment_original_v4.json',
'Supplementary S1-S13 PDF':'supplementary/Supplementary_Information_S1-S13_FINAL.pdf',
'Supplementary S1-S13 machine-readable index':'supplementary/tables_S1_S13/TABLE_INDEX.json',
'reproduction README':'README.md',
'original Stage-2 source':'pipeline/stage2/stage2_major_revision.py',
'original Stage-3 source':'pipeline/stage3/stage3_major_revision.py',
'original Stage-4 source':'pipeline/stage4_rf20/stage4_rf20_major_revision.py',
'stage source hash verification table':'provenance/STAGE_SOURCE_HASH_VERIFICATION.csv',
}
missing=[f'{k}: {v}' for k,v in required.items() if not (ROOT/v).exists()]
if missing: raise RuntimeError('Missing required materials:\n'+'\n'.join(missing))
plan=json.loads((ROOT/'config/seed_plan_20.json').read_text())
if len(plan['all_twenty_seeds'])!=20 or plan['original_five_seeds']!=[13,21,42,87,123] or plan['reference_seed_ablation']!=42:
    raise RuntimeError('Seed plan validation failed')
print('PASS: all professor-requested repository material categories are present.')
for k,v in required.items(): print(f'  - {k}: {v}')
