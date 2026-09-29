# Reproducibility protocol

## Primary retained implementation

`src/nbem_article_experiments_v4_EXACT_USED.py` is the exact base implementation preserved in the final-2 package.

## Final Bank correction

Use `config/dataset_config_FINAL_bank_duration_removed.json`. It adds `duration` to the Bank Marketing `drop_feature_candidates` list. Do not use the original exact-used config for the final Bank results.

## Seeds

### Five-seed in-family protocol
`13, 21, 42, 87, 123`

### Reference-seed ablation
`42`

### Locked 20-seed focal protocol
`13, 21, 42, 87, 123, 370882, 65932, 69887, 156304, 535886, 648362, 663649, 519965, 934243, 41762, 273103, 627018, 266458, 503966, 43529`

## Cross-fitting

The main empirical Hybrid NBEM implementation is non-cross-fitted. The cross-fitted analysis is a supplementary subset check preserved in Stage 2, not the protocol used for the main reported Hybrid NBEM results.

## Statistical unit

For the corrected Table 5 ablation tests, the unit is the dataset: 20 dataset-level reference-seed scores are compared using two-sided Wilcoxon signed-rank tests, with Holm adjustment over the three planned reduced-Hybrid contrasts within each metric.
