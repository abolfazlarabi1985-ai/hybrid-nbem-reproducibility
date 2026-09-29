# Corrected Bank Marketing ablation, reference seed 42

This directory is the authoritative revision addendum for the ablation correction requested after the Bank Marketing leakage audit.

## Corrected protocol

- `duration` removed from Bank Marketing predictors.
- seed = 42.
- 10-fold stratified CV.
- Table-5 models: Hybrid NBEM, Hybrid w/o Dependency, Hybrid w/o Adaptive, Hybrid w/o Deep.

## Files

- `ablation_dataset_seed42_corrected.csv`: dataset-level source across all 20 datasets after replacing Bank with the corrected seed-42 values.
- `table5_ablation_corrected_seed42.csv`: corrected aggregate Table 5 values.
- `ablation_wilcoxon_holm_corrected_seed42.csv`: corrected two-sided Wilcoxon signed-rank tests with Holm adjustment across the three ablation contrasts, separately for Weighted-F1 and Macro-F1.
- `bank_without_duration_stage1_all_fold_results.csv`: retained Stage-1 Bank-without-duration fold-level results for the four primary in-family models across five seeds.
- `bank_seed42_hybrid_fold_results.csv`: seed-42 Hybrid NBEM fold-level subset extracted from the Stage-1 corrected Bank run.
- `bank_seed42_hybrid_wo_deep_fold_results.csv`: seed-42 fold-level rerun for Hybrid w/o Deep.
- `bank_seed42_ablation_dataset_level.csv`: corrected Bank dataset-level values for all four Table-5 variants.

## Fold-level provenance limitation

The final working session retained fold-level files for corrected `Hybrid NBEM` and `Hybrid w/o Deep`. The validated dataset-level corrected values for `Hybrid w/o Dependency` and `Hybrid w/o Adaptive` are retained in the source table above, but their fold-level rerun files were not preserved in the supplied `final-2` archive. `scripts/run_bank_ablation_seed42.py` is therefore provided to regenerate all four fold-level files from the retained exact v4 implementation and corrected Bank config. This limitation is documented rather than silently fabricating missing fold-level artifacts.
