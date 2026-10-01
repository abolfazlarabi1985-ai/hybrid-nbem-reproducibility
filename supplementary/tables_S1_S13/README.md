# Supplementary Tables S1-S13

These CSV files were extracted directly from the final Supplementary Information DOCX included in this release. They are provided as machine-readable companions; the DOCX/PDF remain the formatted authoritative presentation.

- **S1** (`S01.csv`): Implementation-level complexity summary for the evaluated NBEM extensions.
- **S2** (`S02.csv`): Fixed model configurations used in the reported experiments.
- **S3** (`S03.csv`): Dataset-wise weighted F1-score averaged over the five predefined seeds under the primary Bank Marketing protocol with duration excluded.
- **S4** (`S04.csv`): Dataset-wise Macro-F1 averaged over the five predefined seeds under the primary Bank Marketing protocol with duration excluded.
- **S5** (`S05.csv`): Dataset-wise weighted F1-score for the focal four-model comparison, averaged over the same twenty locked seeds.
- **S6** (`S06.csv`): Dataset-wise Macro-F1 for the focal four-model comparison, averaged over the same twenty locked seeds.
- **S7** (`S07.csv`): Dataset-wise weighted Precision (Recall), averaged over the five predefined seeds under the primary Bank Marketing protocol with duration excluded.
- **S8** (`S08.csv`): Model-only timing in the twenty-seed focal runs. Values are equal-dataset means in the measured software/hardware environment; preprocessing time is excluded.
- **S9** (`S09.csv`): Five-seed in-family Wilcoxon comparisons. Holm correction is applied within each F1 metric across the three planned comparators.
- **S10** (`S10.csv`): Twenty-seed focal paired comparisons using one twenty-seed mean per dataset as the inferential unit. Holm correction is across the three planned comparator contrasts within each metric.
- **S11** (`S11.csv`): Actual cross-fitted Hybrid NBEM robustness check. Values are cross-fitted minus main non-cross-fitted means across the original five seeds.
- **S12** (`S12.csv`): Cross-validation sensitivity analysis for Hybrid NBEM. Values are mean ± standard deviation across the original five seeds.
- **S13** (`S13.csv`): Duplicate-grouped sensitivity for the two datasets exceeding the predeclared duplicate-exposure threshold. Values are grouped minus record-level means for Hybrid NBEM.
