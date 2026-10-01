# Major Revision - Stage 2

این بسته مرحله دوم اصلاحات مقاله است و باید فقط بعد از Stage 1 و Stage 1B اجرا شود.

## چه چیزهایی در این مرحله انجام می‌شود؟

1. **Baseline خارجی روی پروتکل نهایی Freeze شده**
   - Logistic Regression
   - Random Forest
   - همان 20 دیتاست، همان 5 seed اصلی، همان CV اصلی مقاله.
   - Bank Marketing بدون `duration`.
   - Diabetes با patient-grouped CV.
   - همه preprocessingها فقط داخل training fold fit می‌شوند.

2. **Actual cross-fitted Hybrid NBEM**
   - برای meta-learner، probabilityهای training به صورت OOF ساخته می‌شوند.
   - حتی preprocessor نیز داخل inner fold دوباره fit می‌شود.
   - بنابراین نمونه OOF در fit کردن preprocessor یا component model مربوط به خودش حضور ندارد.
   - ابتدا روی یک subset از پیش تعیین‌شده و نماینده اجرا می‌شود؛ full-20 نیز داخل بسته وجود دارد ولی فعلاً اجرا نشود.

3. **Timing audit برای Table 6**
   - زمان preprocessing، model fit و inference جداگانه ثبت می‌شود.
   - هدف: رفع ابهام یکسان بودن زمان NBEM و Adaptive در جدول فعلی.

## ترتیب اجرای دقیق

### 0) نصب کتابخانه‌ها (فقط اگر محیط قبلی را ندارید)

```bat
python -m pip install -r requirements.txt
```

### 1) حتماً Smoke Test

```bat
run_00_smoke_test.bat
```

باید `SMOKE PASS` و `exact hash matches: 20/20` دیده شود.
خروجی Smoke علمی نیست و نباید در مقاله استفاده شود.

### 2) Baselineهای اصلی (الزامی برای Stage 2)

```bat
run_01_external_baselines_full.bat
```

این مرحله RF و Logistic Regression را روی 20 دیتاست و 5 seed اجرا می‌کند. اجرا resume-capable است؛ اگر سیستم خاموش شد، همان فایل را مجدداً اجرا کنید.

### 3) Cross-fitted stacking روی subset از پیش تعیین‌شده

```bat
run_02_crossfit_subset_recommended.bat
```

Subset قبل از دیدن نتایج Stage 2 تعیین شده است:
- bank_marketing
- car_evaluation
- dry_bean_dataset
- heart_disease
- internet_advertisements
- productivity_prediction_of_garment_employees

این انتخاب برای پوشش binary/multiclass، categorical/numeric/mixed، کوچک/متوسط/پُربعد است و برای جلوگیری از cherry-picking قبل از اجرا در کد ثابت شده است.

### 4) Timing Audit

```bat
run_03_timing_audit.bat
```

### 5) فعلاً اجرا نکنید

```bat
run_04_crossfit_full20_OPTIONAL_DO_NOT_RUN_YET.bat
```

Full nested cross-fitting بسیار سنگین است. ابتدا خروجی subset را ارسال کنید؛ سپس درباره نیاز به اجرای full-20 تصمیم می‌گیریم.

## بعد از پایان چه چیزی بفرستم؟

کل فولدر زیر را ZIP کنید و ارسال کنید:

```text
results\major_revision_stage2
```

## خروجی‌های مهم

Baseline:
- `10_external_baselines_all_fold_results.csv`
- `10_external_baselines_per_seed_summary.csv`
- `10_external_baselines_overall_summary.csv`
- `13_external_baselines_aggregate_across_datasets.csv`

Cross-fit subset:
- `19_crossfit_subset_predeclared.csv`
- `20_crossfit_subset_all_fold_results.csv`
- `20_crossfit_subset_per_seed_summary.csv`
- `20_crossfit_subset_overall_summary.csv`
- `20_crossfit_subset_vs_revised_noncross_per_seed.csv`
- `20_crossfit_subset_vs_revised_noncross_delta_summary.csv`

Timing:
- `30_timing_audit_all_fold_results.csv`
- `30_timing_audit_per_seed_summary.csv`
- `30_timing_audit_overall_summary.csv`
- `33_timing_table6_clarification.csv`

Integrity:
- `00_exact_snapshot_verification.csv`
- `RUN_MANIFEST_STAGE2.json`

## نکات علمی مهم

- نام baseline قبلی در گزارش نهایی باید `NBEM-prop` یا `implementation-level NBEM comparator` باشد، نه ادعای replication دقیق published NBEM.
- RF و Logistic Regression **standard/untuned baselines** هستند؛ هیچ tuning مبتنی بر test fold انجام نمی‌شود.
- Cross-fitted Hybrid در Stage 2 یک پیاده‌سازی واقعی OOF stacker است، نه فقط تغییر متن مقاله.
- نتایج Stage 1B duplicate-grouped حذف نمی‌شوند و به عنوان robustness analysis مستقل حفظ می‌شوند.
- مقاله هنوز ویرایش نشود تا خروجی Stage 2 بررسی شود.
