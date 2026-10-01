# Major Revision - Stage 1B

این بسته برای **semantic leakage review + duplicate leakage sensitivity** است. کد اصلی v4 و CSVهای خام تغییر نمی‌کنند.

## Stage 1 که در این بسته تثبیت شده

- `duration` در Bank Marketing در config اصلاحی حذف شده است.
- فایل خام Bank دست‌نخورده باقی می‌ماند.
- نتایج Stage 1 در `previous_stage1_results/` فقط برای traceability نگهداری شده‌اند.

## Stage 1B چه می‌کند؟

1. SHA-256 هر 20 دیتاست را دوباره کنترل می‌کند.
2. پنج flag نام‌محور Stage 1 را به تصمیم semantic مستند تبدیل می‌کند.
3. برای شش دیتاستی که Stage 1 در آنها predictor duplicate پیدا کرده، اندازه می‌گیرد چند test row در CV اصلی دارای predictor کاملاً یکسان در train بوده است. برای 14 دیتاست بدون duplicate، این exposure دقیقاً صفر است و نیازی به محاسبه سنگین نیست.
4. با آستانه از پیش تعیین‌شده 10%، دیتاست‌های duplicate-heavy را انتخاب می‌کند.
5. برای دیتاست‌های منتخب، duplicate-group-aware CV را اجرا می‌کند؛ predictorهای کاملاً یکسان هرگز بین train و test تقسیم نمی‌شوند.
6. حساسیت اصلی فقط روی دو مدل مرتبط با ادعای اصلی اجرا می‌شود: `NBEM-prop` و `Hybrid NBEM`.

**هیچ duplicate ای خودکار حذف نمی‌شود.**

## کاهش زمان اجرا بدون کاهش اعتبار مقایسه

سمت record-level از `final_article_run_v4/all_fold_results.csv` دقیق قبلی خوانده می‌شود، چون Haberman/Internet Ads با اصلاح Bank تغییر نکرده‌اند. فقط سمت duplicate-grouped جدید اجرا می‌شود. اگر بخواهید سمت record-level هم مجدداً اجرا شود، گزینه `--rerun-record` وجود دارد، ولی برای Revision لازم نیست.

## دقیقاً چه اجرا کنم؟

### 1) Audit

```bat
python stage1b_duplicate_leakage_sensitivity.py --project-root "." --mode audit
```

یا:

```text
run_01_stage1b_audit.bat
```

باید حتماً ببینید:

```text
exact hash matches: 20/20
```

و در پایان فایل زیر ساخته شود:

```text
results\major_revision_stage1b\25_selected_duplicate_heavy_datasets.csv
```

با audit قبلی انتظار می‌رود آستانه 10%، `haberman_s_survival` و `internet_advertisements` را انتخاب کند.

### 2) Sensitivity اصلی

بعد از موفقیت Audit:

```bat
python stage1b_duplicate_leakage_sensitivity.py --project-root "." --mode sensitivity
```

یا:

```text
run_02_stage1b_sensitivity_core.bat
```

`--quick` نزنید. این اجرای اصلی مقاله است.

### 3) بعد از اتمام

کل فولدر زیر را ZIP کنید و ارسال کنید:

```text
results\major_revision_stage1b
```

تا قبل از تحلیل این خروجی‌ها، RF/LogReg، cross-fitting و ویرایش نتایج مقاله را شروع نکنید.

## خروجی‌های اصلی

- `00_exact_snapshot_verification.csv`
- `20_semantic_review_decisions.csv`
- `21_duplicate_profile_all_datasets.csv`
- `22_duplicate_split_exposure_all_folds.csv`
- `23_duplicate_split_exposure_per_seed.csv`
- `24_duplicate_split_exposure_summary.csv`
- `25_selected_duplicate_heavy_datasets.csv`
- `30_grouped_split_integrity.csv`
- `31_record_level_all_fold_results.csv`
- `32_duplicate_grouped_all_fold_results.csv`
- `33_duplicate_sensitivity_per_seed_summary.csv`
- `34_duplicate_sensitivity_overall_summary.csv`
- `35_duplicate_sensitivity_paired_seed_deltas.csv`
- `36_duplicate_sensitivity_delta_summary.csv`
- `RUN_MANIFEST_STAGE1B.json`
