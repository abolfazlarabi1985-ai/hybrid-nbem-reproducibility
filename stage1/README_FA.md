# Major Revision - Stage 1 (NBEM)

این بسته فقط برای مرحله اول Major Revision ساخته شده است و نسخه اصلی v4 را تغییر نمی‌دهد.

## هدف این مرحله
1. تأیید 20/20 فایل دیتاست با SHA-256 اجرای v4.
2. Audit خودکار leakage/identifier/duplicate برای هر 20 دیتاست.
3. اجرای کنترل‌شده Bank Marketing با و بدون `duration` تحت همان کد، preprocessing، CV، پنج seed و معیارهای v4.

## بسیار مهم
- فایل‌های CSV اصلی هرگز ویرایش نمی‌شوند.
- `base/nbem_article_experiments_v4_EXACT_USED.py` یک کپی read-only از کد مبنای v4 است؛ آن را تغییر ندهید.
- نتایج `--quick` فقط برای تست اجرای کد هستند و نباید وارد مقاله شوند.
- تا پایان این مرحله، متن مقاله را تغییر ندهید.

## محیط پیشنهادی
Python 3.11 (یا همان Python محیط اجرای v4) و سپس:

```bash
python -m pip install -r requirements.txt
```

## اجرای پیشنهادی
### 1) فقط Audit اولیه (سریع، بدون آموزش مدل)
Windows:
```bat
run_01_audit_only.bat
```
Linux/macOS:
```bash
bash run_01_audit_only.sh
```

باید در خروجی ببینید:
`exact hash matches: 20/20`

اگر 20/20 نبود، اجرای آزمایش را متوقف کنید.

### 2) Smoke test اختیاری
فقط برای اطمینان از سالم بودن pipeline:
```bat
run_02_smoke_test.bat
```
خروجی این مرحله برای مقاله نیست.

### 3) اجرای کامل Bank Marketing
Windows:
```bat
run_03_bank_full.bat
```
Linux/macOS:
```bash
bash run_03_bank_full.sh
```

این اجرا از 5 seed اصلی زیر استفاده می‌کند:
`13, 21, 42, 87, 123`
و از 10-fold CV مطابق config v4 استفاده می‌کند.

چهار مدل اصلی دقیقاً مطابق v4 اجرا می‌شوند:
- WNB
- NBEM (implementation-level comparator; نام در مقاله بعداً اصلاح می‌شود)
- Adaptive Weighted NBEM
- Hybrid NBEM

برای هر seed هر دو حالت اجرا می‌شوند:
- with_duration
- without_duration

## خروجی‌های مهم
در `results/major_revision_stage1/`:
- `00_exact_snapshot_verification.csv`
- `01_dataset_leakage_duplicate_audit_summary.csv`
- `02_all_dataset_feature_review_inventory.csv`
- `03_automatic_and_name_based_flags_for_manual_review.csv`
- `04_bank_duration_status.csv`
- `10_bank_with_duration_all_fold_results.csv`
- `11_bank_without_duration_all_fold_results.csv`
- `12_bank_duration_per_seed_summary.csv`
- `13_bank_duration_overall_summary.csv`
- `14_bank_duration_paired_seed_deltas.csv`
- `15_bank_duration_delta_summary.csv`
- `16_bank_revised_config_duration_removed.json`
- `RUN_MANIFEST.json`

## بعد از اتمام
کل فولدر `results/major_revision_stage1` را ZIP کنید و ارسال کنید.
قبل از بررسی این خروجی‌ها، RF/LogReg یا cross-fitted stacking را اجرا نکنید تا preprocessing نهایی Freeze شود.
