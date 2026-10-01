# Major Revision – Stage 4 (Random Forest, exact same 20 seeds)

این بسته فقط برای تکمیل مقایسه‌ی **Hybrid NBEM vs Random Forest** با همان ۲۰ seed قفل‌شده‌ی Stage 3 ساخته شده است.

## اصل مهم
هدف «بهترین نتیجه‌ی قابل دفاع» است، نه انتخاب seed یا تنظیمات پس از دیدن نتیجه. بنابراین:
- هیچ seed حذف/اضافه نمی‌شود.
- همان ۲۰ seed Stage 3 استفاده می‌شود.
- پنج seed اولیه RF از Stage 2، پس از اعتبارسنجی، reuse می‌شوند.
- فقط ۱۵ seed جدید RF واقعاً fit می‌شوند.
- تنظیمات RF دقیقاً همان Stage 2 باقی می‌ماند؛ Stage 4 tuning جدید انجام نمی‌دهد.
- Bank Marketing بدون `duration` است و Diabetes از patient-grouped CV استفاده می‌کند.

## تنظیم RF
`n_estimators=100, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=None, random_state=seed, n_jobs=-1`

`n_jobs=-1` فقط تنظیم محاسباتی است و روی تعریف آماری مدل/seed plan اثر ندارد.

## اجرا روی سرور Linux 64-core / 64GB RAM
بعد از Extract:

```bash
cd NBEM_Major_Revision_STAGE4_RF20_READY_TO_RUN
python3 -m pip install -r requirements.txt
bash run_00_validate_and_smoke.sh
```

در خروجی باید ببینید:
- `exact hash matches: 20/20`
- `frozen protocol: PASS`
- `Stage-3 finalized 20-seed matrix: PASS`
- `Stage-2 RF original five seeds: PASS`
- `SMOKE PASS`

سپس اجرای اصلی:

```bash
bash run_01_rf20_full_64core.sh
```

اجرا resume-safe است. اگر SSH قطع شد بهتر است از `tmux` یا `screen` استفاده کنید:

```bash
tmux new -s rf20
bash run_01_rf20_full_64core.sh
```

برای خروج از tmux بدون قطع اجرا: `Ctrl+B` سپس `D`.

## نکته مهم درباره 64 هسته
اسکریپت BLAS/OpenMP را روی 1 thread می‌بندد و خود Random Forest را با `n_jobs=-1` اجرا می‌کند تا از oversubscription جلوگیری شود. در هر زمان یک dataset×seed اجرا می‌شود، اما درون هر RF همه هسته‌های در دسترس برای درخت‌ها قابل استفاده‌اند.

## خروجی‌ای که باید برای بررسی ارسال کنید
پس از پایان، فقط این فولدر را ZIP کنید:

`results/major_revision_stage4_rf20`

فایل‌های کلیدی:
- `10_rf_all_fold_results_20seeds.csv`
- `12_rf_dataset_model_20seed_summary.csv`
- `13_four_model_dataset_summary_20seeds.csv`
- `14_four_model_aggregate_summary_20seeds.csv`
- `15_hybrid_vs_rf_primary_dataset_level_tests.csv`
- `16_hybrid_vs_rf_dataset_level_differences.csv`
- `17_hybrid_vs_rf_win_tie_loss.csv`
- `18_hybrid_vs_rf_seed_level_stability_DESCRIPTIVE.csv`
- `21_MANUSCRIPT_READY_RF20_SUMMARY.txt`
- `RUN_MANIFEST_STAGE4_RF20.json`

## خروجی ممنوع برای گزارش
`09_rf_additional15_PARTIAL_DO_NOT_REPORT.csv` فقط checkpoint موقت است و نباید در مقاله یا پاسخ داور گزارش شود.

## تفسیر آماری
آزمون اصلی Hybrid vs RF در سطح **۲۰ دیتاست** انجام می‌شود. ۲۰ seed برای robustness است؛ seedها به‌عنوان ۲۰ مشاهده مستقل برای آزمون معناداری استفاده نمی‌شوند. این کار جلوی pseudoreplication را می‌گیرد.
