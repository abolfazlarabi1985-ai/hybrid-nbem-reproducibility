# Major Revision - Stage 3

## هدف این مرحله
این مرحله فقط برای محکم‌کردن پاسخ آماری و تجربی در دو مقایسه از پیش تعیین‌شده طراحی شده است:

1. **Hybrid NBEM در برابر NBEM-prop** (همان implementation-level NBEM comparator)
2. **Hybrid NBEM در برابر Logistic Regression**

سه مدل روی **همان 20 دیتاست، همان preprocessing نهایی، همان outer-CV و دقیقاً همان split برای هر dataset+seed** مقایسه می‌شوند.

## چرا 20 seed؟
پنج seed قبلی بدون حذف نگه داشته شده‌اند:

`13, 21, 42, 87, 123`

پانزده seed جدید **قبل از مشاهده هر نتیجه Stage 3** به صورت deterministic از `numpy.random.default_rng(20260924)` تولید و در `config/stage3_seed_plan.json` قفل شده‌اند. بنابراین seedها بعد از دیدن نتیجه انتخاب نمی‌شوند.

برای صرفه‌جویی در زمان، نتایج پنج seed قبلی که قبلاً تحت پروتکل نهایی validate شده‌اند reuse می‌شوند و فقط 15 seed جدید fit می‌شوند. خروجی نهایی همیشه 20 seed کامل دارد.

## پروتکل علمی ثابت
- 20/20 dataset snapshot باید با SHA-256 match شود.
- `duration` در Bank Marketing حذف است.
- Diabetes همان patient-grouped CV را دارد.
- duplicateها کورکورانه حذف نمی‌شوند؛ sensitivity مربوط به Stage 1B جدا باقی می‌ماند.
- NBEM و Hybrid از **کد exact v4** وارد می‌شوند و بازنویسی نشده‌اند.
- Logistic Regression دقیقاً همان specification مرحله 2 است: `C=1`, `lbfgs`, one-hot fold-local برای categorical/Boolean و standardization fold-local برای numeric.
- واحد اصلی استنباط آماری **dataset** است، نه fold و نه seed.

## اجرا روی Windows
ابتدا ZIP را Extract کن. داخل فولدر CMD باز کن.

### 1) فقط validation و smoke test
```bat
run_00_validate_and_smoke.bat
```
باید این پیام‌ها را ببینی:
- `exact hash matches: 20/20`
- `frozen protocol: PASS`
- `validated preloaded first five: PASS`
- `SMOKE PASS`

خروجی smoke با برچسب `SMOKE_DO_NOT_REPORT` است و نباید در مقاله استفاده شود.

### 2) اجرای اصلی Stage 3
```bat
run_01_stage3_full_20seed.bat
```

این اجرا resume-safe است. اگر سیستم خاموش شد یا اجرا قطع شد، همان فایل را دوباره اجرا کن؛ dataset+seedهای کامل‌شده از cache خوانده می‌شوند.

**نکته:** اجرای اصلی فقط 15 seed جدید را fit می‌کند، اما تحلیل نهایی شامل تمام 20 seed قفل‌شده است.

### 3) فقط بازتولید تحلیل از cache (اختیاری)
```bat
run_02_reanalyze_only.bat
```

## خروجی‌هایی که باید برای بررسی ارسال شوند
کل فولدر زیر را ZIP کن:

`results\major_revision_stage3`

مهم‌ترین فایل‌ها:
- `10_stage3_all_fold_results_20seeds.csv`
- `11_stage3_per_seed_dataset_model.csv`
- `12_stage3_dataset_model_20seed_summary.csv`
- `13_stage3_aggregate_model_summary.csv`
- `14_primary_pairwise_dataset_level_tests.csv`
- `15_primary_pairwise_dataset_level_differences.csv`
- `16_win_tie_loss_dataset_level.csv`
- `17_seed_level_aggregate_stability_DESCRIPTIVE.csv`
- `18_hybrid_win_fraction_across_seeds_DESCRIPTIVE.csv`
- `19_dataset_characteristics.csv`
- `20_exploratory_gain_by_dataset_characteristics.csv`
- `21_exploratory_gain_correlations.csv`
- `22_MANUSCRIPT_READY_SUMMARY.txt`
- `RUN_MANIFEST_STAGE3.json`
- `CHECKSUMS_OUTPUT_SHA256.txt`

## تحلیل آماری اصلی
برای هر dataset ابتدا foldها در هر seed میانگین گرفته می‌شوند؛ سپس عملکرد 20 seed برای همان dataset خلاصه می‌شود. مقایسه Hybrid با هر comparator روی **20 مقدار dataset-level** انجام می‌شود.

گزارش اصلی شامل این موارد است:
- Mean performance across datasets
- Mean/median paired difference
- 95% bootstrap CI برای mean paired difference (resampling روی datasetها)
- Win/Tie/Loss روی 20 dataset
- Paired Wilcoxon signed-rank test
- Holm correction برای دو contrast از پیش تعیین‌شده در هر metric
- Paired rank-biserial correlation به عنوان effect size

Weighted-F1 و Macro-F1 معیارهای اصلی هستند.

## تحلیل capability / dataset characteristics
فایل‌های 19 تا 21 فقط **Exploratory** هستند. آن‌ها بررسی می‌کنند gain هیبرید نسبت به NBEM-prop و Logistic Regression با مواردی مثل اندازه دیتاست، تعداد feature، binary/multiclass و mixed-type بودن چه الگویی دارد.

تا وقتی نتیجه Stage 3 دیده نشده، نباید جمله‌ای مثل «Hybrid روی دیتاست‌های بزرگ بهتر است» وارد مقاله شود. فقط اگر خروجی واقعاً آن را پشتیبانی کند، با برچسب مناسب گزارش می‌شود.

## نکته مهم برای مقاله
این Stage برای «بهتر نشان دادن» یک مدل با انتخاب seed طراحی نشده است. هر 20 seed از قبل قفل شده و هیچ seed پس از مشاهده نتیجه حذف نمی‌شود. اگر Logistic Regression یا NBEM-prop در بخشی بهتر باشند، همان نتیجه گزارش می‌شود.
