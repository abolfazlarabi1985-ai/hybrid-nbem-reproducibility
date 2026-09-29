# چک لیست آپلود GitHub و Zenodo

## قبل از GitHub

- فایل های این بسته را در ریشه مخزن قرار دهید.
- `python scripts/verify_release.py` را اجرا کنید.
- دیتاست های شخص ثالث را بدون بررسی مجوز بازنشر عمومی نکنید؛ manifest و hashها کافی هستند.
- در صورت تمایل به اجازه استفاده مجدد از کد، یک License مناسب انتخاب و اضافه کنید.

## برای Zenodo

- حساب GitHub را به Zenodo متصل کنید.
- مخزن `hybrid-nbem-reproducibility` را در بخش GitHub در Zenodo فعال کنید.
- یک Release واقعی (مثلاً `v1.0.0`) در GitHub بسازید.
- اجازه دهید Zenodo همان Release را archive کند.
- DOI واقعی Zenodo را بعد از ایجاد Record بردارید.
- فقط بعد از ایجاد DOI، آن را در مقاله و Response Letter وارد کنید.

هیچ DOI ساختگی یا موقت در این بسته وارد نشده است.
