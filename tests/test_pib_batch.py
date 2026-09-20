from src.collectors.pib_rss import fetch_rss, parse_rss
from src.collectors.pib_article import collect_article
from src.processors.normalize_validate import (
    normalize_record,
    validate_record
)
from src.storage.minio_client import MinIOStorage


rss_data = fetch_rss()
articles = parse_rss(rss_data)

storage = MinIOStorage()

total = len(articles)
collected = 0
valid = 0
uploaded = 0
already_exists = 0
failed = 0


for item in articles:

    link = item.get("link")

    if not link:
        failed += 1
        continue

    prid = link.split("PRID=")[-1]

    try:

        record = collect_article(prid)

        if record is None:
            failed += 1
            print(f"[FAILED] PRID {prid}")
            continue

        collected += 1

        record = normalize_record(record)

        is_valid, error = validate_record(record)

        if not is_valid:
            failed += 1
            print(
                f"[INVALID] PRID {prid}: {error}"
            )
            continue

        valid += 1

        object_name, was_uploaded = (
            storage.upload_pib_record(record)
        )

        if was_uploaded:
            uploaded += 1
            status = "UPLOADED"
        else:
            already_exists += 1
            status = "EXISTS"

        print(
            f"[{status}] "
            f"{record['english_prid']} "
            f"→ {object_name}"
        )

    except Exception as e:

        failed += 1

        print(
            f"[ERROR] PRID {prid}: {e}"
        )


print("\n========== SUMMARY ==========")
print("TOTAL RSS ITEMS :", total)
print("COLLECTED       :", collected)
print("VALID           :", valid)
print("UPLOADED        :", uploaded)
print("ALREADY EXISTS  :", already_exists)
print("FAILED          :", failed)
print("=============================")
