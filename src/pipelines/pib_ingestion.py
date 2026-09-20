from datetime import datetime, timezone

from src.collectors.pib_rss import fetch_rss, parse_rss
from src.collectors.pib_article import collect_article
from src.processors.normalize_validate import (
    normalize_record,
    validate_record
)
from src.storage.minio_client import MinIOStorage
from src.database.models import IngestionRun
from src.database.session import SessionLocal


def run_pib_ingestion(limit=None):

    total = 0
    collected = 0
    valid = 0
    uploaded = 0
    already_exists = 0
    failed = 0

    failure_types = {
        "collection": 0,
        "validation": 0,
        "storage": 0,
        "unexpected": 0
    }

    db = SessionLocal()

    run = IngestionRun(
        source="PIB",
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        total_items=0,
        collected=0,
        valid=0,
        uploaded=0,
        already_exists=0,
        failed=0,
        collection_failures=0,
        validation_failures=0,
        storage_failures=0,
        unexpected_failures=0,
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        rss_data = fetch_rss()
        articles = parse_rss(rss_data)

        if limit is not None:
            articles = articles[:limit]

        storage = MinIOStorage()

        total = len(articles)

        for item in articles:

            link = item.get("link")

            if not link:
                failed += 1
                failure_types["collection"] += 1
                print("[COLLECTION FAILED] RSS item has no link")
                continue

            prid = link.split("PRID=")[-1]

            try:

                record = collect_article(prid)

                if record is None:
                    failed += 1
                    failure_types["collection"] += 1
                    print(f"[COLLECTION FAILED] PRID {prid}")
                    continue

                collected += 1

                record = normalize_record(record)

                is_valid, error = validate_record(record)

                if not is_valid:
                    failed += 1
                    failure_types["validation"] += 1
                    print(
                        f"[VALIDATION FAILED] PRID {prid}: {error}"
                    )
                    continue

                valid += 1

                try:
                    object_name, was_uploaded = (
                        storage.upload_pib_record(record)
                    )

                except Exception as e:
                    failed += 1
                    failure_types["storage"] += 1
                    print(
                        f"[STORAGE ERROR] PRID {prid}: {e}"
                    )
                    continue

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
                failure_types["unexpected"] += 1

                print(
                    f"[UNEXPECTED ERROR] PRID {prid}: {e}"
                )

        print("\n========== PIB INGESTION SUMMARY ==========")
        print("TOTAL RSS ITEMS :", total)
        print("COLLECTED       :", collected)
        print("VALID           :", valid)
        print("UPLOADED        :", uploaded)
        print("ALREADY EXISTS  :", already_exists)
        print("FAILED          :", failed)

        print("\nFAILURE TYPES")
        print("COLLECTION      :", failure_types["collection"])
        print("VALIDATION      :", failure_types["validation"])
        print("STORAGE         :", failure_types["storage"])
        print("UNEXPECTED      :", failure_types["unexpected"])

        print("===========================================")

        report = {
            "total": total,
            "collected": collected,
            "valid": valid,
            "uploaded": uploaded,
            "already_exists": already_exists,
            "failed": failed,
            "failure_types": failure_types
        }

        run.completed_at = datetime.now(timezone.utc)
        run.total_items = total
        run.collected = collected
        run.valid = valid
        run.uploaded = uploaded
        run.already_exists = already_exists
        run.failed = failed
        run.collection_failures = failure_types["collection"]
        run.validation_failures = failure_types["validation"]
        run.storage_failures = failure_types["storage"]
        run.unexpected_failures = failure_types["unexpected"]

        db.commit()

        return report

    except Exception:
        run.completed_at = datetime.now(timezone.utc)
        run.total_items = total
        run.collected = collected
        run.valid = valid
        run.uploaded = uploaded
        run.already_exists = already_exists
        run.failed = failed + 1
        run.collection_failures = failure_types["collection"]
        run.validation_failures = failure_types["validation"]
        run.storage_failures = failure_types["storage"]
        run.unexpected_failures = failure_types["unexpected"] + 1

        db.commit()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    report = run_pib_ingestion()
    print("\nREPORT:", report)

