from src.collectors.pib_article import collect_article
from src.processors.normalize_validate import (
    normalize_record,
    validate_record
)
from src.storage.minio_client import MinIOStorage


prid = "2311611"

record = collect_article(prid)

if record is None:
    raise RuntimeError("PIB article collection failed")

record = normalize_record(record)

valid, error = validate_record(record)

if not valid:
    raise RuntimeError(f"Validation failed: {error}")

storage = MinIOStorage()

object_name, uploaded = storage.upload_pib_record(record)
print("Object:", object_name)
print("Uploaded:", uploaded)
print("SUCCESS")
print("Title:", record["title"])
print("Published:", record["published_at"])
print("MinIO object:", object_name)