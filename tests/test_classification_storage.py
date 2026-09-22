from src.classifiers.hf import HFClassifier
from src.storage.minio_client import MinIOStorage
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION


PRID = "2312227"


storage = MinIOStorage()

# Find the raw article.
target_object = None
for obj in storage.list_objects("pib/raw/"):
    if obj.object_name.endswith(f"/{PRID}.json"):
        target_object = obj.object_name
        break

if not target_object:
    raise RuntimeError(f"PRID {PRID} not found in MinIO")

record = storage.get_json(target_object)

# Classify the article.
classifier = HFClassifier()
result = classifier.classify(record)

# Convert the dataclass result into storage metadata.
classification = {
    "english_prid": PRID,
    "raw_object": target_object,
    "relevant": result.relevant,
    "chapter_number": result.chapter_number,
    "topic": result.topic,
    "subtopic": result.subtopic,
    "reason": result.reason,
    "confidence": result.confidence,
    "model": classifier.MODEL,
    "provider": classifier.PROVIDER,
    "classified_at": result.classified_at.isoformat(),
}

# Save the classification marker for every article.
classification_object, classification_created = (
    storage.upload_classification(PRID, classification)
)

print("\n=== CLASSIFICATION STORAGE ===")
print(f"Classification object: {classification_object}")
print(f"Classification created: {classification_created}")

# Only relevant articles are materialized under pib/classified/.
if result.relevant:
    chapter_name = LAXMIKANTH_8TH_EDITION[result.chapter_number]

    classified_object, classified_created = (
        storage.upload_classified_article(
            record,
            classification,
            chapter_name,
        )
    )

    print(f"Chapter name:          {chapter_name}")
    print(f"Classified object:     {classified_object}")
    print(f"Classified created:    {classified_created}")
else:
    print("Article is irrelevant; no classified article was created.")
