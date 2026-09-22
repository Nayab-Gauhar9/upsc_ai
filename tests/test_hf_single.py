from src.classifiers.hf import HFClassifier
from src.storage.minio_client import MinIOStorage


PRID = "2312227"


storage = MinIOStorage()

objects = storage.list_objects("pib/raw/")

target_object = None

for obj in objects:
    if obj.object_name.endswith(f"/{PRID}.json"):
        target_object = obj.object_name
        break

if not target_object:
    raise RuntimeError(f"PRID {PRID} not found in MinIO")

record = storage.get_json(target_object)

classifier = HFClassifier()

result = classifier.classify(record)

print("\n=== CLASSIFICATION RESULT ===")
print(f"PRID:        {PRID}")
print(f"Relevant:    {result.relevant}")
print(f"Chapter:     {result.chapter_number}")
print(f"Topic:       {result.topic}")
print(f"Subtopic:    {result.subtopic}")
print(f"Reason:      {result.reason}")
print(f"Confidence:  {result.confidence}")
print(f"Classified:  {result.classified_at}")