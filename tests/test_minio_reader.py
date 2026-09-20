from src.storage.minio_client import MinIOStorage


storage = MinIOStorage()

objects = storage.list_objects("pib/raw/")

for index, obj in enumerate(objects):

    if index >= 5:
        break

    print("\n" + "=" * 70)
    print(f"OBJECT: {obj.object_name}")
    print("=" * 70)

    record = storage.get_json(obj.object_name)

    print("TITLE:", record.get("title"))
    print("SOURCE PRID:", record.get("source_prid"))
    print("SOURCE LANGUAGE:", record.get("source_language"))
    print("ENGLISH PRID:", record.get("english_prid"))
    print("PUBLISHED:", record.get("published_at"))
    print("URL:", record.get("url"))

    article_text = record.get("article_text", "")

    print("\nARTICLE PREVIEW:")
    print(article_text[:1000])