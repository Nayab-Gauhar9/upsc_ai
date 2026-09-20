from src.storage.minio_client import MinIOStorage


def main(limit=20):
    storage = MinIOStorage()

    objects = storage.list_objects("pib/raw/")

    count = 0

    for obj in objects:

        if count >= limit:
            break

        record = storage.get_json(obj.object_name)

        print("\n" + "=" * 80)
        print(f"OBJECT: {obj.object_name}")
        print("=" * 80)

        print(f"TITLE: {record.get('title')}")
        print(f"PRID: {record.get('english_prid')}")
        print(f"PUBLISHED: {record.get('published_at')}")

        article_text = record.get("article_text", "")

        print("\nARTICLE:")
        print(article_text[:1500])

        count += 1


if __name__ == "__main__":
    main()
