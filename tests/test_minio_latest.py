import os
import re
from src.storage.minio_client import MinIOStorage

def run_test():
    storage = MinIOStorage()
    bucket = "upsc-ai"
    prefix = "pib/smart_notes/"
    pattern = re.compile(r"pib/smart_notes/([^/]+)/([^/]+)/smart_notes_(\d+)\.json")

    print(f"Connecting to bucket: {bucket}")
    print(f"Scanning prefix: {prefix} ...\n")

    matched_keys = []

    # Detect whether client is boto3 (S3) or official minio SDK
    if hasattr(storage.client, "get_paginator"):
        # Boto3 client
        paginator = storage.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                match = pattern.search(key)
                if match:
                    matched_keys.append({
                        "prid": int(match.group(3)),
                        "chapter": match.group(1).strip(),
                        "topic": match.group(2).strip(),
                        "key": key
                    })
    else:
        # MinIO official Python client
        objects = storage.client.list_objects(bucket, prefix=prefix, recursive=True)
        for obj in objects:
            key = getattr(obj, "object_name", str(obj))
            match = pattern.search(key)
            if match:
                matched_keys.append({
                    "prid": int(match.group(3)),
                    "chapter": match.group(1).strip(),
                    "topic": match.group(2).strip(),
                    "key": key
                })

    print(f"Total matching smart notes found: {len(matched_keys)}")

    # Sort descending: highest PRID = latest
    matched_keys.sort(key=lambda x: x["prid"], reverse=True)

    print("\n--- TOP 5 LATEST SMART NOTES ---")
    top_5 = matched_keys[:5]
    for idx, item in enumerate(top_5, start=1):
        print(f"{idx}. PRID: {item['prid']}")
        print(f"   Chapter: {item['chapter']}")
        print(f"   Topic:   {item['topic']}")
        print(f"   Path:    {item['key']}")
        
        # Test fetching one JSON file to verify read access
        if idx == 1:
            try:
                data = storage.get_json(item["key"])
                keys_present = list(data.keys()) if isinstance(data, dict) else len(data)
                print(f"   [JSON Read Test Passed]: Keys found -> {keys_present}")
            except Exception as e:
                print(f"   [JSON Read Test Failed]: {e}")
        print()

if __name__ == "__main__":
    run_test()
