import io
import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from src.storage.minio_client import MinIOStorage

load_dotenv()

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "upsc-ai")

if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    raise ValueError("Missing Cloudflare R2 credentials in .env (R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)")

# Initialize Cloudflare R2 client using standard S3 compatibility
r2_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


def get_existing_r2_keys(bucket_name: str) -> set:
    """Fetches all keys currently stored in Cloudflare R2."""
    existing_keys = set()
    paginator = r2_client.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=bucket_name):
            if "Contents" in page:
                for obj in page["Contents"]:
                    existing_keys.add(obj["Key"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print(f"Bucket '{bucket_name}' not found on R2. Creating it now...")
            r2_client.create_bucket(Bucket=bucket_name)
        else:
            raise e

    return existing_keys


def sync_minio_to_r2():
    print("=" * 60)
    print("STARTING MINIO -> CLOUDFLARE R2 SYNC")
    print("=" * 60)

    # 1. Use your existing local storage client
    minio_storage = MinIOStorage()
    bucket_name = minio_storage.bucket

    if not minio_storage.bucket_exists():
        print(f"Local bucket '{bucket_name}' does not exist. Nothing to sync.")
        return

    # 2. Get keys already present in Cloudflare R2
    print(f"Scanning Cloudflare R2 bucket '{R2_BUCKET_NAME}'...")
    r2_keys = get_existing_r2_keys(R2_BUCKET_NAME)
    print(f"Found {len(r2_keys)} object(s) already in R2.")

    # 3. List all objects in local MinIO
    print(f"Scanning local MinIO bucket '{bucket_name}'...")
    local_objects = list(minio_storage.list_objects())
    print(f"Found {len(local_objects)} object(s) locally in MinIO.")

    # 4. Filter for items that need syncing
    to_sync = [obj for obj in local_objects if obj.object_name not in r2_keys]

    if not to_sync:
        print("\nR2 is already fully up to date! (0 new objects)")
        print("=" * 60)
        return

    print(f"\nPushing {len(to_sync)} new object(s) to Cloudflare R2...")

    # 5. Stream objects across
    for idx, obj in enumerate(to_sync, start=1):
        object_name = obj.object_name
        print(f"[{idx}/{len(to_sync)}] Uploading: {object_name} ...", end=" ")

        response = minio_storage.client.get_object(bucket_name, object_name)
        try:
            data = response.read()
            r2_client.put_object(
                Bucket=R2_BUCKET_NAME,
                Key=object_name,
                Body=data,
                ContentType="application/json" if object_name.endswith(".json") else "application/octet-stream"
            )
            print("Done")
        finally:
            response.close()
            response.release_conn()

    print("\n" + "=" * 60)
    print("ALL OBJECTS SYNCHRONIZED SUCCESSFULLY TO CLOUDFLARE R2")
    print("=" * 60)


if __name__ == "__main__":
    sync_minio_to_r2()
