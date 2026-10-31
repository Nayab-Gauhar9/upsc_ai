import io
import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from minio import Minio

load_dotenv()

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "upsc-ai")

LOCAL_MINIO_HOST = os.getenv("MINIO_HOST", "localhost")
LOCAL_MINIO_PORT = int(os.getenv("MINIO_PORT", 9000))
LOCAL_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
LOCAL_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
LOCAL_MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "upsc-ai")

if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    raise ValueError("Missing Cloudflare R2 credentials in .env")

# 1. Initialize R2 Client
r2_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

# 2. Initialize Direct Local MinIO Client
local_minio = Minio(
    f"{LOCAL_MINIO_HOST}:{LOCAL_MINIO_PORT}",
    access_key=LOCAL_MINIO_ACCESS_KEY,
    secret_key=LOCAL_MINIO_SECRET_KEY,
    secure=False,
)


def get_all_r2_keys(bucket_name: str) -> list[str]:
    """Fetches all keys from Cloudflare R2."""
    keys = []
    paginator = r2_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        if "Contents" in page:
            for obj in page["Contents"]:
                keys.append(obj["Key"])
    return keys


def get_local_minio_keys(bucket_name: str) -> set[str]:
    """Fetches all keys currently in local MinIO."""
    if not local_minio.bucket_exists(bucket_name):
        local_minio.make_bucket(bucket_name)
        return set()

    objects = local_minio.list_objects(bucket_name, recursive=True)
    return {obj.object_name for obj in objects}


def sync_r2_to_minio():
    print("=" * 60)
    print("STARTING CLOUDFLARE R2 -> LOCAL MINIO REVERSE SYNC")
    print("=" * 60)

    # Fetch inventory
    print(f"Scanning Cloudflare R2 bucket '{R2_BUCKET_NAME}'...")
    r2_keys = get_all_r2_keys(R2_BUCKET_NAME)
    print(f"Found {len(r2_keys)} object(s) in Cloudflare R2.")

    print(f"Scanning local MinIO bucket '{LOCAL_MINIO_BUCKET}'...")
    local_keys = get_local_minio_keys(LOCAL_MINIO_BUCKET)
    print(f"Found {len(local_keys)} object(s) locally in MinIO.")

    # Missing in MinIO
    to_download = [k for k in r2_keys if k not in local_keys]

    if not to_download:
        print("\nLocal MinIO is already up to date with R2! (0 new objects)")
        print("=" * 60)
        return

    print(f"\nPulling {len(to_download)} new object(s) down to local MinIO...")

    for idx, key in enumerate(to_download, start=1):
        print(f"[{idx}/{len(to_download)}] Downloading from R2 -> MinIO: {key} ...", end=" ")

        # Stream directly from R2 to MinIO in memory
        r2_obj = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        body_bytes = r2_obj["Body"].read()

        local_minio.put_object(
            bucket_name=LOCAL_MINIO_BUCKET,
            object_name=key,
            data=io.BytesIO(body_bytes),
            length=len(body_bytes),
            content_type=r2_obj.get("ContentType", "application/json"),
        )
        print("Done")

    print("\n" + "=" * 60)
    print("ALL R2 OBJECTS SYNCHRONIZED LOCALLY TO MINIO")
    print("=" * 60)


if __name__ == "__main__":
    sync_r2_to_minio()
