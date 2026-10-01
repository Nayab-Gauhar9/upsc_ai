from io import BytesIO
import json
import os
import socket
from datetime import datetime
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error


load_dotenv()

def is_local_minio_alive(host: str = "127.0.0.1", port: int = 9000, timeout: float = 0.5) -> bool:
    """Quick check to see if local MinIO port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False



class MinIOStorage:
    def __init__(self, force_r2: bool = False):
        env_mode = os.getenv("STORAGE_TARGET", "auto").lower()
        use_r2 = force_r2 or env_mode == "r2"

        # Check local MinIO health unless forced to R2
        if not use_r2 and is_local_minio_alive():
            # 1. PRIMARY: Local MinIO (When laptop & Docker are active)
            access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
            secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
            self.client = Minio(
                "localhost:9000",
                access_key=access_key,
                secret_key=secret_key,
                secure=False,
            )
            self.bucket = os.getenv("MINIO_BUCKET_NAME", "upsc-ai")
            self.target_name = "Local MinIO"
        else:
            # 2. FALLBACK: Cloudflare R2 (Laptop off / Render cloud / MinIO stopped)
            r2_endpoint = os.getenv("R2_ENDPOINT_URL")
            r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
            r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
            bucket_name = os.getenv("R2_BUCKET_NAME", "upsc-ai")

            if not (r2_endpoint and r2_access_key and r2_secret_key):
                raise RuntimeError(
                    "Local MinIO is down and R2 credentials are missing from environment."
                )
            clean_endpoint = (
                r2_endpoint.replace("https://", "")
                .replace("http://", "")
                .strip("/")
            )
            self.client = Minio(
                clean_endpoint,
                access_key=r2_access_key,
                secret_key=r2_secret_key,
                secure=True,
                region="auto",
            )
            self.bucket = bucket_name
            self.target_name = "Cloudflare R2"





    def bucket_exists(self):

        return self.client.bucket_exists(
            self.bucket
        )

    def create_bucket(self):

        if not self.bucket_exists():

            self.client.make_bucket(
                self.bucket
            )

    def upload_json(self, object_name, data):

        content = json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        self.client.put_object(
            self.bucket,
            object_name,
            BytesIO(content),
            length=len(content),
            content_type="application/json"
        )

    def upload_pib_record(self, record):

        dt = datetime.fromisoformat(
            record["published_at"]
        )

        object_name = (
            f"pib/raw/"
            f"{dt:%Y/%m/%d}/"
            f"{record['english_prid']}.json"
        )
        if self.object_exists(object_name):
            return object_name, False

        self.upload_json(
            object_name,
            record
        )

        return object_name, True

    def upload_classification(self, prid, classification):
        """Store classification metadata for every processed PIB article."""
        object_name = f"pib/classifications/{prid}.json"

        if self.object_exists(object_name):
            return object_name, False

        self.upload_json(object_name, classification)
        return object_name, True

    def upload_classified_article(self, record, classification, chapter_name):
        """Store a relevant PIB article under chapter/topic folders."""
        topic = classification.get("topic") or "Unspecified Topic"

        def safe_path_component(value):
            value = str(value).strip()
            value = value.replace("/", "-").replace(chr(92), "-")
            return value or "Unspecified"

        chapter_dir = safe_path_component(chapter_name)
        topic_dir = safe_path_component(topic)
        prid = record["english_prid"]

        object_name = (
            f"pib/classified/"
            f"{chapter_dir}/"
            f"{topic_dir}/"
            f"{prid}.json"
        )

        if self.object_exists(object_name):
            return object_name, False

        data = {
            **record,
            "classification": {
                **classification,
                "chapter_name": chapter_name,
            },
        }

        self.upload_json(object_name, data)
        return object_name, True

    def object_exists(self, object_name):
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def get_json(self, object_name):
        response = self.client.get_object(
            self.bucket,
            object_name
        )

        try:
            data = response.read()
            return json.loads(data.decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def list_objects(self, prefix=""):
        return self.client.list_objects(
            self.bucket,
            prefix=prefix,
            recursive=True
        )

    def upload_smart_notes(self, chapter_name, topic, prid, study_intelligence):
        """Store synthesized UPSC smart notes in MinIO."""
        def safe_path(val):
            return str(val).strip().replace("/", "-").replace(chr(92), "-")

        object_name = f"pib/smart_notes/{safe_path(chapter_name)}/{safe_path(topic)}/smart_notes_{prid}.json"
        self.upload_json(object_name, study_intelligence.model_dump(mode="json"))
        return object_name