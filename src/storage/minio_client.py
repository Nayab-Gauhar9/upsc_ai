from io import BytesIO
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error


load_dotenv()


class MinIOStorage:

    def __init__(self):

        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not access_key or not secret_key:
            raise ValueError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set"
            )

        self.client = Minio(
            "localhost:9000",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        self.bucket = "upsc-ai"

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