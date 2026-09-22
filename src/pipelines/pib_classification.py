from src.classifiers.hf import HFClassifier
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION
from src.storage.minio_client import MinIOStorage


class PIBClassificationPipeline:
    """Classify unprocessed PIB articles and materialize relevant articles."""

    def __init__(self):
        self.storage = MinIOStorage()
        self.classifier = HFClassifier()

    def _classification_object(self, prid):
        return f"pib/classifications/{prid}.json"

    def _build_classification(self, record, result, raw_object):
        return {
            "english_prid": record["english_prid"],
            "raw_object": raw_object,
            "relevant": result.relevant,
            "chapter_number": result.chapter_number,
            "topic": result.topic,
            "subtopic": result.subtopic,
            "reason": result.reason,
            "confidence": result.confidence,
            "model": self.classifier.MODEL,
            "provider": self.classifier.PROVIDER,
            "classified_at": result.classified_at.isoformat(),
        }

    def process_one(self, raw_object):
        """Process one raw PIB object without reclassifying an existing PRID."""
        record = self.storage.get_json(raw_object)
        prid = record["english_prid"]

        classification_object = self._classification_object(prid)

        # If classification already exists, do not call the LLM again.
        if self.storage.object_exists(classification_object):
            classification = self.storage.get_json(classification_object)

            print(f"SKIP LLM: {prid} already classified")

            # Repair/materialize a missing relevant article without re-running the model.
            if classification.get("relevant"):
                chapter_number = classification.get("chapter_number")
                chapter_name = LAXMIKANTH_8TH_EDITION[chapter_number]

                classified_object, created = (
                    self.storage.upload_classified_article(
                        record,
                        classification,
                        chapter_name,
                    )
                )

                return {
                    "prid": prid,
                    "status": "already_classified",
                    "relevant": True,
                    "classification_object": classification_object,
                    "classified_object": classified_object,
                    "classified_created": created,
                }

            return {
                "prid": prid,
                "status": "already_classified",
                "relevant": False,
                "classification_object": classification_object,
                "classified_object": None,
                "classified_created": False,
            }

        # No classification marker: this is a genuinely new article.
        result = self.classifier.classify(record)

        classification = self._build_classification(
            record,
            result,
            raw_object,
        )

        classification_object, classification_created = (
            self.storage.upload_classification(
                prid,
                classification,
            )
        )

        classified_object = None
        classified_created = False

        if result.relevant:
            chapter_name = LAXMIKANTH_8TH_EDITION[result.chapter_number]

            classified_object, classified_created = (
                self.storage.upload_classified_article(
                    record,
                    classification,
                    chapter_name,
                )
            )

        return {
            "prid": prid,
            "status": "classified",
            "relevant": result.relevant,
            "classification_object": classification_object,
            "classification_created": classification_created,
            "classified_object": classified_object,
            "classified_created": classified_created,
        }

    def process_prid(self, prid):
        """Find and process one article by English PRID."""
        for obj in self.storage.list_objects("pib/raw/"):
            if obj.object_name.endswith(f"/{prid}.json"):
                return self.process_one(obj.object_name)

        raise RuntimeError(f"PRID {prid} not found in MinIO")

    def process_all(self, limit=None):
        """Process raw PIB articles, skipping already-classified articles."""
        processed = 0
        skipped = 0
        newly_classified = 0

        for obj in self.storage.list_objects("pib/raw/"):
            if limit is not None and newly_classified >= limit:
                break

            result = self.process_one(obj.object_name)
            processed += 1

            if result["status"] == "already_classified":
                skipped += 1
            else:
                newly_classified += 1

        return {
            "processed": processed,
            "skipped": skipped,
            "newly_classified": newly_classified,
        }


if __name__ == "__main__":
    pipeline = PIBClassificationPipeline()
    print(pipeline.process_all(limit=20))
