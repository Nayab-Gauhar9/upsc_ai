import time
from src.classifiers.llm import LangChainGroqClassifier
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION
from src.storage.minio_client import MinIOStorage


class PIBClassificationPipeline:
    """Classify unprocessed PIB articles and materialize relevant articles."""

    def __init__(self):
        self.storage = MinIOStorage()
        self.classifier = LangChainGroqClassifier()

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
            "model": getattr(self.classifier, "MODEL", "llama-3.1-8b-instant"),
            "provider": getattr(self.classifier, "PROVIDER", "groq"),
            "classified_at": result.classified_at.isoformat(),
        }

    def process_one(self, raw_object):
        """Process one raw PIB object without reclassifying an existing PRID."""
        record = self.storage.get_json(raw_object)
        prid = record["english_prid"]
        title = record.get("title", "No Title")

        classification_object = self._classification_object(prid)

        print(f"\n--------------------------------------------------")
        print(f"Processing Article PRID: {prid}")
        print(f"Title: {title[:75]}...")

        # If classification already exists, do not call the LLM again.
        if self.storage.object_exists(classification_object):
            classification = self.storage.get_json(classification_object)
            print(f" [SKIP LLM] Already classified. Found existing file at: '{classification_object}'")

            classified_object_path = None
            if classification.get("relevant"):
                chapter_number = classification.get("chapter_number")
                chapter_name = LAXMIKANTH_8TH_EDITION.get(chapter_number, "Unknown_Chapter")

                classified_object_path, created = (
                    self.storage.upload_classified_article(
                        record,
                        classification,
                        chapter_name,
                    )
                )
                action_msg = "Materialized (repaired missing file)" if created else "Already present"
                print(f" -> Folder [CLASSIFIED]: pib/classified/{chapter_name}/{prid}.json [{action_msg}]")
            else:
                print(f" -> Folder [CLASSIFICATIONS ONLY]: Article marked NOT relevant. No classified folder output.")

            return {
                "prid": prid,
                "status": "already_classified",
                "relevant": classification.get("relevant"),
                "classification_object": classification_object,
                "classified_object": classified_object_path,
            }

        # No classification marker: process with LLM via Groq
        print(f" -> Calling Groq LLM...")
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
        print(f" -> Folder [CLASSIFICATIONS]: Saved output to '{classification_object}'")

        classified_object = None
        if result.relevant:
            chapter_name = LAXMIKANTH_8TH_EDITION.get(result.chapter_number, "Unknown_Chapter")

            classified_object, classified_created = (
                self.storage.upload_classified_article(
                    record,
                    classification,
                    chapter_name,
                )
            )
            print(f" -> Folder [CLASSIFIED]: Materialized relevant article into 'pib/classified/{chapter_name}/{prid}.json'")
        else:
            print(f" -> Status: Article is NOT relevant to Indian Polity (Laxmikanth). Skipped classified folder.")

        # Print breakdown details in terminal
        print(f"   [Details] Relevant: {result.relevant} | Chapter: {result.chapter_number} | Topic: {result.topic} | Confidence: {result.confidence}")

        return {
            "prid": prid,
            "status": "classified",
            "relevant": result.relevant,
            "classification_object": classification_object,
            "classified_object": classified_object,
        }

    def process_all(
        self,
        limit=None,
        chunk_size: int = 15,
        delay_between_chunks_sec: int = 60,
        delay_between_calls_sec: float = 30,
    ):
        """Process raw PIB articles in controlled batches with detailed terminal logs."""
        processed = 0
        skipped = 0
        newly_classified = 0
        current_chunk_count = 0

        objects_list = list(self.storage.list_objects("pib/raw/"))
        print(f"Found {len(objects_list)} total raw articles in MinIO storage.")

        for obj in objects_list:
            if limit is not None and newly_classified >= limit:
                print(f"\nReached processing limit of {limit} newly classified articles. Stopping.")
                break

            result = self.process_one(obj.object_name)
            processed += 1

            if result["status"] == "already_classified":
                skipped += 1
            else:
                newly_classified += 1
                current_chunk_count += 1

                if current_chunk_count >= chunk_size:
                    if limit is not None and newly_classified >= limit:
                        break
                    print(f"\n==================================================")
                    print(f" [RATE-LIMIT PACING] Completed chunk of {chunk_size} newly classified articles.")
                    print(f" Pausing for {delay_between_chunks_sec} seconds to reset Groq rolling window...")
                    print(f"==================================================\n")
                    time.sleep(delay_between_chunks_sec)
                    current_chunk_count = 0
                else:
                    time.sleep(delay_between_calls_sec)

        print(f"\n==================================================")
        print(f" PIPELINE EXECUTION SUMMARY")
        print(f" Total Checked: {processed}")
        print(f" Skipped (Already in MinIO): {skipped}")
        print(f" Newly Processed via LLM: {newly_classified}")
        print(f"==================================================")

        return {
            "processed": processed,
            "skipped": skipped,
            "newly_classified": newly_classified,
        }


if __name__ == "__main__":
    pipeline = PIBClassificationPipeline()
    # Runs with a limit of 10 for safe inspection, change or remove limit=10 as needed
    pipeline.process_all(limit=30)
