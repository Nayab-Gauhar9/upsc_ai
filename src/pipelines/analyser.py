import time
from typing import Optional
from src.celery.llm_rotator import execute_with_key_failover
from src.analyser.study_llm import UPSCStudySynthesizer
from src.storage.minio_client import MinIOStorage
from src.services.telegram_notifier import TelegramSmartNotesNotifier
from src.classifiers.taxonomy import CHAPTER_BY_TITLE, is_valid_chapter, get_chapter_title


class StudySynthesisPipeline:
    """Pipeline to synthesize in-depth UPSC study intelligence, persist to R2, and broadcast via Telegram."""

    def __init__(self):
        self.storage = MinIOStorage()
        self.synthesizer = UPSCStudySynthesizer()
        self.notifier = TelegramSmartNotesNotifier()

    def _normalize_chapter(self, raw_chapter: str) -> str:
        """Ensures the chapter strictly maps to canonical taxonomy.py titles."""
        clean = raw_chapter.strip()
        if clean in CHAPTER_BY_TITLE:
            return clean

        # Check if folder name contains a chapter number like "Chapter 76" or "76 - ..."
        try:
            tokens = clean.replace("-", " ").replace("_", " ").split()
            for token in tokens:
                token_clean = token.replace("Chapter", "").replace("ch", "").strip()
                if token_clean.isdigit():
                    ch_num = int(token_clean)
                    if is_valid_chapter(ch_num):
                        return get_chapter_title(ch_num)
        except Exception:
            pass

        return clean

    def _invoke_synthesizer(self, chapter_name, topic, record, api_key=None):
        """Helper to invoke synthesis while dynamically switching API keys."""
        if api_key:
            if hasattr(self.synthesizer, "llm") and hasattr(self.synthesizer.llm, "api_key"):
                self.synthesizer.llm.api_key = api_key
            elif hasattr(self.synthesizer, "client") and hasattr(self.synthesizer.client, "api_key"):
                self.synthesizer.client.api_key = api_key

        return self.synthesizer.synthesize(
            chapter_name=chapter_name,
            topic=topic,
            record=record,
        )

    def process_all_relevant(
        self,
        limit: Optional[int] = None,
        delay_between_calls: float = 15.0,
        broadcast_telegram: bool = True
    ):
        """
        Scans 'pib/classified/' folders, generates deep UPSC smart notes,
        saves to R2, and broadcasts Telegram capsules with deep links and polls.
        """
        processed = 0
        synthesized = 0
        skipped = 0

        objects = list(self.storage.list_objects("pib/classified/"))
        print(f"\n[PIPELINE] Found {len(objects)} total objects under 'pib/classified/' tree.")

        for obj in objects:
            object_name = obj.object_name

            if object_name.endswith("/") or "smart_notes_" in object_name:
                continue

            if limit is not None and synthesized >= limit:
                print(f"\n[LIMIT REACHED] Stopped after synthesizing {limit} articles.")
                break

            processed += 1
            parts = object_name.split("/")
            if len(parts) < 4:
                continue

            raw_chapter = parts[2]
            topic = parts[3]
            prid_file = parts[4]
            prid = prid_file.replace(".json", "")

            # Canonical chapter mapping from taxonomy.py
            chapter_name = self._normalize_chapter(raw_chapter)

            notes_object_name = f"pib/smart_notes/{chapter_name}/{topic}/smart_notes_{prid}.json"

            if self.storage.object_exists(notes_object_name):
                print(f"[SKIP] Smart notes already exist for PRID: {prid}")
                skipped += 1
                continue

            print(f"\n==================================================")
            print(f" [SYNTHESIS START] PRID: {prid}")
            print(f" Chapter: {chapter_name} (Source: {raw_chapter}) | Topic: {topic}")
            print(f" Source: {object_name}")
            print(f"==================================================")

            try:
                payload = self.storage.get_json(object_name)
                record = payload.get("record", {}) or payload

                # Robust content resolution across ingestion schemas
                content = (
                    record.get("article_text")
                    or record.get("content")
                    or record.get("body")
                    or record.get("summary")
                    or payload.get("text", "")
                )
                title = record.get("title") or record.get("headline") or payload.get("title", "")

                prepared_record = {
                    "title": title,
                    "article_text": content,
                }

                # 1. Synthesize intelligence via Groq rotation
                print(" -> Running LLM study intelligence synthesis...")
                study_result = execute_with_key_failover(
                    self._invoke_synthesizer,
                    chapter_name,
                    topic,
                    prepared_record,
                )

                # 2. Persist to MinIO / Cloudflare R2
                saved_path = self.storage.upload_smart_notes(
                    chapter_name=chapter_name,
                    topic=topic,
                    prid=prid,
                    study_intelligence=study_result,
                )
                synthesized += 1
                print(f" -> [STORAGE SUCCESS] Uploaded to: '{saved_path}'")

                # Terminal summary of the generated note
                print(f"    Headline      : {study_result.headline}")
                print(f"    Backward Links: {len(study_result.backward_linkages)} linkages")
                print(f"    Prelims Points: {len(study_result.prelims_practice_points)} items")

                # 3. Publish Telegram capsule + quiz
                if broadcast_telegram:
                    print(f" -> [TELEGRAM] Broadcasting note and quiz to Telegram...")
                    sent = self.notifier.broadcast_note_and_quiz(
                        study_intelligence=study_result,
                        chapter=chapter_name,
                        topic=topic,
                        prid=prid,
                        object_key=saved_path
                    )
                    if sent:
                        print("    [TELEGRAM SUCCESS] Capsule post and Quiz Poll published!")
                    else:
                        print("    [TELEGRAM WARNING] Notification partially or completely failed.")

                print(f" -> Pausing for {delay_between_calls}s before next article...")
                time.sleep(delay_between_calls)

            except Exception as e:
                print(f"[ERROR] Failed synthesizing study intelligence for PRID {prid}: {e}")

        print(f"\n==================================================")
        print(f" STUDY SYNTHESIS PIPELINE SUMMARY")
        print(f" Total Checked               : {processed}")
        print(f" Skipped (Already Generated) : {skipped}")
        print(f" Newly Synthesized & Alerted : {synthesized}")
        print(f"==================================================")

        return {
            "processed": processed,
            "skipped": skipped,
            "synthesized": synthesized,
        }


if __name__ == "__main__":
    pipeline = StudySynthesisPipeline()
    pipeline.process_all_relevant()