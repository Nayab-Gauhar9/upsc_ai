
import time
from typing import Optional
from src.analyser.study_llm import UPSCStudySynthesizer
from src.storage.minio_client import MinIOStorage


class StudySynthesisPipeline:
    """Pipeline to generate in-depth UPSC study intelligence from classified relevant articles."""

    def __init__(self):
        self.storage = MinIOStorage()
        self.synthesizer = UPSCStudySynthesizer()

    def process_all_relevant(self, limit: Optional[int] = None, delay_between_calls: float = 20.0):
        """
        Scans 'pib/classified/' folders, reads each relevant article, 
        and generates deep UPSC smart notes + active recall points, displaying output in the terminal.
        """
        processed = 0
        synthesized = 0
        skipped = 0

        objects = list(self.storage.list_objects("pib/classified/"))
        print(f"Found {len(objects)} total objects under pib/classified/ tree.")

        for obj in objects:
            object_name = obj.object_name

            # Skip directory markers or already generated smart notes files
            if object_name.endswith("/") or "smart_notes_" in object_name:
                continue

            if limit is not None and synthesized >= limit:
                print(f"\n[LIMIT REACHED] Stopped after synthesizing {limit} articles.")
                break

            processed += 1
            parts = object_name.split("/")
            if len(parts) < 4:
                continue

            chapter_name = parts[2]
            topic = parts[3]
            prid_file = parts[4]
            prid = prid_file.replace(".json", "")

            notes_object_name = f"pib/smart_notes/{chapter_name}/{topic}/smart_notes_{prid}.json"

            # Check if smart notes already exist
            if self.storage.object_exists(notes_object_name):
                print(f"[SKIP] Smart notes already exist for PRID: {prid}")
                skipped += 1
                continue

            print(f"\n==================================================")
            print(f" [SYNTHESIS START] Processing PRID: {prid}")
            print(f" Chapter: {chapter_name}")
            print(f" Topic: {topic}")
            print(f"==================================================")

            try:
                payload = self.storage.get_json(object_name)
                record = payload.get("record", {})
                
                if not record:
                    record = payload

                # Invoke Groq LLM for study intelligence
                study_result = self.synthesizer.synthesize(
                    chapter_name=chapter_name,
                    topic=topic,
                    record=record,
                )

                # Save smart notes back into MinIO
                saved_path = self.storage.upload_smart_notes(
                    chapter_name=chapter_name,
                    topic=topic,
                    prid=prid,
                    study_intelligence=study_result,
                )

                synthesized += 1
                
                # Display detailed results directly in the terminal
                print(f"\n [SUCCESS] Saved smart notes to: '{saved_path}'")
                print(f" Headline      : {study_result.headline}")
                print(f" Summary       : {study_result.summary}")
                print(f" Why It Matters: {study_result.why_it_matters}")
                print(f" Backward Links: {', '.join(study_result.backward_linkages)}")
                print(f" Mains Angle   : {study_result.upsc_relevance_mains}")
                print(f" Mains Question: {study_result.mains_question}")
                print(f" Smart Notes   :")
                for note in study_result.smart_notes:
                    print(f"   * {note}")
                print(f" Prelims Points:")
                for p in study_result.prelims_practice_points:
                    print(f"   * [{str(p.is_correct).upper()}] {p.statement} -> {p.explanation}")
                print(f"--------------------------------------------------\n")

                time.sleep(delay_between_calls)

            except Exception as e:
                print(f"[ERROR] Failed synthesizing study intelligence for PRID {prid}: {e}")

        print(f"\n==================================================")
        print(f" STUDY SYNTHESIS PIPELINE SUMMARY")
        print(f" Total Checked: {processed}")
        print(f" Skipped (Already Synthesized): {skipped}")
        print(f" Newly Synthesized: {synthesized}")
        print(f"==================================================")

        return {
            "processed": processed,
            "skipped": skipped,
            "synthesized": synthesized,
        }


if __name__ == "__main__":
    pipeline = StudySynthesisPipeline()
    # Test on a small batch of 2 first to inspect rich terminal outputs
    pipeline.process_all_relevant(limit=10)
