import time
from src.rag.embeddings import UPSCChunkerAndEmbedder
from src.storage.minio_client import MinIOStorage
from src.database.session import SessionLocal
from src.database.models import DocumentChunkModel


class VectorIngestionPipeline:
    def __init__(self):
        self.storage = MinIOStorage()
        self.embedder = UPSCChunkerAndEmbedder(model_name="voyage-3-large")

    def process_all_smart_notes(self, limit=None):
        objects = list(self.storage.list_objects("pib/smart_notes/"))
        print(f"Found {len(objects)} total smart notes files in MinIO.")

        db = SessionLocal()
        try:
            processed_count = 0
            for obj in objects:
                if limit is not None and processed_count >= limit:
                    break

                object_name = obj.object_name
                if object_name.endswith("/"):
                    continue

                parts = object_name.split("/")
                if len(parts) < 4:
                    continue

                chapter_name = parts[2]
                topic = parts[3]
                filename = parts[4]
                prid = filename.replace("smart_notes_", "").replace(".json", "")

                # Skip if already ingested
                already_ingested = (
                    db.query(DocumentChunkModel)
                    .filter(DocumentChunkModel.prid == prid)
                    .first()
                )
                if already_ingested:
                    print(f"[-] Skipping PRID: {prid} (Already exists in database)")
                    continue

                print("\n" + "=" * 60)
                print(f"[+] PROCESSING ARTICLE")
                print(f"    PRID         : {prid}")
                print(f"    Chapter      : {chapter_name}")
                print(f"    Topic        : {topic}")
                print(f"    Object Key   : {object_name}")
                print("=" * 60)

                smart_notes_data = self.storage.get_json(object_name)
                if not smart_notes_data:
                    print(f"[!] Warning: Empty JSON for PRID {prid}")
                    continue

                chunks = self.embedder.chunk_smart_notes(smart_notes_data)
                if not chunks:
                    print(f"[!] Warning: No chunks generated for PRID {prid}")
                    continue

                print(f"    Generated    : {len(chunks)} chunks. Calling Voyage AI...")

                # Single batch call for all chunks in this article
                texts = [c["text"] for c in chunks]
                embeddings = self.embedder.generate_embeddings(texts, input_type="document")

                # Insert clean chunks with 1024-dim vectors
                for chunk, embedding in zip(chunks, embeddings):
                    db_chunk = DocumentChunkModel(
                        prid=prid,
                        chapter_name=chapter_name,
                        topic=topic,
                        chunk_type=chunk["chunk_type"],
                        chunk_text=chunk["text"],
                        token_count=chunk["token_count"],
                        embedding_1024=embedding,
                        embedding_model="voyage-3-large",
                    )
                    db.add(db_chunk)

                db.commit()
                processed_count += 1
                print(f"[SUCCESS] Ingested {len(chunks)} chunks for PRID {prid}.")

                # Voyage rate limiter: 25s pause maintains ~2.4 RPM (< 3 RPM limit)
                print("Pausing 25s to respect Voyage AI free rate limit...")
                time.sleep(30)

            print(f"\n[COMPLETED] Ingestion complete. Processed {processed_count} files.")

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Ingestion failed: {e}")
            raise
        finally:
            db.close()


if __name__ == "__main__":
    pipeline = VectorIngestionPipeline()
    # Test with 2 articles first, then run with limit=None
    pipeline.process_all_smart_notes(limit=25)
