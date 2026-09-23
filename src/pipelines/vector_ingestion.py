
from src.rag.embeddings import UPSCChunkerAndEmbedder
from src.storage.minio_client import MinIOStorage
from src.database.session import SessionLocal
from src.database.models import DocumentChunkModel


class VectorIngestionPipeline:
    def __init__(self):
        self.storage = MinIOStorage()
        self.embedder = UPSCChunkerAndEmbedder()

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

                # Check if this PRID has already been embedded and stored in the database
                existing_chunks_count = (
                    db.query(DocumentChunkModel)
                    .filter(DocumentChunkModel.prid == prid)
                    .count()
                )

                if existing_chunks_count > 0:
                    print(f"[-] Skipping PRID: {prid} (Already exists in database for '{chapter_name} -> {topic}')")
                    continue

                print("\n" + "=" * 60)
                print(f"[+] PROCESSING NEW ARTICLE")
                print(f"    PRID         : {prid}")
                print(f"    Chapter      : {chapter_name}")
                print(f"    Topic        : {topic}")
                print(f"    MinIO Object : {object_name}")
                print("=" * 60)

                # Fetch smart notes json from MinIO
                smart_notes_data = self.storage.get_json(object_name)
                if not smart_notes_data:
                    print(f"[!] Warning: Empty or unreadable JSON for PRID {prid}")
                    continue

                # Print summary details if available in json
                if isinstance(smart_notes_data, dict):
                    print(f"    Title        : {smart_notes_data.get('title', 'N/A')}")
                    print(f"    Ministry     : {smart_notes_data.get('ministry', 'N/A')}")

                # Break down into semantic chunks
                chunks = self.embedder.chunk_smart_notes(smart_notes_data)
                if not chunks:
                    print(f"[!] Warning: No chunks generated for PRID {prid}")
                    continue

                print(f"    Generated    : {len(chunks)} semantic chunks. Generating embeddings...")

                texts = [c["text"] for c in chunks]
                embeddings = self.embedder.generate_embeddings(texts)

                # Store chunks and vectors into PostgreSQL pgvector table
                for chunk, embedding in zip(chunks, embeddings):
                    db_chunk = DocumentChunkModel(
                        prid=prid,
                        chapter_name=chapter_name,
                        topic=topic,
                        chunk_type=chunk["chunk_type"],
                        chunk_text=chunk["text"],
                        token_count=chunk["token_count"],
                        embedding=embedding,
                        embedding_model="qwen3-8b",
                    )
                    db.add(db_chunk)

                processed_count += 1
                print(f"[SUCCESS] Stored PRID {prid} with {len(chunks)} chunks into pgvector.\n")

            db.commit()
            print(f"[COMPLETED] Total new files embedded and saved in this run: {processed_count}")

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Vector ingestion failed: {e}")
            raise
        finally:
            db.close()


if __name__ == "__main__":
    pipeline = VectorIngestionPipeline()
    # Set limit=None to process all new files, or a number like 1 or 2 for testing
    pipeline.process_all_smart_notes(limit=10)
