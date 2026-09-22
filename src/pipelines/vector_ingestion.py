
from src.rag.embeddings import UPSCChunkerAndEmbedder
from src.storage.minio_client import MinIOStorage
from src.database.session import SessionLocal
from src.database.models import DocumentChunkModel


class VectorIngestionPipeline:
    def __init__(self):
        self.storage = MinIOStorage()
        self.embedder = UPSCChunkerAndEmbedder()

    def process_all_smart_notes(self, limit= None):
        objects = list(self.storage.list_objects("pib/smart_notes/"))
        print(f"Found {len(objects)} smart notes files in MinIO for vector ingestion.")

        db = SessionLocal()
        try:
            for obj in objects:
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

                print(f"Processing vector embeddings for PRID: {prid} ({chapter_name} -> {topic})")

                # Fetch smart notes json from MinIO
                smart_notes_data = self.storage.get_json(object_name)
                if not smart_notes_data:
                    continue

                # Break down into semantic chunks
                chunks = self.embedder.chunk_smart_notes(smart_notes_data)
                if not chunks:
                    continue

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
                        embedding_model="all-MiniLM-L6-v2",
                    )
                    db.add(db_chunk)

            db.commit()
            print("[SUCCESS] All smart notes successfully embedded and stored in PostgreSQL pgvector!")

        except Exception as e:
            db.rollback()
            print(f"[ERROR] Vector ingestion failed: {e}")
            raise
        finally:
            db.close()


if __name__ == "__main__":
    pipeline = VectorIngestionPipeline()
    pipeline.process_all_smart_notes(limit=1)
