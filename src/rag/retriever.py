import json
from src.database.session import SessionLocal
from src.database.models import DocumentChunkModel
from src.storage.minio_client import MinIOStorage
from src.rag.embeddings import UPSCChunkerAndEmbedder

class UPSCRetriever:
    def __init__(self):
        self.db = SessionLocal()
        self.embedder = UPSCChunkerAndEmbedder()
        self.storage = MinIOStorage()

    def _safe_path(self, val: str) -> str:
        return str(val).strip().replace("/", "-").replace("\\", "-")

    def retrieve_smart_notes_for_query(self, query: str, top_k: int = 3) -> list:
        # 1. Generate query embedding using Qwen3-Embedding
        query_embedding = self.embedder.generate_embeddings([query])[0]

        # 2. Query pgvector for closest chunks
        chunks = (
            self.db.query(DocumentChunkModel)
            .order_by(DocumentChunkModel.embedding.cosine_distance(query_embedding))
            .limit(top_k * 3)
            .all()
        )

        seen_prids = set()
        resolved_articles = []

        # 3. Directly construct MinIO path using chunk metadata
        for chunk in chunks:
            if chunk.prid not in seen_prids:
                seen_prids.add(chunk.prid)
                
                try:
                    # Rebuild exact path format matching minio_client.py upload logic
                    safe_chap = self._safe_path(chunk.chapter_name)
                    safe_top = self._safe_path(chunk.topic)
                    object_name = f"pib/smart_notes/{safe_chap}/{safe_top}/smart_notes_{chunk.prid}.json"
                    
                    # Fetch JSON directly from MinIO
                    raw_data = self.storage.get_json(object_name)
                    resolved_articles.append({
                        "prid": chunk.prid,
                        "chapter": chunk.chapter_name,
                        "topic": chunk.topic,
                        "matched_chunk_type": chunk.chunk_type,
                        "smart_notes": raw_data
                    })
                except Exception as e:
                    print(f"Failed to fetch MinIO record for PRID {chunk.prid} at path: {e}")

            if len(resolved_articles) >= top_k:
                break

        return resolved_articles
