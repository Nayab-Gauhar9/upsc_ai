import os
from src.database.session import SessionLocal
from src.database.models import DocumentChunkModel
from src.storage.minio_client import MinIOStorage
from src.rag.embeddings import UPSCChunkerAndEmbedder

class UPSCRetriever:
    def __init__(self):
        # Do not persist a single session on self
        self.embedder = UPSCChunkerAndEmbedder(model_name="voyage-3-large")
        self.storage = MinIOStorage()

    def _safe_path(self, val: str) -> str:
        return str(val).strip().replace("/", "-").replace("\\", "-")

    def retrieve_smart_notes_for_query(self, query: str, top_k: int = 3) -> list:
        # 1. Generate query embedding using Voyage AI (input_type="query")
        query_embeddings = self.embedder.generate_embeddings([query], input_type="query")
        if not query_embeddings:
            return []
        query_embedding = query_embeddings[0]

        # 2. Compute cosine distance explicitly in SQL query
        db = SessionLocal()
        try:
            distance_col = DocumentChunkModel.embedding_1024.cosine_distance(query_embedding).label("distance")
            
            # Select both the chunk model and the distance column
            results = (
                db.query(DocumentChunkModel, distance_col)
                .filter(DocumentChunkModel.embedding_1024.isnot(None))
                .order_by(distance_col.asc())
                .limit(top_k * 3)
                .all()
            )
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        seen_prids = set()
        resolved_articles = []

        # 3. Construct MinIO path and calculate confidence score
        for chunk, distance in results:
            if chunk.prid not in seen_prids:
                seen_prids.add(chunk.prid)
                
                # Convert cosine distance to 0.0 - 1.0 similarity and percentage
                cosine_dist = float(distance) if distance is not None else 1.0
                similarity_score = max(0.0, 1.0 - cosine_dist)
                confidence_pct = round(similarity_score * 100, 2)
                
                try:
                    safe_chap = self._safe_path(chunk.chapter_name)
                    safe_top = self._safe_path(chunk.topic)
                    object_name = f"pib/smart_notes/{safe_chap}/{safe_top}/smart_notes_{chunk.prid}.json"
                    
                    raw_data = self.storage.get_json(object_name)
                    resolved_articles.append({
                        "prid": chunk.prid,
                        "chapter": chunk.chapter_name,
                        "topic": chunk.topic,
                        "matched_chunk_type": chunk.chunk_type,
                        "cosine_distance": round(cosine_dist, 4),
                        "similarity_score": round(similarity_score, 4),
                        "confidence_percentage": confidence_pct,
                        "smart_notes": raw_data
                    })
                except Exception as e:
                    print(f"Failed to fetch MinIO/R2 record for PRID {chunk.prid}: {e}")

            if len(resolved_articles) >= top_k:
                break

        return resolved_articles

