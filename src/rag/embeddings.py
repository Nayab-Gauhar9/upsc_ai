import os
import time
from typing import List, Dict, Any
import voyageai
from dotenv import load_dotenv

load_dotenv()

class UPSCChunkerAndEmbedder:
    def __init__(self, model_name: str = "voyage-3-large"):
        self.model_name = model_name
        self.embedding_dim = 1024  # voyage-3 defaults to 1024 dimensions
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    def chunk_smart_notes(self, smart_notes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Breaks down a smart notes record into comprehensive semantic chunks 
        covering every field for vector search retrieval.
        """
        chunks = []
        
        # 1. Headline, Summary & Significance Chunk
        summary_text = (
            f"Headline: {smart_notes_data.get('headline', '')}\n"
            f"Summary: {smart_notes_data.get('summary', '')}\n"
            f"Significance: {smart_notes_data.get('why_it_matters', '')}"
        )
        chunks.append({
            "chunk_type": "summary_and_significance",
            "text": summary_text,
            "token_count": len(summary_text.split())
        })

        # 2. Backward Linkages & Constitutional Context Chunk
        linkages = smart_notes_data.get('backward_linkages', [])
        if linkages:
            link_text = f"Backward Linkages and Constitutional Context: {', '.join(linkages)}"
            chunks.append({
                "chunk_type": "backward_linkages",
                "text": link_text,
                "token_count": len(link_text.split())
            })

        # 3. Mains Analysis, Syllabus Mapping & Practice Question Chunk
        mains_text = (
            f"Mains Syllabus Mapping: {smart_notes_data.get('upsc_relevance_mains', '')}\n"
            f"Mains Practice Question: {smart_notes_data.get('mains_question', '')}"
        )
        chunks.append({
            "chunk_type": "mains_analysis",
            "text": mains_text,
            "token_count": len(mains_text.split())
        })

        # 4. Individual Smart Notes Bullet Points Chunks
        for note in smart_notes_data.get('smart_notes', []):
            note_text = f"Smart Note Point: {note}"
            chunks.append({
                "chunk_type": "smart_note_bullet",
                "text": note_text,
                "token_count": len(str(note).split())
            })

        # 5. Prelims Practice Points Chunks
        for pp in smart_notes_data.get('prelims_practice_points', []):
            pp_text = (
                f"Prelims Practice Statement: {pp.get('statement', '')}\n"
                f"Correctness: {pp.get('is_correct', False)}\n"
                f"Explanation: {pp.get('explanation', '')}"
            )
            chunks.append({
                "chunk_type": "prelims_practice",
                "text": pp_text,
                "token_count": len(pp_text.split())
            })

        return chunks

    def generate_embeddings(
        self, texts: List[str], input_type: str = "document", delay_seconds: float = 0.0
    ) -> List[List[float]]:
        """
        Generates dense 1024-dim vector embeddings via Voyage AI.
        input_type='document' optimizes for stored passages.
        input_type='query' optimizes for user questions.
        """
        if not texts:
            return []

        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type,
            truncation=True
        )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

        return response.embeddings
