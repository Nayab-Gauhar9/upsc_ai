import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class UPSCChunkerAndEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Local embedding model recommended for efficient vector generation
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def chunk_smart_notes(self, smart_notes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Breaks down a smart notes record into logical semantic chunks 
        suitable for vector search retrieval.
        """
        chunks = []
        
        # Chunk 1: Summary & Why it matters
        summary_text = (
            f"Headline: {smart_notes_data.get('headline', '')}\n"
            f"Summary: {smart_notes_data.get('summary', '')}\n"
            f"Significance: {smart_notes_data.get('why_it_matters', '')}"
        )
        chunks.append({
            "chunk_type": "summary",
            "text": summary_text,
            "token_count": len(summary_text.split())
        })

        # Chunk 2: Backward Linkages & Constitutional Context
        linkages = smart_notes_data.get('backward_linkages', [])
        if linkages:
            link_text = f"Backward Linkages and Constitutional Context: {', '.join(linkages)}"
            chunks.append({
                "chunk_type": "backward_linkages",
                "text": link_text,
                "token_count": len(link_text.split())
            })

        # Chunk 3: Mains Analysis & Syllabus Mapping
        mains_text = (
            f"Mains Syllabus Mapping: {smart_notes_data.get('upsc_relevance_mains', '')}\n"
            f"Mains Practice Question: {smart_notes_data.get('mains_question', '')}"
        )
        chunks.append({
            "chunk_type": "mains_analysis",
            "text": mains_text,
            "token_count": len(mains_text.split())
        })

        # Chunk 4+: Individual Smart Notes Bullet Points
        for idx, note in enumerate(smart_notes_data.get('smart_notes', [])):
            note_text = f"Smart Note Point: {note}"
            chunks.append({
                "chunk_type": "smart_note_bullet",
                "text": note_text,
                "token_count": len(note_text.split())
            })

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a batch of text chunks."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
