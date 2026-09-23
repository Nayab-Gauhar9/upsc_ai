import ollama
from typing import List, Dict, Any

class UPSCChunkerAndEmbedder:
    def __init__(self, model_name: str = "qwen3-embedding"):
        print(f"Using local Ollama embedding model '{model_name}'...")
        self.model_name = model_name
        # Note: Set embedding dimension based on your Ollama model specs (e.g., 1536 or 1024)
        self.embedding_dim = 1536 

    def chunk_smart_notes(self, smart_notes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Breaks down a smart notes record into comprehensive semantic chunks 
        covering every field for robust vector search retrieval via Ollama.
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
                "token_count": len(note_text.split())
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

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings locally using Ollama's embed API."""
        response = ollama.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']
