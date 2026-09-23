import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class UPSCRAGGenerator:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        # Initialize Groq client (reads automatically from GROQ_API_KEY environment variable)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_expert_response(self, query: str, retrieved_articles: list) -> str:
        context_block = ""
        
        for idx, item in enumerate(retrieved_articles, 1):
            notes = item.get("smart_notes", {})
            context_block += f"\n=== Source Article {idx} (PRID: {item['prid']}) ===\n"
            context_block += f"Headline: {notes.get('headline', 'N/A')}\n"
            context_block += f"Summary: {notes.get('summary', 'N/A')}\n"
            context_block += f"Backward Linkages: {notes.get('backward_linkages', [])}\n"
            context_block += f"Mains Relevance: {notes.get('upsc_relevance_mains', 'N/A')}\n"
            context_block += f"Core Smart Notes:\n" + "\n".join([f" - {p}" for p in notes.get('smart_notes', [])]) + "\n"

        system_prompt = """
You are an elite UPSC Civil Services mentor, examiner perspective expert, and faculty specializing in Indian Polity, Constitution, and Governance. 

Your job is to answer the aspirant's query comprehensively. You must structure your teaching logically:

1. Foundational Concept & Evolution (Independence to Modern Times): 
   - First, teach the core topic from scratch as if writing a masterclass.
   - Trace its historical evolution from post-independence constitutional debates and early framework implementation (referencing standard texts like M. Laxmikanth, M.P. Jain, or landmark developments) up to its modern-day application.
2. Context & Current Affair Analysis: Break down the specific current news event or policy change.
3. Static & Constitutional Linkages: Map the topic to specific Articles, constitutional amendments, or landmark Supreme Court judgments (e.g., D.K. Basu guidelines, Sarkaria/Punchhi commissions where relevant).
4. Mains Critical Perspective: Provide multidimensional arguments (challenges, structural issues, and way forward) matching GS Paper syllabus standards.
"""

        user_prompt = f"""
Aspirant Query: {query}

Retrieved Smart Notes & Context from MinIO:
{context_block}
"""

        # Call Groq's high-speed chat completion API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model_name,
            temperature=0.3,
            max_tokens=2048,
        )
        
        return chat_completion.choices[0].message.content
