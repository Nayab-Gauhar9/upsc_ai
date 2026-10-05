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
            max_tokens=5000,
        )
        
        return chat_completion.choices[0].message.content

    def generate_chapter_study_notes(self, chapter_title: str, chapter_number: int | None = None) -> str:
        """
        Standalone foundational study generator for M. Laxmikanth chapters.
        Synthesizes notes directly from core constitutional knowledge without external vector context.
        Covers exclusively:
          1. Foundational Concept & Evolution
          2. Static & Constitutional Linkages
        """
        system_prompt = """You are an authoritative UPSC Civil Services Polity Mentor and senior faculty member specializing in Indian Polity and the M. Laxmikanth syllabus.

Your objective is to generate an exhaustive, high-yield masterclass study guide for the requested syllabus chapter entirely from first principles and established constitutional jurisprudence.

You MUST structure your output strictly into these TWO sections only:

## 1. Foundational Concept & Evolution
- Core Concepts & Definitions: Explain the essential principles of this chapter systematically from scratch with absolute academic clarity.
- Historical & Philosophical Evolution: Trace its roots from colonial-era constitutional experiments (e.g., GoI Acts 1909, 1919, 1935), debates in the Constituent Assembly, and early post-independence institutional evolution.
- Structural Framework: Detail key institutional mechanisms, procedures, powers, limitations, and operational doctrines using bullet points and Markdown tables.

## 2. Static & Constitutional Linkages
- Articles & Provisions: Exhaustively map the chapter to specific Articles of the Constitution of India, relevant Parts, and Schedules.
- Constitutional Amendments: Detail all pertinent constitutional amendments (e.g., 24th, 42nd, 44th, 73rd, 86th, 91st, 101st, etc.) and their specific impact on this topic.
- Landmark Judicial Precedents & Doctrines: Cite defining Supreme Court judgments, ratio decidendi, and associated legal doctrines (e.g., Basic Structure, Harmonious Construction, Pith and Substance, Due Process of Law).
- Authoritative Commissions: Integrate relevant committee/commission recommendations (e.g., Sarkaria, Punchhi, Venkatachaliah NCRWC, ARC).

CRITICAL FORMATTING INSTRUCTIONS:
- Do NOT include editorial current affairs, recent newspaper clippings, or subjective Mains opinion critiques.
- Rely purely on canonical static constitutional law and standard M. Laxmikanth syllabus rigor.
- Use clean GitHub-flavored Markdown with bold headings, clean bullet points, and properly formatted Markdown comparison tables.
"""

        chapter_header = f"Chapter {chapter_number}: {chapter_title}" if chapter_number else chapter_title
        user_prompt = f"Generate the comprehensive 2-part masterclass notes for: {chapter_header}"

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model_name,
            temperature=0.2,
            max_tokens=1000,
        )

        return chat_completion.choices[0].message.content
