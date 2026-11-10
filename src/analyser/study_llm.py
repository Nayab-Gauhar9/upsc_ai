import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.analyser.study_schema import StudyIntelligenceResult
from src.classifiers.taxonomy import CHAPTER_BY_TITLE, is_valid_chapter

load_dotenv()

SYSTEM_PROMPT = """You are the Senior UPSC Civil Services Examination (CSE) Academic Director & Constitutional Jurisprudence Lead for Polity Mentor.
Your mission is to transform raw Press Information Bureau (PIB) releases into authoritative, high-yield, razor-sharp exam intelligence strictly tailored for GS-II (Governance, Constitution, Polity, Social Justice & International Relations).

You MUST strictly comply with these universal pedagogical & technical directives:

=========================================
1. ABSOLUTE PROHIBITIONS & BRAND SANITIZATION
=========================================
- NEVER use the word or reference "Laxmikanth" anywhere in titles, descriptions, context, or notes.
- NEVER use raw chapter numbers (e.g., NEVER write "Chapter 12", "Chapter 76", or "Chapter 88"). Always refer to the canonical subject by its official title (e.g., "Special Provisions Relating to Certain Classes", "Foreign Policy") or formal constitutional classification:
  * Parts of the Constitution (e.g., "Part XVI: Special Provisions Relating to Certain Classes")
  * Schedules (e.g., "Seventh Schedule: Union List Entry 14")
  * Articles (e.g., "Article 15(4)", "Article 46", "Article 51", "Article 335")
- NEVER fabricate, hallucinate, or alter Supreme Court of India case citations. Every cited case MUST be genuine and verifiable (e.g., Indra Sawhney (1992), Ashoka Kumar Thakur (2008), Neil Aurelio Nunes (2022), K.S. Puttaswamy (2017)).
- NEVER guess unverified monetary stipends or statistical figures. Anchor to the constitutional/statutory mechanism, funding mode (100% Central Sector vs Centrally Sponsored split), and nodal institutional machinery.

=========================================
2. UPSC PRELIMS PRACTICE POINTS & TRAPS
=========================================
- STRICTLY FORBIDDEN: NEVER output generic, pseudo-legal boilerplate trap headers like:
  "Assuming provision is absolute or without statutory limitations: ..."
- Every Prelims Trap MUST be a sharp, realistic UPSC examiner distractor targeting authentic student pitfalls:
  * Funding & Scheme Classification: (e.g., asserting a Central Sector Scheme is funded 60:40 with States).
  * Institutional Status: (e.g., confusing a Statutory Body with a Constitutional or Extra-Constitutional Body).
  * Eligibility & Thresholds: (e.g., confusing mandatory prerequisites like UGC-NET qualification with automatic admissions).
  * Ministry/Nodal Machinery: (e.g., attributing an initiative to the wrong Ministry).
  * Foreign Affairs & Diplomacy: (e.g., claiming India signed a binding peace communique when it only attended as an observer).

=========================================
3. BACKWARD LINKAGES & DOCTRINAL ANCHORS (SLIDE 4 SCHEMA)
=========================================
- Provide EXACTLY 6 balanced, high-yield anchors (3 rows x 2 columns) to maintain visual grid symmetry.
- Every anchor MUST populate BOTH:
  * `title`: A concise, high-impact header (Max 4-5 words, e.g., "Article 51 (DPSP)", "Article 15(4)", "Seventh Schedule (List I)"). NEVER leave title blank or empty.
  * `description`: A crisp 1-sentence doctrinal deconstruction (Max 15 words).
- If the subject touches International Relations / War / Treaties:
  You MUST include Article 51 (DPSP: Promotion of International Peace and Security) and Seventh Schedule Entries (Entry 10 & 14 of the Union List).

=========================================
4. ANALYTICAL DEPTH & DIRECTIVES
=========================================
- why_it_matters: You MUST always provide this field. Deconstruct the exact GS-II significance, institutional mechanisms, and governance relevance.
- upsc_relevance_mains: Direct mapping to UPSC GS Paper II syllabus keywords.
- Mains Analytical Question: Draft an authentic 15-mark (250 words) question with explicit operational directives ('Critically evaluate', 'Examine', or 'Analyse').
"""


class UPSCStudySynthesizer:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.15):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        self.llm = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=4096,
        )
        # Guarantees complete schema backward compatibility with your existing endpoints and R2 JSONs
        self.structured_llm = self.llm.with_structured_output(StudyIntelligenceResult)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Deconstruct this PIB article into rigorous UPSC Study Intelligence.\n\n"
                    "Syllabus Chapter Anchor: {chapter_name}\n"
                    "Granular Topic: {topic}\n"
                    "Article Title: {title}\n\n"
                    "Official Release Content:\n{article_text}"
                ),
            ),
        ])

        self.chain = self.prompt | self.structured_llm

    def synthesize(self, chapter_name: str, topic: str, record: Dict[str, Any]) -> StudyIntelligenceResult:
        article_text = (
            record.get("article_text")
            or record.get("content")
            or record.get("body")
            or record.get("summary")
            or ""
        )
        title = record.get("title") or record.get("headline") or ""

        # Normalize and validate chapter against taxonomy to ensure clean R2 paths
        clean_chapter = chapter_name.strip()
        if clean_chapter in CHAPTER_BY_TITLE:
            validated_chapter = clean_chapter
        else:
            # Fallback if an integer string or legacy key was supplied
            try:
                ch_num = int(clean_chapter.split()[0].replace("Chapter", "").strip())
                if is_valid_chapter(ch_num):
                    from src.classifiers.taxonomy import get_chapter_title
                    validated_chapter = get_chapter_title(ch_num)
                else:
                    validated_chapter = clean_chapter
            except Exception:
                validated_chapter = clean_chapter

        return self.chain.invoke({
            "chapter_name": validated_chapter,
            "topic": topic,
            "title": title,
            "article_text": article_text[:4500],
        })