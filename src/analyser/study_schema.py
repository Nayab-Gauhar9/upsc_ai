from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class PrelimsAngle(BaseModel):
    statement: str = Field(description="A factual conceptual statement or trap statement suitable for UPSC Prelims MCQs.")
    is_correct: bool = Field(description="Whether this statement is factually correct based on the article.")
    explanation: str = Field(description="Brief explanation for why the statement is correct/incorrect.")


class StudyIntelligenceResult(BaseModel):
    headline: str = Field(description="Crisp, exam-oriented title of the current affairs event.")
    summary: str = Field(description="Concise 3-4 sentence summary tailored for UPSC aspirants.")
    why_it_matters: str = Field(description="Significance of the development in India's governance framework.")
    backward_linkages: List[str] = Field(description="List of related M. Laxmikanth chapters, constitutional articles, or historical precedents (e.g., 'Article 21', 'Keshavananda Bharati Case').")
    upsc_relevance_mains: str = Field(description="Direct mapping to UPSC GS Paper II / Governance syllabus keywords.")
    prelims_practice_points: List[PrelimsAngle] = Field(description="2-3 analytical Prelims practice statements.")
    mains_question: str = Field(description="A high-level analytical Mains practice question (GS-II style).")
    smart_notes: List[str] = Field(description="Bulleted revision points capturing key facts, bodies involved, and data points.")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))