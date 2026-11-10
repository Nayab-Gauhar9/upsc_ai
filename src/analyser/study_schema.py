from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class BackwardLinkageItem(BaseModel):
    title: str = Field(
        default="",
        description="A concise header (max 4-5 words, e.g. 'Article 15(4)', 'Article 46 (DPSP)').",
    )
    description: str = Field(
        default="",
        description="A crisp 1-sentence doctrinal deconstruction (max 15 words).",
    )


class PrelimsAngle(BaseModel):
    statement: str = Field(
        default="",
        description="A factual conceptual statement or realistic UPSC examiner trap.",
    )
    is_correct: bool = Field(
        default=True,
        description="Whether this statement is factually correct based on the constitutional/legal mechanism.",
    )
    explanation: str = Field(
        default="",
        description="Authoritative explanation justifying why the statement is correct or incorrect.",
    )


class StudyIntelligenceResult(BaseModel):
    headline: str = Field(
        default="",
        description="Crisp, exam-oriented title of the current affairs development.",
    )
    summary: str = Field(
        default="",
        description="Concise 3-4 sentence analytical synthesis tailored for UPSC GS-II.",
    )
    why_it_matters: str = Field(
        default="",
        description="Significance of the development in India's constitutional governance framework.",
    )
    # Accepts structured objects emitted by Groq tool-calling, while preserving legacy string coercion
    backward_linkages: List[BackwardLinkageItem] = Field(
        default_factory=list,
        description="Exactly 6 balanced constitutional/statutory backward linkages with title and doctrinal description.",
    )
    upsc_relevance_mains: str = Field(
        default="",
        description="Direct mapping to UPSC GS Paper II syllabus keywords.",
    )
    prelims_practice_points: List[PrelimsAngle] = Field(
        default_factory=list,
        description="2-3 analytical Prelims practice statements with deliberate examiner distractors.",
    )
    mains_question: str = Field(
        default="",
        description="A high-level 15-mark analytical Mains practice question with explicit operational directives.",
    )
    smart_notes: List[str] = Field(
        default_factory=list,
        description="Bulleted revision points capturing key statutory facts, constitutional bodies, and funding mechanisms.",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("backward_linkages", mode="before")
    @classmethod
    def coerce_backward_linkages(cls, value: Any) -> List[Dict[str, str]]:
        """
        Coerces legacy string-only lists, dict representations, or mixed payloads 
        into standard BackwardLinkageItem dictionaries.
        """
        if not isinstance(value, list):
            return []

        coerced: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("header") or "").strip()
                desc = str(item.get("description") or item.get("desc") or "").strip()
                coerced.append({"title": title, "description": desc})
            elif isinstance(item, str):
                parts = [p.strip() for p in item.split(":", 1)]
                if len(parts) == 2:
                    coerced.append({"title": parts[0], "description": parts[1]})
                else:
                    coerced.append({"title": item.strip(), "description": ""})
            elif hasattr(item, "title") and hasattr(item, "description"):
                coerced.append({"title": getattr(item, "title", ""), "description": getattr(item, "description", "")})
        return coerced