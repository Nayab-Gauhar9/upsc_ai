# src/classifiers/schema.py
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, model_validator

from src.classifiers.taxonomy import is_valid_chapter


class ClassificationResult(BaseModel):
    relevant: bool = Field(
        description="True if the PIB article is relevant to Indian Polity (Laxmikanth syllabus), False otherwise."
    )
    chapter_number: Optional[int] = Field(
        default=None,
        description="The relevant Laxmikanth chapter number (1 to 76/80) if relevant, otherwise null."
    )
    topic: Optional[str] = Field(
        default=None,
        description="A specific, granular news event or core theme from the article (distinct from the chapter title), if relevant."
    )
    subtopic: Optional[str] = Field(
        default=None,
        description="Specific micro-topic or constitutional aspect discussed, otherwise null."
    )
    reason: str = Field(
        description="Clear justification explaining why this article maps or does not map to the Laxmikanth chapter/polity."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    classified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the classification was made."
    )

    @model_validator(mode="after")
    def validate_classification(self):
        # Reason non-empty check
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        # Confidence range
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Relevance-dependent checks
        if not self.relevant:
            if self.chapter_number is not None:
                raise ValueError("chapter_number must be None when relevant is False")
            if self.topic is not None:
                raise ValueError("topic must be None when relevant is False")
            if self.subtopic is not None:
                raise ValueError("subtopic must be None when relevant is False")
        else:
            if self.chapter_number is None:
                raise ValueError("chapter_number is required when relevant is True")
            if not is_valid_chapter(self.chapter_number):
                raise ValueError(f"Invalid Laxmikanth chapter: {self.chapter_number}")
            if not self.topic or not self.topic.strip():
                raise ValueError("topic is required when relevant is True")

        return self
