from dataclasses import dataclass
from datetime import datetime

from src.classifiers.taxonomy import is_valid_chapter


@dataclass
class ClassificationResult:
    relevant: bool
    chapter_number: int | None
    topic: str | None
    subtopic: str | None
    reason: str
    confidence: float
    classified_at: datetime

    def __post_init__(self):

        # -----------------------------------------------------
        # Basic type validation
        # -----------------------------------------------------
        if not isinstance(self.relevant, bool):
            raise ValueError("relevant must be a boolean")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        # -----------------------------------------------------
        # Confidence validation
        # -----------------------------------------------------
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        # -----------------------------------------------------
        # Relevance-dependent validation
        # -----------------------------------------------------
        if not self.relevant:

            if self.chapter_number is not None:
                raise ValueError(
                    "chapter_number must be None when relevant is False"
                )

            if self.topic is not None:
                raise ValueError(
                    "topic must be None when relevant is False"
                )

            if self.subtopic is not None:
                raise ValueError(
                    "subtopic must be None when relevant is False"
                )

        else:

            if self.chapter_number is None:
                raise ValueError(
                    "chapter_number is required when relevant is True"
                )

            if not isinstance(self.chapter_number, int):
                raise ValueError(
                    "chapter_number must be an integer"
                )

            if not is_valid_chapter(self.chapter_number):
                raise ValueError(
                    f"Invalid Laxmikanth chapter: "
                    f"{self.chapter_number}"
                )

            if not isinstance(self.topic, str) or not self.topic.strip():
                raise ValueError(
                    "topic is required when relevant is True"
                )
