import json
from datetime import datetime, timezone

from src.classifiers.schema import ClassificationResult


def parse_classification(response: str) -> ClassificationResult:
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError("LLM returned invalid JSON") from e

    required_fields = {
        "relevant",
        "chapter_number",
        "topic",
        "subtopic",
        "reason",
        "confidence",
    }

    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise ValueError(
            f"Missing classification fields: {sorted(missing_fields)}"
        )

    return ClassificationResult(
        relevant=data["relevant"],
        chapter_number=data["chapter_number"],
        topic=data["topic"],
        subtopic=data["subtopic"],
        reason=data["reason"],
        confidence=data["confidence"],
        classified_at=datetime.now(timezone.utc),
    )
