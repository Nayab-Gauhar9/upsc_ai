from datetime import datetime, timezone

from src.classifiers.schema import ClassificationResult


def valid_result():
    return ClassificationResult(
        relevant=True,
        chapter_number=23,
        topic="Parliamentary Procedure",
        subtopic=None,
        reason="The article discusses the functioning of Parliament.",
        confidence=0.95,
        classified_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------
# Valid result
# ---------------------------------------------------------

result = valid_result()

print("VALID RESULT:", result)
print("✓ Valid classification accepted")


# ---------------------------------------------------------
# Invalid chapter
# ---------------------------------------------------------

try:
    ClassificationResult(
        relevant=True,
        chapter_number=999,
        topic="Test",
        subtopic=None,
        reason="Test",
        confidence=0.95,
        classified_at=datetime.now(timezone.utc),
    )
except ValueError as e:
    print("✓ Invalid chapter rejected:", e)


# ---------------------------------------------------------
# Irrelevant article with chapter
# ---------------------------------------------------------

try:
    ClassificationResult(
        relevant=False,
        chapter_number=23,
        topic=None,
        subtopic=None,
        reason="Not relevant.",
        confidence=0.95,
        classified_at=datetime.now(timezone.utc),
    )
except ValueError as e:
    print("✓ Invalid irrelevant result rejected:", e)


# ---------------------------------------------------------
# Invalid confidence
# ---------------------------------------------------------

try:
    ClassificationResult(
        relevant=True,
        chapter_number=23,
        topic="Parliament",
        subtopic=None,
        reason="Test",
        confidence=1.5,
        classified_at=datetime.now(timezone.utc),
    )
except ValueError as e:
    print("✓ Invalid confidence rejected:", e)
