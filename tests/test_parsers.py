from src.classifiers.parser import parse_classification


valid_response = """
{
    "relevant": true,
    "chapter_number": 23,
    "topic": "Parliamentary Procedure",
    "subtopic": "Sessions of Parliament",
    "reason": "The article contains substantive information about Parliament.",
    "confidence": 0.95
}
"""


result = parse_classification(valid_response)

print("✓ Valid JSON parsed")
print("✓ ClassificationResult created")
print(result)


invalid_response = """
{
    "relevant": true,
    "chapter_number": 999,
    "topic": "Test",
    "subtopic": null,
    "reason": "Test",
    "confidence": 0.95
}
"""


try:
    parse_classification(invalid_response)
except ValueError as e:
    print("✓ Invalid LLM classification rejected:", e)
