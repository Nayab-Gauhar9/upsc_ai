from datetime import datetime
import re


def normalize_record(record):

    normalized = record.copy()

    # Normalize title
    if normalized.get("title"):
        normalized["title"] = " ".join(
            normalized["title"].split()
        )

    # Normalize article text
    if normalized.get("article_text"):
        normalized["article_text"] = "\n\n".join(
            paragraph.strip()
            for paragraph in normalized["article_text"].split("\n\n")
            if paragraph.strip()
        )

    # Normalize URL
    if normalized.get("url"):
        normalized["url"] = normalized["url"].strip()

    # Normalize PRIDs
    if normalized.get("source_prid"):
        normalized["source_prid"] = str(
            normalized["source_prid"]
        )

    if normalized.get("english_prid"):
        normalized["english_prid"] = str(
            normalized["english_prid"]
        )

    normalized["published_at"] = normalize_published_at(
    normalized.get("published_at")
    )

    return normalized

def normalize_published_at(value):

    if not value:
        return None

    match = re.search(
        r"Posted On:\s*(\d{1,2}\s+[A-Z]{3}\s+\d{4}\s+\d{1,2}:\d{2}(?:AM|PM))",
        value
    )

    if not match:
        return None

    dt = datetime.strptime(
        match.group(1),
        "%d %b %Y %I:%M%p"
    )

    return dt.isoformat()


def validate_record(record):

    required_fields = [
        "source",
        "source_prid",
        "source_language",
        "english_prid",
        "url",
        "title",
        "published_at",
        "article_text"
    ]

    missing_fields = [
        field
        for field in required_fields
        if not record.get(field)
    ]

    if missing_fields:
        return False, {
            "error": "MISSING_REQUIRED_FIELDS",
            "fields": missing_fields
        }

    if not record["source_prid"].isdigit():
        return False, {
            "error": "INVALID_SOURCE_PRID"
        }

    if not record["english_prid"].isdigit():
        return False, {
            "error": "INVALID_ENGLISH_PRID"
        }

    if len(record["article_text"].strip()) == 0:
        return False, {
            "error": "EMPTY_ARTICLE"
        }

    return True, None

if __name__ == "__main__":

    test_record = {
        "source": "PIB",
        "source_prid": "2311611",
        "source_language": "hi",
        "english_prid": "2311551",
        "url": " https://pib.gov.in/PressReleasePage.aspx?PRID=2311551 ",
        "title": "  Secretary, MSDE inaugurates Skills4Future  ",
        "published_at": "Posted On: 17 SEP 2026 6:22PM by PIB Delhi",
        "article_text": "  First paragraph.  \n\n\n  Second paragraph.  "
    }

    normalized = normalize_record(test_record)

    print("=" * 80)
    print("NORMALIZED RECORD")
    print("=" * 80)

    for key, value in normalized.items():
        print(f"{key}: {value}")

    valid, error = validate_record(normalized)

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print("VALID:", valid)
    print("ERROR:", error)
