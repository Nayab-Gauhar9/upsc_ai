import json
import time

from ollama import chat

from src.classifiers.prompts import SYSTEM_PROMPT, build_user_prompt
from src.classifiers.parser import parse_classification
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION
from src.storage.minio_client import MinIOStorage


MODEL = "qwen3:8b"

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "boolean"
        },
        "chapter_number": {
            "type": ["integer", "null"],
            "enum": list(range(1, 93)) + [None]
        },
        "topic": {
            "type": ["string", "null"]
        },
        "subtopic": {
            "type": ["string", "null"]
        },
        "reason": {
            "type": "string"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": [
        "relevant",
        "chapter_number",
        "topic",
        "subtopic",
        "reason",
        "confidence"
    ]
}


def main():

    # ---------------------------------------------------------
    # 1. Read one real PIB article from MinIO
    # ---------------------------------------------------------
    storage = MinIOStorage()

    object_name = "pib/raw/2026/09/18/2312227.json"

    print(f"\nReading: {object_name}")

    record = storage.get_json(object_name)

    print(f"TITLE: {record['title']}")
    print(f"PRID: {record['english_prid']}")
    print()


    # ---------------------------------------------------------
    # 2. Build our existing prompts
    # ---------------------------------------------------------
    user_prompt = build_user_prompt(
        record,
        LAXMIKANTH_8TH_EDITION
    )


    # ---------------------------------------------------------
    # 3. Send article to DeepSeek-R1
    # ---------------------------------------------------------
    print("Sending article to DeepSeek-R1 8B...")
    print("This may take some time on the Moto Book 60.\n")

    start = time.perf_counter()

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format=CLASSIFICATION_SCHEMA,
        think=True,
        options={
            "temperature": 0
        }
    )

    elapsed = time.perf_counter() - start


    # ---------------------------------------------------------
    # 4. Show raw JSON returned by the model
    # ---------------------------------------------------------
    raw_response = response.message.content

    print("RAW MODEL RESPONSE:")
    print(raw_response)
    parsed = json.loads(raw_response)

    print()
    print("CONFIDENCE VALUE:")
    print(parsed["confidence"])
    print("CONFIDENCE TYPE:")
    print(type(parsed["confidence"]).__name__)


    # ---------------------------------------------------------
    # 5. Parse using OUR existing parser
    # ---------------------------------------------------------
    result = parse_classification(raw_response)


    # ---------------------------------------------------------
    # 6. Show final ClassificationResult
    # ---------------------------------------------------------
    print("=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)

    print(result)

    print()
    print(f"MODEL: {MODEL}")
    print(f"TIME: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
