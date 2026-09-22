import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .base import ArticleClassifier
from .parser import parse_classification
from .prompts import SYSTEM_PROMPT

load_dotenv()


class HFClassifier(ArticleClassifier):
    """Hugging Face Inference Provider classifier."""

    MODEL = "Qwen/Qwen3-32B"
    PROVIDER = "nscale"
    MAX_TOKENS = 2048

    def __init__(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN must be set")

        self.client = InferenceClient(
            provider=self.PROVIDER,
            api_key=token,
        )

    def classify(self, record):
        chapters = self._chapter_text()

        user_prompt = f"""
Classify the following PIB article for Indian Polity relevance.

Use exactly one Laxmikanth chapter when the article is relevant.
If it is not relevant, set chapter_number, topic, and subtopic to null.

Keep the reason concise, no more than 25 words.
Return only the requested JSON object.

Laxmikanth chapters:
{chapters}

Article title:
{record["title"]}

Article text:
{record["article_text"]}
"""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "polity_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevant": {"type": "boolean"},
                        "chapter_number": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "maximum": 92,
                        },
                        "topic": {"type": ["string", "null"]},
                        "subtopic": {"type": ["string", "null"]},
                        "reason": {"type": "string", "maxLength": 300},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "relevant",
                        "chapter_number",
                        "topic",
                        "subtopic",
                        "reason",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        }

        completion = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.MAX_TOKENS,
            response_format=response_format,
        )

        raw = completion.choices[0].message.content

        result = parse_classification(raw)

        # Keep the classifier result in the existing dataclass contract.
        if result.classified_at is None:
            result.classified_at = datetime.now(timezone.utc)

        return result

    @staticmethod
    def _chapter_text():
        from .taxonomy import LAXMIKANTH_8TH_EDITION

        return "\n".join(
            f"{number}. {name}"
            for number, name in LAXMIKANTH_8TH_EDITION.items()
        )
