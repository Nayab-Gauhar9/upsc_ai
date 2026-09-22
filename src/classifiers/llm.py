
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.classifiers.base import ArticleClassifier
from src.classifiers.schema import ClassificationResult
from src.classifiers.prompts import SYSTEM_PROMPT
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION

load_dotenv()


class LangChainGroqClassifier(ArticleClassifier):
    PROVIDER = "groq"
    MODEL = "openai/gpt-oss-120b"
    MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable or pass api_key."
            )

        self.MODEL = model_name

        # 1. Initialize Groq Chat Model
        self.llm = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=self.MAX_TOKENS,
        )

        # 2. Bind the Pydantic schema for strict structured output
        self.structured_llm = self.llm.with_structured_output(ClassificationResult)

        # 3. Define Prompt Template matching your exact hf.py user prompt structure
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Classify the following PIB article for Indian Polity relevance.\n\n"
                    "Use exactly one Laxmikanth chapter when the article is relevant.\n"
                    "If it is not relevant, set chapter_number, topic, and subtopic to null.\n\n"
                    "Keep the reason concise, no more than 25 words.\n"
                    "Return only the requested JSON object.\n\n"
                    "Laxmikanth chapters:\n{chapters}\n\n"
                    "Article title:\n{title}\n\n"
                    "Article text:\n{article_text}"
                ),
            ),
        ])

        # 4. Build LCEL Chain
        self.chain = self.prompt | self.structured_llm

    @staticmethod
    def _chapter_text() -> str:
        return "\n".join(
            f"{number}. {name}"
            for number, name in LAXMIKANTH_8TH_EDITION.items()
        )

    def classify(self, record: Dict[str, Any]) -> ClassificationResult:
        chapters = self._chapter_text()
        title = record.get("title", "")
        article_text = record.get("article_text", "") or record.get("content", "")

        # Invoke the LCEL chain with exact prompt variables matching hf.py
        result: ClassificationResult = self.chain.invoke({
            "chapters": chapters,
            "title": title,
            "article_text": article_text,
        })

        # Ensure timestamp alignment
        if result.classified_at is None:
            result.classified_at = datetime.now(timezone.utc)

        return result

    def classify_batch(
        self, items: List[Dict[str, Any]], max_concurrency: int = 2
    ) -> List[ClassificationResult]:
        """Classify multiple article records in batch mode."""
        formatted_items = [
            {
                "chapters": self._chapter_text(),
                "title": item.get("title", ""),
                "article_text": item.get("article_text", "") or item.get("content", ""),
            }
            for item in items
        ]
        return self.chain.batch(formatted_items, config={"max_concurrency": max_concurrency})
