
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.analyser.study_schema import StudyIntelligenceResult

load_dotenv()


class UPSCStudySynthesizer:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.2):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        self.llm = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=3072,
        )
        self.structured_llm = self.llm.with_structured_output(StudyIntelligenceResult)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a top-tier UPSC Civil Services mentor and senior faculty member. "
                    "Your task is to transform current affairs releases into rigorous exam intelligence. "
                    "Connect events directly to constitutional articles, static syllabus concepts (M. Laxmikanth), "
                    "and provide high-yield smart notes, backward linkages, and active recall angles."
                ),
            ),
            (
                "human",
                (
                    "Analyze this classified PIB article and generate UPSC study intelligence.\n\n"
                    "Chapter Name: {chapter_name}\n"
                    "Topic Name: {topic}\n"
                    "Article Title: {title}\n\n"
                    "Article Text:\n{article_text}"
                ),
            ),
        ])

        self.chain = self.prompt | self.structured_llm

    def synthesize(self, chapter_name: str, topic: str, record: Dict[str, Any]) -> StudyIntelligenceResult:
        return self.chain.invoke({
            "chapter_name": chapter_name,
            "topic": topic,
            "title": record.get("title", ""),
            "article_text": record.get("article_text", "") or record.get("content", ""),
        })
