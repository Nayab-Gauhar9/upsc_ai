import json
from pydantic import BaseModel
from typing import List, Optional
from src.storage.minio_client import MinIOStorage
from dotenv import load_dotenv

load_dotenv()
# Minimal Pydantic schema to mimic your StudyIntelligence model


class PrelimsPoint(BaseModel):
    statement: str
    is_correct: bool
    explanation: str

class StudyIntelligenceMock(BaseModel):
    headline: str
    summary: str
    why_it_matters: str
    backward_linkages: List[str]
    upsc_relevance_mains: str
    prelims_practice_points: List[PrelimsPoint]
    mains_question: str
    smart_notes: List[str]
    generated_at: Optional[str] = "2026-10-03T12:00:00+05:30"

def run_upload_test():
    storage = MinIOStorage()

    test_prid = 9999999
    chapter_name = "Panchayati Raj"
    topic = "Grassroots Decentralisation"

    # Mock synthesized intelligence payload
    payload = StudyIntelligenceMock(
        headline="Integration Test: Empowering Local Self-Government",
        summary="A test run to verify end-to-end MinIO/R2 upload and immediate Telegram broadcast delivery.",
        why_it_matters="Directly exercises Article 243G and real-time dissemination pipeline.",
        backward_linkages=[
            "Article 243 (Definitions)",
            "73rd Constitutional Amendment Act, 1992"
        ],
        upsc_relevance_mains="Governance – Decentralisation; E-Governance.",
        prelims_practice_points=[
            PrelimsPoint(
                statement="Panchayati Raj Institutions are constitutionally protected under Part IX.",
                is_correct=True,
                explanation="The 73rd Constitutional Amendment added Part IX comprising Articles 243 to 243O."
            )
        ],
        mains_question="Examine the technological and administrative bottlenecks hindering genuine decentralization in rural local bodies.",
        smart_notes=[
            "Pipeline verification test point 1: Validating storage upload.",
            "Pipeline verification test point 2: Validating Telegram deep link propagation."
        ]
    )

    print(f"Uploading mock smart note for PRID: {test_prid}...")
    uploaded_key = storage.upload_smart_notes(
        chapter_name=chapter_name,
        topic=topic,
        prid=test_prid,
        study_intelligence=payload
    )

    print(f"\n✅ Uploaded successfully!")
    print(f"Returned Object Key: {uploaded_key}")

    # Verify that the object actually exists in the bucket
    if storage.object_exists(uploaded_key):
        print(f"Verified: {uploaded_key} is present in bucket '{storage.bucket}'.")
        fetched_data = storage.get_json(uploaded_key)
        print(f"Fetched headline from bucket: {fetched_data.get('headline')}")
    else:
        print(f"❌ Failed to verify {uploaded_key} in storage.")

if __name__ == "__main__":
    run_upload_test()
