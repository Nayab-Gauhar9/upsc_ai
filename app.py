from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session
from src.storage.minio_client import MinIOStorage
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION, CHAPTER_BY_TITLE
from src.database.session import SessionLocal
from src.database.models import User, UserQueryHistory, UserChapterProgress, UserNote
from src.rag.retriever import UPSCRetriever
from src.pipelines.rag_generator import UPSCRAGGenerator
from src.cache import get_cached_chapter, set_cached_chapter
from datetime import datetime

storage = MinIOStorage()
SMART_NOTES_PREFIX = "pib/smart_notes/"
app = FastAPI(title="UPSC AI Mentor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health_check():
    return {"status": "ok"}

retriever = UPSCRetriever()
generator = UPSCRAGGenerator(model_name="openai/gpt-oss-120b")

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# ---------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------

class UserSyncRequest(BaseModel):
    firebase_uid: str
    email: EmailStr
    display_name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class GenerateRequest(BaseModel):
    question: str
    top_k: int = 3
    firebase_uid: Optional[str] = None

class ChapterStudyRequest(BaseModel):
    chapter_id: Optional[int] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    firebase_uid: Optional[str] = None

class CompleteChapterRequest(BaseModel):
    firebase_uid: str
    chapter_number: Optional[int] = None
    chapter_title: str

class NoteCreateRequest(BaseModel):
    firebase_uid: str
    title: str
    content: str
    category: Optional[str] = "Polity GS-II"

class NoteUpdateRequest(BaseModel):
    firebase_uid: str
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None

# ---------------------------------------------------------
# User Sync & Search Endpoints
# ---------------------------------------------------------

@app.post("/api/users/sync")
async def sync_firebase_user(payload: UserSyncRequest, db: Session = Depends(get_db)):
    """Creates or updates a user from Firebase Auth."""
    try:
        user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
        if not user:
            user = User(
                firebase_uid=payload.firebase_uid,
                email=payload.email,
                display_name=payload.display_name,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            if payload.display_name and user.display_name != payload.display_name:
                user.display_name = payload.display_name
                db.commit()

        return {
            "status": "success",
            "user_id": user.id,
            "firebase_uid": user.firebase_uid,
            "email": user.email,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/search")
async def search_notes(payload: SearchRequest):
    try:
        results = retriever.retrieve_smart_notes_for_query(payload.query, top_k=payload.top_k)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/generate")
async def generate_answer(payload: GenerateRequest, db: Session = Depends(get_db)):
    try:
        # 1. Vector retrieval from MinIO
        retrieved_articles = retriever.retrieve_smart_notes_for_query(payload.question, top_k=payload.top_k)

        # 2. RAG answer generation
        answer = generator.generate_expert_response(payload.question, retrieved_articles)

        # 3. Log query to database if a user session is present
        if payload.firebase_uid:
            user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
            if user:
                history_entry = UserQueryHistory(
                    user_id=user.id,
                    question=payload.question,
                    model_answer=answer,
                )
                db.add(history_entry)
                db.commit()

        return {
            "status": "success",
            "sources_count": len(retrieved_articles),
            "model_answer": answer,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/history")
async def get_user_history(
    firebase_uid: str, limit: int = 15, db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        return {"status": "success", "history": []}

    history = (
        db.query(UserQueryHistory)
        .filter(UserQueryHistory.user_id == user.id)
        .order_by(UserQueryHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "status": "success",
        "history": [
            {
                "id": h.id,
                "question": h.question,
                "model_answer": h.model_answer,
                "created_at": h.created_at.isoformat() if h.created_at else None,
            }
            for h in history
        ],
    }

# ---------------------------------------------------------
# Laxmikanth Chapter Modules & Progress Endpoints
# ---------------------------------------------------------

@app.get("/api/laxmikanth/chapters")
async def get_laxmikanth_chapters():
    """Returns the ordered list of all 92 chapters from Laxmikanth 8th Edition."""
    chapters = [
        {"id": ch_num, "title": title}
        for ch_num, title in sorted(LAXMIKANTH_8TH_EDITION.items())
    ]
    return {
        "status": "success",
        "total_chapters": len(chapters),
        "chapters": chapters,
    }

@app.post("/api/laxmikanth/complete")
async def mark_chapter_completed(payload: CompleteChapterRequest, db: Session = Depends(get_db)):
    """Marks a chapter as completed in the lightweight progress tracking table."""
    try:
        user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        chapter_num = payload.chapter_number or CHAPTER_BY_TITLE.get(payload.chapter_title.strip())
        if not chapter_num:
            raise HTTPException(status_code=400, detail=f"Invalid chapter title: '{payload.chapter_title}'")

        # Idempotent check
        existing = (
            db.query(UserChapterProgress)
            .filter(
                UserChapterProgress.user_id == user.id,
                UserChapterProgress.chapter_number == chapter_num,
            )
            .first()
        )

        if not existing:
            progress = UserChapterProgress(
                user_id=user.id,
                chapter_number=chapter_num,
                chapter_title=payload.chapter_title.strip(),
            )
            db.add(progress)
            db.commit()

        return {
            "status": "success",
            "message": f"Chapter {chapter_num} marked as completed",
            "chapter_number": chapter_num,
            "chapter_title": payload.chapter_title,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/laxmikanth/progress")
async def get_user_chapter_progress(firebase_uid: str, db: Session = Depends(get_db)):
    """Retrieves list of completed chapter IDs and summary metrics for the user."""
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        return {"status": "success", "completed_count": 0, "completed_chapters": []}

    completed = (
        db.query(UserChapterProgress)
        .filter(UserChapterProgress.user_id == user.id)
        .order_by(UserChapterProgress.chapter_number.asc())
        .all()
    )

    completed_ids = [c.chapter_number for c in completed]

    return {
        "status": "success",
        "total_chapters": len(LAXMIKANTH_8TH_EDITION),
        "completed_count": len(completed),
        "completed_chapter_ids": completed_ids,
        "completed_chapters": [
            {
                "chapter_number": c.chapter_number,
                "chapter_title": c.chapter_title,
                "completed_at": c.completed_at.isoformat(),
            }
            for c in completed
        ],
    }

@app.post("/api/laxmikanth/study")
async def study_laxmikanth_chapter(payload: ChapterStudyRequest):
   
    chapter_id = payload.chapter_id or payload.chapter_number
    if not chapter_id and payload.chapter_title and payload.chapter_title.strip() != "string":
        chapter_id = CHAPTER_BY_TITLE.get(payload.chapter_title.strip())

    if not chapter_id or chapter_id not in LAXMIKANTH_8TH_EDITION:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chapter identifier: ID '{chapter_id}', Title '{payload.chapter_title}'",
        )

    chapter_title = LAXMIKANTH_8TH_EDITION[chapter_id]

    # 2. Check Redis cache first
    cached_payload = get_cached_chapter(chapter_id)
    if cached_payload:
        return {
            "status": "success",
            "source": "cache",
            **cached_payload,
        }

    # 3. Cache Miss: Generate notes via LLM (Points 1 & 3 only)
    try:
        study_notes = generator.generate_chapter_study_notes(
            chapter_title=chapter_title,
            chapter_number=chapter_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # 4. Compute navigation pointers
    prev_id = chapter_id - 1 if chapter_id > 1 else None
    next_id = chapter_id + 1 if chapter_id < len(LAXMIKANTH_8TH_EDITION) else None

    result = {
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "navigation": {
            "prev_chapter_id": prev_id,
            "prev_chapter_title": LAXMIKANTH_8TH_EDITION.get(prev_id) if prev_id else None,
            "next_chapter_id": next_id,
            "next_chapter_title": LAXMIKANTH_8TH_EDITION.get(next_id) if next_id else None,
        },
        "study_notes": study_notes,
    }

    # 5. Store in Redis
    set_cached_chapter(chapter_id, result)

    return {
        "status": "success",
        "source": "generated",
        **result,
    }

@app.get("/api/current-affairs/chapters")
def list_ca_chapters():
    """Lists top-level chapters in MinIO under pib/smart_notes/"""
    try:
        objects = storage.client.list_objects(storage.bucket, prefix=SMART_NOTES_PREFIX, recursive=False)
        chapters = []
        for obj in objects:
            if obj.is_dir:
                folder_name = obj.object_name[len(SMART_NOTES_PREFIX):].strip("/")
                if folder_name:
                    chapters.append({
                        "id": folder_name,
                        "title": folder_name
                    })
        return {
            "status": "success",
            "total": len(chapters),
            "chapters": sorted(chapters, key=lambda x: x["title"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/current-affairs/topics")
def list_ca_topics(chapter: str = Query(..., description="Chapter name")):
    """Lists sub-topics under a selected chapter in MinIO"""
    try:
        prefix = f"{SMART_NOTES_PREFIX}{chapter.strip('/')}/"
        objects = storage.client.list_objects(storage.bucket, prefix=prefix, recursive=False)
        topics = []
        for obj in objects:
            if obj.is_dir:
                topic_name = obj.object_name[len(prefix):].strip("/")
                if topic_name:
                    topics.append({
                        "id": topic_name,
                        "chapter": chapter,
                        "full_path": f"{chapter.strip('/')}/{topic_name}",
                        "title": topic_name
                    })
        return {
            "status": "success",
            "chapter": chapter,
            "total": len(topics),
            "topics": sorted(topics, key=lambda x: x["title"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/current-affairs/notes")
def list_ca_notes(topic_path: str = Query(..., description="e.g. Election Commission/Election Commission")):
    """Lists smart note files inside a topic, extracting the real headline from each note."""
    try:
        # Check both raw path and space/hyphen normalized variants
        prefixes_to_try = [
            f"{SMART_NOTES_PREFIX}{topic_path.strip('/')}/",
            f"{SMART_NOTES_PREFIX}{topic_path.replace('-', ' ').strip('/')}/",
            f"{SMART_NOTES_PREFIX}{topic_path.replace(' ', '-').strip('/')}/"
        ]

        objects = []
        for pref in prefixes_to_try:
            matched = list(storage.client.list_objects(storage.bucket, prefix=pref, recursive=False))
            if matched:
                objects = matched
                break

        notes = []
        for obj in objects:
            if not obj.is_dir and obj.object_name.endswith(".json"):
                filename = obj.object_name.split("/")[-1]
                prid = filename.replace("smart_notes_", "").replace(".json", "")
                
                # Fetch note content to extract actual headline
                real_title = f"PIB Release #{prid}"
                summary_preview = ""
                try:
                    data = storage.get_json(obj.object_name)
                    if isinstance(data, dict):
                        real_title = data.get("headline") or data.get("title") or real_title
                        summary_preview = data.get("summary", "")[:120] + "..." if data.get("summary") else ""
                except Exception:
                    pass

                notes.append({
                    "object_key": obj.object_name,
                    "filename": filename,
                    "prid": prid,
                    "title": real_title,
                    "preview": summary_preview,
                    "size_kb": round(obj.size / 1024, 2)
                })

        return {
            "status": "success",
            "topic_path": topic_path,
            "total": len(notes),
            "notes": sorted(notes, key=lambda x: x["filename"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/current-affairs/note-content")
def get_ca_note_content(object_key: str = Query(..., description="MinIO object key")):
    """
    Reads a PIB UPSC smart note from MinIO and converts all 8 intelligence
    dimensions into a comprehensive, beautifully styled Markdown document.
    """
    try:
        raw_data = storage.get_json(object_key)

        if not isinstance(raw_data, dict):
            return {
                "status": "success",
                "title": "Smart Note",
                "object_key": object_key,
                "markdown": str(raw_data)
            }

        headline = raw_data.get("headline") or raw_data.get("title") or "PIB Smart Study Note"
        
        # Format timestamp if present
        meta_date = ""
        if "generated_at" in raw_data and raw_data["generated_at"]:
            try:
                dt = datetime.fromisoformat(str(raw_data["generated_at"]))
                meta_date = f" *| Generated: {dt.strftime('%d %B %Y')}*"
            except Exception:
                meta_date = f" *| {raw_data['generated_at']}*"

        md: list[str] = [
            f"# {headline}",
            f"> PIB UPSC Intelligence Note{meta_date}\n",
            "---"
        ]

        # 1. Executive Summary
        if raw_data.get("summary"):
            md.append("## 📌 Executive Summary")
            md.append(f"{raw_data['summary'].strip()}\n")

        # 2. Why It Matters
        if raw_data.get("why_it_matters"):
            md.append("## 💡 Why It Matters for UPSC")
            md.append(f"{raw_data['why_it_matters'].strip()}\n")

        # 3. Smart Key Notes (High-Yield Bullets)
        smart_notes = raw_data.get("smart_notes")
        if isinstance(smart_notes, list) and smart_notes:
            md.append("## 📝 Smart Notes (Key Exam Highlights)")
            for pt in smart_notes:
                md.append(f"* {str(pt).strip()}")
            md.append("")

        # 4. Backward Linkages (Static Polity & Laxmikanth Foundation)
        linkages = raw_data.get("backward_linkages")
        if isinstance(linkages, list) and linkages:
            md.append("## 🔗 Backward Linkages (Static Syllabus & Case Laws)")
            for item in linkages:
                clean_item = str(item).replace("⚑", "—").strip()
                md.append(f"* {clean_item}")
            md.append("")

        # 5. Mains Relevance Syllabus Mapping
        mains_tags = raw_data.get("upsc_relevance_mains")
        if mains_tags:
            md.append("## 🎯 UPSC Mains Syllabus Mapping")
            md.append(f"Keywords / Themes: {str(mains_tags).strip()}\n")

        # 6. Prelims Practice Statements & Explanations
        prelims = raw_data.get("prelims_practice_points")
        if isinstance(prelims, list) and prelims:
            md.append("## 🧭 Prelims Practice Statements")
            for idx, item in enumerate(prelims, start=1):
                if isinstance(item, dict):
                    stmt = item.get("statement", "").strip()
                    correct = item.get("is_correct")
                    exp = item.get("explanation", "").strip()
                    badge = "✅ Correct" if correct is True else ("❌ Incorrect" if correct is False else "")

                    md.append(f"Statement {idx}: {stmt}")
                    if badge:
                        md.append(f"> Verdict: {badge}")
                    if exp:
                        md.append(f"> Explanation: {exp}")
                    md.append("")
                else:
                    md.append(f"* {str(item).strip()}")

        # 7. Mains Practice Question
        if raw_data.get("mains_question"):
            md.append("## ✍️ Mains Analytical Practice Question")
            md.append(f"> *\"{raw_data['mains_question'].strip()}\"*\n")

        return {
                    "status": "success",
                    "title": headline,
                    "object_key": object_key,
                    "markdown": "\n".join(md)
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notes")
async def get_user_notes(firebase_uid: str, db: Session = Depends(get_db)):
    """Fetch all personal notes for an authenticated student."""
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        return {"status": "success", "total": 0, "notes": []}

    notes = (
        db.query(UserNote)
        .filter(UserNote.user_id == user.id)
        .order_by(UserNote.updated_at.desc())
        .all()
    )

    return {
        "status": "success",
        "total": len(notes),
        "notes": [
            {
                "id": n.id,
                "title": n.title,
                "content": n.content,
                "category": n.category,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "updated_at": n.updated_at.isoformat() if n.updated_at else None,
            }
            for n in notes
        ],
    }

@app.post("/api/notes")
async def create_user_note(payload: NoteCreateRequest, db: Session = Depends(get_db)):
    """Create a new personal note."""
    user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    note = UserNote(
        user_id=user.id,
        title=payload.title.strip() or "Untitled Note",
        content=payload.content,
        category=payload.category.strip() if payload.category else "Polity GS-II",
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return {
        "status": "success",
        "note": {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "category": note.category,
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat(),
        },
    }

@app.put("/api/notes/{note_id}")
async def update_user_note(
    note_id: int, payload: NoteUpdateRequest, db: Session = Depends(get_db)
):
    """Update an existing note."""
    user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    note = db.query(UserNote).filter(UserNote.id == note_id, UserNote.user_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if payload.title is not None:
        note.title = payload.title.strip()
    if payload.content is not None:
        note.content = payload.content
    if payload.category is not None:
        note.category = payload.category.strip()

    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)

    return {
        "status": "success",
        "note": {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "category": note.category,
            "updated_at": note.updated_at.isoformat(),
        },
    }

@app.delete("/api/notes/{note_id}")
async def delete_user_note(
    note_id: int, firebase_uid: str = Query(...), db: Session = Depends(get_db)
):
    """Delete a note belonging to the student."""
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    note = db.query(UserNote).filter(UserNote.id == note_id, UserNote.user_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()

    return {"status": "success", "message": f"Note {note_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
