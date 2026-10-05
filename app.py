from fastapi import FastAPI, HTTPException, Depends, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session
from src.storage.minio_client import MinIOStorage
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION, CHAPTER_BY_TITLE
from src.database.session import SessionLocal
from src.database.models import User, UserQueryHistory, UserChapterProgress, UserNote, IssueDossierModel, IssueGraphEdgeModel
from src.rag.retriever import UPSCRetriever
from src.pipelines.rag_generator import UPSCRAGGenerator
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from src.cache import get_cached_chapter, get_cached_json, set_cached_chapter, set_cached_json
from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

storage = MinIOStorage()

retriever = UPSCRetriever()
generator = UPSCRAGGenerator(model_name="openai/gpt-oss-120b")
SMART_NOTES_PREFIX = "pib/smart_notes/"

def fetch_quality_gated_dossiers(
    query_str: str,
    db: Session,
    min_keyword_match_ratio: float = 0.65,
    max_dossiers: int = 3
) -> list:
    """
    Dynamically fetches dossiers only if they cross the relevance threshold.
    Incorporates more if relevant, ignores if low match.
    """
    stopwords = {"what", "when", "where", "which", "with", "from", "that", "this", "explain", "analyze", "evaluate", "discuss"}
    query_terms = [
        w.lower().strip() for w in query_str.split() 
        if len(w) > 3 and w.lower().strip() not in stopwords
    ]
    if not query_terms:
        return []

    # Build SQL ILIKE clauses across terms
    conditions = []
    for term in query_terms:
        pattern = f"%{term}%"
        conditions.append(IssueDossierModel.issue_slug.ilike(pattern))
        conditions.append(IssueDossierModel.title.ilike(pattern))
        conditions.append(IssueDossierModel.canonical_summary.ilike(pattern))

    stmt = (
        select(IssueDossierModel)
        .where(or_(*conditions))
        .options(selectinload(IssueDossierModel.edges))
        .order_by(IssueDossierModel.last_updated_at.desc())
        .limit(10)
    )

    candidates = db.scalars(stmt).all()
    scored_dossiers = []

    for d in candidates:
        text_corpus = f"{d.issue_slug} {d.title} {d.canonical_summary or ''}".lower()
        matched_terms = [t for t in query_terms if t in text_corpus]
        match_ratio = len(matched_terms) / len(query_terms)

        # STRICT QUALITY FILTER: Ignore if below threshold
        if match_ratio >= min_keyword_match_ratio:
            scored_dossiers.append((match_ratio, d))

    # Sort descending by match quality
    scored_dossiers.sort(key=lambda x: x[0], reverse=True)

    # Format selected high-quality dossiers
    selected_contexts = []
    for score, d in scored_dossiers[:max_dossiers]:
        perspectives = [
            f"[{e.publication.upper()} | {e.event_date}]: {e.core_claim}"
            for e in d.edges[:4]
        ]
        body_dim = d.mains_framework.get("body_dimensions", "") if d.mains_framework else ""
        way_forward = d.mains_framework.get("way_forward_reforms", "") if d.mains_framework else ""

        content_text = f"""
--- LIVING CONSTITUTIONAL DOSSIER: {d.title} (Match Confidence: {round(score * 100)}%) ---
Chapter: {d.chapter_name} | Status: {d.status.upper()} | Perspectives Synthesized: {d.article_count}
Canonical Synthesis:
{d.canonical_summary}

Dialectical Dimensions:
{body_dim}

Key Evolutionary Perspectives:
{chr(10).join(perspectives)}

Way Forward:
{way_forward}
------------------------------------------------
"""
        selected_contexts.append({
            "prid": d.issue_slug,  # Fixes KeyError: 'prid'
            "headline": f"Master Topic Dossier: {d.title}",
            "title": d.title,
            "summary": d.canonical_summary,
            "content": content_text,
            "text": content_text,  # Fallback for generators accessing article['text']
            "source": "Evolving Topic Dossier (Multi-Source)",
            "source_type": "dossier",
            "chapter": d.chapter_name,
            "issue_slug": d.issue_slug,
            "relevance_score": score,
        })

    return selected_contexts

def filter_quality_pib_notes(notes: list, min_score: float = 0.45) -> list:
    """Drops noise and retains solid topical PIB matches."""
    if not notes:
        return []
    
    quality_notes = []
    for item in notes:
        # Check all possible score keys including similarity_score
        score = (
            item.get("similarity_score")
            or item.get("score")
            or item.get("similarity")
            or item.get("distance_score")
        )
        
        # If score is present, enforce the 0.45 threshold
        if score is not None:
            if float(score) >= min_score:
                quality_notes.append(item)
        else:
            # Fallback if no score was attached
            quality_notes.append(item)
            
    return quality_notes
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)

# Initialize Limiter
limiter = Limiter(key_func=get_client_ip)

app = FastAPI(
    title="UPSC AI Mentor API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

security = HTTPBasic()

DOCS_USERNAME = os.getenv("DOCS_USERNAME", "admin")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "")

def verify_docs_access(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, DOCS_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DOCS_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# 2. Authenticated Swagger UI route
@app.get("/docs", include_in_schema=False)
async def get_documentation(auth: None = Depends(verify_docs_access)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Polity Mentor API Docs")

# 3. Authenticated OpenAPI schema route
@app.get("/openapi.json", include_in_schema=False)
async def openapi_endpoint(auth: None = Depends(verify_docs_access)):
    return get_openapi(title="UPSC AI Mentor API", version="1.0.0", routes=app.routes)

# Register Limiter on App State and Error Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


ALLOWED_ORIGINS = [
    "https://politymentor.com",
    "https://www.politymentor.com",
    "https://app.flutterflow.io",
    "https://preview.flutterflow.io",
]

ORIGIN_REGEX = (
    r"^(https://.*\.ngrok-free\.app"
    r"|https://.*\.ngrok-free\.dev"
    r"|https://.*\.ngrok\.io"
    r"|https://.*\.onrender\.com"
    r"|http://localhost(:\d+)?"
    r"|http://127\.0\.0\.1(:\d+)?)$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}

@app.get("/api/notes/latest")
def get_latest_note():
    cache_key = "ca:notes:latest"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

    try:
        notes = retriever.storage.list_latest_smart_notes(limit=1)
        if not notes:
            return {"status": "empty", "note": None}
        
        response = {
            "status": "success",
            "note": notes[0]
        }
        set_cached_json(cache_key, response, ex_seconds=600)  # 10 minutes
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    firebase_uid:str

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
async def search_notes(payload: SearchRequest, db: Session = Depends(get_db)):
  try:
    dossiers = fetch_quality_gated_dossiers(query_str=payload.query, db=db, limit=2)
    notes = retriever.retrieve_smart_notes_for_query(
        payload.query, top_k=payload.top_k
    )
    return {
        "status": "success",
        "dossiers": dossiers,
        "data": notes,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/generate")
async def generate_answer(
    payload: GenerateRequest, db: Session = Depends(get_db)
):
  try:
    # 1. Quality-gated retrieval of master dossiers (SQL macro anchor)
    dossier_contexts = fetch_quality_gated_dossiers(
        query_str=payload.question,
        db=db,
        min_keyword_match_ratio=0.65,
        max_dossiers=3,
    )

    # 2. Retrieve candidate PIB notes from pgvector + MinIO
    candidate_notes = retriever.retrieve_smart_notes_for_query(
        payload.question, top_k=max(payload.top_k, 5)
    )

    # 3. Filter PIB notes using calibrated 0.45 threshold (replaces 0.65)
    quality_pib_notes = filter_quality_pib_notes(
        candidate_notes, min_score=0.45
    )

    # 4. Dynamic Blending
    blended_sources = []
    if dossier_contexts:
      blended_sources.extend(dossier_contexts)
      blended_sources.extend(quality_pib_notes[:2])
    else:
      blended_sources.extend(quality_pib_notes[: payload.top_k])

    # 5. Guard against empty context
    if not blended_sources:
      return {
          "status": "success",
          "sources_count": 0,
          "has_dossier_anchor": False,
          "references": [],
          "model_answer": (
              "No directly relevant current affairs dossiers or PIB releases met"
              " the confidence threshold for this query. Please refine your"
              " query or specify a constitutional provision."
          ),
      }

    # 6. Generate answer using strictly high-relevance sources
    answer = generator.generate_expert_response(
        payload.question, blended_sources
    )

    # 7. Log query history
    if payload.firebase_uid:
      user = (
          db.query(User)
          .filter(User.firebase_uid == payload.firebase_uid)
          .first()
      )
      if user:
        history_entry = UserQueryHistory(
            user_id=user.id,
            question=payload.question,
            model_answer=answer,
        )
        db.add(history_entry)
        db.commit()

    # 8. Resolve real titles and confidence percentages for references
    references = []
    for s in blended_sources:
      smart_note_data = (
          s.get("smart_notes") if isinstance(s.get("smart_notes"), dict) else {}
      )

      resolved_title = (
          s.get("headline")
          or s.get("title")
          or smart_note_data.get("headline")
          or smart_note_data.get("title")
          or f"PIB Release #{s.get('prid', '')}"
      )

      # Extract similarity or confidence
      conf_val = s.get("confidence_percentage")
      if conf_val is None:
        if s.get("similarity_score") is not None:
          conf_val = round(float(s["similarity_score"]) * 100, 2)
        elif s.get("relevance_score") is not None:
          conf_val = round(float(s["relevance_score"]) * 100, 2)

      references.append({
          "title": resolved_title,
          "type": s.get("source_type", "article"),
          "source": s.get("source") or s.get("publication") or "PIB Note",
          "slug": s.get("issue_slug"),
          "prid": s.get("prid"),
          "confidence": f"{conf_val}%" if conf_val is not None else None,
      })

    return {
        "status": "success",
        "sources_count": len(blended_sources),
        "has_dossier_anchor": len(dossier_contexts) > 0,
        "references": references,
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
    cache_key = "laxmikanth:chapters_list"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

    chapters = [
        {"id": ch_num, "title": title}
        for ch_num, title in sorted(LAXMIKANTH_8TH_EDITION.items())
    ]
    response = {
        "status": "success",
        "total_chapters": len(chapters),
        "chapters": chapters,
    }
    set_cached_json(cache_key, response, ex_seconds=2592000)  # 30 days
    return response

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
@limiter.limit("10/minute")
async def study_laxmikanth_chapter(request:Request,payload: ChapterStudyRequest):
   
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
    cache_key = "ca:chapters"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

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
        response = {
            "status": "success",
            "total": len(chapters),
            "chapters": sorted(chapters, key=lambda x: x["title"])
        }
        set_cached_json(cache_key, response, ex_seconds=7200)  # 2 hours
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/current-affairs/topics")
def list_ca_topics(chapter: str = Query(..., description="Chapter name")):
    """Lists sub-topics under a selected chapter in MinIO"""
    cache_key = f"ca:topics:{chapter.strip('/')}"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

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
        response = {
            "status": "success",
            "chapter": chapter,
            "total": len(topics),
            "topics": sorted(topics, key=lambda x: x["title"])
        }
        set_cached_json(cache_key, response, ex_seconds=7200)  # 2 hours
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/current-affairs/notes")
def list_ca_notes(topic_path: str = Query(..., description="e.g. Election Commission/Election Commission")):
    """Lists smart note files inside a topic, extracting the real headline from each note."""
    cache_key = f"ca:notes:{topic_path.strip('/')}"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

    try:
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

        response = {
            "status": "success",
            "topic_path": topic_path,
            "total": len(notes),
            "notes": sorted(notes, key=lambda x: x["filename"], reverse=True)
        }
        set_cached_json(cache_key, response, ex_seconds=3600)  # 1 hour
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/current-affairs/note-content")
def get_ca_note_content(object_key: str = Query(..., description="MinIO object key")):
    """
    Reads a PIB UPSC smart note from MinIO and converts all 8 intelligence
    dimensions into a comprehensive, beautifully styled Markdown document.
    """
    cache_key = f"ca:content:{object_key}"
    cached = get_cached_json(cache_key)
    if cached:
        return cached

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

        if raw_data.get("summary"):
            md.append("## 📌 Executive Summary")
            md.append(f"{raw_data['summary'].strip()}\n")

        if raw_data.get("why_it_matters"):
            md.append("## 💡 Why It Matters for UPSC")
            md.append(f"{raw_data['why_it_matters'].strip()}\n")

        smart_notes = raw_data.get("smart_notes")
        if isinstance(smart_notes, list) and smart_notes:
            md.append("## 📝 Smart Notes (Key Exam Highlights)")
            for pt in smart_notes:
                md.append(f"* {str(pt).strip()}")
            md.append("")

        linkages = raw_data.get("backward_linkages")
        if isinstance(linkages, list) and linkages:
            md.append("## 🔗 Backward Linkages (Static Syllabus & Case Laws)")
            for item in linkages:
                clean_item = str(item).replace("⚑", "—").strip()
                md.append(f"* {clean_item}")
            md.append("")

        mains_tags = raw_data.get("upsc_relevance_mains")
        if mains_tags:
            md.append("## 🎯 UPSC Mains Syllabus Mapping")
            md.append(f"Keywords / Themes: {str(mains_tags).strip()}\n")

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

        if raw_data.get("mains_question"):
            md.append("## ✍️ Mains Analytical Practice Question")
            md.append(f"> *\"{raw_data['mains_question'].strip()}\"*\n")

        response = {
            "status": "success",
            "title": headline,
            "object_key": object_key,
            "markdown": "\n".join(md)
        }
        set_cached_json(cache_key, response, ex_seconds=604800)  # 7 days
        return response
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


class SaveSmartNoteToWorkspaceRequest(BaseModel):
    firebase_uid: str
    object_key: str
    category: Optional[str] = "Current Affairs GS-II"
    custom_reflection: Optional[str] = None

@app.post("/api/notes/save-smart-note")
async def save_smart_note_to_workspace(
    payload: SaveSmartNoteToWorkspaceRequest, 
    db: Session = Depends(get_db)
):
    """
    Imports a PIB Smart Note from MinIO directly into the user's personal UserNote list.
    """
    # 1. Verify user exists
    user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Fetch the smart note content from MinIO
    try:
        raw_data = storage.get_json(payload.object_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load smart note from storage: {e}")

    headline = raw_data.get("headline") or raw_data.get("title") or "PIB Smart Note"

    # 3. Check if the user already saved this note (by title match)
    existing = (
        db.query(UserNote)
        .filter(UserNote.user_id == user.id, UserNote.title == headline)
        .first()
    )
    if existing:
        return {
            "status": "success",
            "message": "Note already exists in your workspace",
            "note_id": existing.id
        }

    # 4. Generate structured markdown note content
    note_lines = [
        f"# {headline}\n",
        f"Source: PIB Reference | {payload.object_key}\n"
    ]

    if payload.custom_reflection:
        note_lines.append(f"### 💡 My Revision Notes / Reflections\n{payload.custom_reflection}\n")

    if raw_data.get("summary"):
        note_lines.append(f"### 📌 Executive Summary\n{raw_data['summary']}\n")

    smart_notes = raw_data.get("smart_notes")
    if isinstance(smart_notes, list) and smart_notes:
        note_lines.append("### 📝 Key Analysis & Takeaways")
        for pt in smart_notes:
            note_lines.append(f"* {pt}")
        note_lines.append("")

    linkages = raw_data.get("backward_linkages")
    if isinstance(linkages, list) and linkages:
        note_lines.append("### 🔗 Backward Linkages (Static Polity & Cases)")
        for item in linkages:
            note_lines.append(f"* {item}")
        note_lines.append("")

    if raw_data.get("mains_question"):
        note_lines.append(f"### ✍️ Mains Analytical Question\n> {raw_data['mains_question']}\n")

    formatted_content = "\n".join(note_lines)

    # 5. Insert into existing UserNote table
    new_note = UserNote(
        user_id=user.id,
        title=headline,
        content=formatted_content,
        category=payload.category or "Current Affairs GS-II",
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    return {
        "status": "success",
        "message": "Saved to personal workspace",
        "note_id": new_note.id
    }

from fastapi.responses import StreamingResponse
import urllib.parse
from src.services.pdf_generator import generate_pdf_from_markdown

@app.get("/api/current-affairs/download-pdf")
async def download_smart_note_pdf(object_key: str = Query(..., description="MinIO object key")):
    """Generates and streams a downloadable PDF for any smart note directly from MinIO."""
    # Reuse your existing get_ca_note_content logic
    note_data = get_ca_note_content(object_key=object_key)
    title = note_data.get("title", "UPSC_Smart_Note")
    markdown_text = note_data.get("markdown", "")

    pdf_stream = generate_pdf_from_markdown(title=title, markdown_content=markdown_text)

    safe_filename = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    encoded_filename = urllib.parse.quote(f"{safe_filename[:40]}.pdf")

    return StreamingResponse(
        pdf_stream,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}"
        }
    )

@app.get("/api/notes/{note_id}/download-pdf")
async def download_user_note_pdf(
    note_id: int, 
    firebase_uid: str = Query(...), 
    db: Session = Depends(get_db)
):
    """Generates and streams a downloadable PDF from a saved UserNote."""
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    note = db.query(UserNote).filter(UserNote.id == note_id, UserNote.user_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    pdf_stream = generate_pdf_from_markdown(title=note.title, markdown_content=note.content)

    safe_filename = "".join(c for c in note.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    encoded_filename = urllib.parse.quote(f"{safe_filename[:40]}.pdf")

    return StreamingResponse(
        pdf_stream,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

