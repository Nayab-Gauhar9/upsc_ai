from fastapi import FastAPI, HTTPException, Depends, Query, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session
from src.storage.minio_client import MinIOStorage
from src.classifiers.taxonomy import LAXMIKANTH_8TH_EDITION, CHAPTER_BY_TITLE
from src.database.session import SessionLocal
from src.database.models import User, UserQueryHistory, UserChapterProgress, UserNote, IssueDossierModel, IssueGraphEdgeModel, UniqueConstraint, UserSession
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
from sqlalchemy import or_, select, func
from sqlalchemy.orm import selectinload
import time
from src.cache import delete_cached_key, delete_cached_pattern
import urllib.parse
from fastapi.responses import StreamingResponse
from src.services.pdf_generator import generate_pdf_from_markdown

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
    "https://admin.politymentor.com",           # <--- Custom admin portal domain
    "https://polity-admin-portal.pages.dev",
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
def get_latest_note(response: Response):
    start_time = time.perf_counter()
    cache_key = "ca:notes:latest"
    cached = get_cached_json(cache_key)
    
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=600, stale-while-revalidate=30"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    try:
        notes = retriever.storage.list_latest_smart_notes(limit=1)
        if not notes:
            return {"status": "empty", "note": None}
        
        result_payload = {
            "status": "success",
            "note": notes[0]
        }
        set_cached_json(cache_key, result_payload, ex_seconds=600)  # 10 minutes
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=600, stale-while-revalidate=30"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return result_payload
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

class UserSessionEventRequest(BaseModel):
    firebase_uid: str
    display_name: Optional[str] = None
    event_type: str = "LOGOUT"

from datetime import datetime, timezone, timedelta

@app.post("/api/users/sync")
async def sync_firebase_user(
    payload: UserSyncRequest, 
    request: Request, 
    db: Session = Depends(get_db)
):
    """
    Creates or updates a user from Firebase Auth with dual UID/Email lookup
    and logs a LOGIN session event in user_sessions idempotently without
    resetting active user sessions on page re-navigation.
    """
    clean_uid = payload.firebase_uid.strip() if payload.firebase_uid else None
    clean_email = payload.email.lower().strip() if payload.email else None
    clean_display = payload.display_name.strip() if payload.display_name else None

    if not clean_uid or not clean_email:
        raise HTTPException(
            status_code=400,
            detail="Both 'firebase_uid' and 'email' are mandatory."
        )

    try:
        user = (
            db.query(User)
            .filter(
                or_(
                    User.firebase_uid == clean_uid,
                    func.lower(User.email) == clean_email
                )
            )
            .first()
        )

        if not user:
            user = User(
                firebase_uid=clean_uid,
                email=clean_email,
                display_name=clean_display or clean_email.split("@")[0],
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            has_changes = False
            if user.firebase_uid != clean_uid:
                user.firebase_uid = clean_uid
                has_changes = True
            if user.email.lower() != clean_email:
                user.email = clean_email
                has_changes = True
            if clean_display and user.display_name != clean_display:
                user.display_name = clean_display
                has_changes = True

            if has_changes:
                db.commit()
                db.refresh(user)

        # Idempotent Session Login Tracking
        now_utc = datetime.now(timezone.utc)
        latest_session = (
            db.query(UserSession)
            .filter(UserSession.user_id == user.id)
            .order_by(UserSession.created_at.desc())
            .first()
        )

        should_log_login = False

        if not latest_session:
            # First time user ever logs in
            should_log_login = True
        elif latest_session.event_type == "LOGOUT":
            # User previously logged out, this is a fresh login
            should_log_login = True
        else:
            # Last event was LOGIN. Check inactivity window (30 minutes)
            session_time = latest_session.created_at
            if session_time.tzinfo is None:
                session_time = session_time.replace(tzinfo=timezone.utc)

            # If user was active over 30 mins ago without logging out, treat as new session
            if (now_utc - session_time) > timedelta(minutes=30):
                should_log_login = True

        if should_log_login:
            session_event = UserSession(
                user_id=user.id,
                firebase_uid=clean_uid,
                display_name=user.display_name,
                event_type="LOGIN",
                duration_seconds=None,
                ip_address=get_client_ip(request),
                user_agent=request.headers.get("User-Agent", "")[:250],
                created_at=now_utc,
            )
            db.add(session_event)
            db.commit()

        return {
            "status": "success",
            "user_id": user.id,
            "firebase_uid": user.firebase_uid,
            "email": user.email,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database user sync failed: {str(e)}")


from datetime import datetime, timezone

@app.post("/api/users/logout")
async def log_user_logout(
    payload: UserSessionEventRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    clean_uid = payload.firebase_uid.strip() if payload.firebase_uid else None
    if not clean_uid:
        raise HTTPException(status_code=400, detail="firebase_uid is required.")

    user = db.query(User).filter(User.firebase_uid == clean_uid).first()
    if not user:
        return {"status": "ignored", "detail": "User not found"}

    # Use timezone-aware UTC current time
    now_utc = datetime.now(timezone.utc)

    # Find the most recent LOGIN session for this user
    latest_login = (
        db.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.event_type == "LOGIN"
        )
        .order_by(UserSession.created_at.desc())
        .first()
    )

    duration = None
    if latest_login and latest_login.created_at:
        login_time = latest_login.created_at
        # Ensure login_time is timezone-aware
        if login_time.tzinfo is None:
            login_time = login_time.replace(tzinfo=timezone.utc)
        
        elapsed = int((now_utc - login_time).total_seconds())
        duration = max(0, elapsed)

    session_event = UserSession(
        user_id=user.id,
        firebase_uid=clean_uid,
        display_name=payload.display_name or user.display_name,
        event_type="LOGOUT",
        duration_seconds=duration,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent", "")[:250],
        created_at=now_utc,
    )
    db.add(session_event)
    db.commit()

    return {
        "status": "success",
        "event_type": "LOGOUT",
        "duration_seconds": duration,
        "logged_out_at": session_event.created_at.isoformat()
    }

@app.get("/api/admin/users")
def get_admin_users_overview(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Administrative overview: returns registered aspirants, recent login sessions,
    and chapter completion progress counts.
    """
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_notes = db.query(func.count(UserNote.id)).scalar() or 0
    total_queries = db.query(func.count(UserQueryHistory.id)).scalar() or 0

    users = (
        db.query(User)
        .order_by(User.created_at.desc() if hasattr(User, 'created_at') else User.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    user_list = []
    for u in users:
        # Get count of completed chapters
        completed_count = (
            db.query(func.count(UserChapterProgress.id))
            .filter(UserChapterProgress.user_id == u.id)
            .scalar() or 0
        )
        
        # Get latest session event
        latest_sess = (
            db.query(UserSession)
            .filter(UserSession.user_id == u.id)
            .order_by(UserSession.created_at.desc())
            .first()
        )

        user_list.append({
            "id": u.id,
            "firebase_uid": u.firebase_uid,
            "email": u.email,
            "display_name": u.display_name,
            "chapters_completed": completed_count,
            "last_active": latest_sess.created_at.isoformat() if latest_sess else None,
            "last_event": latest_sess.event_type if latest_sess else "NONE",
            "last_ip": latest_sess.ip_address if latest_sess else None,
        })

    return {
        "status": "success",
        "stats": {
            "total_users": total_users,
            "total_notes": total_notes,
            "total_ai_queries": total_queries,
        },
        "users": user_list
    }


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
async def get_laxmikanth_chapters(response: Response):
    start_time = time.perf_counter()
    cache_key = "laxmikanth:chapters_list"
    cached = get_cached_json(cache_key)
    
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=2592000, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    chapters = [
        {"id": ch_num, "title": title}
        for ch_num, title in sorted(LAXMIKANTH_8TH_EDITION.items())
    ]
    result_payload = {
        "status": "success",
        "total_chapters": len(chapters),
        "chapters": chapters,
    }
    set_cached_json(cache_key, result_payload, ex_seconds=2592000)  # 30 days
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=2592000, stale-while-revalidate=86400"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return result_payload


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
@limiter.limit("30/minute")
async def study_laxmikanth_chapter(request: Request, payload: ChapterStudyRequest):
    chapter_id = payload.chapter_id or payload.chapter_number
    if not chapter_id and payload.chapter_title and payload.chapter_title.strip() != "string":
        chapter_id = CHAPTER_BY_TITLE.get(payload.chapter_title.strip())

    if not chapter_id or chapter_id not in LAXMIKANTH_8TH_EDITION:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chapter identifier: ID '{chapter_id}', Title '{payload.chapter_title}'",
        )

    chapter_title = LAXMIKANTH_8TH_EDITION[chapter_id]
    minio_key = f"static/laxmikanth/chapter_{chapter_id:02d}.json"
    cache_key = f"chapter:static:v1:{chapter_id}"

    # 1. Redis Cache Check
    cached = get_cached_json(cache_key)
    if cached:
        return {"status": "success", "source": "cache", **cached}

    # 2. Fetch from MinIO (upsc-ai)
    try:
        raw_data = storage.get_json(minio_key)
    except Exception:
        raise HTTPException(
            status_code=404, 
            detail=f"Chapter {chapter_id} not found in MinIO. Please run the seed script."
        )

    # 3. Navigation
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
        "study_notes": raw_data.get("markdown", ""),
    }

    set_cached_json(cache_key, result, ex_seconds=2592000)
    return {"status": "success", "source": "minio_store", **result}


@app.get("/api/current-affairs/chapters")
def list_ca_chapters(response: Response):
    """Lists top-level chapters in MinIO under pib/smart_notes/"""
    start_time = time.perf_counter()
    cache_key = "ca:chapters"
    cached = get_cached_json(cache_key)
    
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=7200, stale-while-revalidate=120"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    try:
        objects = storage.client.list_objects(storage.bucket, prefix=SMART_NOTES_PREFIX, recursive=False)
        chapters = []
        for obj in objects:
            if obj.is_dir:
                folder_name = obj.object_name[len(SMART_NOTES_PREFIX):].strip("/")
                if folder_name:
                    chapters.append({"id": folder_name, "title": folder_name})
        result_payload = {
            "status": "success",
            "total": len(chapters),
            "chapters": sorted(chapters, key=lambda x: x["title"])
        }
        set_cached_json(cache_key, result_payload, ex_seconds=7200)  # 2 hours
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=7200, stale-while-revalidate=120"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return result_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/current-affairs/topics")
def list_ca_topics(
    response: Response,
    chapter: str = Query(..., description="Chapter name")
):
    """Lists sub-topics under a selected chapter in MinIO"""
    start_time = time.perf_counter()
    clean_chapter = chapter.strip('/')
    cache_key = f"ca:topics:{clean_chapter}"
    cached = get_cached_json(cache_key)
    
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=7200, stale-while-revalidate=120"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    try:
        prefix = f"{SMART_NOTES_PREFIX}{clean_chapter}/"
        objects = storage.client.list_objects(storage.bucket, prefix=prefix, recursive=False)
        topics = []
        for obj in objects:
            if obj.is_dir:
                topic_name = obj.object_name[len(prefix):].strip("/")
                if topic_name:
                    topics.append({
                        "id": topic_name,
                        "chapter": chapter,
                        "full_path": f"{clean_chapter}/{topic_name}",
                        "title": topic_name
                    })
        result_payload = {
            "status": "success",
            "chapter": chapter,
            "total": len(topics),
            "topics": sorted(topics, key=lambda x: x["title"])
        }
        set_cached_json(cache_key, result_payload, ex_seconds=7200)  # 2 hours
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=7200, stale-while-revalidate=120"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return result_payload
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

from fastapi import Response
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

@app.get("/sitemap.xml", include_in_schema=True)
async def generate_sitemap(db: Session = Depends(get_db)):
    """
    Dynamically generates an XML sitemap pulling real-time slugs from
    PostgreSQL dossiers and core constitutional foundation chapters.
    """
    cache_key = "seo:sitemap:xml"
    cached = get_cached_json(cache_key)
    
    headers = {
        "Cache-Control": "public, max-age=3600, s-maxage=43200, stale-while-revalidate=3600",
        "X-Cache": "HIT" if (cached and "xml" in cached) else "MISS"
    }

    if cached and "xml" in cached:
        return Response(content=cached["xml"], media_type="application/xml", headers=headers)

    base_url = "https://politymentor.com"
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

    def add_url(loc: str, priority: str = "0.8", changefreq: str = "weekly", lastmod: str = today_str):
        url_elem = ET.SubElement(urlset, "url")
        ET.SubElement(url_elem, "loc").text = loc
        ET.SubElement(url_elem, "lastmod").text = lastmod
        ET.SubElement(url_elem, "changefreq").text = changefreq
        ET.SubElement(url_elem, "priority").text = priority

    # 1. Main Navigation & Public Landing Pages
    add_url(f"{base_url}/", priority="1.0", changefreq="daily")
    add_url(f"{base_url}/prelims-pyqs", priority="0.9", changefreq="weekly")
    add_url(f"{base_url}/syllabus", priority="0.8", changefreq="monthly")

    # 2. Static Constitutional Foundations
    for ch_num, title in LAXMIKANTH_8TH_EDITION.items():
        clean_slug = "".join(c.lower() if c.isalnum() or c == " " else "" for c in title).strip().replace(" ", "-")
        add_url(f"{base_url}/static-polity/{clean_slug}", priority="0.8", changefreq="monthly")

    # 3. Dynamic Living Dossiers from Neon PostgreSQL
    try:
        dossiers = (
            db.query(IssueDossierModel.issue_slug, IssueDossierModel.last_updated_at)
            .filter(IssueDossierModel.issue_slug.isnot(None))
            .all()
        )
        for d in dossiers:
            mod_date = d.last_updated_at.strftime("%Y-%m-%d") if d.last_updated_at else today_str
            add_url(f"{base_url}/dossier/{d.issue_slug}", priority="0.9", changefreq="daily", lastmod=mod_date)
    except Exception as e:
        print(f"Warning: Could not fetch dossiers for sitemap: {e}")

    rough_string = ET.tostring(urlset, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    set_cached_json(cache_key, {"xml": pretty_xml}, ex_seconds=43200)

    return Response(content=pretty_xml, media_type="application/xml", headers=headers)


from fastapi.responses import PlainTextResponse

@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=True)
def get_robots_txt():
    lines = [
        "User-agent: *",
        "Allow: /",
        "Allow: /dossier/*",
        "Allow: /static-polity/*",
        "Allow: /prelims-pyqs",
        "Disallow: /api/",
        "Disallow: /docs",
        "Disallow: /openapi.json",
        "Disallow: /workspace",
        "",
        "Sitemap: https://politymentor.com/sitemap.xml",
    ]
    return "\n".join(lines)
from src.database.models import ContentReport, QuestionReport
from pydantic import BaseModel

class ReportStatusUpdate(BaseModel):
    status: str  # 'RESOLVED' or 'DISMISSED'

@app.get("/api/admin/reports")
def get_all_admin_reports(
    status_filter: Optional[str] = Query(None, description="PENDING, RESOLVED, or None for all"),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Fetches combined content hallucination flags and MCQ error reports."""
    # 1. Content Reports (Hallucinations, Outdated Law, etc.)
    c_query = db.query(ContentReport)
    if status_filter and status_filter.upper() != "ALL":
        c_query = c_query.filter(ContentReport.status == status_filter.upper())
    content_reports = c_query.order_by(ContentReport.created_at.desc()).limit(limit).all()

    # 2. Daily Drill Question Reports
    q_query = db.query(QuestionReport)
    if status_filter and status_filter.upper() != "ALL":
        q_query = q_query.filter(QuestionReport.status == status_filter.upper())
    question_reports = q_query.order_by(QuestionReport.created_at.desc()).limit(limit).all()

    # Aggregated counts
    pending_content = db.query(func.count(ContentReport.id)).filter(ContentReport.status == "PENDING").scalar() or 0
    pending_questions = db.query(func.count(QuestionReport.id)).filter(QuestionReport.status == "PENDING").scalar() or 0

    return {
        "status": "success",
        "metrics": {
            "pending_content_reports": pending_content,
            "pending_question_reports": pending_questions,
            "total_pending": pending_content + pending_questions
        },
        "content_reports": [
            {
                "id": cr.id,
                "type": "AI_CONTENT",
                "firebase_uid": cr.firebase_uid,
                "issue_category": cr.issue_category,
                "source_page": cr.source_page,
                "flagged_content": cr.flagged_content,
                "user_feedback": cr.user_feedback,
                "status": cr.status,
                "created_at": cr.created_at.isoformat() if cr.created_at else None,
                "resolved_at": cr.resolved_at.isoformat() if cr.resolved_at else None
            }
            for cr in content_reports
        ],
        "question_reports": [
            {
                "id": str(qr.id),
                "type": "DRILL_MCQ",
                "user_id": qr.user_id,
                "drill_date": qr.drill_date,
                "question_index": qr.question_index,
                "question_text": qr.question_text,
                "issue_type": qr.issue_type,
                "user_notes": qr.user_notes,
                "status": qr.status,
                "created_at": qr.created_at.isoformat() if qr.created_at else None
            }
            for qr in question_reports
        ]
    }

@app.patch("/api/admin/reports/content/{report_id}")
def update_content_report_status(
    report_id: int, 
    payload: ReportStatusUpdate, 
    db: Session = Depends(get_db)
):
    rep = db.query(ContentReport).filter(ContentReport.id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Content report not found")
    
    rep.status = payload.status.upper()
    if payload.status.upper() == "RESOLVED":
        rep.resolved_at = datetime.utcnow()
    db.commit()
    return {"status": "success", "message": f"Content report #{report_id} updated to {rep.status}"}

@app.patch("/api/admin/reports/question/{report_id}")
def update_question_report_status(
    report_id: str, 
    payload: ReportStatusUpdate, 
    db: Session = Depends(get_db)
):
    rep = db.query(QuestionReport).filter(QuestionReport.id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Question report not found")
    
    rep.status = payload.status.upper()
    db.commit()
    return {"status": "success", "message": f"Question report {report_id} updated to {rep.status}"}

from src.database.models import ConstitutionalArticleModel
@app.get("/api/admin/articles")
def list_admin_articles(
    response: Response,
    search: Optional[str] = Query(None, description="Search by number or title"),
    part: Optional[str] = Query(None, description="Filter by Part number"),
    cluster: Optional[str] = Query(None, description="Filter by doctrine cluster"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Lists constitutional provisions with metadata, matrix flags, and Redis caching."""
    start_time = time.perf_counter()

    # Deterministic cache key including all query filters
    clean_search = search.strip().lower() if search else "none"
    clean_part = part.strip() if part else "all"
    clean_cluster = cluster.strip() if cluster else "all"
    cache_key = f"articles:catalogue:c={clean_cluster}:p={clean_part}:s={clean_search}:l={limit}:o={offset}"

    # 1. Check Redis Cache
    cached = get_cached_json(cache_key)
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    # 2. Database Fallback (Cache Miss)
    query = db.query(ConstitutionalArticleModel)

    if search:
        pat = f"%{search.strip()}%"
        query = query.filter(
            or_(
                ConstitutionalArticleModel.article_number.ilike(pat),
                ConstitutionalArticleModel.title.ilike(pat),
                ConstitutionalArticleModel.article_slug.ilike(pat)
            )
        )
    if part:
        query = query.filter(ConstitutionalArticleModel.part_number == part.strip())
    if cluster:
        query = query.filter(ConstitutionalArticleModel.doctrine_cluster_id == cluster.strip())

    total = query.count()
    articles = (
        query.order_by(ConstitutionalArticleModel.id.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    result_data = {
        "status": "success",
        "total": total,
        "articles": [
            {
                "id": a.id,
                "article_number": a.article_number,
                "article_slug": a.article_slug,
                "title": a.title,
                "part_number": a.part_number,
                "doctrine_cluster_id": a.doctrine_cluster_id,
                "has_comparative": a.comparative_matrix is not None,
                "traps_count": len(a.aspirant_summary.get("prelims_traps", [])) if isinstance(a.aspirant_summary, dict) else 0,
            }
            for a in articles
        ]
    }

    # 3. Cache for 7 days (604,800s)
    set_cached_json(cache_key, result_data, ex_seconds=604800)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return result_data


@app.get("/api/admin/articles/{article_id}")
def get_admin_article_detail(
    article_id: int, 
    response: Response, 
    db: Session = Depends(get_db)
):
    """Fetches complete verbatim text, bare act analysis, 4-epoch trajectory, and matrix."""
    start_time = time.perf_counter()
    cache_key = f"articles:detail:{article_id}"

    # 1. Fast Cache Hit Check
    cached = get_cached_json(cache_key)
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        # Instruct browser & Cloudflare to cache for 7 days, but revalidate if needed
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    # 2. Database Fallback (Cache Miss)
    article = db.query(ConstitutionalArticleModel).filter(ConstitutionalArticleModel.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Constitutional Article not found")

    payload = {
        "status": "success",
        "article": {
            "id": article.id,
            "article_number": article.article_number,
            "article_slug": article.article_slug,
            "title": article.title,
            "part_number": article.part_number,
            "doctrine_cluster_id": article.doctrine_cluster_id,
            "verbatim_text": article.verbatim_text,
            "bare_act_analysis": article.bare_act_analysis,
            "aspirant_summary": article.aspirant_summary,
            "evolution_trajectory": article.evolution_trajectory,
            "comparative_matrix": article.comparative_matrix,
            "created_at": article.created_at.isoformat() if article.created_at else None,
            "updated_at": article.updated_at.isoformat() if article.updated_at else None,
        }
    }

    # 3. Cache for 7 days (604,800s)
    set_cached_json(cache_key, payload, ex_seconds=604800)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return payload


class UpdateArticleRequest(BaseModel):
    title: Optional[str] = None
    part_number: Optional[str] = None
    doctrine_cluster_id: Optional[str] = None
    verbatim_text: Optional[str] = None
    bare_act_analysis: Optional[list] = None
    aspirant_summary: Optional[dict] = None
    evolution_trajectory: Optional[dict] = None
    comparative_matrix: Optional[dict] = None

@app.patch("/api/admin/articles/{article_id}")
def update_admin_article(
    article_id: int, 
    payload: UpdateArticleRequest, 
    db: Session = Depends(get_db)
):
    """Updates constitutional article content and purges stale Redis cache immediately."""
    article = db.query(ConstitutionalArticleModel).filter(ConstitutionalArticleModel.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Constitutional Article not found")

    # Update only fields provided in the request
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(article, field, value)

    article.updated_at = datetime.utcnow()
    
    # 1. Commit to PostgreSQL
    db.commit()
    db.refresh(article)

    # 2. IMMEDIATE CACHE PURGE (Place it right here!)
    delete_cached_key(f"articles:detail:{article.id}")
    if hasattr(article, "article_slug") and article.article_slug:
        delete_cached_key(f"article:detail:{article.article_slug.strip().lower()}")
    
    # Invalidate all filtered list catalogues (parts, clusters, searches)
    delete_cached_pattern("articles:catalogue:*")

    return {
        "status": "success", 
        "message": f"Article #{article_id} updated and cache invalidated.",
        "article_id": article.id
    }


from src.database.models import PrelimsPYQModel
from pydantic import BaseModel
from typing import List

class UpdatePYQMappingRequest(BaseModel):
    chapter_number: Optional[int] = None
    chapter_name: Optional[str] = None
    mapped_articles: Optional[List[int]] = None
    difficulty_level: Optional[str] = None
    is_conceptual: Optional[bool] = None

@app.get("/api/admin/pyqs")
def list_admin_pyqs(
    response: Response,
    search: Optional[str] = Query(None, description="Search question text or explanation"),
    year: Optional[int] = Query(None, description="Filter by exam year"),
    chapter_number: Optional[int] = Query(None, description="Filter by Laxmikanth chapter (1-92)"),
    article_number: Optional[int] = Query(None, description="Filter by mapped article integer"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Lists Prelims PYQs with article array containment, taxonomy filters, and Redis caching."""
    start_time = time.perf_counter()

    # Deterministic query key across all active filters
    clean_search = search.strip().lower() if search else "none"
    cache_key = f"pyqs:catalogue:y={year or 'all'}:ch={chapter_number or 'all'}:art={article_number or 'all'}:s={clean_search}:l={limit}:o={offset}"

    # 1. Fast Cache Hit Check
    cached = get_cached_json(cache_key)
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    # 2. Database Fallback (Cache Miss)
    query = db.query(PrelimsPYQModel)

    if search:
        pat = f"%{search.strip()}%"
        query = query.filter(
            or_(
                PrelimsPYQModel.question_text.ilike(pat),
                PrelimsPYQModel.explanation.ilike(pat)
            )
        )
    if year:
        query = query.filter(PrelimsPYQModel.year == year)
    if chapter_number:
        query = query.filter(PrelimsPYQModel.chapter_number == chapter_number)
    if article_number:
        # PostgreSQL GIN array containment: WHERE article_number = ANY(mapped_articles)
        query = query.filter(PrelimsPYQModel.mapped_articles.any(article_number))

    total = query.count()
    pyqs = (
        query.order_by(PrelimsPYQModel.year.desc(), PrelimsPYQModel.question_number.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    result_data = {
        "status": "success",
        "total": total,
        "pyqs": [
            {
                "id": p.id,
                "year": p.year,
                "question_number": p.question_number,
                "question_text": p.question_text,
                "options": p.options,
                "correct_option": p.correct_option,
                "chapter_number": p.chapter_number,
                "chapter_name": p.chapter_name,
                "mapped_articles": p.mapped_articles,
                "explanation": p.explanation,
                "difficulty_level": p.difficulty_level,
                "is_conceptual": p.is_conceptual,
            }
            for p in pyqs
        ]
    }

    # 3. Cache for 7 days (604,800s)
    set_cached_json(cache_key, result_data, ex_seconds=604800)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=604800, stale-while-revalidate=86400"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return result_data

@app.patch("/api/admin/pyqs/{pyq_id}")
def update_pyq_metadata(
    pyq_id: int, 
    payload: UpdatePYQMappingRequest, 
    db: Session = Depends(get_db)
):
    """Updates taxonomy chapter links, difficulty, or mapped constitutional article IDs and purges cache."""
    pyq = db.query(PrelimsPYQModel).filter(PrelimsPYQModel.id == pyq_id).first()
    if not pyq:
        raise HTTPException(status_code=404, detail="PYQ item not found")

    if payload.chapter_number is not None:
        pyq.chapter_number = payload.chapter_number
    if payload.chapter_name is not None:
        pyq.chapter_name = payload.chapter_name.strip()
    if payload.mapped_articles is not None:
        pyq.mapped_articles = sorted(list(set(payload.mapped_articles)))
    if payload.difficulty_level is not None:
        pyq.difficulty_level = payload.difficulty_level.strip()
    if payload.is_conceptual is not None:
        pyq.is_conceptual = payload.is_conceptual

    db.commit()
    db.refresh(pyq)

    # --- INSTANT CACHE INVALIDATION ---
    # 1. Purge all cached PYQ list filter combinations
    delete_cached_pattern("pyqs:catalogue:*")
    # 2. Also purge any module drill caches built in blueprint.py
    delete_cached_pattern("blueprint:drill:*")
    delete_cached_pattern("blueprint:prelims:*")

    return {"status": "success", "message": f"PYQ #{pyq.id} updated and caches purged"}


from src.database.models import IngestionRun, DocumentChunkModel
from sqlalchemy import desc

@app.get("/api/admin/ingestion-runs")
def list_admin_ingestion_runs(
    source: Optional[str] = Query(None, description="Filter by source e.g. PIB, The Hindu"),
    limit: int = 30,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Lists recent ingestion runs with execution metrics and failure breakdowns."""
    query = db.query(IngestionRun)
    if source:
        query = query.filter(IngestionRun.source.ilike(f"%{source.strip()}%"))

    total_runs = query.count()
    runs = query.order_by(IngestionRun.started_at.desc()).offset(offset).limit(limit).all()

    # Aggregate total system health metrics
    total_valid = db.query(func.coalesce(func.sum(IngestionRun.valid), 0)).scalar()
    total_failed = db.query(func.coalesce(func.sum(IngestionRun.failed), 0)).scalar()
    total_chunks = db.query(func.count(DocumentChunkModel.id)).scalar() or 0

    return {
        "status": "success",
        "total_runs": total_runs,
        "metrics": {
            "total_valid_articles": total_valid,
            "total_failures": total_failed,
            "total_vector_chunks": total_chunks
        },
        "runs": [
            {
                "run_id": str(r.run_id),
                "source": r.source,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "total_items": r.total_items,
                "collected": r.collected,
                "valid": r.valid,
                "uploaded": r.uploaded,
                "already_exists": r.already_exists,
                "failed": r.failed,
                "failures": {
                    "collection": r.collection_failures,
                    "validation": r.validation_failures,
                    "storage": r.storage_failures,
                    "unexpected": r.unexpected_failures
                }
            }
            for r in runs
        ]
    }

@app.get("/api/admin/ingestion-chunks")
def list_recent_vector_chunks(
    limit: int = 25,
    db: Session = Depends(get_db)
):
    """Inspects recently generated text chunks and Qwen 1024-dim embedding vectors."""
    chunks = (
        db.query(DocumentChunkModel)
        .order_by(DocumentChunkModel.id.desc())
        .limit(limit)
        .all()
    )

    return {
        "status": "success",
        "chunks": [
            {
                "id": c.id,
                "prid": c.prid,
                "chapter_name": c.chapter_name,
                "topic": c.topic,
                "chunk_type": c.chunk_type,
                "chunk_text": c.chunk_text[:180] + ("..." if len(c.chunk_text) > 180 else ""),
                "token_count": c.token_count,
                "embedding_model": c.embedding_model,
                "has_embedding": c.embedding_1024 is not None
            }
            for c in chunks
        ]
    }

@app.post("/api/admin/purge-cache")
def purge_system_cache(tag: str = Query(..., description="blueprint, articles, pyqs, or dossiers")):
    """Purges Redis cache keys by functional tag."""
    clean_tag = tag.strip().lower()
    cleared = 0

    if clean_tag == "articles":
        cleared += delete_cached_pattern("articles:*")
        cleared += delete_cached_pattern("article:*")
    elif clean_tag == "pyqs":
        cleared += delete_cached_pattern("pyqs:*")
        cleared += delete_cached_pattern("blueprint:drill:*")
    elif clean_tag == "blueprint":
        cleared += delete_cached_pattern("blueprint:*")
        cleared += delete_cached_pattern("laxmikanth:*")
    elif clean_tag == "dossiers":
        cleared += delete_cached_pattern("admin:dossiers:*")
        cleared += delete_cached_pattern("dossier:*")
        cleared += delete_cached_pattern("dossiers:*")
        delete_cached_key("seo:sitemap:xml")
        cleared += 1
    elif clean_tag == "smart-notes":
        cleared += delete_cached_pattern("ca:content:*")
        cleared += delete_cached_pattern("ca:notes:*")
        cleared += delete_cached_pattern("ca:topics:*")
        delete_cached_key("ca:notes:latest")
        cleared += 1
    else:
        raise HTTPException(status_code=400, detail=f"Unrecognized cache tag '{tag}'")

    return {"status": "success", "tag": clean_tag, "keys_evicted": cleared}

@app.get("/api/admin/dossiers")
def get_admin_dossiers(
    response: Response,
    search: Optional[str] = Query(None, description="Search dossiers by title, slug, or chapter"),
    status: Optional[str] = Query(None, description="active, archived, etc."),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Administrative overview for living constitutional topic dossiers,
    prelims factsheet summaries, and connected evolutionary perspective edges.
    """
    start_time = time.perf_counter()

    # Deterministic cache key across filters
    clean_search = search.strip().lower() if search else "none"
    clean_status = status.strip().lower() if status else "all"
    cache_key = f"admin:dossiers:list:s={clean_search}:st={clean_status}:l={limit}:o={offset}"

    # 1. Fast Cache Hit Check
    cached = get_cached_json(cache_key)
    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=180, s-maxage=3600, stale-while-revalidate=60"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    # 2. Database Fallback (Cache Miss)
    query = db.query(IssueDossierModel).options(selectinload(IssueDossierModel.edges))

    if search:
        pat = f"%{search.strip()}%"
        query = query.filter(
            or_(
                IssueDossierModel.title.ilike(pat),
                IssueDossierModel.chapter_name.ilike(pat),
                IssueDossierModel.issue_slug.ilike(pat)
            )
        )
    if status:
        query = query.filter(IssueDossierModel.status == clean_status)

    total = query.count()
    dossiers = (
        query.order_by(IssueDossierModel.last_updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    data = []
    for d in dossiers:
        data.append({
            "id": str(d.issue_id),
            "title": d.title,
            "issue_slug": d.issue_slug,
            "chapter_name": d.chapter_name,
            "status": d.status,
            "article_count": d.article_count,
            "canonical_summary": d.canonical_summary,
            "prelims_factsheet": d.prelims_factsheet or {},
            "mains_framework": d.mains_framework or {},
            "first_detected_at": d.first_detected_at.isoformat() if d.first_detected_at else None,
            "last_updated_at": d.last_updated_at.isoformat() if d.last_updated_at else None,
            "edges_count": len(d.edges) if d.edges else 0,
            "perspectives": [
                {
                    "publication": e.publication,
                    "event_date": str(e.event_date) if hasattr(e, 'event_date') else None,
                    "perspective_stance": e.perspective_stance,
                    "core_claim": e.core_claim,
                    "legal_anchors": e.legal_anchors or []
                }
                for e in (d.edges if d.edges else [])
            ]
        })

    result_payload = {
        "status": "success",
        "total": total,
        "dossiers": data
    }

    # 3. Cache for 2 hours (7,200s)
    set_cached_json(cache_key, result_payload, ex_seconds=7200)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["Cache-Control"] = "public, max-age=180, s-maxage=3600, stale-while-revalidate=60"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return result_payload

from src.database.models import IssueDossierModel, IssueGraphEdgeModel
from sqlalchemy.orm import selectinload
import os
import secrets
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecretpassword")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "your-secret-token-key")

class AdminLoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/admin/login")
def admin_login(payload: AdminLoginRequest):
    valid_user = secrets.compare_digest(payload.username.strip(), ADMIN_USERNAME)
    valid_pass = secrets.compare_digest(payload.password.strip(), ADMIN_PASSWORD)

    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid administrative username or password."
        )

    return {
        "status": "success",
        "token": ADMIN_SECRET_KEY
    }


# ---------------------------------------------------------
# Admin Smart Notes Studio & R2 Quality Cockpit Endpoints
# ---------------------------------------------------------

@app.get("/api/admin/smart-notes/queue")
def list_admin_smart_notes_queue(
    limit: int = 250,
    chapter: Optional[str] = Query(None, description="Filter by canonical chapter name"),
    sort_by: str = Query("recent", description="'recent' (descending PRID) or 'raw'")
):
    """
    Lists smart notes from R2 with descending PRID sorting and chapter filtering.
    """
    try:
        search_prefix = SMART_NOTES_PREFIX
        if chapter and chapter.strip() not in ("All", ""):
            search_prefix = f"{SMART_NOTES_PREFIX}{chapter.strip('/')}"

        raw_objects = []
        if hasattr(storage, "list_objects"):
            try:
                raw_objects = list(storage.list_objects(search_prefix))
            except TypeError:
                raw_objects = list(storage.list_objects(search_prefix, recursive=True))
        elif hasattr(storage, "client"):
            raw_objects = list(
                storage.client.list_objects(storage.bucket, prefix=search_prefix, recursive=True)
            )

        notes = []
        available_chapters = set()

        for obj in raw_objects:
            key = getattr(obj, "object_name", None) or getattr(obj, "key", "")
            if not key or key.endswith("/") or not key.endswith(".json") or "smart_notes_" not in key:
                continue

            parts = key.split("/")
            # pib/smart_notes/<chapter>/<topic>/smart_notes_<prid>.json
            ch_name = parts[2] if len(parts) >= 5 else "General"
            tp_name = parts[3] if len(parts) >= 5 else "Current Affairs"
            filename = parts[-1]
            prid_str = filename.replace("smart_notes_", "").replace(".json", "")

            available_chapters.add(ch_name)

            headline = f"PIB Note #{prid_str}"
            try:
                data = storage.get_json(key)
                if isinstance(data, dict):
                    headline = data.get("headline") or data.get("title") or headline
            except Exception:
                pass

            notes.append({
                "object_key": key,
                "filename": filename,
                "prid": prid_str,
                "prid_num": int(prid_str) if prid_str.isdigit() else 0,
                "chapter": ch_name,
                "topic": tp_name,
                "headline": headline,
                "size_kb": round((getattr(obj, "size", 0) or 0) / 1024, 2),
                "last_modified": getattr(obj, "last_modified", None),
            })

        # Sort: Highest PRID first (most recent PIB releases)
        if sort_by == "recent":
            notes.sort(key=lambda x: x["prid_num"], reverse=True)

        return {
            "status": "success",
            "total": len(notes),
            "chapters": sorted(list(available_chapters)),
            "items": notes[:limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan R2 smart notes: {str(e)}")
    
@app.get("/api/admin/smart-notes/detail")
def get_admin_smart_note_detail(
    object_key: str = Query(..., description="MinIO/R2 object key e.g. pib/smart_notes/..."),
):
    try:
        data = storage.get_json(object_key)
        if not isinstance(data, dict):
            raise HTTPException(status_code=404, detail="Smart note not found or corrupt in storage")

        # Normalize prelims items so frontend gets explicit statement, rule, trap, and is_correct
        raw_prelims = data.get("prelims_practice_points") or []
        normalized_prelims = []
        for p in raw_prelims:
            if isinstance(p, dict):
                stmt = p.get("statement") or p.get("rule") or ""
                expl = p.get("explanation") or p.get("trap") or ""
                # Default is_correct: check bool, or inspect rule heuristic
                is_corr = p.get("is_correct")
                if is_corr is None:
                    # If it was saved with 'rule' vs 'trap', the statement itself was usually False (a trap) or True
                    is_corr = False if p.get("trap") else True

                normalized_prelims.append({
                    "statement": stmt,
                    "rule": stmt,
                    "explanation": expl,
                    "trap": expl,
                    "is_correct": bool(is_corr)
                })
            else:
                normalized_prelims.append({
                    "statement": str(p),
                    "rule": str(p),
                    "explanation": "",
                    "trap": "",
                    "is_correct": True
                })

        data["prelims_practice_points"] = normalized_prelims

        parts = object_key.split("/")
        return {
            "status": "success",
            "object_key": object_key,
            "chapter": parts[2] if len(parts) >= 5 else "General",
            "topic": parts[3] if len(parts) >= 5 else "Polity",
            "prid": parts[-1].replace("smart_notes_", "").replace(".json", ""),
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AdminSmartNoteUpdateRequest(BaseModel):
    object_key: str
    headline: Optional[str] = None
    sub_headline: Optional[str] = None
    pib_executive_context: Optional[str] = None
    strategic_upsc_value: Optional[str] = None
    summary: Optional[str] = None
    why_it_matters: Optional[str] = None
    smart_notes: Optional[list] = None
    backward_linkages: Optional[list] = None
    prelims_practice_points: Optional[list] = None
    mains_question: Optional[str] = None
    mains_analytical_question: Optional[str] = None
    raw_json_override: Optional[dict] = None
    re_render_slides: Optional[bool] = False

import io
import json

@app.put("/api/admin/smart-notes/save")
def save_amended_smart_note(payload: AdminSmartNoteUpdateRequest):
    """
    Overwrites the amended smart note JSON in MinIO/R2, harmonizes both
    legacy (rule/trap) and modern (statement/is_correct/explanation) fields,
    and flushes all downstream Redis caches (including ca:content:v5).
    """
    try:
        # 1. Fetch current object from R2 safely
        current_data = {}
        try:
            existing = storage.get_json(payload.object_key)
            if isinstance(existing, dict):
                current_data = existing
        except Exception:
            current_data = {}

        # 2. Merge updates
        if payload.raw_json_override:
            updated_data = payload.raw_json_override
        else:
            updated_data = dict(current_data)

            # Direct scalar & text synchronizations
            if payload.headline is not None:
                updated_data["headline"] = payload.headline.strip()
                updated_data["title"] = payload.headline.strip()

            if payload.sub_headline is not None:
                updated_data["sub_headline"] = payload.sub_headline.strip()

            if payload.summary is not None or payload.pib_executive_context is not None:
                val = (payload.summary or payload.pib_executive_context or "").strip()
                updated_data["summary"] = val
                updated_data["pib_executive_context"] = val

            if payload.why_it_matters is not None or payload.strategic_upsc_value is not None:
                val = (payload.why_it_matters or payload.strategic_upsc_value or "").strip()
                updated_data["why_it_matters"] = val
                updated_data["strategic_upsc_value"] = val

            if payload.smart_notes is not None:
                updated_data["smart_notes"] = [
                    str(pt).strip() for pt in payload.smart_notes if str(pt).strip()
                ]

            # 3. Synchronize Backward Linkages (Ensure both 'title' and 'anchor' exist)
            if payload.backward_linkages is not None:
                cleaned_linkages = []
                for item in payload.backward_linkages:
                    if isinstance(item, dict):
                        t = (item.get("title") or item.get("anchor") or "").strip()
                        d = (item.get("description") or item.get("desc") or "").strip()
                        cleaned_linkages.append({
                            "title": t,
                            "anchor": t,
                            "description": d
                        })
                    elif isinstance(item, str):
                        parts = [p.strip() for p in item.split(":", 1)]
                        if len(parts) == 2:
                            cleaned_linkages.append({"title": parts[0], "anchor": parts[0], "description": parts[1]})
                        else:
                            cleaned_linkages.append({"title": item.strip(), "anchor": item.strip(), "description": ""})
                updated_data["backward_linkages"] = cleaned_linkages

            # 4. Synchronize Prelims Drill Points (Enforce explicit boolean `is_correct`)
            if payload.prelims_practice_points is not None:
                cleaned_prelims = []
                for item in payload.prelims_practice_points:
                    if isinstance(item, dict):
                        stmt = (item.get("statement") or item.get("rule") or "").strip()
                        expl = (item.get("explanation") or item.get("trap") or "").strip()
                        
                        # Preserve explicit boolean from frontend toggle, defaulting to False if marked as trap
                        is_corr = item.get("is_correct")
                        if is_corr is None:
                            is_corr = False if item.get("trap") else True

                        cleaned_prelims.append({
                            "statement": stmt,
                            "rule": stmt,
                            "is_correct": bool(is_corr),
                            "explanation": expl,
                            "trap": expl
                        })
                    else:
                        cleaned_prelims.append({
                            "statement": str(item).strip(),
                            "rule": str(item).strip(),
                            "is_correct": True,
                            "explanation": "",
                            "trap": ""
                        })
                updated_data["prelims_practice_points"] = cleaned_prelims

            if payload.mains_question is not None or payload.mains_analytical_question is not None:
                m_val = (payload.mains_analytical_question or payload.mains_question or "").strip()
                updated_data["mains_question"] = m_val
                updated_data["mains_analytical_question"] = m_val

        # Timestamp audit
        updated_data["last_manually_edited_at"] = datetime.utcnow().isoformat()

        # 5. Direct R2 / MinIO Byte Stream Upload
        json_bytes = json.dumps(updated_data, ensure_ascii=False, indent=2).encode("utf-8")
        data_stream = io.BytesIO(json_bytes)

        if hasattr(storage, "upload_json"):
            storage.upload_json(payload.object_key, updated_data)
        elif hasattr(storage, "client"):
            storage.client.put_object(
                bucket_name=storage.bucket,
                object_name=payload.object_key,
                data=data_stream,
                length=len(json_bytes),
                content_type="application/json"
            )
        else:
            raise AttributeError("Unable to locate S3 upload client on MinIOStorage instance")

        # 6. Flush all relevant Redis cache keys
        delete_cached_key(f"ca:content:v5:{payload.object_key}")
        delete_cached_key(f"ca:content:v4:{payload.object_key}")
        delete_cached_key("ca:notes:latest")
        delete_cached_pattern("ca:notes:*")

        return {
            "status": "success",
            "message": "Smart Note successfully amended in R2 and all caches purged",
            "object_key": payload.object_key,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist smart note to R2: {str(e)}")

# =====================================================================
# 1. ATOMIC SMART NOTE CONTENT ENDPOINT (FLUTTERFLOW / WEB FRONTEND)
# =====================================================================

@app.get("/api/current-affairs/note-content")
def get_ca_note_content(
    response: Response,
    object_key: str = Query(..., description="MinIO object key")
):
    """
    Parses a PIB UPSC Smart Note from MinIO/R2 into clean, modular,
    atomic JSON fields for custom FlutterFlow UI components.
    """
    start_time = time.perf_counter()
    # Bumped cache namespace to v5 to instantly evict legacy string-polluted entries
    cache_key = f"ca:content:v5:{object_key}"
    cached = get_cached_json(cache_key)

    if cached:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return cached

    try:
        raw_data = storage.get_json(object_key)

        if not isinstance(raw_data, dict):
            return {
                "status": "error",
                "message": "Invalid note format in storage",
                "object_key": object_key
            }

        headline = raw_data.get("headline") or raw_data.get("title") or "PIB Smart Study Note"

        # Format publication date
        meta_date = ""
        raw_date = raw_data.get("generated_at")
        if raw_date:
            try:
                dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                meta_date = dt.strftime("%d %B %Y")
            except Exception:
                meta_date = str(raw_date)

        # 1. Summary & Context
        summary_text = (raw_data.get("summary") or raw_data.get("pib_executive_context") or "").strip()
        why_it_matters_text = (raw_data.get("why_it_matters") or raw_data.get("strategic_upsc_value") or "").strip()

        # 2. Highlights / Smart Notes
        raw_highlights = raw_data.get("smart_notes") or []
        highlights_list = []
        for idx, pt in enumerate(raw_highlights, start=1):
            clean_pt = str(pt).strip().lstrip("•").lstrip("-").strip()
            # Support both "Title: Body" format and pure directive sentences
            title_part = ""
            desc_part = clean_pt
            for sep in [":", " – ", " — ", " - "]:
                if sep in clean_pt:
                    parts = clean_pt.split(sep, 1)
                    title_part = parts[0].strip(" '\"")
                    desc_part = parts[1].strip(" '\"")
                    break

            if not title_part:
                title_part = f"Key Directive #{idx}"

            highlights_list.append({
                "title": title_part,
                "body": desc_part
            })

        # 3. Backward Linkages (Robust Dict + String Handler)
        raw_linkages = raw_data.get("backward_linkages") or []
        linkages_list = []
        for item in raw_linkages:
            if isinstance(item, dict):
                # Cleanly pull fields from modern R2 structured schema without stringifying dict
                anchor_val = (
                    item.get("title")
                    or item.get("anchor")
                    or item.get("article")
                    or ""
                ).strip()
                desc_val = (
                    item.get("description")
                    or item.get("desc")
                    or item.get("tagline")
                    or "Constitutional Foundation / Statutory Linkage"
                ).strip()
                linkages_list.append({
                    "anchor": anchor_val,
                    "description": desc_val
                })
            else:
                clean_item = str(item).replace("⚑", "—").strip().lstrip("•").lstrip("-").strip()
                separator = None
                for sep_candidate in ["—", "–", " - ", ":", "-"]:
                    if sep_candidate in clean_item:
                        separator = sep_candidate
                        break

                if separator:
                    parts = clean_item.split(separator, 1)
                    linkages_list.append({
                        "anchor": parts[0].strip(" '\""),
                        "description": parts[1].strip(" '\"")
                    })
                else:
                    linkages_list.append({
                        "anchor": clean_item,
                        "description": "Constitutional Foundation / Statutory Linkage"
                    })

        # 4. Prelims Drill (Dual Schema: statement/explanation OR rule/trap)
        raw_prelims = raw_data.get("prelims_practice_points") or []
        drill_list = []
        for idx, item in enumerate(raw_prelims, start=1):
            if isinstance(item, dict):
                stmt = (item.get("statement") or item.get("rule") or "").strip()
                expl = (item.get("explanation") or "").strip()
                trap_val = (item.get("trap") or "").strip()

                is_corr = item.get("is_correct")
                if is_corr is None and "rule" in item:
                    is_corr = True

                if not expl and trap_val:
                    expl = f"Trap Distinction: {trap_val}"

                drill_list.append({
                    "statement_number": idx,
                    "statement": stmt,
                    "is_correct": is_corr,
                    "verdict_label": "CORRECT" if is_corr is True else ("INCORRECT" if is_corr is False else "EXAM FACT"),
                    "explanation": expl
                })
            else:
                drill_list.append({
                    "statement_number": idx,
                    "statement": str(item).strip(),
                    "is_correct": None,
                    "verdict_label": "INFO",
                    "explanation": ""
                })

        # 5. Mains Analytical Question
        mains_text = (raw_data.get("mains_question") or raw_data.get("mains_analytical_question") or "").strip()

        result_payload = {
            "status": "success",
            "title": headline,
            "object_key": object_key,
            "date": meta_date,
            "upsc_relevance": raw_data.get("upsc_relevance_mains", ""),
            "content": {
                "summary": summary_text,
                "why_it_matters": why_it_matters_text,
                "highlights": highlights_list,
                "linkages": linkages_list,
                "drill": drill_list,
                "mains": {
                    "question": mains_text,
                    "word_limit": "250 Words",
                    "marks": "15 Marks"
                }
            }
        }

        set_cached_json(cache_key, result_payload, ex_seconds=604800)  # 7 days

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800, stale-while-revalidate=86400"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return result_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# 2. SAVE SMART NOTE TO WORKSPACE ENDPOINT
# =====================================================================

@app.post("/api/notes/save-smart-note")
async def save_smart_note_to_workspace(
    payload: SaveSmartNoteToWorkspaceRequest, 
    db: Session = Depends(get_db)
):
    """
    Imports a PIB Smart Note from MinIO/R2 directly into the user's personal UserNote list.
    """
    user = db.query(User).filter(User.firebase_uid == payload.firebase_uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        raw_data = storage.get_json(payload.object_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load smart note from storage: {e}")

    headline = raw_data.get("headline") or raw_data.get("title") or "PIB Smart Note"

    # Deduplication check
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

    # Generate structured markdown note
    note_lines = [
        f"# {headline}\n",
        f"Source: PIB Reference | {payload.object_key}\n"
    ]

    if payload.custom_reflection:
        note_lines.append(f"### 💡 My Revision Notes / Reflections\n{payload.custom_reflection}\n")

    summary_val = raw_data.get("summary") or raw_data.get("pib_executive_context")
    if summary_val:
        note_lines.append(f"### 📌 Executive Summary\n{summary_val.strip()}\n")

    why_val = raw_data.get("why_it_matters") or raw_data.get("strategic_upsc_value")
    if why_val:
        note_lines.append(f"### 🎯 Strategic UPSC Value\n{why_val.strip()}\n")

    smart_notes = raw_data.get("smart_notes")
    if isinstance(smart_notes, list) and smart_notes:
        note_lines.append("### 📝 Key Analysis & Takeaways")
        for pt in smart_notes:
            note_lines.append(f"* {str(pt).strip()}")
        note_lines.append("")

    linkages = raw_data.get("backward_linkages")
    if isinstance(linkages, list) and linkages:
        note_lines.append("### 🔗 Backward Linkages (Static Polity & Cases)")
        for item in linkages:
            if isinstance(item, dict):
                t = item.get("title") or item.get("anchor") or ""
                d = item.get("description") or item.get("desc") or ""
                line = f"{t} – {d}" if t and d else (t or d)
            else:
                line = str(item).strip()
            note_lines.append(f"* {line}")
        note_lines.append("")

    mains_val = raw_data.get("mains_question") or raw_data.get("mains_analytical_question")
    if mains_val:
        note_lines.append(f"### ✍️ Mains Analytical Question\n> {mains_val.strip()}\n")

    formatted_content = "\n".join(note_lines)

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


# =====================================================================
# 3. DOWNLOAD PDF DOSSIER ENDPOINT
# =====================================================================

@app.get("/api/current-affairs/download-pdf")
async def download_smart_note_pdf(object_key: str = Query(..., description="MinIO object key")):
    """
    Generates and streams a print-calibrated, slate-and-brass PDF dossier 
    directly from the MinIO smart note object.
    """
    try:
        raw_data = storage.get_json(object_key)
        if not isinstance(raw_data, dict):
            raise HTTPException(status_code=404, detail="Note not found or invalid format in storage")

        title = raw_data.get("headline") or raw_data.get("title") or "UPSC Current Affairs Dossier"
        
        lines = [
            f"# {title}\n",
            "**UPSC Civil Services Examination • GS-II Intelligence Dossier**\n",
            "---\n"
        ]

        # 1. Executive Summary
        summary_val = raw_data.get("summary") or raw_data.get("pib_executive_context")
        if summary_val:
            lines.append("## Executive Summary\n")
            lines.append(f"{summary_val.strip()}\n\n---\n")

        # 2. Why It Matters
        why_val = raw_data.get("why_it_matters") or raw_data.get("strategic_upsc_value")
        if why_val:
            lines.append("## Why It Matters for UPSC\n")
            lines.append(f"> {why_val.strip()}\n\n---\n")

        # 3. Key Highlights
        smart_notes = raw_data.get("smart_notes")
        if isinstance(smart_notes, list) and smart_notes:
            lines.append("## Strategic Takeaways & Key Facts\n")
            for pt in smart_notes:
                lines.append(f"* {str(pt).strip()}")
            lines.append("\n---\n")

        # 4. Backward Linkages / Constitutional Anchors (Clean formatting for dict & str)
        linkages = raw_data.get("backward_linkages")
        if isinstance(linkages, list) and linkages:
            lines.append("## Backward Linkages & Constitutional Anchors\n")
            for item in linkages:
                if isinstance(item, dict):
                    t = item.get("title") or item.get("anchor") or ""
                    d = item.get("description") or item.get("desc") or ""
                    line = f"{t} – {d}" if t and d else (t or d)
                else:
                    line = str(item).strip()
                lines.append(f"* {line}")
            lines.append("\n---\n")

        # 5. Prelims Drill & Fact Checks
        prelims = raw_data.get("prelims_practice_points")
        if isinstance(prelims, list) and prelims:
            lines.append("## Prelims Drill & Boundary Traps\n")
            for idx, item in enumerate(prelims, start=1):
                if isinstance(item, dict):
                    stmt = (item.get("statement") or item.get("rule") or "").strip()
                    correct = item.get("is_correct")
                    if correct is None and "rule" in item:
                        correct = True
                    exp = (item.get("explanation") or "").strip()
                    trap = (item.get("trap") or "").strip()

                    badge = "**[CORRECT]**" if correct is True else ("**[INCORRECT]**" if correct is False else "")
                    lines.append(f"**Statement {idx}:** {stmt}\n")
                    if badge:
                        lines.append(f"* Verdict: {badge}\n")
                    if exp:
                        lines.append(f"* Forensic Analysis: {exp}\n")
                    if trap:
                        lines.append(f"* Distractor Trap: {trap}\n")
                else:
                    lines.append(f"* {str(item).strip()}")
            lines.append("\n---\n")

        # 6. Mains Analytical Question
        mains_val = raw_data.get("mains_question") or raw_data.get("mains_analytical_question")
        if mains_val:
            lines.append("## Mains Answer-Building Framework\n")
            lines.append(f"> **Practice Question (250 Words | 15 Marks):**  \n> *{mains_val.strip()}*\n")

        markdown_content = "\n".join(lines)

        pdf_stream = generate_pdf_from_markdown(title=title, markdown_content=markdown_content)

        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()
        encoded_filename = urllib.parse.quote(f"{safe_title[:45]}.pdf")

        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

