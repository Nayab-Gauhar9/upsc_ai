from fastapi import FastAPI, HTTPException, Depends, Query, Request, status
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
    Reads a PIB UPSC smart note from MinIO and compiles it into an
    aesthetically styled, high-readability study dossier for mobile.
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
                meta_date = f" • {dt.strftime('%d %B %Y')}"
            except Exception:
                meta_date = f" • {raw_data['generated_at']}"

        # -------------------------------------------------------------
        # Aesthetic Mobile Markdown Construction (No broken blockquotes)
        # -------------------------------------------------------------
        md: list[str] = [
            f"# {headline}\n",
            f"🏷️ GS-II Intelligence Dossier{meta_date} • *Polity Mentor verified*",
            "\n---\n"
        ]

        # 1. Executive Summary
        if raw_data.get("summary"):
            md.append("### 📌 Executive Summary\n")
            md.append(f"{raw_data['summary'].strip()}\n")
            md.append("\n---\n")

        # 2. Why It Matters for UPSC
        if raw_data.get("why_it_matters"):
            md.append("### 💡 Why It Matters for UPSC\n")
            md.append(f"{raw_data['why_it_matters'].strip()}\n")
            md.append("\n---\n")

        # 3. Smart Notes (Key Exam Highlights)
        smart_notes = raw_data.get("smart_notes")
        if isinstance(smart_notes, list) and smart_notes:
            md.append("### 📝 Smart Notes (Key Exam Highlights)\n")
            for pt in smart_notes:
                md.append(f"• {str(pt).strip()}\n")
            md.append("\n---\n")

        # 4. Backward Linkages (Static Syllabus & Case Laws)
        linkages = raw_data.get("backward_linkages")
        if isinstance(linkages, list) and linkages:
            md.append("### 🔗 Backward Linkages (Static Syllabus & Articles)\n")
            for item in linkages:
                clean_item = str(item).replace("⚑", "—").strip()
                # Format constitutional provisions with clean inline tags
                if "—" in clean_item:
                    parts = clean_item.split("—", 1)
                    md.append(f"• {parts[0].strip()} — {parts[1].strip()}\n")
                else:
                    md.append(f"• {clean_item}\n")
            md.append("\n---\n")

        # 5. Prelims Practice Statements (Clean Cards without > blockquotes)
        prelims = raw_data.get("prelims_practice_points")
        if isinstance(prelims, list) and prelims:
            md.append("### 🧭 Prelims Drill & Fact Checks\n")
            for idx, item in enumerate(prelims, start=1):
                if isinstance(item, dict):
                    stmt = item.get("statement", "").strip()
                    correct = item.get("is_correct")
                    exp = item.get("explanation", "").strip()
                    
                    badge = "✅ CORRECT" if correct is True else ("❌ INCORRECT" if correct is False else "")

                    md.append(f"Statement {idx}: {stmt}\n")
                    if badge:
                        md.append(f"• Verdict: {badge}\n")
                    if exp:
                        md.append(f"• Analysis: {exp}\n")
                    md.append("")
                else:
                    md.append(f"• {str(item).strip()}\n")
            md.append("\n---\n")

# 6. Mains Analytical Practice Question
        if raw_data.get("mains_question"):
            md.append("### ✍️ Mains Analytical Question\n")
            md.append(f"Practice Question:\n")
            md.append(f"*{raw_data['mains_question'].strip()}*\n")
            md.append("\n*Framing Focus: 250 Words | 15 Marks | Structural Dialectic*")

        response = {
            "status": "success",
            "title": headline,
            "object_key": object_key,
            "markdown": "\n".join(md)
        }
        
        # Invalidate/cache with new format
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
    if cached and "xml" in cached:
        return Response(content=cached["xml"], media_type="application/xml")

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

    # 2. Static Constitutional Foundations (Neutral, High-Search GS-II Terms)
    for ch_num, title in LAXMIKANTH_8TH_EDITION.items():
        # Formats clean slugs: /static-polity/historical-underpinnings-and-evolution
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

    # Generate pretty-printed XML string
    rough_string = ET.tostring(urlset, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    # Cache for 12 hours so the database is not strained by crawler visits
    set_cached_json(cache_key, {"xml": pretty_xml}, ex_seconds=43200)

    return Response(content=pretty_xml, media_type="application/xml")

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
    search: Optional[str] = Query(None, description="Search by number or title"),
    part: Optional[str] = Query(None, description="Filter by Part number"),
    cluster: Optional[str] = Query(None, description="Filter by doctrine cluster"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Lists constitutional provisions with metadata and matrix flags."""
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

    return {
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

@app.get("/api/admin/articles/{article_id}")
def get_admin_article_detail(article_id: int, db: Session = Depends(get_db)):
    """Fetches complete verbatim text, bare act analysis, 4-epoch trajectory, and matrix."""
    article = db.query(ConstitutionalArticleModel).filter(ConstitutionalArticleModel.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Constitutional Article not found")

    return {
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
    search: Optional[str] = Query(None, description="Search question text or explanation"),
    year: Optional[int] = Query(None, description="Filter by exam year"),
    chapter_number: Optional[int] = Query(None, description="Filter by Laxmikanth chapter (1-92)"),
    article_number: Optional[int] = Query(None, description="Filter by mapped article integer"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Lists Prelims PYQs with article array containment and taxonomy filters."""
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

    return {
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

@app.patch("/api/admin/pyqs/{pyq_id}")
def update_pyq_metadata(
    pyq_id: int, 
    payload: UpdatePYQMappingRequest, 
    db: Session = Depends(get_db)
):
    """Updates taxonomy chapter links, difficulty, or mapped constitutional article IDs."""
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
    return {"status": "success", "message": f"PYQ #{pyq.id} updated"}

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

from src.database.models import IssueDossierModel, IssueGraphEdgeModel
from sqlalchemy.orm import selectinload

@app.get("/api/admin/dossiers")
def get_admin_dossiers(
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
        query = query.filter(IssueDossierModel.status == status.strip().lower())

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

    return {
        "status": "success",
        "total": total,
        "dossiers": data
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

