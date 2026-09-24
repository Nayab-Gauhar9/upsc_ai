from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import User, UserQueryHistory
from src.rag.retriever import UPSCRetriever
from src.pipelines.rag_generator import UPSCRAGGenerator

app = FastAPI(title="UPSC AI Mentor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = UPSCRetriever()
generator = UPSCRAGGenerator(model_name="openai/gpt-oss-120b")

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request schemas
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
    firebase_uid: Optional[str] = None  # Passed from FlutterFlow Authenticated User ID

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
