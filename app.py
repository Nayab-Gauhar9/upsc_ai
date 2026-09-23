
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag.retriever import UPSCRetriever  # Ensure class name matches your retriever
from src.pipelines.rag_generator import UPSCRAGGenerator

app = FastAPI(title="UPSC AI Mentor API")

# Enable CORS so your web and mobile apps can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = UPSCRetriever()
generator = UPSCRAGGenerator(model_name="openai/gpt-oss-120b") # Configured with your local LLM

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class GenerateRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/api/rag/search")
async def search_notes(payload: SearchRequest):
    try:
        results = retriever.retrieve_smart_notes_for_query(payload.query, top_k=payload.top_k)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/generate")
async def generate_answer(payload: GenerateRequest):
    try:
        # 1. Retrieve full notes from MinIO via vector matching
        retrieved_articles = retriever.retrieve_smart_notes_for_query(payload.question, top_k=payload.top_k)
        
        # 2. Generate expert UPSC response with foundational/evolutionary context
        answer = generator.generate_expert_response(payload.question, retrieved_articles)
        
        return {
            "status": "success", 
            "sources_count": len(retrieved_articles),
            "model_answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "main":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
