from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn
import os

app = FastAPI(title="Embedding Service")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int

@app.on_event("startup")
async def load_model():
    global model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
    print("✅ Model loaded successfully")

@app.get("/")
async def root():
    return {
        "message": "Embedding Service Running", 
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(request: EmbedRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        embedding = model.encode(request.text).tolist()
        return EmbedResponse(
            embedding=embedding,
            dimensions=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "embed_service:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )