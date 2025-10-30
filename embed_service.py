from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn

app = FastAPI(title="Embedding Service")

# Load model on startup (adjust model based on your needs)
# Options: 'all-MiniLM-L6-v2' (384 dim), 'all-mpnet-base-v2' (768 dim)
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
    return {"message": "Embedding Service Running", "model": "all-MiniLM-L6-v2"}

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
    uvicorn.run(
        "embed_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )