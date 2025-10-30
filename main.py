# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
from difflib import SequenceMatcher
import os
import httpx
import uvicorn
import logging

# ---------- CONFIG ----------
load_dotenv()
MONGO_URI      = os.getenv("MONGODB_URI")
DB_NAME        = os.getenv("DB_NAME")
COLLECTION     = os.getenv("COLLECTION_NAME")
EMBED_PORT     = os.getenv("EMBED_SERVICE_PORT", "8001")
EMBED_URL      = f"http://localhost:{EMBED_PORT}/embed"
INDEX_NAME     = "hybrid_index"

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-tools-search")

# ---------- FASTAPI ----------
app = FastAPI(title="AI Tools Search API", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODELS ----------
class SearchResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    score: float
    match_type: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    search_type: str = "hybrid"  # hybrid | semantic | exact


# ---------- LIFECYCLE ----------
client, db, collection = None, None, None


@app.on_event("startup")
async def startup():
    global client, db, collection
    client = AsyncIOMotorClient(MONGO_URI)
    db, collection = client[DB_NAME], client[DB_NAME][COLLECTION]
    logger.info("✅ MongoDB connected")


@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()
        logger.info("❌ MongoDB closed")


# ---------- UTILS ----------
async def get_embedding(text: str) -> List[float]:
    """Call local embedding service to get text embedding."""
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            resp = await cli.post(EMBED_URL, json={"text": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
    except Exception as e:
        logger.exception("Embedding service error")
        raise HTTPException(500, f"Embedding service error: {e}")


def format_search_result(doc: dict, score: Optional[float], match_type: str) -> SearchResponse:
    """Convert MongoDB doc to Pydantic response model."""
    return SearchResponse(
        id=str(doc.get("_id", "")),
        name=doc.get("name", ""),
        description=doc.get("description", ""),
        score=float(score or 0.0),
        match_type=match_type,
    )


def text_similarity(a: str, b: str) -> float:
    """Compute string similarity ratio between 0 and 1."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------- SEARCH CORE ----------
async def exact_search(query: str, limit: int) -> List[SearchResponse]:
    """Perform exact text + autocomplete search with reranking."""
    pipeline = [
        {
            "$search": {
                "index": INDEX_NAME,
                "compound": {
                    "should": [
                        {"text": {"query": query, "path": "name", "score": {"boost": {"value": 5}}}},
                        {"autocomplete": {"query": query, "path": "name", "score": {"boost": {"value": 3}}}},
                        {"text": {"query": query, "path": "description"}},
                        {"text": {"query": query, "path": "tags"}},
                    ],
                    "minimumShouldMatch": 1,
                },
            }
        },
        {"$addFields": {"search_score": {"$meta": "searchScore"}}},
        {"$limit": limit * 3},  # fetch more results for better reranking
    ]

    results = [
        format_search_result(doc, doc.get("search_score"), "exact")
        async for doc in collection.aggregate(pipeline)
    ]

    # Custom reranking: exact matches and prefix matches boosted
    def rerank(item: SearchResponse) -> float:
        name_lower = item.name.lower()
        query_lower = query.lower()
        if name_lower == query_lower:
            return item.score + 10.0  # exact match
        elif name_lower.startswith(query_lower):
            return item.score + 5.0   # prefix match
        return item.score

    results.sort(key=lambda x: rerank(x), reverse=True)
    return results[:limit]


async def semantic_search(query: str, limit: int) -> List[SearchResponse]:
    """Perform semantic vector-based search with text similarity rerank."""
    vector = await get_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": limit * 10,
                "limit": limit * 3,
            }
        },
        {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
    ]

    results = [
        format_search_result(doc, doc.get("vector_score"), "semantic")
        async for doc in collection.aggregate(pipeline)
    ]

    # Add string similarity boost
    for r in results:
        sim = text_similarity(query, r.name)
        if r.name.lower() == query.lower():
            r.score += 10.0
        elif r.name.lower().startswith(query.lower()):
            r.score += 5.0
        else:
            r.score += sim * 5.0

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


async def hybrid_search(query: str, limit: int) -> List[SearchResponse]:
    """Combine vector and text search results, then rerank."""
    vector = await get_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": limit * 10,
                "limit": limit * 3,
            }
        },
        {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
        {
            "$unionWith": {
                "coll": COLLECTION,
                "pipeline": [
                    {
                        "$search": {
                            "index": INDEX_NAME,
                            "compound": {
                                "should": [
                                    {"text": {"query": query, "path": "name", "score": {"boost": {"value": 5}}}},
                                    {"autocomplete": {"query": query, "path": "name", "score": {"boost": {"value": 3}}}},
                                    {"text": {"query": query, "path": "description"}},
                                ],
                            },
                        }
                    },
                    {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
                    {"$limit": limit * 3},
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "name": {"$first": "$name"},
                "description": {"$first": "$description"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
            }
        },
        {
            "$project": {
                "name": 1,
                "description": 1,
                "score": {
                    "$add": [
                        {"$ifNull": ["$vs_score", 0]},
                        {"$ifNull": ["$fts_score", 0]},
                    ]
                },
            }
        },
        {"$sort": {"score": -1}},
    ]

    results = [
        format_search_result(doc, doc.get("score"), "hybrid")
        async for doc in collection.aggregate(pipeline)
    ]

    # Post reranking
    for r in results:
        sim = text_similarity(query, r.name)
        if r.name.lower() == query.lower():
            r.score += 10.0
        elif r.name.lower().startswith(query.lower()):
            r.score += 5.0
        else:
            r.score += sim * 3.0

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


# ---------- ROUTES ----------
@app.get("/")
async def root():
    return {"message": "AI Tools Search API 🚀"}


@app.get("/health")
async def health():
    await client.admin.command("ping")
    return {"status": "healthy", "mongodb": "connected", "count": await collection.count_documents({})}


@app.post("/search", response_model=List[SearchResponse])
async def search_tools(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if not 1 <= request.limit <= 100:
        raise HTTPException(400, "Limit must be 1-100")

    try:
        if request.search_type == "exact":
            return await exact_search(request.query, request.limit)
        elif request.search_type == "semantic":
            return await semantic_search(request.query, request.limit)
        else:
            return await hybrid_search(request.query, request.limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(500, f"Search failed: {e}")


@app.get("/search", response_model=List[SearchResponse])
async def search_tools_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    type: str = Query("hybrid", regex="^(hybrid|semantic|exact)$"),
):
    """GET endpoint for browser-based testing."""
    return await search_tools(SearchRequest(query=q, limit=limit, search_type=type))


# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
