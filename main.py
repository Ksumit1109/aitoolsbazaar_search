# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
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
    search_type: Optional[str] = "auto"  # auto | hybrid | semantic | exact


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
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            resp = await cli.post(EMBED_URL, json={"text": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
    except Exception as e:
        logger.exception("Embedding service error")
        raise HTTPException(500, f"Embedding service error: {e}")


def format_search_result(doc: dict, score: Optional[float], match_type: str) -> SearchResponse:
    return SearchResponse(
        id=str(doc.get("_id", "")),
        name=doc.get("name", ""),
        description=doc.get("description", ""),
        score=float(score or 0.0),
        match_type=match_type,
    )


# ---------- SEARCH CORE ----------
async def exact_search(query: str, limit: int) -> List[SearchResponse]:
    pipeline = [
        {
            "$search": {
                "index": INDEX_NAME,
                "compound": {
                    "should": [
                        {"text": {"query": query, "path": "name", "score": {"boost": {"value": 5}}}},
                        {"autocomplete": {"query": query, "path": "name", "score": {"boost": {"value": 3}}}},
                        {"text": {"query": query, "path": "description", "score": {"boost": {"value": 2}}}},
                        {"text": {"query": query, "path": "tags"}},
                    ],
                    "minimumShouldMatch": 1,
                }
            }
        },
        {"$addFields": {"search_score": {"$meta": "searchScore"}}},
        {"$sort": {"search_score": -1}},
        {"$limit": limit},
    ]
    results = [format_search_result(doc, doc.get("search_score"), "exact") async for doc in collection.aggregate(pipeline)]

    # Boost exact title matches
    for res in results:
        if res.name.strip().lower() == query.strip().lower():
            res.score += 5

    return sorted(results, key=lambda x: x.score, reverse=True)


async def semantic_search(query: str, limit: int) -> List[SearchResponse]:
    vector = await get_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
    ]
    return [format_search_result(doc, doc.get("vector_score"), "semantic") async for doc in collection.aggregate(pipeline)]


async def hybrid_search(query: str, limit: int) -> List[SearchResponse]:
    vector = await get_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": limit * 10,
                "limit": limit,
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
                                ]
                            }
                        }
                    },
                    {"$limit": limit},
                    {"$addFields": {"fts_score": {"$meta": "searchScore"}}},
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
        # Normalize & weight
        {
            "$project": {
                "name": 1,
                "description": 1,
                "normalized_vs": {
                    "$divide": [{"$ifNull": ["$vs_score", 0]}, 1.0]
                },
                "normalized_fts": {
                    "$divide": [{"$ifNull": ["$fts_score", 0]}, 100.0]
                },
            }
        },
        {
            "$addFields": {
                "score": {
                    "$add": [
                        {"$multiply": ["$normalized_vs", 0.4]},
                        {"$multiply": ["$normalized_fts", 0.6]},
                    ]
                }
            }
        },
        # Boost exact name match
        {
            "$addFields": {
                "exact_match_boost": {
                    "$cond": [
                        {"$eq": [{"$toLower": "$name"}, query.lower()]},
                        1,
                        0
                    ]
                }
            }
        },
        {"$addFields": {"score": {"$add": ["$score", "$exact_match_boost"]}}},
        {"$sort": {"score": -1}},
        {"$limit": limit},
    ]

    results = []
    async for doc in collection.aggregate(pipeline):
        results.append(format_search_result(doc, doc.get("score"), "hybrid"))
    return results


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
    query = request.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    limit = max(1, min(request.limit, 100))
    search_type = (request.search_type or "auto").lower()

    # ---------- AUTO-DETECT SEARCH TYPE ----------
    if search_type == "auto":
        words = query.split()
        if query.endswith("?") or len(words) > 5:
            search_type = "semantic"
        elif len(words) <= 2 and query.replace(" ", "").isalnum():
            search_type = "exact"
        else:
            search_type = "hybrid"
        logger.info(f"🔍 Auto-detected search type: {search_type}")

    # ---------- EXECUTE SEARCH ----------
    try:
        if search_type == "exact":
            return await exact_search(query, limit)
        elif search_type == "semantic":
            return await semantic_search(query, limit)
        else:
            return await hybrid_search(query, limit)
    except Exception as e:
        logger.exception("Search error")
        raise HTTPException(500, f"Search failed: {e}")


@app.get("/search", response_model=List[SearchResponse])
async def search_tools_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    type: str = Query("auto", regex="^(auto|hybrid|semantic|exact)$"),
):
    return await search_tools(SearchRequest(query=q, limit=limit, search_type=type))


# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
