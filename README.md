# 🧠 AI Tools Search API

An **intelligent FastAPI-based search backend** that combines **Exact**, **Semantic**, and **Hybrid** search using **MongoDB Atlas Search** and **vector embeddings**.  
It can **automatically detect** which search method (exact, semantic, or hybrid) is most appropriate for a user’s query.

---

## 🚀 Features

- 🧩 **Automatic Search Type Detection**

  - Short queries → Exact search
  - Long descriptive queries → Semantic search
  - Medium queries → Hybrid search

- ⚡ **MongoDB Atlas Vector + Text Search**

  - Combines `$search` and `$vectorSearch` for accurate, context-aware results.

- 🧠 **Embeddings Integration**

  - Works with an external embedding service (e.g., SentenceTransformer or OpenAI).

- 🔍 **Three Search Modes (Manual Override)**
  - `exact` → text and autocomplete search
  - `semantic` → vector similarity
  - `hybrid` → merges both using **Reciprocal Rank Fusion (RRF)**

---

## 🧩 Tech Stack

| Component    | Technology                       |
| ------------ | -------------------------------- |
| Backend      | FastAPI                          |
| Database     | MongoDB Atlas                    |
| Async Driver | Motor                            |
| HTTP Client  | httpx                            |
| Embeddings   | External microservice (`/embed`) |
| Server       | uvicorn                          |
| Env Config   | python-dotenv                    |

---

## ⚙️ Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/ai-tools-search.git
cd ai-tools-search
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create `.env`

```env
MONGODB_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net
DB_NAME=aitoolsbazaar
COLLECTION_NAME=tools
EMBED_SERVICE_PORT=8001
```

---

## ⚙️ Example Embedding Service

Create a separate service to generate embeddings:

```python
# embed_service.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed")
async def embed(data: dict):
    text = data["text"]
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}
```

Run it:

```bash
uvicorn embed_service:app --port 8001
```

---

## ▶️ Run the main API

```bash
uvicorn main:app --reload
```

By default runs at:  
**http://localhost:8000**

---

## 🧠 API Endpoints

### 1️⃣ **Health Check**

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "mongodb": "connected",
  "count": 542
}
```

---

### 2️⃣ **Automatic Search**

```http
POST /search
```

**Body:**

```json
{
  "query": "tools that generate PDF using ChatGPT"
}
```

The backend automatically selects:

- `exact` for short queries
- `semantic` for long descriptive queries
- `hybrid` for medium-length ones

**Response Example:**

```json
[
  {
    "id": "6542a7f3e1",
    "name": "ChatGPT",
    "description": "AI chatbot for text generation",
    "score": 15.4,
    "match_type": "hybrid"
  },
  {
    "id": "6542a8f3e9",
    "name": "ChatGPT PDF Generator",
    "description": "Tool to create PDFs using ChatGPT",
    "score": 13.1,
    "match_type": "hybrid"
  }
]
```

---

### 3️⃣ **Manual Search (GET)**

```http
GET /search?q=chatgpt&type=semantic
```

**Query Params:**
| Name | Type | Description |
|------|------|-------------|
| `q` | string | Search text |
| `limit` | int | Number of results (default: 10) |
| `type` | enum | `exact`, `semantic`, or `hybrid` |

---

## 🧮 How Scoring Works

| Search Type  | Mongo Field                  | Description                           |
| ------------ | ---------------------------- | ------------------------------------- |
| **Exact**    | `$meta: "searchScore"`       | Atlas full-text score                 |
| **Semantic** | `$meta: "vectorSearchScore"` | Vector similarity score               |
| **Hybrid**   | Combination                  | Adds both scores for balanced ranking |

> The **score** measures relevance: higher = more accurate match.

---

## 🧪 Example Frontend Integration

Example React code:

```js
async function searchTools(query) {
  const res = await fetch("http://localhost:8000/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await res.json();
  console.log(data);
}
```

---

## 🧰 MongoDB Index Configuration

### 🔹 Full-Text Search Index (`hybrid_index`)

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "name": { "type": "string" },
      "description": { "type": "string" },
      "tags": { "type": "string" }
    }
  }
}
```

### 🔹 Vector Search Index

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

---

## 🧾 License

MIT © 2025 [Your Name]  
Feel free to modify and use in your own projects.

---
