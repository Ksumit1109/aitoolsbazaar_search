import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from tqdm.asyncio import tqdm
import certifi

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION = os.getenv("COLLECTION_NAME")

# Initialize model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

async def generate_embeddings():
    # Connect to MongoDB with SSL certificate
    client = AsyncIOMotorClient(
        MONGO_URI,
        tlsCAFile=certifi.where()
    )
    db = client[DB_NAME]
    collection = db[COLLECTION]
    
    # Get total count
    total = await collection.count_documents({})
    print(f"Total documents: {total}")
    
    # Get documents without embeddings
    documents_without_embeddings = await collection.count_documents({"embedding": {"$exists": False}})
    print(f"Documents without embeddings: {documents_without_embeddings}")
    
    # Process in batches
    batch_size = 100
    updated_count = 0
    
    cursor = collection.find({"embedding": {"$exists": False}})
    
    batch = []
    
    # Create progress bar
    pbar = tqdm(total=documents_without_embeddings, desc="Processing documents")
    
    async for doc in cursor:
        # Create text to embed (combine name and description)
        text_to_embed = f"{doc.get('name', '')} {doc.get('description', '')}"
        
        batch.append({
            "_id": doc["_id"],
            "text": text_to_embed
        })
        
        # Process batch
        if len(batch) >= batch_size:
            texts = [item["text"] for item in batch]
            embeddings = model.encode(texts).tolist()
            
            # Update documents
            for item, embedding in zip(batch, embeddings):
                await collection.update_one(
                    {"_id": item["_id"]},
                    {"$set": {"embedding": embedding}}
                )
            
            updated_count += len(batch)
            pbar.update(len(batch))
            batch = []
    
    # Process remaining batch
    if batch:
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts).tolist()
        
        for item, embedding in zip(batch, embeddings):
            await collection.update_one(
                {"_id": item["_id"]},
                {"$set": {"embedding": embedding}}
            )
        
        updated_count += len(batch)
        pbar.update(len(batch))
    
    pbar.close()
    print(f"\n✅ Total documents updated: {updated_count}")
    client.close()

if __name__ == "__main__":
    asyncio.run(generate_embeddings())