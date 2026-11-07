from pymongo import MongoClient
from typing import List
from app.core.config import settings  # Import settings

def save_embedding(embeddings: List[dict]):
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    collection = db[settings.MONGO_COLLECTION]
    
    for embedding in embeddings:
        collection.insert_one(embedding)
