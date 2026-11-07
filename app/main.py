from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection, get_database
from app.api.v1 import embeddings, ingestion, items


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DB
    await connect_to_mongo()
    yield
    # Shutdown: Close DB connection
    await close_mongo_connection()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Include routers
app.include_router(items.router, prefix="/api/v1/items", tags=["items"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])


@app.get("/")
async def root():
    get_database()
    return {"message": "Service is up and running"}