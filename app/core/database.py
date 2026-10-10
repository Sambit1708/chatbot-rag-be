from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings


class MongoDBManager:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None


db_manager = MongoDBManager()


async def connect_to_mongo():
    """Establish client connection pool on app startup."""
    db_manager.client = AsyncIOMotorClient(settings.MONGODB_URL)
    db_manager.db = db_manager.client[settings.DATABASE_NAME]
    # Ping the server to ensure connection is valid
    await db_manager.client.admin.command("ping")
    print(" Connected to MongoDB successfully.")


async def close_mongo_connection():
    """Close connection pool on app shutdown."""
    if db_manager.client:
        db_manager.client.close()
        print(" MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """FastAPI Dependency for route handlers."""
    if db_manager.db is None:
        raise RuntimeError("Database is not initialized.")
    return db_manager.db