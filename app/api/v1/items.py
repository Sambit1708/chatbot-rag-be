from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.database import get_database

router = APIRouter()


@router.get("/")
async def list_items(db: AsyncIOMotorDatabase = Depends(get_database)):
    cursor = db["items"].find().to_list(length=100)
    items = await cursor
    # Convert MongoDB ObjectId to string if needed
    for item in items:
        item["_id"] = str(item["_id"])
    return items


@router.post("/")
async def create_item(payload: dict, db: AsyncIOMotorDatabase = Depends(get_database)):
    result = await db["items"].insert_one(payload)
    return {"inserted_id": str(result.inserted_id)}