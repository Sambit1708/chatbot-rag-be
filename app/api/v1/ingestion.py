from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from google.genai.errors import ClientError
from motor.motor_asyncio import AsyncIOMotorDatabase
from pypdf.errors import PdfReadError

from app.core.config import settings
from app.core.database import get_database
from app.core.ingestion import DataIngestion

router = APIRouter()
DOCS_PDF = Path(__file__).resolve().parents[2] / "docs" / "RAG_Practice_Learning.pdf"


@router.get("/split-pdf")
def split_pdf_into_chunks() -> dict[str, int | list[dict[str, Any]]]:
    if not DOCS_PDF.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Test PDF was not found at {DOCS_PDF}.",
        )

    try:
        ingestion = DataIngestion(str(DOCS_PDF))
        chunks = ingestion.split_pdf_into_chunks(ingestion.load_pdf())
    except PdfReadError as error:
        raise HTTPException(
            status_code=500,
            detail="The test PDF could not be read.",
        ) from error

    return {
        "chunk_count": len(chunks),
        "chunks": [
            {
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "source": DOCS_PDF.name,
                },
            }
            for chunk in chunks
        ],
    }


@router.post("/ingest-pdf")
async def ingest_pdf(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> dict[str, int]:
    if not DOCS_PDF.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Test PDF was not found at {DOCS_PDF}.",
        )
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API is not configured. Set GEMINI_API_KEY in the project's .env file.",
        )

    ingestion = DataIngestion(str(DOCS_PDF))
    try:
        return await ingestion.document_ingestion(db)
    except ClientError as error:
        if error.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini embedding quota exceeded. Wait for the retry period in the error and try again.",
            ) from error
        raise
