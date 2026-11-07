import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne
from pypdf.errors import PdfReadError

from app.core.config import get_ollama_embeddings, settings
from app.core.database import get_database
from app.core.ingestion import process_pdf

router = APIRouter()
DOCS_PDF = Path(__file__).resolve().parents[2] / "docs" / "RAG_Practice_Learning.pdf"


@router.post("/")
async def create_embedding(
    model: Annotated[str, Query(min_length=1)] = settings.OLLAMA_EMBEDDING_MODEL,
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> dict[str, int | str]:
    if not DOCS_PDF.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"PDF was not found at {DOCS_PDF}.",
        )

    try:
        pdf_content = await asyncio.to_thread(DOCS_PDF.read_bytes)
        chunks = await asyncio.to_thread(
            process_pdf,
            pdf_content,
            DOCS_PDF.name,
        )
    except PdfReadError as error:
        raise HTTPException(
            status_code=400,
            detail="The PDF in the docs folder could not be read.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        embeddings = await asyncio.to_thread(
            get_ollama_embeddings,
            [chunk.page_content for chunk in chunks],
            model,
        )
    except HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {error.code} while creating embeddings.",
        ) from error
    except URLError as error:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Ollama. Check that Ollama is running.",
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    operations = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        text = chunk.page_content
        metadata = chunk.metadata
        chunk_id = hashlib.sha256(
            f"{metadata['source']}\0{metadata['page']}\0{text}".encode("utf-8")
        ).hexdigest()
        operations.append(
            UpdateOne(
                {"_id": chunk_id},
                {
                    "$setOnInsert": {
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata,
                    }
                },
                upsert=True,
            )
        )

    result = await db["document_ingestion"].bulk_write(operations, ordered=False)
    return {
        "message": "PDF processed and embedded successfully.",
        "chunk_count": len(chunks),
        "inserted_count": result.upserted_count,
        "duplicate_count": len(chunks) - result.upserted_count,
    }
