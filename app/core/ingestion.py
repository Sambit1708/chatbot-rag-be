## Data ingestion
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from motor.motor_asyncio import AsyncIOMotorDatabase
from pypdf import PdfReader

from app.core.config import get_embeddings


def split_pdf_into_chunks(data: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    return text_splitter.split_documents(data)


def process_pdf(pdf_content: bytes, filename: str) -> list[Document]:
    reader = PdfReader(BytesIO(pdf_content))
    pages = [
        Document(
            page_content=text,
            metadata={"source": filename, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
        if (text := page.extract_text() or "").strip()
    ]
    if not pages:
        raise ValueError("The PDF contains no extractable text.")

    return split_pdf_into_chunks(pages)


class DataIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_pdf(self):
        loader = PyPDFLoader(self.file_path)
        return loader.load()

    def split_pdf_into_chunks(self, data: list[Document]) -> list[Document]:
        return split_pdf_into_chunks(data)

    def document_embedded(self):
        data = self.load_pdf()
        documents = self.split_pdf_into_chunks(data)
        embeddings = get_embeddings([doc.page_content for doc in documents])
        docs_to_insert = [
            {
                "text": doc.page_content,
                "embedding": embedding,
            }
            for doc, embedding in zip(documents, embeddings, strict=True)
        ]
        return docs_to_insert

    async def document_ingestion(self, db: AsyncIOMotorDatabase) -> dict[str, int]:
        docs_to_insert = self.document_embedded()
        if not docs_to_insert:
            return {"inserted_count": 0}

        result = await db["document_ingestion"].insert_many(docs_to_insert)
        return {"inserted_count": len(result.inserted_ids)}
