from functools import lru_cache
from pathlib import Path

from google import genai

from pydantic_settings import BaseSettings, SettingsConfigDict

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Chatbot App"
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "chatbot_db"
    GEMINI_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )

settings = Settings()


@lru_cache(maxsize=1)
def _get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in the project's .env file.")

    client = _get_genai_client(settings.GEMINI_API_KEY)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=batch,
        )
        if response.embeddings is None or len(response.embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of embeddings.")
        for embedding in response.embeddings:
            if embedding.values is None:
                raise RuntimeError("Gemini returned an empty embedding.")
            embeddings.append(embedding.values)
    return embeddings


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def get_ollama_embeddings(texts: list[str], model: str) -> list[list[float]]:
    import json
    import math
    from urllib.request import Request, urlopen

    if not texts:
        return []

    embeddings: list[list[float]] = []
    endpoint = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        request = Request(
            endpoint,
            data=json.dumps({"model": model, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            try:
                result = json.loads(response.read())
            except json.JSONDecodeError as error:
                raise RuntimeError("Ollama returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an unexpected embedding response.")
        batch_embeddings = result.get("embeddings")
        if (
            not isinstance(batch_embeddings, list)
            or len(batch_embeddings) != len(batch)
        ):
            raise RuntimeError("Ollama returned an unexpected number of embeddings.")

        for embedding in batch_embeddings:
            if (
                not isinstance(embedding, list)
                or not embedding
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in embedding
                )
            ):
                raise RuntimeError("Ollama returned an invalid embedding vector.")
            embeddings.append([float(value) for value in embedding])

    return embeddings
