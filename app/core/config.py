from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Chatbot App"
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "chatbot_db"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )


settings = Settings()