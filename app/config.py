from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和 .env 文件读取的配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Term-Aware Translation API"
    app_version: str = "0.1.0"
    api_key: str

    glossary_path: Path = Path("data/glossary.csv")
    embedding_model: str = "intfloat/multilingual-e5-small"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "glossary"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"


@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的配置对象，只读取一次 .env。"""
    return Settings()