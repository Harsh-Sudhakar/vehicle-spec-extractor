"""Runtime settings, loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (OpenRouter, OpenAI-compatible API)
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: float = 90.0

    # Retrieval models (ONNX via fastembed)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Retrieval sizes
    dense_k: int = 25
    sparse_k: int = 25
    rerank_k: int = 30
    final_k: int = 8

    # Storage
    data_dir: Path = Path("data")

    # Bonus OCR for image-only pages (off by default: the assignment is text-only)
    enable_ocr: bool = False

    # Web app limits
    max_upload_mb: int = 60
    rate_limit_per_min: int = 20

    # What the UI opens with: a manual (filename or doc_id) and a question that is run on load
    default_doc: str = "f150-workshop-manual.pdf"
    default_query: str = "Torque for brake caliper bolts"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
