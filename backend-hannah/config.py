"""
config.py
---------
Central configuration loaded from environment variables.
All tuneable parameters live here — no magic numbers scattered around.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── API ──────────────────────────────────────────────────────────
    app_title: str = "Hannah Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Token Handler ────────────────────────────────────────────────
    # Maximum number of conversation turns to keep in context
    max_history_turns: int = Field(default=8, ge=1, le=32)
    # Maximum tokens allowed in the full context window sent to a model
    max_context_tokens: int = Field(default=1024, ge=128, le=8192)
    # Rough characters-per-token ratio used for fast length estimation
    chars_per_token: float = Field(default=4.0, gt=0)

    # ── Semantic Cache ───────────────────────────────────────────────
    # Cosine similarity threshold above which a cache hit is declared
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    # Sentence-transformers model used to embed queries
    embedding_model: str = "all-MiniLM-L6-v2"
    # How many past queries to search (top-k ANN search)
    cache_top_k: int = Field(default=5, ge=1)
    # Redis connection string — falls back to in-memory if not set
    redis_url: str | None = None
    # TTL for cached entries in seconds (24 h default)
    cache_ttl_seconds: int = Field(default=86400, ge=0)

    # ── Model Selector ───────────────────────────────────────────────
    # Path to the persisted sklearn classifier (.joblib)
    selector_model_path: str = "data/model_selector.joblib"
    # Confidence threshold below which the selector defaults to slow
    selector_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # ── Downstream Models ────────────────────────────────────────────
    # These are the endpoints the backend will route requests to.
    # The frontend or infra team wires the actual URLs via env vars.
    fast_model_url: str = "http://localhost:8001/generate"
    slow_model_url: str = "http://localhost:8001/generate"
    fast_model_timeout: float = 10.0   # seconds
    slow_model_timeout: float = 60.0   # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere
settings = Settings()
