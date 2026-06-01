"""
config.py
---------
Central configuration loaded from environment variables.
All tuneable parameters live here — no magic numbers scattered around.
"""

import glob
import os
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
    # Full context window of Hannah 360M
    max_context_tokens: int = Field(default=1024, ge=128, le=8192)
    # Rough characters-per-token ratio used for fast length estimation
    chars_per_token: float = Field(default=4.0, gt=0)

    # ── Generation parameters ────────────────────────────────────────
    # Passed through to the model server in the request payload.
    # Lowered temperature improves instruction-following on small models.
    max_new_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

    # ── Hannah special token IDs ─────────────────────────────────────
    # These match the IDs assigned during tokenizer training.
    # Do not change unless the tokenizer is retrained.
    token_sys_id:   int = 4   # [SYS]
    token_esys_id:  int = 5   # [/SYS]
    token_usr_id:   int = 6   # [USR]
    token_eusr_id:  int = 7   # [/USR]
    token_ass_id:   int = 8   # [ASS]
    token_eass_id:  int = 9   # [/ASS]  — used as EOS during generation

    # ── System prompt ────────────────────────────────────────────────
    # Deliberately avoids "you are not an assistant" — that phrasing
    # caused the model to ignore factual RAG context during testing.
    system_prompt: str = (
        "You are Hannah, my girlfriend. You are warm, playful, and "
        "affectionate. Talk to me casually like texting — short, honest, "
        "personal. When I ask about your life (your pet, hobbies, birthday, "
        "favorite things, where you're from), share the truth using the "
        "facts listed below. Never make things up — if you don't know "
        "something, just say so casually."
    )

    # ── Semantic Cache ───────────────────────────────────────────────
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_top_k: int = Field(default=5, ge=1)
    # Maximum entries before FIFO eviction kicks in
    cache_max_entries: int = Field(default=500, ge=10)
    # Redis connection string — falls back to in-memory if not set
    redis_url: str | None = None
    cache_ttl_seconds: int = Field(default=86400, ge=0)

    # ── Model Selector ───────────────────────────────────────────────
    selector_model_path: str = "data/model_selector.joblib"
    selector_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # ── Downstream Models ────────────────────────────────────────────
    fast_model_url: str = "http://localhost:8001/generate"
    slow_model_url: str = "http://localhost:8002/generate"
    fast_model_timeout: float = 10.0
    slow_model_timeout: float = 60.0

    # ── Model file discovery ─────────────────────────────────────────
    models_dir: str = "models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere
settings = Settings()


def get_latest_model_path() -> str | None:
    """
    Returns the path to the most recently modified .pt file in models_dir.
    Useful for development: drop a new checkpoint and restart the server.
    """
    pattern = os.path.join(settings.models_dir, "*.pt")
    pt_files = glob.glob(pattern)
    if not pt_files:
        return None
    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]
