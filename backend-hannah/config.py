"""
config.py
---------
Central configuration loaded from environment variables.
All tuneable parameters live here.
"""

import glob
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # API
    app_title: str = "Hannah Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # Token Handler
    max_history_turns: int = Field(default=8, ge=1, le=32)
    max_context_tokens: int = Field(default=1024, ge=128, le=8192)
    chars_per_token: float = Field(default=4.0, gt=0)

    # Generation parameters
    max_new_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

    # Hannah special token IDs
    token_sys_id:   int = 4
    token_esys_id:  int = 5
    token_usr_id:   int = 6
    token_eusr_id:  int = 7
    token_ass_id:   int = 8
    token_eass_id:  int = 9

    # System prompt
    system_prompt: str = (
        "You are Hannah, my girlfriend. You are warm, playful, and "
        "affectionate. Talk to me casually like texting — short, honest, "
        "personal. When I ask about your life (your pet, hobbies, birthday, "
        "favorite things, where you're from), share the truth using the "
        "facts listed below. Never make things up — if you don't know "
        "something, just say so casually."
    )

    # Semantic Cache
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_top_k: int = Field(default=5, ge=1)
    cache_max_entries: int = Field(default=500, ge=10)
    redis_url: str | None = None
    cache_ttl_seconds: int = Field(default=86400, ge=0)

    # Auth
    jwt_secret: str = "hannah-local-secret-2026"
    session_ttl_seconds: int = Field(default=1800, ge=60)

    # Model Selector
    selector_model_path: str = "data/model_selector.joblib"
    selector_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # Downstream Models
    fast_model_url: str = "http://localhost:8001/generate"
    slow_model_url: str = "http://localhost:8002/generate"
    fast_model_timeout: float = 10.0
    slow_model_timeout: float = 60.0

    # Model file discovery
    models_dir: str = "models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def get_latest_model_path() -> str | None:
    """Returns the most recently modified .pt file in models_dir."""
    pattern = os.path.join(settings.models_dir, "*.pt")
    pt_files = glob.glob(pattern)
    if not pt_files:
        return None
    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]
