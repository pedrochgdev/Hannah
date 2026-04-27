"""
api/schemas.py
--------------
Pydantic models for request/response validation.
These are the contracts between the web app and this backend service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


# ── Inbound ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="UUID that uniquely identifies the user's conversation session.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's current message.",
    )


# ── Outbound ──────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    # Agrega esta línea para decirle a Pydantic que ignore el conflicto
    model_config = ConfigDict(protected_namespaces=()) 
    
    session_id: str
    response: str
    model_used: str = Field(
        description="Which model produced the response: 'cache', 'fast', or 'slow'."
    )
    cache_hit: bool = False
    cache_similarity: float | None = None
    model_signal: str | None = Field(
        default=None,
        description="The routing decision: 'fast' or 'slow'. Null on cache hits.",
    )
    selector_confidence: float | None = None
    turns_in_context: int = Field(
        default=0,
        description="Number of history turns included in the context sent to the model.",
    )


# ── Health / diagnostics ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    cache_size: int
    sessions_active: int
    selector_trained: bool


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]
