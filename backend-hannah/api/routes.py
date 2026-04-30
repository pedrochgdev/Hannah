"""
api/routes.py
-------------
FastAPI router wiring the three core components together.

Request pipeline (mirrors the architecture diagram):

  POST /chat
    1. TokenHandler  — build context from session history
    2. SemanticCache — check for a similar cached response
       HIT  -> return cached response, skip steps 3-5
       MISS -> continue
    3. ModelSelector — decide fast or slow signal
    4. Call downstream model (fast or slow) with context
    5. Store response in cache
    6. Record turn in session history
    7. Return response to caller

Additional routes:
  GET /health          — liveness + component diagnostics
  GET /session/{id}    — inspect history for a session
  DELETE /session/{id} — clear session history
"""

from __future__ import annotations

import httpx
import os
from fastapi import APIRouter, HTTPException

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionHistoryResponse,
)
from config import settings
from core.model_selector import ModelSelector, ModelSignal
from core.semantic_cache import SemanticCache
from core.token_handler import TokenHandler
from rag.rag_component import RAGComponent

router = APIRouter()

# ── Shared component instances (initialised once at import time) ──────
# These are intentionally module-level singletons so FastAPI's dependency
# injection is not needed for stateful objects that are expensive to create.

_token_handler  = TokenHandler()
_semantic_cache = SemanticCache()
_model_selector = ModelSelector()
_RAG_DB = os.path.join(os.path.dirname(__file__), "..", "rag", "hannah_knowledge")
_rag = RAGComponent(db_path=_RAG_DB)

# ── Main chat endpoint ────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main entry point.
    Accepts a user prompt tied to a session and returns a model response.
    """

    # Step 1 — Build context from session history
    context = _token_handler.build_context(
        session_id=request.session_id,
        current_prompt=request.prompt,
    )

    # Step 2 — Semantic cache lookup
    cache_result = _semantic_cache.lookup(request.prompt)
    if cache_result.hit:
        return ChatResponse(
            session_id=request.session_id,
            response=cache_result.response,
            model_used="cache",
            cache_hit=True,
            cache_similarity=cache_result.similarity,
            turns_in_context=context["turns_used"],
        )

    # Step 3 — Model routing decision
    signal, confidence = _model_selector.select(
        prompt=request.prompt,
        context=context,
    )

    # Step 3.5 — RAG retrieval ← NUEVO
    rag_mode = "extended" if signal == ModelSignal.SLOW else "simplified"
    rag_result = _rag.retrieve(request.prompt, mode=rag_mode)
    rag_context = rag_result["formatted_context"]

    # Step 4 — Call the selected downstream model
    response_text = await _call_model(signal, context, rag_context)

    # Step 5 — Cache the response for future similar queries
    _semantic_cache.store(request.prompt, response_text)

    # RAG integration note (for when RAG is added):
    # Inject retrieved context INSIDE the [SYS] block, not in [MEMORY] blocks.
    # Hannah 360M was not trained to read [MEMORY] content — empirical tests
    # show the model ignores it. The [SYS] block is where it reliably attends.
    # The Slow Model (Llama) is instruction-tuned and can use [MEMORY] natively.

    # Step 6 — Record this turn in session history
    _token_handler.record_turn(
        session_id=request.session_id,
        user_message=request.prompt,
        assistant_message=response_text,
    )

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        model_used=signal.value,
        cache_hit=False,
        cache_similarity=cache_result.similarity,
        model_signal=signal.value,
        selector_confidence=confidence,
        turns_in_context=context["turns_used"],
    )


# ── Health endpoint ───────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        cache_size=_semantic_cache.size(),
        sessions_active=_token_handler.session_count(),
        selector_trained=_model_selector.is_trained(),
    )


# ── Session management endpoints ──────────────────────────────────────

@router.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str) -> SessionHistoryResponse:
    history = _token_handler.get_history(session_id)
    return SessionHistoryResponse(session_id=session_id, turns=history)


@router.delete("/session/{session_id}", status_code=200)
async def clear_session(session_id: str) -> dict:
    _token_handler.clear_session(session_id)
    return {}

# ── Internal helper ───────────────────────────────────────────────────

async def _call_model(signal: ModelSignal, context: dict, rag_context: str = "") -> str:
    """
    Forward the context to the appropriate downstream model endpoint.

    The downstream models expose a simple HTTP POST interface:
      POST /generate
      Body: {"prompt": "...", "history": [...]}
      Response: {"response": "..."}

    Replace the URL and payload format here to match the actual model
    server interface (llama.cpp, vLLM, custom FastAPI, etc.).
    """
    if signal == ModelSignal.FAST:
        url     = settings.fast_model_url
        timeout = settings.fast_model_timeout
    else:
        url     = settings.slow_model_url
        timeout = settings.slow_model_timeout

    payload = {
        "prompt":             context["prompt"],
        "history":            context["history"],
        "rag_context":        rag_context,
        "max_new_tokens":     settings.max_new_tokens,
        "temperature":        settings.temperature,
        "top_k":              settings.top_k,
        "top_p":              settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Model timeout ({signal.value} model did not respond in {timeout}s).",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Downstream model error: {exc.response.status_code}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error calling {signal.value} model: {exc}",
        )
