"""
api/routes.py
-------------
FastAPI router con autenticacion estatica y soporte multi-sesion.

Pipeline:
  POST /api/v1/auth/token  — login, obtener JWT
  POST /api/v1/chat        — turno de conversacion (requiere token)
  POST /api/v1/transcribe  — ASR (Whisper)
  POST /api/v1/tts         — TTS (Kokoro)
  GET  /api/v1/health      — estado del sistema
"""
from __future__ import annotations

import io
import os
import tempfile

import httpx
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionHistoryResponse,
    TTSRequest,
)
from config import settings
from core.auth import create_token, verify_token, TokenData
from core.inference_queue import InferenceQueue
from core.model_selector import ModelSelector, ModelSignal
from core.semantic_cache import SemanticCache
from core.token_handler import TokenHandler
from rag.rag_component import RAGComponent

from faster_whisper import WhisperModel
from kokoro import KPipeline

router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────
_token_handler  = TokenHandler()
_semantic_cache = SemanticCache()
_model_selector = ModelSelector()

_RAG_DB = os.path.join(os.path.dirname(__file__), "..", "rag", "hannah_knowledge")

# Registry de RAGComponent por tenant_id (lazy init)
_rag_registry: dict[str, RAGComponent] = {}


def _get_rag(tenant_id: str) -> RAGComponent:
    if tenant_id not in _rag_registry:
        _rag_registry[tenant_id] = RAGComponent(db_path=_RAG_DB, tenant_id=tenant_id)
    return _rag_registry[tenant_id]


# Modelos de audio
print("Cargando Faster-Whisper...")
_whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("Cargando Kokoro TTS...")
_tts_pipeline = KPipeline(lang_code='a')
_tts_voice = 'af_heart'

# ── Colas de inferencia ───────────────────────────────────────────────

async def _run_fast(payload: dict) -> str:
    return await _http_call(settings.fast_model_url, payload, settings.fast_model_timeout)

async def _run_slow(payload: dict) -> str:
    return await _http_call(settings.slow_model_url, payload, settings.slow_model_timeout)

_fast_queue = InferenceQueue(worker_fn=_run_fast, max_concurrent=2)
_slow_queue = InferenceQueue(worker_fn=_run_slow, max_concurrent=1)


@router.on_event("startup")
async def _startup() -> None:
    _fast_queue.start()
    _slow_queue.start()


# ── Auth ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/token")
def login(req: LoginRequest) -> dict:
    """Obtener JWT para usar en los demas endpoints."""
    try:
        token = create_token(req.username, req.password)
        return {"access_token": token, "token_type": "bearer"}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


# ── Audio endpoints ───────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        segments, _ = _whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        os.remove(tmp_path)
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en transcripcion: {e}")


@router.post("/tts")
async def generate_tts(request: TTSRequest) -> StreamingResponse:
    try:
        generator = _tts_pipeline(
            request.text, voice=_tts_voice, speed=1.0, split_pattern=r'\n+'
        )
        all_audio: list = []
        for _, _, audio_data in generator:
            all_audio.extend(audio_data)
        if not all_audio:
            raise HTTPException(status_code=500, detail="No se pudo generar audio.")
        buffer = io.BytesIO()
        sf.write(buffer, all_audio, 24000, format="WAV")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en TTS: {e}")


# ── Chat ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: TokenData = Depends(verify_token),
) -> ChatResponse:
    tenant_id = token.tenant_id

    # 1 — Contexto de sesion (desde Redis)
    context = _token_handler.build_context(
        session_id=request.session_id,
        current_prompt=request.prompt,
    )

    # 2 — Cache semantico por tenant
    cache_result = _semantic_cache.lookup(request.prompt, tenant_id=tenant_id)
    if cache_result.hit:
        return ChatResponse(
            session_id=request.session_id,
            response=cache_result.response,
            model_used="cache",
            cache_hit=True,
            cache_similarity=cache_result.similarity,
            turns_in_context=context["turns_used"],
        )

    # 3 — Routing fast/slow
    signal, confidence = _model_selector.select(
        prompt=request.prompt,
        context=context,
    )

    # 3.5 — RAG por tenant
    rag_mode = "extended" if signal == ModelSignal.SLOW else "simplified"
    rag_result = _get_rag(tenant_id).retrieve(request.prompt, mode=rag_mode)
    rag_context = rag_result["formatted_context"]

    # 4 — Inferencia via cola (evita OOM en GPU)
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
    queue = _fast_queue if signal == ModelSignal.FAST else _slow_queue
    response_text = await queue.submit(payload)

    # 5 — Guardar en cache
    _semantic_cache.store(request.prompt, response_text, tenant_id=tenant_id)

    # 6 — Registrar turno en Redis
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


# ── Health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        cache_size=_semantic_cache.size(),
        sessions_active=_token_handler.session_count(),
        selector_trained=_model_selector.is_trained(),
    )


# ── Session management ────────────────────────────────────────────────

@router.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str) -> SessionHistoryResponse:
    history = _token_handler.get_history(session_id)
    return SessionHistoryResponse(session_id=session_id, turns=history)


@router.delete("/session/{session_id}", status_code=200)
async def clear_session(session_id: str) -> dict:
    _token_handler.clear_session(session_id)
    return {}


# ── Internal helper ───────────────────────────────────────────────────

async def _http_call(url: str, payload: dict, timeout: float) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Model timeout ({timeout}s).")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Model error: {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {exc}")
