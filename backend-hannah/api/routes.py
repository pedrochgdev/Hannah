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
import io
import tempfile
import soundfile as sf
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionHistoryResponse,
    TTSRequest,
)
from config import settings
from core.model_selector import ModelSignal
from core.model_selector_v2 import ModelSelectorV2
from core.semantic_cache import SemanticCache
from core.token_handler import TokenHandler
from rag.rag_component import RAGComponent

from faster_whisper import WhisperModel
from kokoro import KPipeline

router = APIRouter()

# ── Shared component instances (initialised once at import time) ──────
# These are intentionally module-level singletons so FastAPI's dependency
# injection is not needed for stateful objects that are expensive to create.

_token_handler  = TokenHandler()
_semantic_cache = SemanticCache()
_model_selector = ModelSelectorV2()
_session_signals: dict[str, str] = {}
_RAG_DB = os.path.join(os.path.dirname(__file__), "..", "rag", "hannah_knowledge")
_rag = RAGComponent(db_path=_RAG_DB)

print("Cargando Faster-Whisper...")
_whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

print("Cargando Kokoro TTS...")
_tts_pipeline = KPipeline(lang_code='a')
_tts_voice = 'af_heart'

# ── Audio Endpoints (NUEVOS) ──────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Recibe el audio del frontend, lo guarda temporalmente y lo transcribe con Faster Whisper.
    """
    try:
        # 1. Guardar el archivo subido en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # 2. Transcribir
        segments, info = _whisper_model.transcribe(tmp_path, beam_size=5)

        # 3. Unir los segmentos de texto
        text = " ".join([segment.text for segment in segments])

        # 4. Limpiar el archivo temporal
        os.remove(tmp_path)

        return {"text": text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en transcripción: {str(e)}")

@router.post("/tts")
async def generate_tts(request: TTSRequest):
    """
    Recibe texto y devuelve un flujo de audio WAV generado por Kokoro.
    """
    try:
        # Generar el audio con Kokoro
        # KPipeline devuelve un generador, tomamos el primer resultado
        generator = _tts_pipeline(request.text, voice=_tts_voice, speed=1.0, split_pattern=r'\n+')

        all_audio = []
        sample_rate = 24000

        for _, _, audio_data in generator:
            all_audio.extend(audio_data)

        if not all_audio:
            raise HTTPException(status_code=500, detail="No se pudo generar audio.")

        # Escribir a un buffer en memoria en formato WAV
        buffer = io.BytesIO()
        sf.write(buffer, all_audio, sample_rate, format='WAV')
        buffer.seek(0)

        # Devolver como StreamingResponse
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en TTS: {str(e)}")

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
    context["prev_signal"] = _session_signals.get(request.session_id, "fast")
    signal, confidence = _model_selector.select(
        prompt=request.prompt,
        context=context,
    )
    _session_signals[request.session_id] = signal.value

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

@router.post("/chat/regenerate", response_model=ChatResponse)
async def regenerate(request: ChatRequest) -> ChatResponse:
    # Quitar el último turno — es la respuesta que queremos reemplazar
    _token_handler.pop_last_turn(request.session_id)

    # Construir contexto sin ese último turno
    context = _token_handler.build_context(
        session_id=request.session_id,
        current_prompt=request.prompt,
    )

    # RAG en modo extended siempre
    rag_result = _rag.retrieve(request.prompt, mode="extended")
    rag_context = rag_result["formatted_context"]

    # Directo al slow model
    response_text = await _call_model(ModelSignal.SLOW, context, rag_context)

    # Guardar nuevo turno
    _token_handler.record_turn(
        session_id=request.session_id,
        user_message=request.prompt,
        assistant_message=response_text,
    )

    _session_signals[request.session_id] = ModelSignal.SLOW.value

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        model_used="slow",
        cache_hit=False,
        cache_similarity=None,
        model_signal="slow",
        selector_confidence=1.0,
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
    _session_signals.pop(session_id, None)
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
