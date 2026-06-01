# Hannah — Virtual AI Companion

> Aplicación de AI companion con modelo de lenguaje propio entrenado desde cero. Arquitectura dual: Hannah 360M (modelo fast, custom) + Qwen2.5-14B (modelo slow). Incluye RAG, síntesis de voz, reconocimiento de audio, y avatar 3D.

---

## Demo

```
Frontend  → http://localhost:3000
Backend   → http://localhost:8000
Hannah    → http://localhost:8001  (modelo 360M)
Qwen      → http://localhost:8002  (modelo 14B)
```

---

## Arquitectura del Sistema

```
Usuario (voz / texto)
        │
        ▼
  ┌─────────────┐
  │  Frontend   │  HTML + CSS + JS  (puerto 3000)
  │  (chat UI)  │  Avatar 3D, TTS playback, mic input
  └──────┬──────┘
         │ HTTP
         ▼
  ┌─────────────────────────────────────────────────┐
  │              Backend FastAPI (8000)              │
  │                                                  │
  │  /api/v1/chat  →  Token Handler                 │
  │                        │                         │
  │              ┌─────────┴──────────┐              │
  │              │   Semantic Cache   │              │
  │              └─────────┬──────────┘              │
  │                        │ MISS                    │
  │              ┌─────────┴──────────┐              │
  │              │   Model Selector   │  SVM/NB      │
  │              └────┬──────────┬────┘              │
  │                FAST         SLOW                 │
  │                  │            │                  │
  │          RAG simplified  RAG extended            │
  │          (~75 tokens)    (~330 tokens)           │
  │                  │            │                  │
  │                  ▼            ▼                  │
  │           Hannah 360M    Qwen2.5-14B             │
  │           (puerto 8001)  (puerto 8002)           │
  └─────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │  TTS / ASR  │  Kokoro (síntesis) · Whisper (transcripción)
  └─────────────┘
```

---

## Modelo Hannah 360M

Modelo de lenguaje entrenado desde cero sobre arquitectura OLMo3.

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | OLMo3 Transformer |
| Parámetros | 367.6M |
| Vocab size | 32,000 |
| d_model | 1024 |
| n_layers | 24 |
| n_heads | 16 |
| Hardware entrenamiento | RTX 5070 Ti 16GB |

Pipeline de entrenamiento (scripts **no incluidos** en este repo):

```
Pretraining → SFT Conversacional → RAG Fine-Tuning → SFT Personalidad
```

El modelo final (`hannah_personality_final.pt`) aprende a:
- Conversar en formato `[SYS][USR][ASS]`
- Leer y usar contexto del bloque `[MEMORY]`
- Responder con la voz y personalidad de Hannah (cálida, juguetona, natural)
- Manejar prompts incomprensibles e idioma incorrecto

Los pesos del modelo **no están en este repositorio** por tamaño (~1.4GB). Se cargan en `backend-hannah/model/`.

---

## Estructura del Repositorio

```
Hannah/
│
├── backend-hannah/
│   ├── app.py                      # FastAPI principal — endpoints /chat, /tts
│   ├── config.py                   # Configuración global
│   ├── requirements.txt
│   │
│   ├── server/
│   │   ├── hannah_model_server.py  # Servidor Hannah 360M (puerto 8001)
│   │   └── qwen_model_server.py    # Servidor Qwen2.5-14B (puerto 8002)
│   │
│   ├── core/
│   │   ├── model_selector.py       # Clasificador fast/slow (SVM/NaiveBayes)
│   │   └── token_handler.py        # Gestión de tokens y contexto
│   │
│   ├── rag/
│   │   ├── rag_component.py        # Orquestador principal del RAG
│   │   ├── hannah_pipeline.py      # Pipeline completo RAG + modelo
│   │   ├── vector_store.py         # ChromaDB — base vectorial
│   │   ├── embeddings.py           # all-MiniLM-L6-v2 (384 dims)
│   │   ├── semantic_cache.py       # Caché semántico (threshold 0.92)
│   │   ├── query_enhancer.py       # Expansión de queries (modo extended)
│   │   ├── context_handler.py      # Formateo [MEMORY]...[/MEMORY]
│   │   ├── user_profile.py         # Perfil de sesión del usuario
│   │   ├── ingest_knowledge.py     # Ingesta de conocimiento a ChromaDB
│   │   └── hannah_knowledge/       # Base vectorial persistente (ChromaDB)
│   │
│   ├── model/
│   │   ├── hannah_personality_final.pt   # ← modelo activo (no en repo)
│   │   └── hannah_dpo_v1_final.pt        # versión anterior (no en repo)
│   │
│   ├── tokenizer/
│   │   └── hannah_tok/             # Tokenizador SentencePiece (32k vocab)
│   │
│   └── data/
│       └── model_selector.joblib   # Clasificador fast/slow entrenado
│
├── frontend-hannah/
│   ├── templates/
│   │   └── index.html              # UI principal del chat
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── chat.js             # Lógica de chat, TTS, micrófono
│
├── scripts/                        # Utilidades y herramientas
├── src/                            # Código fuente auxiliar
├── start.sh                        # Script de arranque (fish shell)
├── requirements.txt
└── README.md
```

---

## Instalación

### Requisitos

- Python 3.11+
- CUDA 12+ con GPU NVIDIA (para Hannah 360M y Qwen)
- fish shell (para `start.sh`)

### Dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales:

- `torch`, `transformers`, `fastapi`, `uvicorn`
- `chromadb`, `sentence-transformers`
- `olmo-core` (arquitectura del modelo)
- `kokoro` (TTS)
- `openai-whisper` (ASR)

### Configuración inicial

```bash
# 1. Colocar el modelo en backend-hannah/model/
#    hannah_personality_final.pt  (~1.4GB, no incluido en repo)

# 2. Ingestar conocimiento de Hannah en ChromaDB
cd backend-hannah
python rag/ingest_knowledge.py

# 3. Entrenar el model selector
python train_selector.py
```

---

## Arranque

```bash
./start.sh
```

El script levanta los 4 servicios en paralelo:

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Frontend | 3000 | UI del chat (HTTP server) |
| Backend FastAPI | 8000 | API principal |
| Hannah 360M | 8001 | Modelo fast (custom) |
| Qwen2.5-14B | 8002 | Modelo slow (Ollama/HF) |

---

## Pipeline de Procesamiento de Mensajes

Cada mensaje del usuario pasa por:

1. **Token Handler** — prepara el mensaje y el historial
2. **Semantic Cache** — si hay una respuesta similar cacheada (score ≥ 0.92), la retorna directamente
3. **Model Selector** — clasifica el mensaje como `fast` o `slow` usando un SVM entrenado con 9 features
4. **RAG retrieval** — busca en ChromaDB conocimiento relevante de Hannah
   - Modo `simplified`: 1-3 chunks, ~75 tokens
   - Modo `extended`: 5-10 chunks, ~330 tokens
   - Si el score < 0.35: no se inyecta contexto
5. **User Profile** — inyecta hechos del usuario detectados en sesión (nombre, trabajo, etc.) en `[MEMORY]`
6. **Generación** — Hannah 360M (fast) o Qwen2.5-14B (slow)
7. **TTS** — Kokoro sintetiza la respuesta en audio

---

## RAG — Base de Conocimiento

Hannah tiene una base vectorial (ChromaDB) con conocimiento sobre sí misma: personalidad, hechos, preferencias, y contexto conversacional.

Embeddings: `all-MiniLM-L6-v2` (384 dimensiones)

Para añadir conocimiento nuevo:

```bash
python rag/ingest_knowledge.py
```

Para diagnosticar qué score obtiene una query:

```python
from rag.rag_component import RAGComponent
rag = RAGComponent(db_path="rag/hannah_knowledge")
results = rag.debug_relevance("what's your favorite movie?")
for r in results:
    print(f"score={r['score']:.3f} | {r['text'][:60]}")
```

---

## Model Selector

Clasificador ligero (SVM o Naive Bayes) que decide si el mensaje va a Hannah 360M (fast) o Qwen2.5-14B (slow), basado en 9 features:

| Feature | Descripción |
|---------|-------------|
| prompt_token_len | Longitud estimada en tokens |
| complexity_kw_count | Keywords de complejidad (explain, write, compare...) |
| sentence_count | Número de oraciones |
| question_count | Cantidad de signos `?` |
| temporal_marker_count | Marcadores temporales (since, lately...) |
| multi_topic_count | Conjunciones multi-tema (and also, besides...) |
| history_turns | Turnos de historial disponibles |
| avg_assistant_len | Largo promedio de respuestas anteriores |
| prompt_char_len | Largo en caracteres |

Para reentrenar con nuevos ejemplos:

```bash
cd backend-hannah
python train_selector.py
```

---

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| Hannah no recuerda el nombre del usuario | User Profile no detectó el patrón | Verificar `rag/user_profile.py` |
| RAG siempre retorna vacío | Score < 0.35 para todos los docs | Bajar threshold con `rag.adjust_relevance_threshold(0.25)` |
| CUDA OOM al cargar ambos modelos | Insuficiente VRAM | Cargar Qwen en CPU o usar cuantización |
| `model_selector.joblib` no encontrado | No se corrió `train_selector.py` | `python train_selector.py` |
| TTS sin audio | Kokoro no iniciado | Verificar servicio en `start.sh` |

---

**Última actualización:** Junio 2026
