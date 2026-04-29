# Hannah AI — Manual de Integración

## RAG Pipeline · Fast Hannah · Slow Hannah · Classifier Model

> **Audiencia:** Equipo de backend (John) y cualquier miembro que conecte componentes.  
> **Alcance:** Cómo cada pieza se enchufa al sistema completo una vez que el RAG está corriendo.

---

## 1. Visión General del Sistema

```
Mensaje del usuario
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    Backend (Web App)                      │
│                                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │                  HannahPipeline                    │   │
│  │                                                    │   │
│  │  1. Caché de respuestas completas                  │   │
│  │        │HIT → retorna directo (saltea LLM + RAG)   │   │
│  │        │MISS ↓                                     │   │
│  │  2. Classifier Model ──► "simplified" | "extended" │   │
│  │        │                                           │   │
│  │  3. RAGComponent.retrieve(query, mode)             │   │
│  │     ├─ Caché de contexto RAG (nivel interno)       │   │
│  │     ├─ QueryEnhancer  ──► expand / HyDE            │   │
│  │     ├─ VectorStore    ──► búsqueda ChromaDB        │   │
│  │     └─ ContextHandler ──► [MEMORY]…[/MEMORY]       │   │
│  │        │                                           │   │
│  │  4a. Fast Hannah 360M    [si mode = simplified]    │   │
│  │  4b. Slow Hannah Qwen    [si mode = extended]      │   │
│  │        │                                           │   │
│  │  5. Guardar respuesta completa en caché            │   │
│  │        │                                           │   │
│  │     Response dict ──► Web App ──► Usuario          │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

### Dos niveles de caché

El sistema tiene caché en **dos niveles** distintos — es importante entenderlos:

| Nivel       | Clase                               | Qué cachea                              | Dónde vive              |
| ----------- | ----------------------------------- | --------------------------------------- | ----------------------- |
| Pipeline    | `SemanticCache` en `HannahPipeline` | Respuesta completa (texto + metadata)   | RAM del proceso backend |
| RAG interno | `SemanticCache` en `RAGComponent`   | Solo el contexto `[MEMORY]...[/MEMORY]` | RAM del proceso backend |

Un HIT en el nivel Pipeline saltea **todo**: ChromaDB, embeddings y la generación LLM.  
Un HIT en el nivel RAG solo saltea la búsqueda en ChromaDB, pero el LLM igual genera.

---

## 2. Prerrequisitos

```bash
# Python 3.11+ recomendado
pip install chromadb sentence-transformers numpy torch transformers fastapi uvicorn
```

> **Nota GPU:** Fast Hannah y Slow Hannah necesitan CUDA.  
> Los módulos del RAG (ChromaDB, embeddings) corren solo en CPU.  
> **No mover el modelo de embeddings a GPU** — competiría por VRAM con el LLM.

---

## 3. Estructura de Directorios

```
proyecto/
├── rag/                   ← módulos RAG
│   ├── embeddings.py                 # Módulo 1 — EmbeddingService
│   ├── vector_store.py               # Módulo 2 — VectorStore (ChromaDB)
│   ├── semantic_cache.py             # Módulo 3 — SemanticCache
│   ├── query_enhancer.py             # Módulo 4 — QueryEnhancer
│   ├── context_handler.py            # Módulo 5 — ContextHandler
│   ├── rag_component.py              # Módulo 6 — RAGComponent (orquestador)
│   ├── hannah_pipeline.py            # Capa de integración (RAG + modelos)
│   ├── ingest_knowledge.py           # Script de ingesta inicial (correr 1 vez)
│   └── hannah_knowledge/             # BD ChromaDB persistente (se autocrea)
│
├── repositorio/                      ← checkpoints de los LLMs
│   ├── checkpoints/
│   │   ├── hannah_dpo/
│   │   │   └── hannah_dpo_final.pt   # Pesos Fast Hannah
│   │   └── slow_hannah/              # Pesos Slow Hannah
│   └── tokenizer/
│       ├── hannah_tok/               # Tokenizer Fast Hannah
│       └── slow_tok/                 # Tokenizer Slow Hannah
│
└── backend/                          ← Web App (John)
    └── main.py                       # Servidor FastAPI
```

---

## 4. Paso 0 — Poblar la Base de Conocimiento (una sola vez)

Antes de arrancar el backend, la BD de ChromaDB debe estar poblada.

```bash
cd rag
python ingest_knowledge.py
```

Esto crea la carpeta `./hannah_knowledge/`.  
**No borrar esta carpeta** — es la memoria de largo plazo de Hannah.  
Si se re-ejecuta el script, los documentos existentes se actualizan (upsert) y se agregan los nuevos. Nada se duplica.

---

## 5. Conectar el Backend

### 5.1 Uso básico — lo único que el backend necesita llamar

```python
# backend/main.py
from hannah_pipeline import HannahPipeline

# Inicializar UNA SOLA VEZ al arrancar el servidor
# Esto carga los LLMs en GPU — hacerlo en el startup, no por request
pipeline = HannahPipeline(
    load_fast_model=True,
    load_slow_model=True
)

# Llamar en cada mensaje del usuario
response = pipeline.process_message(
    user_msg="Hey, what's your favorite movie?",
    history=[
        ("Hi!", "Hey babe~"),
        ("How are you?", "Doing great, thinking about you 😊"),
    ]
)

# Estructura del dict de respuesta:
# {
#   "text":        "Spirited Away, obviously. Have you seen it?",
#   "source":      "fast" | "slow" | "fast_fallback" | "cache" | "rag_only",
#   "rag_context": "[MEMORY]...[/MEMORY]",
#   "mode":        "simplified" | "extended",
#   "cache_hit":   False,
#   "latency":     0.312,      # segundos totales
#   "rag_chunks":  2
# }
print(response["text"])
```

> **Significado de `source`:**
>
> - `"fast"` → generó Fast Hannah, modo simplified
> - `"slow"` → generó Slow Hannah, modo extended
> - `"fast_fallback"` → el Classifier eligió extended pero Slow no estaba disponible, usó Fast
> - `"cache"` → respuesta vino del caché de nivel pipeline (no se llamó al LLM)
> - `"rag_only"` → modo test sin modelos cargados

### 5.2 Endpoint FastAPI completo

```python
from fastapi    import FastAPI
from pydantic   import BaseModel
from contextlib import asynccontextmanager
from hannah_pipeline import HannahPipeline
import asyncio

pipeline: HannahPipeline = None   # global del proceso

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = HannahPipeline(load_fast_model=True, load_slow_model=True)
    yield
    # cleanup si fuera necesario

app = FastAPI(lifespan=lifespan)

class MessageRequest(BaseModel):
    user_msg: str
    history:  list[tuple[str, str]] = []

@app.post("/chat")
async def chat(req: MessageRequest):
    # process_message es síncrono (CPU + GPU — bloqueante).
    # run_in_executor evita que congele el event loop de FastAPI.
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        pipeline.process_message,
        req.user_msg,
        req.history
    )
    return {
        "reply":      result["text"],
        "source":     result["source"],
        "latency_ms": int(result["latency"] * 1000)
    }

@app.get("/stats")
def stats():
    return pipeline.get_stats()
```

> ⚠️ **Importante:** Nunca llamar `pipeline.process_message()` directamente dentro
> de un endpoint `async` sin `run_in_executor`. La inferencia de PyTorch es bloqueante
> y congela el event loop de FastAPI, lo que deja a todos los usuarios esperando.

---

## 6. Conectar Fast Hannah (360M)

Fast Hannah ya está cableada en `hannah_pipeline.py`. Cuando el Classifier retorna
`"simplified"`, este camino corre.

**Lo que se necesita:**

- Checkpoint en: `repositorio/checkpoints/hannah_dpo/hannah_dpo_final.pt`
- Tokenizer en: `repositorio/tokenizer/hannah_tok/`
- GPU con ≥ 4 GB VRAM

**Formato esperado del checkpoint:**

```python
# El checkpoint debe ser un dict con la key "model" conteniendo el state dict.
# Las keys pueden tener el prefijo "_orig_mod." (de torch.compile) — se limpian automáticamente.
{
    "model": {
        "_orig_mod.embed_tokens.weight": tensor(...),
        "_orig_mod.layers.0.attn.q_proj.weight": tensor(...),
        ...
    }
}
```

**Contexto que recibe (modo simplified):**

```
[SYS] You are Hannah, my girlfriend... [/SYS]
[MEMORY]Hannah's favorite movie is Spirited Away...[/MEMORY]
[USR] what's your fav movie? [/USR][ASS]
```

Presupuesto de tokens: Sistema ≈ 100 · Memoria ≈ 200 · Historia ≈ 200 → total ~500, dentro de la ventana de 512 de Hannah.

---

## 7. Conectar Slow Hannah (Qwen2.5-14B)

**Estado actual:** El pipeline ya tiene la estructura completa para Slow Hannah
(`_load_slow_model`, `_build_slow_prompt`, `_generate_slow`). Solo falta que el
checkpoint esté disponible en la ruta configurada.

**Lo que se necesita:**

- Carpeta de modelo en: `repositorio/checkpoints/slow_hannah/`
  (formato HuggingFace estándar: `config.json` + archivos de pesos `.safetensors`)
- Tokenizer en: `repositorio/tokenizer/slow_tok/`
  (si no existe, el pipeline intenta usar el tokenizer dentro del mismo checkpoint)
- GPU(s) con ≥ 28 GB VRAM total en bfloat16
  (1× GPU de 40 GB, o 2× GPU de 16 GB con `device_map="auto"`)

**Verificar que Slow Hannah está cargada:**

```python
stats = pipeline.get_stats()
print(stats["models"]["slow_hannah_loaded"])   # True si cargó bien
```

**Contexto que recibe (modo extended, formato ChatML de Qwen):**

```
<|im_start|>system
You are Hannah, my girlfriend...

[MEMORY]
[Fuente: preferences] Hannah's favorite movie is Spirited Away...
---
[Fuente: personal] Hannah has a cat named Mochi...
[/MEMORY]
<|im_end|>
<|im_start|>user
Tell me everything about yourself
<|im_end|>
<|im_start|>assistant
```

Presupuesto de tokens: Sistema + Memoria ≈ 1600 tokens. La ventana de 32K de Qwen elimina cualquier problema de truncación.

**Comportamiento si Slow no está disponible:**  
Si el checkpoint no existe o falla al cargar, el pipeline usa Fast Hannah como fallback
y retorna `source = "fast_fallback"` en el dict de respuesta. No hay crash.

---

## 8. Conectar el Classifier Model

El Classifier Model (también llamado "Decisor") decide si el mensaje va a Fast o Slow Hannah.

**Estado actual:** `_select_model()` en `hannah_pipeline.py` es una heurística de placeholder
basada en longitud del mensaje y palabras clave. Funciona, pero debe reemplazarse por el
clasificador real cuando esté listo.

**Contrato de interfaz que el Classifier debe cumplir:**

```python
def classify(user_msg: str, history: list[tuple[str, str]]) -> str:
    """
    Args:
        user_msg: Mensaje actual del usuario.
        history:  Lista de (turno_usuario, turno_hannah) de la conversación.

    Returns:
        "simplified"  →  Fast Hannah (360M), contexto RAG ~200 tokens
        "extended"    →  Slow Hannah (Qwen), contexto RAG ~1500 tokens
    """
    ...
```

**Cómo conectarlo en `hannah_pipeline.py`:**

```python
# Opción A — import directo y reemplazar el método
from mi_clasificador import MiClasificador

class HannahPipeline:
    def __init__(self, ...):
        ...
        self.classifier = MiClasificador()

    def _select_model(self, user_msg, history):
        return self.classifier.classify(user_msg, history)
```

**Lo que el Classifier recibirá en la práctica:**

- Saludos cortos, emojis, < 20 palabras → debería retornar `"simplified"`
- Preguntas multi-parte, "explain", "compare", "describe in detail" → `"extended"`
- Preguntas sobre Hannah misma (identidad, conocimiento) → depende de la complejidad

---

## 9. Agregar Conocimiento en Tiempo Real

El pipeline puede aprender información nueva durante la conversación:

```python
pipeline.add_knowledge(
    text="The user's name is Jorge and he studies computer science at PUCP.",
    metadata={"source": "conversation", "topic": "user_info"},
    doc_id="user_jorge_001"
)
```

Los IDs se upsertean: si `user_jorge_001` ya existe, se actualiza. El cambio
se persiste en disco inmediatamente (ChromaDB es persistente por defecto).

---

## 10. Problemas Conocidos y TODOs

### ⚠️ Bug corregido: mutación del historial en `_build_fast_prompt`

En la versión anterior de `hannah_pipeline.py`, el método `_build_fast_prompt`
llamaba `history.pop(0)` directamente sobre la lista recibida como argumento.
Esto mutaba la lista **del caller** (el backend), lo que podía eliminar turnos
del historial de la conversación en curso silenciosamente.

**Corrección aplicada en esta versión:**

```python
def _build_fast_prompt(self, user_msg, history, rag_context):
    history = list(history)   # ← copia defensiva, no muta el original
    ...
```

El mismo patrón se aplica en `_build_slow_prompt`.

---

| #   | Problema                                                   | Gravedad  | Archivo                               |
| --- | ---------------------------------------------------------- | --------- | ------------------------------------- |
| 1   | ~~Mutación del historial en `_build_prompt`~~              | ~~Media~~ | ~~`hannah_pipeline.py`~~ ✅ corregido |
| 2   | `view_docs.py` tiene import roto (`from rag.vector_store`) | Baja      | `view_docs.py` línea 1                |
| 3   | HyDE usa templates estáticos, no un LLM real               | Baja      | `query_enhancer._generate_hyde()`     |

**Fix para #2 (`view_docs.py`):**

```python
# Cambiar esto:
from rag.vector_store import VectorStore

# Por esto:
from vector_store import VectorStore
```

---

## 11. Referencia Rápida — Contratos de Datos

### `pipeline.process_message()` → retorna:

```python
{
  "text":        str,    # Respuesta de Hannah
  "source":      str,    # "fast" | "slow" | "fast_fallback" | "cache" | "rag_only"
  "rag_context": str,    # "[MEMORY]...[/MEMORY]" inyectado al prompt
  "mode":        str,    # "simplified" | "extended"
  "cache_hit":   bool,   # True si el caché de respuestas completas respondió
  "latency":     float,  # segundos totales de proceso
  "rag_chunks":  int     # chunks de ChromaDB usados
}
```

### `rag.retrieve()` → retorna:

```python
{
  "formatted_context": str,        # "[MEMORY]...[/MEMORY]"
  "raw_chunks":        list[str],  # textos planos de los chunks
  "scores":            list[float],# similitud coseno (0–1) por chunk
  "mode":              str,
  "num_chunks":        int,
  "approx_tokens":     int,
  "cache_hit":         bool,       # HIT del caché interno del RAG
  "enhanced_query":    dict        # info del QueryEnhancer
}
```

### Classifier → retorna:

```python
"simplified"   # Fast Hannah · RAG simplified · ~200 tokens de contexto
"extended"     # Slow Hannah · RAG extended   · ~1500 tokens de contexto
```

### `pipeline.get_stats()` → retorna:

```python
{
  "vector_store":   {"total_documents": int},
  "cache":          {"entries": int, "max_size": int, "threshold": float, "total_hits": int},
  "response_cache": {"entries": int, "max_size": int, "threshold": float, "total_hits": int},
  "status":         "operational",
  "models": {
    "fast_hannah_loaded": bool,
    "slow_hannah_loaded": bool,
    "fast_device":        "cuda:0" | "cpu" | "N/A"
  }
}
```
