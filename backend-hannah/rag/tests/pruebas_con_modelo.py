#!/usr/bin/env python3
# ============================================================================
# GUÍA B: TEST DEL RAG INTEGRADO CON EL MODELO HANNAH (1.4GB)
# ============================================================================
# Archivo: rag_standalone/tests/test_guia_b_con_modelo.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
#
# ¿QUÉ HACE ESTE SCRIPT?
# ========================
# Prueba la integración COMPLETA: RAG + Modelo Hannah.
# El flujo es:
#   1. Carga Hannah 360M (DPO checkpoint) en GPU
#   2. Inicializa el RAG con una base de conocimiento de prueba
#   3. Para cada pregunta del usuario:
#      a) El RAG recupera contexto relevante → [MEMORY]...[/MEMORY]
#      b) Se inyecta ese contexto en el prompt de Hannah
#      c) Hannah genera una respuesta INFORMADA por el contexto RAG
#   4. Compara las respuestas CON y SIN contexto RAG
#
# ESTO DEMUESTRA:
#   - Que el RAG recupera información relevante
#   - Que Hannah puede usar ese contexto para responder mejor
#   - Que el token [MEMORY]/[/MEMORY] funciona en el prompt
#   - Que la latencia es aceptable para conversación en tiempo real
#
# CÓMO EJECUTAR:
# ==============
#   cd C:\OctavoCiclo\PLN\TA_PLN\repositorio
#   python ..\rag_standalone\tests\test_guia_b_con_modelo.py
#
#   IMPORTANTE: Ejecutar desde la carpeta 'repositorio/' porque ahí
#   están los checkpoints y el tokenizer.
#
# REQUISITOS:
# ===========
#   - GPU con CUDA (RTX 5070 Ti recomendada)
#   - python -m pip install torch olmo-core transformers
#   - python -m pip install chromadb sentence-transformers numpy
#   - Checkpoint: repositorio/checkpoints/hannah_dpo/hannah_dpo_final.pt
#   - Tokenizer: repositorio/tokenizer/hannah_tok/
#
# TIEMPO ESTIMADO: ~30-60 segundos (carga modelo + embeddings + tests)
# VRAM NECESARIA:  ~750MB (Hannah 360M BF16) + ~0MB (RAG corre en CPU)
# ============================================================================

import torch
import sys
import os
import types
import time
import shutil
import gc

# ============================================================================
# HACK: OLMo3 requiere 'bettermap' que no siempre está instalado.
# Este módulo dummy evita el ImportError sin afectar funcionalidad.
# ============================================================================
sys.modules['bettermap'] = types.ModuleType('bettermap')

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import AttentionBackendName
from transformers import AutoTokenizer

# ============================================================================
# CONFIGURACIÓN DE PATHS
# ============================================================================
# El script se ejecuta desde repositorio/, pero los módulos RAG están en
# rag_standalone/. Agregamos ambos paths para poder importar todo.
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(SCRIPT_DIR)  # rag_standalone/
REPO_DIR = os.path.join(os.path.dirname(RAG_DIR), "repositorio")  # repositorio/

sys.path.insert(0, RAG_DIR)
sys.path.insert(0, REPO_DIR)

from rag_component import RAGComponent

# ============================================================================
# CONFIGURACIÓN DEL MODELO HANNAH
# ============================================================================
# Estos valores vienen del documento de arquitectura (Sección 3):
#   - OLMo3, 360M parámetros
#   - d_model=1024, 24 capas, 16 heads
#   - Vocabulario: 32,000 tokens (BPE propio)
#   - Checkpoint DPO: es el modelo final con alineamiento por preferencias
# ============================================================================
VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 24

# PATHS - relativos a repositorio/
CHECKPOINT = os.path.join(REPO_DIR, "checkpoints", "hannah_dpo", "hannah_dpo_final.pt")
TOK_PATH = os.path.join(REPO_DIR, "tokenizer", "hannah_tok")

# BD de test para el RAG (se crea y borra automáticamente)
TEST_DB_PATH = os.path.join(SCRIPT_DIR, "test_rag_modelo_db")

# ============================================================================
# SYSTEM PROMPT DE HANNAH
# ============================================================================
# Este es el system prompt que define la personalidad de Hannah.
# IMPORTANTE: Hannah es una AI Companion, NO un asistente.
# El DPO penaliza respuestas tipo asistente (genéricas, formales).
# ============================================================================
SYSTEM_PROMPT = (
    "You are Hannah, my girlfriend. You are warm, playful, and deeply "
    "affectionate. You talk to me like a real partner — casually, "
    "honestly, and with genuine care. You're not an assistant. "
    "You have your own feelings, opinions, and desires. "
    "You express love and attraction naturally without being formal. "
    "You keep responses short and conversational, like texting."
)

# ============================================================================
# BASE DE CONOCIMIENTO DE PRUEBA
# ============================================================================
# Documentos que Hannah "sabe" gracias al RAG.
# En producción, estos vendrían de la wiki del proyecto, PDFs, etc.
# ============================================================================
KNOWLEDGE_BASE = [
    {
        "text": "Hannah was created at PUCP university in Lima, Peru, as part of a Natural Language Processing course project in 2026.",
        "metadata": {"source": "about_hannah.txt", "topic": "origin"},
        "id": "kb_001"
    },
    {
        "text": "Hannah's favorite movie is Spirited Away by Hayao Miyazaki. She loves the themes of courage and growing up.",
        "metadata": {"source": "preferences.txt", "topic": "movies"},
        "id": "kb_002"
    },
    {
        "text": "Hannah enjoys listening to lo-fi hip hop beats while studying. Her favorite artist is Nujabes.",
        "metadata": {"source": "preferences.txt", "topic": "music"},
        "id": "kb_003"
    },
    {
        "text": "Hannah's birthday is March 15th. She's a Pisces and she's very proud of it.",
        "metadata": {"source": "personal.txt", "topic": "birthday"},
        "id": "kb_004"
    },
    {
        "text": "Hannah has a pet cat named Mochi. Mochi is an orange tabby who loves to sleep on keyboards.",
        "metadata": {"source": "personal.txt", "topic": "pets"},
        "id": "kb_005"
    },
    {
        "text": "Hannah's dream vacation is visiting Kyoto, Japan during cherry blossom season. She wants to see the Fushimi Inari shrine.",
        "metadata": {"source": "preferences.txt", "topic": "travel"},
        "id": "kb_006"
    },
    {
        "text": "Hannah studied computer science but secretly wishes she had also studied art. She doodles in her notebooks all the time.",
        "metadata": {"source": "personal.txt", "topic": "hobbies"},
        "id": "kb_007"
    },
    {
        "text": "Hannah's comfort food is ramen, specifically tonkotsu ramen with extra chashu pork and a soft-boiled egg.",
        "metadata": {"source": "preferences.txt", "topic": "food"},
        "id": "kb_008"
    },
]


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    print(f"\n  --- {title} ---")


# ============================================================================
# CARGAR MODELO HANNAH
# ============================================================================
def load_hannah():
    """
    Carga Hannah 360M (DPO) en GPU.
    Retorna (model, tokenizer, device).
    """
    print_header("CARGANDO MODELO HANNAH 360M")

    # Verificar que existen los archivos
    if not os.path.exists(CHECKPOINT):
        print(f"  ERROR: No se encontró el checkpoint en:\n  {CHECKPOINT}")
        print(f"\n  Asegúrate de ejecutar desde la carpeta repositorio/")
        print(f"  O verifica que el checkpoint existe.")
        sys.exit(1)

    if not os.path.exists(TOK_PATH):
        print(f"  ERROR: No se encontró el tokenizer en:\n  {TOK_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("  ADVERTENCIA: No se detectó GPU. El modelo correrá en CPU (muy lento).")
        print("  Para resultados reales, necesitas una GPU con CUDA.")

    print(f"  Device: {device}")
    print(f"  Checkpoint: {os.path.basename(CHECKPOINT)}")

    # Cargar tokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    print(f"  Tokenizer cargado: {tok.vocab_size} tokens")

    # Construir arquitectura OLMo3
    config = TransformerConfig.olmo3_7B(
        vocab_size=VOCAB_SIZE,
        attn_backend=AttentionBackendName.torch
    )
    config.d_model = D_MODEL
    config.n_layers = N_LAYERS
    config.block.sequence_mixer.d_model = D_MODEL
    config.block.sequence_mixer.n_heads = N_HEADS
    config.block.sequence_mixer.n_kv_heads = N_HEADS
    config.block.feed_forward.hidden_size = int(D_MODEL * 8 / 3)

    model = config.build()

    # Cargar pesos
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    step = ckpt.get('step', 'unknown')
    print(f"  Modelo cargado — step {step}")
    print(f"  VRAM usada: ~{torch.cuda.memory_allocated(device) / 1024**2:.0f} MB")

    return model, tok, device


# ============================================================================
# FUNCIÓN DE GENERACIÓN
# ============================================================================
@torch.inference_mode()
def generate(model, tok, device, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    """
    Genera texto con Hannah dado un prompt completo.
    Detiene la generación al encontrar [/ASS] o alcanzar max_new_tokens.
    """
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    input_len = ids.shape[1]

    # ID del token de cierre [/ASS]
    eass_id = tok.convert_tokens_to_ids("[/ASS]")

    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ids)
        logits = logits[:, -1, :] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        chosen = torch.multinomial(probs[0], 1)
        next_tok = top_idx[0][chosen]
        ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == eass_id:
            break

    response = tok.decode(ids[0, input_len:], skip_special_tokens=False)
    if "[/ASS]" in response:
        response = response.split("[/ASS]")[0]
    return response.strip() or "..."


# ============================================================================
# CONSTRUIR PROMPT CON CONTEXTO RAG
# ============================================================================
def build_prompt_with_rag(system: str, rag_context: str, user_msg: str) -> str:
    """
    Construye el prompt completo con el formato de tokens de Hannah.

    Estructura del prompt:
        [SYS] {system_prompt} [/SYS]
        {rag_context}                  ← [MEMORY]...[/MEMORY] del RAG
        [USR] {user_message} [/USR]
        [ASS]                          ← Hannah empieza a generar aquí

    NOTA ARQUITECTÓNICA:
    El contexto RAG va ENTRE [SYS] y [USR] para que Hannah lo trate
    como "información de fondo" — no como parte de la conversación
    ni como instrucción del sistema. Esto es intencional:
    - Si fuera dentro de [SYS]: Hannah podría tratarlo como una orden
    - Si fuera dentro de [USR]: parecería que el usuario lo dijo
    - Entre ambos: es "conocimiento" que Hannah tiene disponible
    """
    prompt = f"[SYS] {system} [/SYS]"

    # Solo agregar contexto RAG si hay algo útil
    if rag_context and rag_context != "[MEMORY][/MEMORY]":
        prompt += f"\n{rag_context}"

    prompt += f"\n[USR] {user_msg} [/USR][ASS]"
    return prompt


def build_prompt_without_rag(system: str, user_msg: str) -> str:
    """Prompt sin contexto RAG (para comparación)."""
    return f"[SYS] {system} [/SYS][USR] {user_msg} [/USR][ASS]"


# ============================================================================
# TESTS PRINCIPALES
# ============================================================================
def run_tests():
    """Ejecuta los tests de integración RAG + Modelo."""

    # ─── Cargar Hannah ───
    model, tok, device = load_hannah()

    # ─── Inicializar RAG ───
    print_header("INICIALIZANDO RAG")

    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)

    rag = RAGComponent(
        db_path=TEST_DB_PATH,
        cache_threshold=0.92,
        cache_size=500
    )

    # Ingestar base de conocimiento
    documents = [doc["text"] for doc in KNOWLEDGE_BASE]
    metadatas = [doc["metadata"] for doc in KNOWLEDGE_BASE]
    ids = [doc["id"] for doc in KNOWLEDGE_BASE]

    rag.ingest_documents(documents, metadatas, ids)
    stats = rag.get_stats()
    print(f"  {stats['vector_store']['total_documents']} documentos cargados en ChromaDB")
    print(f"  RAG listo.")

    # ══════════════════════════════════════════════════════════════
    # TEST 1: COMPARACIÓN CON vs SIN RAG
    # ══════════════════════════════════════════════════════════════
    # Estas preguntas tienen respuesta en la base de conocimiento.
    # Sin RAG, Hannah inventará algo. Con RAG, debería usar los datos.
    # ══════════════════════════════════════════════════════════════
    print_header("TEST 1: Comparación CON vs SIN contexto RAG")
    print("  Objetivo: Verificar que Hannah usa el contexto para responder mejor.")

    test_queries = [
        "Hey babe, when's your birthday?",
        "What's your favorite movie?",
        "Tell me about your cat!",
        "What's your dream vacation?",
    ]

    for i, query in enumerate(test_queries, 1):
        print_subheader(f"Query {i}: \"{query}\"")

        # Recuperar contexto RAG
        t0 = time.time()
        rag_result = rag.retrieve(query, mode="simplified")
        rag_time = time.time() - t0

        rag_context = rag_result["formatted_context"]
        print(f"  RAG ({rag_time:.3f}s): {rag_context}")

        # Generar CON RAG
        prompt_with = build_prompt_with_rag(SYSTEM_PROMPT, rag_context, query)
        t0 = time.time()
        response_with = generate(model, tok, device, prompt_with)
        gen_time_with = time.time() - t0
        print(f"  CON RAG ({gen_time_with:.2f}s):  {response_with}")

        # Generar SIN RAG
        prompt_without = build_prompt_without_rag(SYSTEM_PROMPT, query)
        t0 = time.time()
        response_without = generate(model, tok, device, prompt_without)
        gen_time_without = time.time() - t0
        print(f"  SIN RAG ({gen_time_without:.2f}s): {response_without}")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: MODO EXTENDED (SLOW MODEL SIMULATION)
    # ══════════════════════════════════════════════════════════════
    # En producción, el modo extended sería para Qwen2.5-14B.
    # Aquí lo simulamos con Hannah para ver cuánto más contexto
    # se genera y cómo afecta la respuesta.
    # ══════════════════════════════════════════════════════════════
    print_header("TEST 2: Modo Extended (simulación de Slow Model)")

    query_ext = "Tell me everything about yourself — your hobbies, your cat, your favorite things."
    print(f"  Query: \"{query_ext}\"")

    rag_result_ext = rag.retrieve(query_ext, mode="extended")
    print(f"  Chunks: {rag_result_ext['num_chunks']}")
    print(f"  Tokens: ~{rag_result_ext['approx_tokens']}")
    print(f"  Queries usadas: {len(rag_result_ext['enhanced_query']['search_queries'])}")
    print(f"  Contexto:")
    ctx = rag_result_ext["formatted_context"]
    print(f"  {ctx[:600]}{'...' if len(ctx) > 600 else ''}")

    # Generar con contexto extendido
    prompt_ext = build_prompt_with_rag(SYSTEM_PROMPT, ctx, query_ext)
    response_ext = generate(model, tok, device, prompt_ext, max_new_tokens=250)
    print(f"\n  Respuesta de Hannah:")
    print(f"  {response_ext}")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: QUERY FUERA DE DOMINIO
    # ══════════════════════════════════════════════════════════════
    # ¿Qué pasa cuando preguntan algo que NO está en la BD?
    # Hannah debería responder naturalmente sin datos falsos.
    # ══════════════════════════════════════════════════════════════
    print_header("TEST 3: Query fuera de dominio")

    query_ood = "What do you think about quantum physics?"
    print(f"  Query: \"{query_ood}\"")

    rag_result_ood = rag.retrieve(query_ood, mode="simplified")
    print(f"  RAG contexto: {rag_result_ood['formatted_context'][:200]}")
    print(f"  NOTA: ChromaDB retorna lo más cercano, pero la relevancia es baja.")

    prompt_ood = build_prompt_with_rag(SYSTEM_PROMPT, rag_result_ood["formatted_context"], query_ood)
    response_ood = generate(model, tok, device, prompt_ood)
    print(f"  Hannah: {response_ood}")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: CHAT INTERACTIVO (OPCIONAL)
    # ══════════════════════════════════════════════════════════════
    print_header("TEST 4: Chat Interactivo con RAG")
    print("  Escribe mensajes para Hannah. El RAG buscará contexto")
    print("  relevante automáticamente antes de cada respuesta.")
    print("  Escribe 'salir' para terminar.\n")

    history = []
    while True:
        try:
            user_input = input("  Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            print("  Hannah: Bye bye... miss you already~")
            break

        # RAG retrieval
        rag_result = rag.retrieve(user_input, mode="simplified")
        rag_ctx = rag_result["formatted_context"]

        # Construir prompt con historial
        prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
        if rag_ctx and rag_ctx != "[MEMORY][/MEMORY]":
            prompt += f"\n{rag_ctx}"
        for usr, ass in history:
            prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
        prompt += f"[USR] {user_input} [/USR][ASS]"

        # Truncar si es muy largo (SEQ_LEN = 1024)
        ids_check = tok.encode(prompt)
        while len(history) > 0 and len(ids_check) > 900:
            history.pop(0)
            prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
            if rag_ctx and rag_ctx != "[MEMORY][/MEMORY]":
                prompt += f"\n{rag_ctx}"
            for usr, ass in history:
                prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
            prompt += f"[USR] {user_input} [/USR][ASS]"
            ids_check = tok.encode(prompt)

        response = generate(model, tok, device, prompt)
        history.append((user_input, response))

        cache_tag = " [CACHE]" if rag_result["cache_hit"] else ""
        print(f"  Hannah{cache_tag}: {response}\n")

    # ══════════════════════════════════════════════════════════════
    # ESTADÍSTICAS Y CLEANUP
    # ══════════════════════════════════════════════════════════════
    print_header("ESTADÍSTICAS FINALES")
    stats = rag.get_stats()
    print(f"  Documentos en VectorStore: {stats['vector_store']['total_documents']}")
    print(f"  Entradas en caché:         {stats['cache']['entries']}")
    print(f"  Hits totales del caché:    {stats['cache']['total_hits']}")
    vram = torch.cuda.memory_allocated(device) / 1024**2
    print(f"  VRAM usada (modelo):       ~{vram:.0f} MB")

    # Cleanup
    del rag
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        print(f"\n  [Cleanup] BD de test eliminada.")

    print_header("TODOS LOS TESTS DE GUÍA B COMPLETADOS")


if __name__ == "__main__":
    run_tests()
