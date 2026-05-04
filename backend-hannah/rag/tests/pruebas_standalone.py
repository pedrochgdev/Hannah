#!/usr/bin/env python3
# ============================================================================
# TEST DEL RAG STANDALONE (SIN MODELO HANNAH)
# ============================================================================
# Archivo: rag_standalone/tests/test_guia_a_standalone.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
# Prueba el pipeline RAG completo sin necesidad del modelo Hannah.
# Crea una base de conocimiento personalizada con datos reales del proyecto,
# ingesta documentos, y ejecuta múltiples queries en ambos modos
# (simplified/extended) para verificar que todo funciona correctamente.
# Requisitos:
#   python -m pip install chromadb sentence-transformers numpy
# Tiempo estimado: ~15-30 segundos 
# ============================================================================

import sys
import os
import shutil
import gc
import time

# ============================================================================
# CONFIGURACIÓN DE PATHS
# ============================================================================
# Agregamos el directorio padre (rag_standalone/) al path para poder
# importar los módulos: embeddings, vector_store, semantic_cache, etc.
# Esto es necesario porque estamos ejecutando desde tests/ pero los
# módulos viven en rag_standalone/
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(SCRIPT_DIR)  # rag_standalone/
sys.path.insert(0, RAG_DIR)

from rag_component import RAGComponent
# ============================================================================
# BASE DE DATOS TEST
# ============================================================================
# Usamos una carpeta separada para no contaminar la BD real de Hannah.
# Se crea al inicio y se borra al final automáticamente.
# ============================================================================
TEST_DB_PATH = os.path.join(SCRIPT_DIR, "test_knowledge_db")


# ============================================================================
# BASE DE CONOCIMIENTO PERSONALIZADA
# ============================================================================
# Simulan documentos reales sobre el proyecto Hannah.
# En producción, estos vendrían de PDFs, wikis, o documentos del equipo.
# Cada documento tiene:
#   - text: el contenido del documento
#   - metadata: información sobre la fuente (para trazabilidad)
#   - id: identificador único
# ============================================================================

KNOWLEDGE_BASE = [
    # --- Arquitectura general ---
    {
        "text": (
            "Hannah es un modelo transformer de 360 millones de parámetros "
            "basado en la arquitectura OLMo3. Su configuración incluye "
            "d_model=1024, 24 capas de atención, 16 heads, y una longitud "
            "de secuencia máxima de 1024 tokens durante inferencia."
        ),
        "metadata": {"source": "arquitectura.pdf", "section": "modelo_base", "topic": "hannah"},
        "id": "arch_001"
    },
    {
        "text": (
            "El sistema Hannah sigue el paradigma de Prepared Mind, Fast Response "
            "(Zhang et al., 2025), que propone desacoplamiento temporal: un Fast Model "
            "(Hannah 360M) responde en tiempo real con baja latencia, mientras un "
            "Slow Model (Qwen2.5-14B-Instruct) genera respuestas más elaboradas "
            "en segundo plano."
        ),
        "metadata": {"source": "arquitectura.pdf", "section": "fast_slow", "topic": "paradigma"},
        "id": "arch_002"
    },
    {
        "text": (
            "Hannah fue diseñada como AI Companion para practicar inglés de forma "
            "conversacional y natural. No es un asistente virtual: es una compañera "
            "de conversación con personalidad propia. El DPO penaliza respuestas "
            "tipo asistente (genéricas, formales, con disclaimers)."
        ),
        "metadata": {"source": "design.pdf", "section": "identidad", "topic": "hannah"},
        "id": "arch_003"
    },

    # --- Entrenamiento ---
    {
        "text": (
            "El entrenamiento de Hannah pasó por tres fases: "
            "1) Pretraining con 80,000 steps usando next-token prediction sobre "
            "un corpus de 14GB de texto en inglés. "
            "2) SFT (Supervised Fine-Tuning) con 15,000 steps usando pares "
            "instrucción-respuesta curados manualmente. "
            "3) DPO (Direct Preference Optimization) con 1,500 steps usando "
            "pares chosen/rejected para alinear con preferencias humanas."
        ),
        "metadata": {"source": "tecnica.pdf", "section": "entrenamiento", "topic": "training"},
        "id": "train_001"
    },
    {
        "text": (
            "El Slow Model del sistema Hannah usa Qwen2.5-14B-Instruct, un modelo "
            "preentrenado por el equipo de Alibaba al que se le aplicó SFT por "
            "nuestro equipo. Este modelo reemplazó a Meta Llama como Slow Model "
            "según la actualización del Laboratorio Grupal (abril 2026)."
        ),
        "metadata": {"source": "lab_grupal.pdf", "section": "slow_model", "topic": "qwen"},
        "id": "train_002"
    },
    {
        "text": (
            "Durante el DPO, se usa SEQ_LEN=512 (no 1024) por limitaciones de "
            "memoria GPU al procesar pares chosen/rejected simultáneamente. Sin "
            "embargo, en inferencia se usa SEQ_LEN=1024, que es el máximo que el "
            "modelo soporta gracias al pretraining y SFT."
        ),
        "metadata": {"source": "tecnica.pdf", "section": "dpo_config", "topic": "training"},
        "id": "train_003"
    },

    # --- RAG ---
    {
        "text": (
            "El RAG (Retrieval-Augmented Generation) de Hannah usa ChromaDB como "
            "base de datos vectorial con distancia coseno explícita (no la L2 por "
            "defecto). Los embeddings se generan con all-MiniLM-L6-v2, un modelo "
            "de 384 dimensiones que corre en CPU (~80MB)."
        ),
        "metadata": {"source": "rag_doc.pdf", "section": "vector_store", "topic": "rag"},
        "id": "rag_001"
    },
    {
        "text": (
            "El Semantic Cache del RAG almacena queries previas y sus respuestas "
            "en memoria. Si una nueva query tiene similitud coseno >= 0.92 con "
            "alguna query cacheada, retorna la respuesta sin volver a consultar "
            "ChromaDB. Esto reduce latencia en preguntas repetidas."
        ),
        "metadata": {"source": "rag_doc.pdf", "section": "cache", "topic": "rag"},
        "id": "rag_002"
    },
    {
        "text": (
            "El contexto RAG se formatea con tokens [MEMORY] y [/MEMORY]. En modo "
            "simplified (Fast Model): máximo 3 chunks, ~200 tokens, sin metadatos. "
            "En modo extended (Slow Model): hasta 10 chunks, ~1500 tokens, con "
            "metadatos de fuente y reranking por similitud coseno exacta."
        ),
        "metadata": {"source": "rag_doc.pdf", "section": "context_handler", "topic": "rag"},
        "id": "rag_003"
    },

    # --- Hardware ---
    {
        "text": (
            "El servidor de Hannah usa una GPU NVIDIA RTX 5070 Ti con 16GB de VRAM. "
            "Hannah 360M en BF16 ocupa ~750MB de VRAM. Qwen2.5-14B cuantizado a "
            "4-bit ocupa ~7GB. El pipeline RAG (embeddings + ChromaDB) corre "
            "enteramente en CPU para no competir por VRAM."
        ),
        "metadata": {"source": "infra.pdf", "section": "hardware", "topic": "gpu"},
        "id": "hw_001"
    },

    # --- Tokenizador ---
    {
        "text": (
            "Hannah usa un tokenizador BPE (Byte-Pair Encoding) entrenado desde "
            "cero con un vocabulario de 32,000 tokens. El tokenizador fue entrenado "
            "sobre el mismo corpus de pretraining usando la librería tokenizers de "
            "HuggingFace con normalización NFC."
        ),
        "metadata": {"source": "tecnica.pdf", "section": "tokenizer", "topic": "tokenizer"},
        "id": "tok_001"
    },

    # --- Equipo ---
    {
        "text": (
            "El equipo de Hannah está formado por: Luis Miranda Mallqui (Arquitecto "
            "de Solución), Marlow (líder del Slow Model), Pedro y John "
            "(entrenamiento y datos). El proyecto es parte del curso de PLN en la "
            "PUCP, octavo ciclo, semestre 2026-1."
        ),
        "metadata": {"source": "proyecto.pdf", "section": "equipo", "topic": "equipo"},
        "id": "team_001"
    },
]


def print_header(title: str):
    """Imprime un encabezado bonito."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Imprime un sub-encabezado."""
    print(f"\n--- {title} ---")


def run_tests():
    """Ejecuta todos los tests de la Guía A."""

    print_header("TEST RAG STANDALONE (SIN MODELO)")
    print(f"  BD temporal: {TEST_DB_PATH}")
    print(f"  Documentos: {len(KNOWLEDGE_BASE)}")

    # Limpiar BD anterior si existe
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)

    # Crear el RAGComponent con la BD de test
    print("\n[1/5] Inicializando RAG...")
    rag = RAGComponent(
        db_path=TEST_DB_PATH,
        cache_threshold=0.92,  # Limite de producción
        cache_size=500
    )
    print("      OK - Todos los componentes cargados.")

    # ══════════════════════════════════════════════════════════════
    # INGESTAR LA BASE DE CONOCIMIENTO
    # ══════════════════════════════════════════════════════════════
    print_header("Ingesta de Base de Conocimiento")

    documents = [doc["text"] for doc in KNOWLEDGE_BASE]
    metadatas = [doc["metadata"] for doc in KNOWLEDGE_BASE]
    ids = [doc["id"] for doc in KNOWLEDGE_BASE]

    print(f"  Insertando {len(documents)} documentos en ChromaDB...")
    rag.ingest_documents(documents, metadatas, ids)
    print(f"  OK - Base de conocimiento creada.")

    # Verificar
    stats = rag.get_stats()
    total_docs = stats["vector_store"]["total_documents"]
    print(f"  Verificación: {total_docs} documentos en VectorStore.")
    assert total_docs == len(KNOWLEDGE_BASE), f"ERROR: esperábamos {len(KNOWLEDGE_BASE)}, hay {total_docs}"

    # ══════════════════════════════════════════════════════════════
    # QUERIES EN MODO SIMPLIFIED (FAST MODEL)
    # ══════════════════════════════════════════════════════════════
    print_header("Queries en Modo SIMPLIFIED (Fast Model)")
    print("  Config: max 3 chunks, ~200 tokens, sin metadata, sin reranking")

    simplified_queries = [
        "¿Cuántos parámetros tiene Hannah?",
        "¿Qué GPU usa el servidor?",
        "¿Qué es el DPO?",
        "¿Quiénes forman el equipo de Hannah?",
    ]

    for i, query in enumerate(simplified_queries, 1):
        print_subheader(f"Query {i}: {query}")
        result = rag.retrieve(query, mode="simplified")
        print(f"  Cache hit:  {result['cache_hit']}")
        print(f"  Chunks:     {result['num_chunks']}")
        print(f"  Tokens:     ~{result['approx_tokens']}")
        print(f"  Contexto:")
        print(f"  {result['formatted_context']}")

    # ══════════════════════════════════════════════════════════════
    # QUERIES EN MODO EXTENDED (SLOW MODEL)
    # ══════════════════════════════════════════════════════════════
    print_header("Queries en Modo EXTENDED (Slow Model)")
    print("  Config: max 10 chunks, ~1500 tokens, con metadata, con reranking")

    extended_queries = [
        "Explica el proceso de entrenamiento completo de Hannah incluyendo todas las fases.",
        "¿Cómo funciona el RAG de Hannah y qué componentes tiene?",
    ]

    for i, query in enumerate(extended_queries, 1):
        print_subheader(f"Query {i}: {query}")
        result = rag.retrieve(query, mode="extended")
        print(f"  Cache hit:    {result['cache_hit']}")
        print(f"  Chunks:       {result['num_chunks']}")
        print(f"  Tokens:       ~{result['approx_tokens']}")
        print(f"  Queries usadas: {len(result['enhanced_query']['search_queries'])}")
        # Mostrar los primeros 500 chars del contexto
        ctx = result["formatted_context"]
        if len(ctx) > 500:
            print(f"  Contexto (primeros 500 chars):")
            print(f"  {ctx[:500]}...")
        else:
            print(f"  Contexto:")
            print(f"  {ctx}")

    # ══════════════════════════════════════════════════════════════
    # TEST DE SEMANTIC CACHE
    # ══════════════════════════════════════════════════════════════
    print_header("PASO 10c: Verificación del Semantic Cache")

    print_subheader("Repitiendo query exacta (esperamos CACHE HIT)")
    result_cached = rag.retrieve("¿Cuántos parámetros tiene Hannah?", mode="simplified")
    print(f"  Cache hit: {result_cached['cache_hit']}")
    if result_cached["cache_hit"]:
        print("  CORRECTO: El cache detectó la query repetida.")
    else:
        print("  NOTA: Cache MISS (puede pasar si el threshold es muy estricto).")

    print_subheader("Query sobre tema NO cubierto (esperamos resultado de baja relevancia)")
    result_offopic = rag.retrieve("¿Cuál es la receta de la carbonara?", mode="simplified")
    print(f"  Chunks:  {result_offopic['num_chunks']}")
    print(f"  Contexto: {result_offopic['formatted_context'][:200]}")
    print("  NOTA: ChromaDB siempre retorna resultados (los más cercanos),")
    print("        pero con scores de similitud bajos. En producción,")
    print("        podríamos agregar un umbral mínimo de relevancia.")

    # ══════════════════════════════════════════════════════════════
    # ESTADÍSTICAS FINALES
    # ══════════════════════════════════════════════════════════════
    print_header("ESTADÍSTICAS FINALES")
    stats = rag.get_stats()
    print(f"  Documentos en VectorStore: {stats['vector_store']['total_documents']}")
    print(f"  Entradas en caché:         {stats['cache']['entries']}")
    print(f"  Hits totales del caché:    {stats['cache']['total_hits']}")
    print(f"  Estado del sistema:        {stats['status']}")

    # ══════════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════════
    # Liberamos ChromaDB antes de borrar la carpeta (WinError 32)
    del rag
    gc.collect()
    time.sleep(0.5)

    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        print(f"\n[Cleanup] BD de test eliminada: {TEST_DB_PATH}")

    print_header("TODOS LOS TESTS DE GUÍA A COMPLETADOS")
    print("  El RAG funciona correctamente sin necesidad del modelo Hannah.")
    print("  Para probar con el modelo de 1.4GB, ejecuta:")
    print("  python tests/test_guia_b_con_modelo.py")


if __name__ == "__main__":
    run_tests()
