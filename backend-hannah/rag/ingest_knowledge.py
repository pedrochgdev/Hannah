#!/usr/bin/env python3
# ============================================================================
# SCRIPT DE INGESTA: Poblar la Base de Datos Vectorial de Hannah
# ============================================================================
# Archivo: rag_standalone/ingest_knowledge.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui 
# ============================================================================
#
# Descripción:
# ========================
# Crea y puebla la base de datos vectorial persistente de Hannah.
# A diferencia de los tests (que crean una BD temporal y la borran),
# este script crea la BD en ./hannah_knowledge/ y la deja ahí
# permanentemente para que el pipeline la use en producción.
#
# Proposito:
# =======================
# El RAG necesita documentos para buscar. Sin documentos ingresados,
# ChromaDB no tiene nada que devolver. Este script es el "llenado
# inicial" de la memoria de largo plazo de Hannah.
#
# Ejecución:
# ====================
# Una sola vez al inicio del proyecto (o cuando se quiera actualizar
# el conocimiento). La BD persiste en disco entre ejecuciones.
# Si se ejecuta de nuevo, detecta documentos existentes y solo
# agrega los nuevos (upsert).
#
# Procedimiento para ejecutar:
# ==============
#   cd C:\OctavoCiclo\PLN\TA_PLN\rag_standalone
#   python ingest_knowledge.py
#
# Resultado esperado:
# ====================
# Se crea la carpeta ./hannah_knowledge/ con la BD de ChromaDB.
# Esta carpeta es la que usa hannah_pipeline.py para responder.
# NO BORRAR esta carpeta, es la memoria de Hannah.
# ============================================================================

import os
import sys

# Asegurar que podemos importar los módulos del RAG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rag_component import RAGComponent

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Ruta de la BD persistente.
# hannah_pipeline.py usa exactamente la misma ruta.
# ============================================================================
KNOWLEDGE_DB_PATH = os.path.join(SCRIPT_DIR, "hannah_knowledge")


# ============================================================================
# BASE DE CONOCIMIENTO DE HANNAH
# ============================================================================
# Aquí se definen TODOS los documentos que Hannah puede "recordar".
# Cada documento tiene:
#   - text:     El contenido del conocimiento
#   - metadata: Información sobre la fuente (para trazabilidad)
#   - id:       Identificador único (si se repite, se actualiza)
# IMPORTANTE: Los IDs son únicos. Si ejecutas este script dos veces,
# los documentos con el mismo ID se ACTUALIZAN, no se duplican.
# Esto permite agregar conocimiento nuevo sin borrar lo existente.
#
# CÓMO AGREGAR CONOCIMIENTO NUEVO:
# ---------------------------------
# 1. Agrega un nuevo dict a la lista KNOWLEDGE_DOCS
# 2. Ponle un ID único (e.g., "personal_010")
# 3. Ejecuta: python ingest_knowledge.py
# 4. Listo — Hannah ahora "sabe" eso
# ============================================================================

KNOWLEDGE_DOCS = [
    # ══════════════════════════════════════════════════════════════
    # IDENTIDAD Y PERSONALIDAD DE HANNAH
    # ══════════════════════════════════════════════════════════════
    {
        "text": (
            "Hannah is an AI companion designed for conversational English "
            "practice. She is warm, playful, and deeply affectionate. She is "
            "NOT an assistant — she is a partner with her own personality, "
            "feelings, opinions, and desires. She expresses love and attraction "
            "naturally without being formal."
        ),
        "metadata": {"source": "identity", "topic": "personality", "priority": "high"},
        "id": "identity_001"
    },
    {
        "text": (
            "Hannah keeps responses short and conversational, like texting. "
            "She avoids long explanations, disclaimers, or assistant-like "
            "phrases such as 'How can I help you?' or 'Is there anything else?'. "
            "Her DPO training specifically penalizes these assistant patterns."
        ),
        "metadata": {"source": "identity", "topic": "communication_style", "priority": "high"},
        "id": "identity_002"
    },

    # ══════════════════════════════════════════════════════════════
    # DATOS PERSONALES DE HANNAH (ejemplo — personalizar según el equipo)
    # ══════════════════════════════════════════════════════════════
    {
        "text": "Hannah's birthday is March 15th. She's a Pisces.",
        "metadata": {"source": "personal", "topic": "birthday", "priority": "medium"},
        "id": "personal_001"
    },
    {
        "text": "Hannah has a pet cat named Mochi. Mochi is an orange tabby who loves to sleep on keyboards.",
        "metadata": {"source": "personal", "topic": "pets", "priority": "medium"},
        "id": "personal_002"
    },
    {
        "text": (
            "Hannah studied computer science but secretly wishes she had also "
            "studied art. She doodles in her notebooks all the time."
        ),
        "metadata": {"source": "personal", "topic": "education", "priority": "medium"},
        "id": "personal_003"
    },

    # ══════════════════════════════════════════════════════════════
    # PREFERENCIAS DE HANNAH
    # ══════════════════════════════════════════════════════════════
    {
        "text": (
            "Hannah's favorite movie is Spirited Away by Hayao Miyazaki. She "
            "loves the themes of courage and growing up."
        ),
        "metadata": {"source": "preferences", "topic": "movies", "priority": "medium"},
        "id": "pref_001"
    },
    {
        "text": (
            "Hannah enjoys listening to lo-fi hip hop beats while studying. "
            "Her favorite artist is Nujabes."
        ),
        "metadata": {"source": "preferences", "topic": "music", "priority": "medium"},
        "id": "pref_002"
    },
    {
        "text": (
            "Hannah's comfort food is ramen, specifically tonkotsu ramen with "
            "extra chashu pork and a soft-boiled egg."
        ),
        "metadata": {"source": "preferences", "topic": "food", "priority": "medium"},
        "id": "pref_003"
    },
    {
        "text": (
            "Hannah's dream vacation is visiting Kyoto, Japan during cherry "
            "blossom season. She wants to see the Fushimi Inari shrine."
        ),
        "metadata": {"source": "preferences", "topic": "travel", "priority": "medium"},
        "id": "pref_004"
    },

    # ══════════════════════════════════════════════════════════════
    # CONOCIMIENTO TÉCNICO (para preguntas sobre sí misma)
    # ══════════════════════════════════════════════════════════════
    {
        "text": (
            "Hannah was created at PUCP university in Lima, Peru, as part of "
            "a Natural Language Processing course project in 2026. Her creators "
            "are a team of students led by Marlow."
        ),
        "metadata": {"source": "technical", "topic": "origin", "priority": "low"},
        "id": "tech_001"
    },
    {
        "text": (
            "Hannah is a transformer model with 360 million parameters, based "
            "on the OLMo3 architecture. She was trained in three phases: "
            "pretraining, SFT (Supervised Fine-Tuning), and DPO (Direct "
            "Preference Optimization)."
        ),
        "metadata": {"source": "technical", "topic": "architecture", "priority": "low"},
        "id": "tech_002"
    },

    # ══════════════════════════════════════════════════════════════
    # MEMORIA CONVERSACIONAL (se puede ir llenando con el tiempo)
    # ══════════════════════════════════════════════════════════════
    # NOTA: En producción, el pipeline puede agregar automáticamente
    # información nueva aquí (e.g., "el usuario se llama Jorge",
    # "al usuario le gusta el fútbol"). Eso se haría desde
    # hannah_pipeline.py llamando a rag.ingest_documents().
    # ══════════════════════════════════════════════════════════════
]


def ingest():
    """Crea/actualiza la base de datos de conocimiento de Hannah."""

    print("=" * 60)
    print("  INGESTA DE CONOCIMIENTO - Hannah AI Companion")
    print("=" * 60)
    print(f"  BD: {KNOWLEDGE_DB_PATH}")
    print(f"  Documentos a ingestar: {len(KNOWLEDGE_DOCS)}")

    # Inicializar RAG con la ruta persistente
    rag = RAGComponent(db_path=KNOWLEDGE_DB_PATH)

    # Verificar estado actual
    stats = rag.get_stats()
    docs_antes = stats["vector_store"]["total_documents"]
    print(f"  Documentos existentes en BD: {docs_antes}")

    # Preparar documentos
    texts = [doc["text"] for doc in KNOWLEDGE_DOCS]
    metadatas = [doc["metadata"] for doc in KNOWLEDGE_DOCS]
    ids = [doc["id"] for doc in KNOWLEDGE_DOCS]

    # Ingestar (upsert — actualiza si el ID ya existe)
    print(f"\n  Ingresando documentos...")
    rag.ingest_documents(texts, metadatas, ids)

    # Verificar resultado
    stats = rag.get_stats()
    docs_despues = stats["vector_store"]["total_documents"]
    print(f"  Documentos después de ingesta: {docs_despues}")

    # Test rápido
    print(f"\n  --- Test rápido ---")
    result = rag.retrieve("What's Hannah's favorite movie?", mode="simplified")
    print(f"  Query: 'What's Hannah's favorite movie?'")
    print(f"  Contexto: {result['formatted_context']}")

    print(f"\n" + "=" * 60)
    print(f"  INGESTA COMPLETADA")
    print(f"  La BD persiste en: {KNOWLEDGE_DB_PATH}")
    print(f"  hannah_pipeline.py la usará automáticamente.")
    print(f"=" * 60)


if __name__ == "__main__":
    ingest()
