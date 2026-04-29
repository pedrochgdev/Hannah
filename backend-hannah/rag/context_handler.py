# context_handler.py
# ============================================================================
# MÓDULO 5 DE 6: Manejador de Contexto (ContextHandler)
# ============================================================================
# Archivo: rag_standalone/context_handler.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
#
# Descripción:
# ========================
# Recibe los chunks recuperados por VectorStore y los procesa para generar
# el CONTEXTO FINAL que se inyectará al modelo de generación (Hannah 360M
# o Qwen2.5-14B-Instruct).
#
# Es como un editor de noticias: de todos los artículos que el buscador
# encontró, selecciona los mejores, los ordena, los recorta al tamaño
# adecuado, y los empaqueta en un formato que el modelo entiende.
#
# FORMATO DE SALIDA:
# ==================
# El contexto se formatea con los tokens especiales [MEMORY] y [/MEMORY]
# del tokenizer de Hannah. Estos tokens le indican al modelo dónde empieza
# y termina el conocimiento externo inyectado por el RAG.
#
# Ejemplo Simplified (Fast):
#   [MEMORY]Hannah es un modelo transformer de 360M de parámetros basado en OLMo3.[/MEMORY]
#
# Ejemplo Extended (Slow):
#   [MEMORY][Fuente: arquitectura.pdf] Hannah es un modelo transformer de 360M...
#   ---
#   [Fuente: tecnica.pdf] El entrenamiento pasó por tres fases...
#   ---
#   [Fuente: lab_grupal.pdf] El Slow Model usa Qwen2.5-14B-Instruct...[/MEMORY]
#
#
# SOBRE LOS LÍMITES DE TOKENS:
# ============================
# Hannah fue "alineada" con secuencias de 512. Si le mandamos un
# prompt de 1024 tokens, la calidad de la personalidad podría degradarse
# aunque el modelo técnicamente lo procese.
# Decisión:
# El documento de arquitectura define:
#   - Simplified (Fast): ~200 tokens de contexto RAG
#   - Extended (Slow): ~1500 tokens de contexto RAG
# El prompt completo sería: System (~100) + Memory (~200) + User (~50) +
# Historial (~200) = ~550, que cabe en 1024 y también en 512. .
#
# Para Slow (Qwen2.5-14B-Instruct), el SEQ_LEN no es un problema porque
# Qwen2.5 tiene ventana de contexto de 32K tokens.
#
# MODOS DE OPERACIÓN (Sección 2.5 del doc de arquitectura):
# =========================================================
# ┌─────────────┬───────────────────────────────────────────────────────────┐
# │ Modo        │ Configuración                                           │
# ├─────────────┼───────────────────────────────────────────────────────────┤
# │ Simplified  │ 2-3 chunks, sin HyDE, sin QE, ~200 tokens de contexto  │
# │ (Fast)      │ Respuesta directa y concisa para Hannah 360M            │
# ├─────────────┼───────────────────────────────────────────────────────────┤
# │ Extended    │ 5-10 chunks, con HyDE, con QE + reranking,             │
# │ (Slow)      │ ~1500 tokens de contexto para Qwen2.5-14B-Instruct     │
# └─────────────┴───────────────────────────────────────────────────────────┘
# DEPENDENCIAS:
# =============
# numpy (para operaciones vectoriales en reranking)
# embeddings.py (para re-calcular similitudes en reranking)
# ============================================================================

from rag.embeddings import EmbeddingService
import numpy as np


class ContextHandler:
    """
    Procesa chunks recuperados y genera contexto formateado con [MEMORY]/[/MEMORY].

    FLUJO INTERNO:
    1. Recibe resultados crudos de ChromaDB
    2. Convierte distancias a scores de similitud
    3. [Solo Extended] Reranking: recalcula similitudes exactas
    4. Selecciona top-N chunks según modo
    5. Trunca al límite de caracteres
    6. Formatea con [MEMORY]/[/MEMORY]
    """

    # ─── CONFIGURACIÓN POR MODO ───
    MODE_CONFIG = {
        "simplified": {
            "max_chunks": 3,          # Máximo 3 chunks (Sección 2.5: "2-3 chunks")
            "min_chunks": 1,          # Mínimo 1 chunk (siempre algo de contexto)
            "max_context_tokens": 200, # ~200 tokens según arquitectura
            "max_context_chars": 800,  # ~4 chars por token en español promedio
            "use_reranking": False,    # Sin reranking (Sección 2.5: "sin HyDE, sin QE")
            "include_metadata": False, # Contexto limpio, sin etiquetas de fuente
            "separator": " ",          # Un solo bloque continuo (para Hannah 360M)
        },
        "extended": {
            "max_chunks": 10,          # Hasta 10 chunks (Sección 2.5: "5-10 chunks")
            "min_chunks": 3,           # Mínimo 3 para dar contexto rico
            "max_context_tokens": 1500, # ~1500 tokens según arquitectura
            "max_context_chars": 6000,  # ~4 chars por token
            "use_reranking": True,      # Con reranking (Sección 2.5)
            "include_metadata": True,   # Incluir [Fuente: X] por chunk
            "separator": "\n---\n",     # Separar chunks visualmente
        }
    }

    def __init__(self):
        """Inicializa con un EmbeddingService para reranking."""
        self.embedder = EmbeddingService()

    def process(self, search_results: dict, query: str, mode: str = "simplified") -> dict:
        """
        Procesa resultados de búsqueda y genera contexto formateado.
        Args:
            search_results: Resultados de ChromaDB. Formato:
                {
                    "documents": [["texto1", "texto2", ...]],    # Lista anidada
                    "metadatas": [[{meta1}, {meta2}, ...]],       # Lista anidada
                    "distances": [[0.15, 0.25, ...]]              # Cosine distance
                }
            query: Query original del usuario (para reranking en Extended).
            mode: "simplified" (Fast Hannah) o "extended" (Slow Qwen)
        Returns:
            {
                "formatted_context": "[MEMORY]...contexto...[/MEMORY]",
                "raw_chunks": ["chunk1", "chunk2", ...],
                "scores": [0.85, 0.75, ...],
                "mode": "simplified" | "extended",
                "num_chunks": 3,
                "approx_tokens": 195
            }
        """
        config = self.MODE_CONFIG.get(mode)
        if not config:
            raise ValueError(f"Modo '{mode}' no reconocido. Usar 'simplified' o 'extended'.")

        # ─── Paso 1: Extraer datos de ChromaDB ───
        # ChromaDB devuelve listas ANIDADAS: results['documents'][0]
        # El [0] es porque le pasamos 1 sola query (query_embeddings=[...])
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        if not documents:
            return self._empty_response(mode)

        # ─── Paso 2: Convertir distancias a scores de similitud ───
        # En ChromaDB con cosine distance:
        #   distancia = 0 → idénticos (similitud = 1)
        #   distancia = 1 → no relacionados (similitud = 0)
        #   distancia = 2 → opuestos (similitud = -1)
        # Fórmula: similitud = 1 - distancia
        scores = [1.0 - d for d in distances]

        # ─── Paso 3: Crear lista de chunks con metadata ───
        chunks = []
        for doc, meta, score in zip(documents, metadatas, scores):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "score": score
            })

        # ─── Paso 4: Reranking (solo en modo Extended) ───
        # ¿Por qué rerankear si ChromaDB ya ordenó por relevancia?
        # Porque ChromaDB usa búsqueda APROXIMADA (HNSW). El reranking
        # recalcula la similitud exacta para refinar el orden.
        # También es útil cuando combinamos resultados de múltiples
        # queries (Query Expansion) que pueden tener ordenes diferentes.
        if config["use_reranking"] and len(chunks) > 1:
            chunks = self._rerank(chunks, query)

        # ─── Paso 5: Limitar cantidad de chunks ───
        chunks = chunks[:config["max_chunks"]]

        # ─── Paso 6: Truncar al límite de caracteres ───
        # Nos aseguramos de no exceder el límite de tokens del modo
        selected_chunks, total_chars = self._truncate_to_limit(
            chunks, config["max_context_chars"], config["separator"]
        )

        # ─── Paso 7: Formatear con [MEMORY]/[/MEMORY] ───
        formatted = self._format_context(selected_chunks, config)

        return {
            "formatted_context": formatted,
            "raw_chunks": [c["text"] for c in selected_chunks],
            "scores": [c["score"] for c in selected_chunks],
            "mode": mode,
            "num_chunks": len(selected_chunks),
            "approx_tokens": total_chars // 4  # Estimación: ~4 chars por token
        }

    # ─────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ─────────────────────────────────────────────
    def _rerank(self, chunks: list[dict], query: str) -> list[dict]:
        """
        Re-ranking basado en similitud coseno exacta con la query.
        El primer ranking viene de ChromaDB (búsqueda ANN/HNSW, aproximada).
        El reranking recalcula la similitud de forma exacta para cada chunk.
        Esto es especialmente importante cuando se combinan resultados de
        múltiples queries (Query Expansion en modo Extended), porque el
        orden relativo entre resultados de diferentes queries puede ser
        inconsistente.

        Costo: ~1ms por chunk (384 dims × dot product). Para 10 chunks = ~10ms.
        """
        # Calcular embedding de la query original
        query_emb = np.array(self.embedder.get_embedding(query))

        for chunk in chunks:
            # Calcular embedding del chunk
            chunk_emb = np.array(self.embedder.get_embedding(chunk["text"]))
            # Producto escalar de vectores normalizados = similitud coseno exacta
            chunk["rerank_score"] = float(np.dot(query_emb, chunk_emb))

        # Ordenar por score (mayor = más relevante)
        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        # Actualizar score principal
        for chunk in chunks:
            chunk["score"] = chunk["rerank_score"]

        return chunks

    def _truncate_to_limit(self, chunks: list[dict], max_chars: int, separator: str) -> tuple:
        """
        Selecciona chunks hasta llenar el límite de caracteres.
        Va agregando chunks uno por uno. Si un chunk no cabe completo,
        lo trunca (solo si quedan >50 chars de espacio, para no dejar
        fragmentos inútiles).
        Returns:
            Tupla (chunks_seleccionados, total_caracteres)
        """
        selected = []
        total_chars = 0

        for chunk in chunks:
            chunk_len = len(chunk["text"])
            sep_len = len(separator) if selected else 0  # Sin separador antes del primero

            if total_chars + chunk_len + sep_len > max_chars:
                # ¿Cabe una versión truncada?
                remaining = max_chars - total_chars - sep_len
                if remaining > 50:  # Solo si quedan >50 chars útiles
                    truncated_chunk = chunk.copy()
                    truncated_chunk["text"] = chunk["text"][:remaining] + "..."
                    selected.append(truncated_chunk)
                    total_chars += remaining + sep_len
                break  # No hay más espacio

            selected.append(chunk)
            total_chars += chunk_len + sep_len

        return selected, total_chars

    def _format_context(self, chunks: list[dict], config: dict) -> str:
        """
        Genera el contexto final con tokens [MEMORY]/[/MEMORY].
        MODO SIMPLIFIED:
          [MEMORY]Texto del chunk 1. Texto del chunk 2.[/MEMORY]
          Un solo bloque continuo, sin metadata
        MODO EXTENDED:
          [MEMORY][Fuente: arquitectura.pdf] Texto del chunk 1.
          ---
          [Fuente: tecnica.pdf] Texto del chunk 2.[/MEMORY]
          Separadores entre chunks, metadata de fuente incluida
        """
        if not chunks:
            return "[MEMORY][/MEMORY]"

        separator = config["separator"]
        parts = []

        for chunk in chunks:
            if config["include_metadata"] and chunk.get("metadata"):
                # Modo Extended: incluir fuente
                source = chunk["metadata"].get("source", "unknown")
                parts.append(f"[Fuente: {source}] {chunk['text']}")
            else:
                # Modo Simplified: solo texto
                parts.append(chunk["text"])

        context_body = separator.join(parts)

        # Envolver con tokens [MEMORY]/[/MEMORY]
        # Estos tokens están registrados en el tokenizer de Hannah
        # (ver hannah_tok/tokenizer_config.json)
        return f"[MEMORY]{context_body}[/MEMORY]"

    def _empty_response(self, mode: str) -> dict:
        """Respuesta cuando no hay resultados de búsqueda."""
        return {
            "formatted_context": "[MEMORY][/MEMORY]",
            "raw_chunks": [],
            "scores": [],
            "mode": mode,
            "num_chunks": 0,
            "approx_tokens": 0
        }
