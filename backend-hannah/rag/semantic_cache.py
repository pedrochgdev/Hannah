# semantic_cache.py
# ============================================================================
# MÓDULO 3 DE 6: Caché Semántico (SemanticCache)
# ============================================================================
# Archivo: rag_standalone/semantic_cache.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
#
# Descripción:
# ========================
# Este programa implementa una "memoria rápida" que evita búsquedas repetidas en ChromaDB.
# Si un usuario pregunta algo muy parecido a una pregunta anterior, devolvemos
# el resultado cacheado directamente SIN buscar de nuevo en la base de datos.
#
# Ubicación en la arquitectura:
# ================================================================================
# En la arquitectura oficial, el Semantic Cache está en la CAPA DE DECISIÓN,
# Antes del Model Selector. El flujo es:
#   Usuario → Web App → Token Sequence Handler → [Semantic Cache]
#                                                      │
#                                              ┌───────┴───────┐
#                                            [HIT]           [MISS]
#                                              │               │
#                                         Web App          Model Selector
#                                        (responde)         → RAG → Modelo
#
# Sin embargo, la arquitectura para el RAG standalone, el Semantic Cache está
# integrado dentro  del RAGComponent como primera verificación antes de buscar.
# IMPORTANTE: Una vez integrado el sistema completo, se debe mover el caché a su ubicación oficial en la capa de decisión.
#
# Flujo:
# ================
# 1. Cuando llega una query, la convertimos a vector (embedding)
# 2. Comparamos ese vector contra TODOS los vectores de queries anteriores
# 3. Si la similitud coseno con alguna query anterior es >= 0.92 → HIT
# 4. Si no hay match → MISS, se busca normalmente y se guarda el resultado
#
# ¿Por que similitud coseno?
# ===============================================
# Porque la gente pregunta lo mismo de muchas formas:
#   - "¿Cuántos parámetros tiene Hannah?"
#   - "¿Cuántos params tiene el modelo Hannah?"
#   - "¿Cuál es el tamaño de Hannah en parámetros?"
# Todas son la misma pregunta. Con matching exacto, fallaríamos en las 3.
# Con similitud coseno, las 3 tienen similitud >0.90 entre sí → HIT.
#
# ¿Por que 0.92 como umbral?
# ============================
# - Si es muy alto (ej. 0.98): casi nunca habrá HIT, el caché es inútil
# - Si es muy bajo (ej. 0.80): habrá falsos positivos (queries diferentes
#   que suenan parecido darían HIT con respuestas incorrectas)
# - 0.92 es un buen balance: captura paráfrasis genuinas sin falsos positivos
# - El doc oficial (Sección 2.3) recomienda "comenzar conservador (≥0.92)"
#
# DEPENDENCIAS:
# =============
# numpy (para operaciones vectoriales)
# embeddings.py (nuestro módulo de embeddings)
# ============================================================================

import numpy as np
from rag.embeddings import EmbeddingService

class SemanticCache:
    """
    Caché semántico en memoria para el RAG de Hannah.

    Almacena pares (query_embedding, response) y busca matches por
    similitud coseno antes de ir a ChromaDB.

    NOTA DE RENDIMIENTO:
    Este caché está en RAM (no en disco). Se pierde al reiniciar Python.
    Para un sistema en producción, se podría persistir en Redis o SQLite.
    Para nuestro entregable del 24 abril, RAM es suficiente.
    """

    def __init__(self, similarity_threshold: float = 0.92, max_cache_size: int = 500):
        """
        Args:
            similarity_threshold: Umbral mínimo de similitud coseno para HIT.
                                  Valor recomendado por la arquitectura: 0.92
                                  → Mayor = más preciso pero menos hits
                                  → Menor = más hits pero riesgo de respuestas incorrectas

            max_cache_size: Máximo de entradas en caché.
                           Cuando se llena, elimina la más antigua (FIFO).
                           500 entradas ≈ 500 vectores × 384 dims × 4 bytes ≈ 750KB de RAM.
                           Es nada comparado con los GB que usa el modelo.
        """
        self.threshold = similarity_threshold
        self.max_size = max_cache_size
        self.embedder = EmbeddingService()

        # Almacenamiento interno: lista de diccionarios
        # Cada entrada: {
        #   "query": str (texto original, para debugging),
        #   "embedding": np.array (vector de 384 dims),
        #   "response": dict (resultado completo del RAG),
        #   "hits": int (contador de veces que fue retornado)
        # }
        self._cache: list[dict] = []

    def lookup(self, query: str) -> dict | None:
        """
        Busca en el caché si existe una query semánticamente similar.

        Proceso:
        1. Convierte la query a vector
        2. Calcula dot product contra todos los vectores del caché
           (como están normalizados, dot product = similitud coseno)
        3. Si el mejor score >= threshold → retorna la respuesta cacheada
        4. Si no → retorna None (MISS)

        Args:
            query: Texto de la query del usuario.

        Returns:
            - dict con el contexto cacheado si hay HIT
            - None si hay MISS

        COMPLEJIDAD: O(n) donde n = entradas en caché.
        Con 500 entradas y vectores de 384 dims, toma ~0.1ms (instantáneo).
        """
        if not self._cache:
            return None

        # Convertir query a vector
        query_emb = np.array(self.embedder.get_embedding(query))

        best_score = -1.0
        best_idx = -1

        for idx, entry in enumerate(self._cache):
            # Producto escalar de vectores normalizados = SIMILITUD COSENO
            # Valores van de -1 (opuestos) a 1 (idénticos)
            score = float(np.dot(query_emb, entry["embedding"]))

            if score > best_score:
                best_score = score
                best_idx = idx

        # ¿El mejor match supera el umbral?
        if best_score >= self.threshold:
            self._cache[best_idx]["hits"] += 1
            print(f"[SemanticCache] HIT (score={best_score:.4f}): "
                  f"'{query[:50]}...' ≈ '{self._cache[best_idx]['query'][:50]}...'")
            return self._cache[best_idx]["response"]

        print(f"[SemanticCache] MISS (best_score={best_score:.4f}): '{query[:50]}...'")
        return None

    def store(self, query: str, response: dict):
        """
        Almacena una nueva entrada en el caché.
        Args:
            query: Texto original de la query.
            response: Resultado completo del RAG pipeline (dict con
                     formatted_context, raw_chunks, scores, etc.)
        POLÍTICA DE LIBERACIÓN: FIFO (First In, First Out)
        Cuando el caché está lleno (500 entradas), eliminamos la más antigua.
        Alternativas más sofisticadas:
          - LRU (Least Recently Used): eliminar la menos usada
          - LFU (Least Frequently Used): eliminar la menos popular
        Para nuestro caso, FIFO es suficiente.
        """
        # Si el caché está lleno, eliminar la entrada más antigua
        if len(self._cache) >= self.max_size:
            removed = self._cache.pop(0)
            print(f"[SemanticCache] Lleno ({self.max_size}), eliminando: "
                  f"'{removed['query'][:30]}...'")

        # Convertir query a vector y almacenar
        query_emb = np.array(self.embedder.get_embedding(query))

        self._cache.append({
            "query": query,
            "embedding": query_emb,
            "response": response,
            "hits": 0
        })
        print(f"[SemanticCache] Almacenado: '{query[:50]}...' "
              f"(total={len(self._cache)})")

    def get_stats(self) -> dict:
        """Retorna estadísticas del caché para monitoreo y debugging."""
        total_hits = sum(entry["hits"] for entry in self._cache)
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "threshold": self.threshold,
            "total_hits": total_hits
        }

    def clear(self):
        """Limpia todo el caché. Útil para testing."""
        self._cache.clear()
        print("[SemanticCache] Caché limpiado.")

