# rag_component.py
# ============================================================================
# MÓDULO 6 DE 6: Orquestador Principal (RAGComponent)
# ============================================================================
# Archivo: rag_standalone/rag_component.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
# Descripción:
# ========================
# Es el "cerebro" del RAG. Une los 5 módulos anteriores en un solo pipeline:
#   1. SemanticCache   → ¿Ya respondimos esto antes?
#   2. QueryEnhancer   → Mejorar la pregunta para buscar mejor
#   3. VectorStore     → Buscar en ChromaDB
#   4. ContextHandler  → Formatear con [MEMORY]/[/MEMORY]
#   5. SemanticCache   → Guardar resultado para futuras consultas
# DIAGRAMA DE FLUJO COMPLETO:
# ============================
#   ┌─────────────────────────────────────────────────┐
#   │              RAGComponent.retrieve()             │
#   │                                                  │
#   │  0. Pre-filtro: len(query) < 3?                  │
#   │       │                                          │
#   │     [SÍ] ───→ return vacío (skip embeddings)     │
#   │       │                                          │
#   │     [NO]                                         │
#   │       ↓                                          │
#   │  1. SemanticCache.lookup(query)                  │
#   │       │                                          │
#   │     [HIT] ───→ return cached response            │
#   │       │                                          │
#   │     [MISS]                                       │
#   │       ↓                                          │
#   │  2. QueryEnhancer.enhance(query, mode)           │
#   │       ↓                                          │
#   │  3. VectorStore.search() × N queries             │
#   │     (multi-query: busca con cada variante)       │
#   │       ↓                                          │
#   │  3.5 Filtro de relevancia: best_score < 0.35?    │
#   │       │                                          │
#   │     [SÍ] ───→ return vacío (no contexto)         │
#   │       │                                          │
#   │     [NO]                                         │
#   │       ↓                                          │
#   │  4. ContextHandler.process(results, mode)        │
#   │     (selecciona, rerankea, trunca, formatea)     │
#   │       ↓                                          │
#   │  5. SemanticCache.store(query, response)         │
#   │       ↓                                          │
#   │  return {"formatted_context": "[MEMORY]...",     │
#   │          "raw_chunks": [...], "timing": {...}}   │
#   └─────────────────────────────────────────────────┘
# INTERFAZ DE USO:
# ================
#   rag = RAGComponent()
#   # Ingestar conocimiento (una sola vez, se persiste en disco)
#   rag.ingest_documents(docs, metadatas, ids)
#   # Recuperar contexto para Fast Model (Hannah 360M)
#   result = rag.retrieve("¿Qué es Hannah?", mode="simplified")
#   context = result["formatted_context"]
#   # → "[MEMORY]Hannah es un modelo de 360M parámetros...[/MEMORY]"
#   # Recuperar contexto para Slow Model (Qwen2.5-14B-Instruct)
#   result = rag.retrieve("Explica el DPO", mode="extended")
#   # Versión async (para no bloquear el servidor web)
#   result = await rag.aretrieve("¿Qué es Hannah?", mode="simplified")
# Asincronía:
# ========================
# Desacoplamiento temporal: mientras el Model Selector
# decide fast/slow, el RAG ya puede estar buscando en paralelo.
#
# Nuestro aretrieve() usa ThreadPoolExecutor porque:
# - ChromaDB y sentence-transformers son operaciones bloqueantes (CPU-bound)
# - asyncio solo es útil para I/O-bound (red, disco)
# - ThreadPoolExecutor corre código bloqueante en hilos separados
# - Así el event loop de asyncio (FastAPI, por ejemplo) no se congela
#
# DEPENDENCIAS:
# ========================
# Todos los módulos anteriores + asyncio + concurrent.futures (stdlib)
# ============================================================================

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from rag.semantic_cache import SemanticCache
from rag.query_enhancer import QueryEnhancer
from rag.context_handler import ContextHandler

# ─── Logger estructurado ───
# Permite controlar verbosidad desde el backend sin tocar código:
#   logging.getLogger("hannah.rag").setLevel(logging.WARNING)  # producción
#   logging.getLogger("hannah.rag").setLevel(logging.DEBUG)    # desarrollo
logger = logging.getLogger("hannah.rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[RAG %(levelname)s] %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RAGComponent:
    """
    Este es el único módulo que el resto del equipo necesita usar.
    Solo necesitan hacer:
        rag = RAGComponent()
        result = rag.retrieve("pregunta del usuario", mode="simplified")
        context = result["formatted_context"]
        # → Inyectar `context` en el prompt del modelo
    """

    # ─── UMBRAL MÍNIMO DE RELEVANCIA ───
    # Si el MEJOR chunk tiene score (1 - distancia) menor a este valor,
    # NO se inyecta contexto. Esto evita que mensajes casuales como
    # "hi", "love you", "haha" reciban contexto irrelevante del RAG
    # que confunde al modelo.
    #
    # Valores de referencia empíricos:
    #   "what's your favorite movie?" vs doc de película → score ~0.60
    #   "how old are you?"            vs doc de birthday → score ~0.45
    #   "hi"                          vs cualquier doc   → score ~0.15
    #   "love you"                    vs cualquier doc   → score ~0.20
    #
    # Con umbral 0.35:
    #   - Preguntas relevantes (>0.35) → SÍ reciben contexto
    #   - Chat casual (<0.35) → NO recibe contexto (Hannah responde libre)
    MIN_RELEVANCE_SCORE = 0.35

    # ─── LONGITUD MÍNIMA DE QUERY ───
    # Queries de 1-2 caracteres ("hi", "k", "ok") nunca son preguntas
    # de conocimiento. Cortamos antes de gastar cómputo en embeddings.
    MIN_QUERY_LENGTH = 3

    def __init__(self, db_path: str = "./hannah_vectordb",
                 cache_threshold: float = 0.92,
                 cache_size: int = 500):
        """
        Inicializa todos los componentes del pipeline.
        Args:
            db_path: Ruta a la base de datos ChromaDB.
                     Default: "./hannah_vectordb" (carpeta local) Se crea automáticamente si no existe.
            cache_threshold: Umbral de similitud para el Semantic Cache.
                            Default: 0.92 
            cache_size: Tamaño máximo del caché semántico.
                       Default: 500 entradas (~750KB de RAM)
        """
        logger.info("Inicializando componentes...")

        # ─── Componente 1: Base de datos vectorial ───
        # Almacena los documentos como vectores en ChromaDB
        self.vector_store = VectorStore(db_path=db_path)

        # ─── Componente 2: Caché semántico ───
        # Evita búsquedas repetidas para queries similares
        self.cache = SemanticCache(
            similarity_threshold=cache_threshold,
            max_cache_size=cache_size
        )

        # ─── Componente 3: Mejora de queries ───
        # Genera variantes para buscar mejor (solo en modo Extended)
        self.query_enhancer = QueryEnhancer()

        # ─── Componente 4: Manejador de contexto ───
        # Selecciona, rerankea, trunca y formatea los chunks
        self.context_handler = ContextHandler()

        # ─── ThreadPoolExecutor para async ───
        # 2 workers: uno para embeddings, otro para ChromaDB
        # Más workers no ayudan porque ambos usan CPU intensivamente
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info("Todos los componentes inicializados correctamente.")

    def retrieve(self, query: str, mode: str = "simplified") -> dict:
        """
        Pipeline RAG síncrono. Punto de entrada principal.
        Args:
            query: Pregunta del usuario en texto natural.
                   Ejemplo: "¿Cuántos parámetros tiene Hannah?"

            mode: Señal del Model Selector.
                  - "simplified": para Fast Model (Hannah 360M)
                    → 2-3 chunks, ~200 tokens, sin HyDE
                  - "extended": para Slow Model (Qwen2.5-14B-Instruct)
                    → 5-10 chunks, ~1500 tokens, con HyDE + QE
        Returns:
            dict con:
            {
                "formatted_context": "[MEMORY]...[/MEMORY]",
                "raw_chunks": ["chunk1", "chunk2"],
                "scores": [0.85, 0.75],
                "mode": "simplified",
                "num_chunks": 2,
                "approx_tokens": 180,
                "cache_hit": False,
                "enhanced_query": {...}  # Info de QueryEnhancer
            }
        """
        t_start = time.perf_counter()
        timing = {}
        logger.info(f"retrieve(mode={mode}): '{query[:60]}...'")

        # ═══════════════════════════════════════════════════
        # PASO 0: Pre-filtro de queries triviales
        # ═══════════════════════════════════════════════════
        # Queries vacías o muy cortas (1-2 chars) nunca necesitan RAG.
        # Cortamos ANTES de computar embeddings → ahorra ~30ms por request.
        clean_query = query.strip()
        if len(clean_query) < self.MIN_QUERY_LENGTH:
            logger.info(f"Query muy corta ({len(clean_query)} chars) → skip RAG")
            return {
                "formatted_context": "[MEMORY][/MEMORY]",
                "raw_chunks": [],
                "scores": [],
                "num_chunks": 0,
                "approx_tokens": 0,
                "cache_hit": False,
                "enhanced_query": None,
                "filtered_by_relevance": True,
                "best_relevance_score": 0.0,
                "skip_reason": "query_too_short",
                "timing": {"total_ms": 0}
            }

        # ═══════════════════════════════════════════════════
        # PASO 1: Verificar Semantic Cache
        # ═══════════════════════════════════════════════════
        # Si una query similar ya fue procesada (similitud >= 0.92),
        # devolvemos el resultado cacheado sin buscar en ChromaDB.
        # Esto ahorra ~50-100ms por request.
        t0 = time.perf_counter()
        cached = self.cache.lookup(query)
        timing["cache_lookup_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if cached is not None:
            cached["cache_hit"] = True
            cached["timing"] = timing
            return cached

        # ═══════════════════════════════════════════════════
        # TRY-EXCEPT: Degradación graceful
        # ═══════════════════════════════════════════════════
        # Si cualquier componente falla (ChromaDB caído, embedding model
        # corrupto, etc.), el RAG retorna contexto vacío en vez de
        # crashear todo el backend. Hannah puede responder sin contexto.
        try:
            # ═══════════════════════════════════════════════════
            # PASO 2: Mejorar la query
            # ═══════════════════════════════════════════════════
            # Simplified: solo limpia → 1 query de búsqueda
            # Extended: limpia + expande + HyDE → 4-5 queries
            t0 = time.perf_counter()
            enhanced = self.query_enhancer.enhance(query, mode=mode)
            search_queries = enhanced["search_queries"]
            timing["query_enhance_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            logger.debug(f"Queries de búsqueda generadas: {len(search_queries)}")

            # ═══════════════════════════════════════════════════
            # PASO 3: Buscar en VectorStore (ChromaDB)
            # ═══════════════════════════════════════════════════
            # Simplified: busca con 1 query, trae 3 resultados
            # Extended: busca con 4-5 queries, trae 10 resultados, fusiona
            t0 = time.perf_counter()
            n_results = 3 if mode == "simplified" else 10
            all_results = self._multi_query_search(search_queries, n_results)
            timing["vector_search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # ═══════════════════════════════════════════════════
            # PASO 3.5: Filtro de relevancia mínima
            # ═══════════════════════════════════════════════════
            # ChromaDB SIEMPRE devuelve resultados, incluso para "hi" o "love you".
            # Aquí verificamos si el MEJOR resultado es realmente relevante.
            # Score = 1 - distancia (cosine). Si el mejor score < MIN_RELEVANCE_SCORE,
            # significa que NINGÚN documento es relevante → no inyectar contexto.
            distances = all_results.get("distances", [[]])[0]
            if distances:
                best_distance = min(distances)
                best_score = 1.0 - best_distance
                logger.info(f"Mejor score de relevancia: {best_score:.3f} "
                           f"(umbral: {self.MIN_RELEVANCE_SCORE})")

                if best_score < self.MIN_RELEVANCE_SCORE:
                    logger.info(f"Score {best_score:.3f} < {self.MIN_RELEVANCE_SCORE} → "
                               f"contexto NO relevante, retornando vacío")
                    timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
                    empty_result = {
                        "formatted_context": "[MEMORY][/MEMORY]",
                        "raw_chunks": [],
                        "scores": [],
                        "num_chunks": 0,
                        "approx_tokens": 0,
                        "cache_hit": False,
                        "enhanced_query": enhanced,
                        "filtered_by_relevance": True,
                        "best_relevance_score": best_score,
                        "timing": timing
                    }
                    # Cachear también el resultado vacío para que
                    # queries similares sean HIT directo
                    self.cache.store(query, empty_result)
                    return empty_result
            else:
                # Sin resultados de ChromaDB → vacío
                logger.warning("ChromaDB no devolvió resultados")
                timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
                return {
                    "formatted_context": "[MEMORY][/MEMORY]",
                    "raw_chunks": [],
                    "scores": [],
                    "num_chunks": 0,
                    "approx_tokens": 0,
                    "cache_hit": False,
                    "enhanced_query": enhanced,
                    "filtered_by_relevance": True,
                    "best_relevance_score": 0.0,
                    "timing": timing
                }

            # ═══════════════════════════════════════════════════
            # PASO 4: Procesar con ContextHandler
            # ═══════════════════════════════════════════════════
            # Selecciona mejores chunks, rerankea (Extended), trunca al
            # límite de tokens, formatea con [MEMORY]/[/MEMORY]
            t0 = time.perf_counter()
            context_result = self.context_handler.process(
                search_results=all_results,
                query=query,
                mode=mode
            )
            timing["context_handler_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # Añadir metadata extra
            context_result["cache_hit"] = False
            context_result["enhanced_query"] = enhanced
            context_result["best_relevance_score"] = best_score

            # ═══════════════════════════════════════════════════
            # PASO 5: Almacenar en Semantic Cache
            # ═══════════════════════════════════════════════════
            # Para que la próxima query similar sea un HIT
            self.cache.store(query, context_result)

            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            context_result["timing"] = timing

            logger.info(f"Contexto generado: {context_result['num_chunks']} chunks, "
                       f"~{context_result['approx_tokens']} tokens, "
                       f"{timing['total_ms']}ms total")

            return context_result

        except Exception as e:
            # ═══════════════════════════════════════════════════
            # DEGRADACIÓN GRACEFUL
            # ═══════════════════════════════════════════════════
            # Si algo falla, Hannah sigue funcionando — solo sin contexto.
            # El error se loguea para que el equipo pueda investigar.
            logger.error(f"Error en pipeline RAG: {type(e).__name__}: {e}")
            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return {
                "formatted_context": "[MEMORY][/MEMORY]",
                "raw_chunks": [],
                "scores": [],
                "num_chunks": 0,
                "approx_tokens": 0,
                "cache_hit": False,
                "enhanced_query": None,
                "filtered_by_relevance": False,
                "best_relevance_score": 0.0,
                "error": f"{type(e).__name__}: {str(e)}",
                "timing": timing
            }

    async def aretrieve(self, query: str, mode: str = "simplified") -> dict:
        """
        Pipeline RAG asíncrono.
        Hace lo mismo que retrieve() pero sin bloquear el event loop.
        Útil cuando se integra con FastAPI o cualquier framework async.
        Ejemplo con FastAPI:
            @app.post("/rag")
            async def get_context(query: str, mode: str):
                result = await rag.aretrieve(query, mode)
                return result
        Internamente usa ThreadPoolExecutor porque ChromaDB y
        sentence-transformers son operaciones CPU-bound (bloqueantes).
        """
        loop = asyncio.get_event_loop()
        # Ejecutar retrieve() en un thread separado
        result = await loop.run_in_executor(
            self._executor,
            self.retrieve,
            query,
            mode
        )
        return result

    def _multi_query_search(self, queries: list[str], n_results: int) -> dict:
        """
        Busca múltiples queries y fusiona los resultados.
        ¿POR QUÉ MULTI-QUERY?
        En modo Extended, QueryEnhancer genera 4-5 variantes de la pregunta.
        Buscamos CADA variante en ChromaDB y fusionamos los resultados.
        Si un documento aparece en múltiples búsquedas, nos quedamos con
        el mejor score (menor distancia).
        Args:
            queries: Lista de queries de búsqueda.
            n_results: Cuántos resultados traer por query.
        Returns:
            Diccionario en formato ChromaDB (fusionado y deduplicado).
        """
        # Caso simple: 1 sola query (modo Simplified)
        if len(queries) == 1:
            return self.vector_store.search(queries[0], n_results=n_results)

        # Caso multi-query: fusionar resultados de todas las queries
        seen_docs = {}  # key: texto del doc, value: {metadata, distance}

        for q in queries:
            results = self.vector_store.search(q, n_results=n_results)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                if doc in seen_docs:
                    # Documento ya visto: quedarse con el MEJOR score
                    # (menor distancia = más relevante)
                    if dist < seen_docs[doc]["distance"]:
                        seen_docs[doc] = {"metadata": meta, "distance": dist}
                else:
                    seen_docs[doc] = {"metadata": meta, "distance": dist}

        # Ordenar por distancia (menor = más relevante)
        sorted_docs = sorted(seen_docs.items(), key=lambda x: x[1]["distance"])

        # Limitar al n_results pedido
        sorted_docs = sorted_docs[:n_results]

        # Reconstruir formato ChromaDB
        return {
            "documents": [[doc for doc, _ in sorted_docs]],
            "metadatas": [[info["metadata"] for _, info in sorted_docs]],
            "distances": [[info["distance"] for _, info in sorted_docs]]
        }

    def ingest_documents(self, documents: list[str], metadatas: list[dict],
                         ids: list[str]):
        """
        Ingesta documentos en la base de datos vectorial.
        Wrapper sobre VectorStore.add_documents() para 
        solo interactuar con RAGComponent (interfaz única).
        Args:
            documents: Lista de textos a indexar.
                       Ejemplo: ["Hannah tiene 360M params", "RAG usa ChromaDB"]
            metadatas: Info extra por documento.
                       Ejemplo: [{"source": "docs"}, {"source": "manual"}]
            ids: IDs únicos por documento.
                 Ejemplo: ["doc1", "doc2"]
        EJEMPLO COMPLETO:
            rag.ingest_documents(
                documents=["Hannah es un modelo de 360M de parámetros."],
                metadatas=[{"source": "arquitectura.pdf"}],
                ids=["arch_001"]
            )
        """
        self.vector_store.add_documents(documents, metadatas, ids)

    def get_stats(self) -> dict:
        """
        Retorna estadísticas del sistema para monitoreo.

        Útil para debugging y para la presentación:
        - ¿Cuántos documentos hay en la BD?
        - ¿Cuántas entradas en caché?
        - ¿Cuántos cache hits se han dado?
        """
        collection_count = self.vector_store.collection.count()
        cache_stats = self.cache.get_stats()
        return {
            "vector_store": {
                "total_documents": collection_count,
            },
            "cache": cache_stats,
            "relevance_threshold": self.MIN_RELEVANCE_SCORE,
            "min_query_length": self.MIN_QUERY_LENGTH,
            "status": "operational"
        }

    def health_check(self) -> dict:
        """
        Diagnóstico rápido del pipeline RAG.
        Útil para el endpoint /health del backend o para debugging.
        Verifica:
          - VectorStore accesible y con documentos
          - Cache funcional
          - Embedding model cargado
        Returns:
            {"healthy": True/False, "issues": [...], "stats": {...}}
        """
        issues = []

        # Verificar VectorStore
        try:
            doc_count = self.vector_store.collection.count()
            if doc_count == 0:
                issues.append("VectorStore vacío: no hay documentos ingestados")
        except Exception as e:
            issues.append(f"VectorStore inaccesible: {str(e)}")
            doc_count = -1

        # Verificar Cache
        try:
            cache_stats = self.cache.get_stats()
        except Exception as e:
            issues.append(f"Cache error: {str(e)}")
            cache_stats = {}

        # Verificar Embedding Service (intenta generar un embedding de test)
        try:
            test_embedding = self.vector_store.embedder.get_embedding("test")
            if len(test_embedding) != 384:
                issues.append(f"Embedding dimensión inesperada: {len(test_embedding)}")
        except Exception as e:
            issues.append(f"Embedding model error: {str(e)}")

        healthy = len(issues) == 0
        if healthy:
            logger.info("Health check: OK")
        else:
            logger.warning(f"Health check: {len(issues)} issues encontrados")

        return {
            "healthy": healthy,
            "issues": issues,
            "stats": {
                "documents": doc_count,
                "cache": cache_stats,
                "relevance_threshold": self.MIN_RELEVANCE_SCORE
            }
        }

    def adjust_relevance_threshold(self, new_threshold: float) -> dict:
        """
        Ajusta el umbral de relevancia en runtime.
        Útil para A/B testing o para afinar durante desarrollo.
        Args:
            new_threshold: Nuevo valor entre 0.0 y 1.0.
                          Recomendado: 0.25-0.45
                          Más bajo → más permisivo (más contexto inyectado)
                          Más alto → más estricto (menos contexto)
        Returns:
            {"previous": float, "current": float}
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold debe estar entre 0.0 y 1.0, recibido: {new_threshold}")

        previous = self.MIN_RELEVANCE_SCORE
        self.MIN_RELEVANCE_SCORE = new_threshold
        logger.info(f"Relevance threshold ajustado: {previous:.3f} → {new_threshold:.3f}")

        # Limpiar cache porque los resultados previos pueden tener
        # decisiones de filtrado diferentes
        self.cache.clear()
        logger.info("Cache limpiado (threshold cambió)")

        return {"previous": previous, "current": new_threshold}

    def debug_relevance(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Herramienta de diagnóstico para entender qué scores obtiene una query.
        NO modifica el cache ni genera contexto. Solo muestra qué encontraría
        el RAG si se le preguntara.

        Útil para:
          - Afinar MIN_RELEVANCE_SCORE empíricamente
          - Verificar que los documentos ingestados son encontrables
          - Debugging cuando "el RAG no da contexto y debería"

        Ejemplo:
            results = rag.debug_relevance("what's your favorite movie?")
            for r in results:
                print(f"  score={r['score']:.3f} | {r['text'][:60]}")

        Args:
            query: Texto a buscar.
            n_results: Cuántos resultados mostrar.
        Returns:
            Lista de dicts ordenada por score (mayor = más relevante):
            [{"text": "...", "score": 0.65, "distance": 0.35,
              "would_pass_filter": True, "metadata": {...}}]
        """
        results = self.vector_store.search(query, n_results=n_results)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        output = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist
            output.append({
                "text": doc,
                "score": round(score, 4),
                "distance": round(dist, 4),
                "would_pass_filter": score >= self.MIN_RELEVANCE_SCORE,
                "metadata": meta
            })

        # Ya viene ordenado por distancia de ChromaDB, pero lo explicitamos
        output.sort(key=lambda x: x["score"], reverse=True)
        return output


# ===========================================================================================================================================================================================================================================
# PRUEBA END-TO-END
# ============================================================================
# Ejecutar: python rag_component.py
# Este test simula el flujo completo:
# 1. Inicializa el RAG
# 2. Ingesta 8 documentos de prueba
# 3. Busca en modo Simplified (Fast) → debe retornar 2-3 chunks
# 4. Busca la misma query de nuevo → debe ser Cache HIT
# 5. Busca en modo Extended (Slow) → debe retornar más chunks con QE
# 6. Muestra estadísticas del sistema
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Component - Test End-to-End")
    print("=" * 60)

    # ─── Inicializar con BD de test temporal ───
    import shutil, os
    TEST_DB = "./hannah_test_db"
    if os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB)

    rag = RAGComponent(db_path=TEST_DB)

    # ─── Ingestar documentos de prueba ───
    # Estos simulan el conocimiento que tendría la BD real de Hannah
    test_docs = [
        "Hannah es un modelo transformer de 360 millones de parámetros basado en la arquitectura OLMo3.",
        "El entrenamiento de Hannah pasó por tres fases: pretraining con 80k steps, SFT con 15k steps y DPO con 1500 steps.",
        "El Slow Model del sistema Hannah usa Qwen2.5-14B-Instruct, un modelo preentrenado al que se le aplicó SFT.",
        "El RAG de Hannah usa ChromaDB como base de datos vectorial con embeddings de all-MiniLM-L6-v2 (384 dimensiones).",
        "El Semantic Cache evita búsquedas repetidas comparando la similitud coseno de las queries con un umbral de 0.92.",
        "La arquitectura de Hannah sigue el paradigma de Prepared Mind, Fast Response (Zhang et al., 2025).",
        "El Model Selector decide si una query va al Fast Model (Hannah 360M) o al Slow Model (Qwen2.5-14B-Instruct).",
        "Hannah fue diseñada como AI Companion para practicar inglés de forma conversacional y natural."
    ]
    test_metas = [{"source": f"test_doc_{i}"} for i in range(len(test_docs))]
    test_ids = [f"test_{i}" for i in range(len(test_docs))]

    rag.ingest_documents(test_docs, test_metas, test_ids)

    # ═══════════════════════════════════════════════════
    # TEST 1: Modo Simplified (Fast Hannah 360M)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: Modo Simplified (Fast)")
    print("=" * 60)
    result = rag.retrieve("¿Cuántos parámetros tiene Hannah?", mode="simplified")
    print(f"  Cache hit:  {result['cache_hit']}")
    print(f"  Chunks:     {result['num_chunks']}")
    print(f"  Tokens:     ~{result['approx_tokens']}")
    print(f"  Contexto:\n  {result['formatted_context']}")

    # ═══════════════════════════════════════════════════
    # TEST 2: Cache con threshold de producción (0.92)
    # ═══════════════════════════════════════════════════
    # NOTA: "¿Cuántos parámetros tiene el modelo Hannah?" vs la original
    # tiene score coseno ~0.907, que es MENOR que el threshold de producción
    # (0.92). Por lo tanto, en producción esto es MISS — correcto.
    # Con threshold de test (0.90) sería HIT (ver semantic_cache.py Test 4).
    print("\n" + "=" * 60)
    print("  TEST 2: Cache con threshold producción (0.92)")
    print("=" * 60)
    result2 = rag.retrieve("¿Cuántos parámetros tiene el modelo Hannah?", mode="simplified")
    print(f"  Cache hit:  {result2['cache_hit']} (MISS esperado: score ~0.907 < threshold 0.92)")

    # ═══════════════════════════════════════════════════
    # TEST 3: Modo Extended (Slow Qwen2.5-14B)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: Modo Extended (Slow)")
    print("=" * 60)
    result3 = rag.retrieve(
        "¿Cómo funciona el entrenamiento de Hannah y qué fases tiene?",
        mode="extended"
    )
    print(f"  Cache hit:  {result3['cache_hit']}")
    print(f"  Chunks:     {result3['num_chunks']}")
    print(f"  Tokens:     ~{result3['approx_tokens']}")
    print(f"  Queries usadas: {len(result3['enhanced_query']['search_queries'])}")
    print(f"  Contexto (primeros 300 chars):\n  {result3['formatted_context'][:300]}...")

    # ═══════════════════════════════════════════════════
    # TEST 4: Filtro de relevancia (queries irrelevantes)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: Filtro de relevancia - Chat casual")
    print("=" * 60)
    casual_queries = ["hi", "love you", "haha ok", "k"]
    for cq in casual_queries:
        r = rag.retrieve(cq, mode="simplified")
        filtered = r.get("filtered_by_relevance", False)
        skip = r.get("skip_reason", "")
        score = r.get("best_relevance_score", "N/A")
        status = "FILTRADO" if filtered else "PASÓ"
        reason = f" (skip: {skip})" if skip else f" (score: {score})"
        print(f"  '{cq}' → {status}{reason}")

    # ═══════════════════════════════════════════════════
    # TEST 5: Health check
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 5: Health Check")
    print("=" * 60)
    health = rag.health_check()
    print(f"  Healthy: {health['healthy']}")
    if health['issues']:
        for issue in health['issues']:
            print(f"  ⚠ {issue}")

    # ═══════════════════════════════════════════════════
    # TEST 6: Timing del pipeline
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 6: Timing del Pipeline")
    print("=" * 60)
    result_timed = rag.retrieve("¿Qué modelo usa Hannah?", mode="simplified")
    t = result_timed.get("timing", {})
    for step, ms in t.items():
        print(f"  {step}: {ms}ms")

    # ═══════════════════════════════════════════════════
    # TEST 7: Debug de relevancia
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 7: Debug Relevancia")
    print("=" * 60)
    for test_q in ["¿Cuántos parámetros tiene?", "haha ok bye"]:
        print(f"\n  Query: '{test_q}'")
        debug = rag.debug_relevance(test_q, n_results=3)
        for d in debug:
            icon = "PASS" if d["would_pass_filter"] else "FAIL"
            print(f"    [{icon}] score={d['score']:.3f} | {d['text'][:50]}...")

    # ═══════════════════════════════════════════════════
    # ESTADÍSTICAS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ESTADÍSTICAS DEL SISTEMA")
    print("=" * 60)
    stats = rag.get_stats()
    print(f"  Documentos en VectorStore: {stats['vector_store']['total_documents']}")
    print(f"  Entradas en caché:         {stats['cache']['entries']}")
    print(f"  Hits totales del caché:    {stats['cache']['total_hits']}")
    print(f"  Estado:                    {stats['status']}")

    # ─── Cleanup ───
    # En Windows, ChromaDB PersistentClient mantiene file locks
    # sobre data_level0.bin. Hay que liberar el objeto antes de borrar.
    del rag          # destruye RAGComponent → destruye VectorStore → libera ChromaDB
    import gc
    gc.collect()     # fuerza liberación de objetos huérfanos
    import time
    time.sleep(0.5)  # da tiempo a Windows para soltar los file handles

    if os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB, ignore_errors=True)
        print(f"\n[Cleanup] BD de test eliminada.")

    print("\n" + "=" * 60)
    print("TODOS LOS TESTS COMPLETADOS")
    print("=" * 60)
