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
#   ┌───────────────────────────────────────────────┐
#   │              RAGComponent.retrieve()           │
#   │                                                │
#   │  1. SemanticCache.lookup(query)                │
#   │       │                                        │
#   │     [HIT] ───→ return cached response          │
#   │       │                                        │
#   │     [MISS]                                     │
#   │       ↓                                        │
#   │  2. QueryEnhancer.enhance(query, mode)         │
#   │       ↓                                        │
#   │  3. VectorStore.search() × N queries           │
#   │     (multi-query: busca con cada variante)     │
#   │       ↓                                        │
#   │  4. ContextHandler.process(results, mode)      │
#   │     (selecciona, rerankea, trunca, formatea)   │
#   │       ↓                                        │
#   │  5. SemanticCache.store(query, response)       │
#   │       ↓                                        │
#   │  return {"formatted_context": "[MEMORY]...",   │
#   │          "raw_chunks": [...],                  │
#   │          "scores": [...], ...}                 │
#   └───────────────────────────────────────────────┘
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
from concurrent.futures import ThreadPoolExecutor

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from rag.semantic_cache import SemanticCache
from rag.query_enhancer import QueryEnhancer
from rag.context_handler import ContextHandler

class RAGComponent:
    """
    Este es el único módulo que el resto del equipo necesita usar.
    Solo necesitan hacer:
        rag = RAGComponent()
        result = rag.retrieve("pregunta del usuario", mode="simplified")
        context = result["formatted_context"]
        # → Inyectar `context` en el prompt del modelo
    """

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
        print("[RAG] Inicializando componentes...")

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

        print("[RAG] Todos los componentes inicializados correctamente.")

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
        print(f"\n[RAG] retrieve(mode={mode}): '{query[:60]}...'")

        # ═══════════════════════════════════════════════════
        # PASO 1: Verificar Semantic Cache
        # ═══════════════════════════════════════════════════
        # Si una query similar ya fue procesada (similitud >= 0.92),
        # devolvemos el resultado cacheado sin buscar en ChromaDB.
        # Esto ahorra ~50-100ms por request.
        cached = self.cache.lookup(query)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        # ═══════════════════════════════════════════════════
        # PASO 2: Mejorar la query
        # ═══════════════════════════════════════════════════
        # Simplified: solo limpia → 1 query de búsqueda
        # Extended: limpia + expande + HyDE → 4-5 queries
        enhanced = self.query_enhancer.enhance(query, mode=mode)
        search_queries = enhanced["search_queries"]
        print(f"[RAG] Queries de búsqueda generadas: {len(search_queries)}")

        # ═══════════════════════════════════════════════════
        # PASO 3: Buscar en VectorStore (ChromaDB)
        # ═══════════════════════════════════════════════════
        # Simplified: busca con 1 query, trae 3 resultados
        # Extended: busca con 4-5 queries, trae 10 resultados, fusiona
        n_results = 3 if mode == "simplified" else 10
        all_results = self._multi_query_search(search_queries, n_results)

        # ═══════════════════════════════════════════════════
        # PASO 4: Procesar con ContextHandler
        # ═══════════════════════════════════════════════════
        # Selecciona mejores chunks, rerankea (Extended), trunca al
        # límite de tokens, formatea con [MEMORY]/[/MEMORY]
        context_result = self.context_handler.process(
            search_results=all_results,
            query=query,
            mode=mode
        )

        # Añadir metadata extra
        context_result["cache_hit"] = False
        context_result["enhanced_query"] = enhanced

        # ═══════════════════════════════════════════════════
        # PASO 5: Almacenar en Semantic Cache
        # ═══════════════════════════════════════════════════
        # Para que la próxima query similar sea un HIT
        self.cache.store(query, context_result)

        print(f"[RAG] Contexto generado: {context_result['num_chunks']} chunks, "
              f"~{context_result['approx_tokens']} tokens")

        return context_result

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
            "status": "operational"
        }


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
