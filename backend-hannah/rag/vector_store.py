# vector_store.py
# ============================================================================
# MÓDULO 2 DE 6: Base de Datos Vectorial (VectorStore)
# ============================================================================
# Archivo: rag_standalone/vector_store.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
#
# Descripción 
# ========================
# Este programa maneja la base de datos vectorial usando ChromaDB. Es donde se guardan
# y se buscan los documentos del conocimiento de Hannah.
#
# ANALOGÍA: Imagina una biblioteca.
#   - add_documents() = poner libros nuevos en los estantes
#   - search() = pedirle al bibliotecario que te encuentre los libros
#     más relevantes para tu pregunta
#
# La diferencia con una base de datos normal (SQL) es que aquí NO buscamos
# por palabras exactas, sino por SIGNIFICADO. Ejemplo:
#   - Query: "¿Cuántos parámetros tiene Hannah?"
#   - Resultado: "Hannah 360M tiene 360 millones de parámetros"
#   → Encontró el documento aunque NO comparten las mismas palabras exactas.
#     Esto funciona porque comparamos los VECTORES (embeddings) de ambos textos.
#
# ¿QUÉ ES ChromaDB?
# ==================
# ChromaDB es una base de datos vectorial open-source y liviana.
# Alternativas serían Pinecone (cloud, de pago), Weaviate, Milvus, FAISS.
# Elegimos ChromaDB porque:
#   1. Es gratis y open-source
#   2. Se instala con pip (sin Docker ni servicios externos)
#   3. Persiste datos en disco (no se pierden al cerrar Python)
#   4. Suficiente para nuestro volumen de datos
#
# ALGORITMO DE BÚSQUEDA: HNSW (Hierarchical Navigable Small World)
# =================================================================
# ChromaDB usa HNSW internamente. Es un algoritmo de búsqueda aproximada
# de vecinos cercanos (ANN). En vez de comparar tu query contra todos los
# documentos (O(n), lento), HNSW construye un grafo jerárquico que permite
# encontrar los más similares en O(log n). Es ~100x más rápido que fuerza
# bruta para bases de datos grandes.
#
# Respecto a COSINE vs L2
# =====================================
# ChromaDB usa distancia L2 (euclidiana) POR DEFECTO.
# Nosotros usamos embeddings normalizados (normalize_embeddings=True),
# lo que significa que necesitamos distancia COSENO, no L2.
# De no colocar {"hnsw:space": "cosine"}, los resultados serán incorrectos.
# Esto es porque:
#   - L2 mide distancia geométrica (como en un mapa)
#   - Coseno mide el ángulo entre vectores (dirección, no magnitud)
#   - Para texto, la dirección importa más que la magnitud
#
# DEPENDENCIAS:
# =============
# pip install chromadb  (instala también hnswlib internamente)
# ============================================================================

import chromadb
import gc
import time
import shutil
import os
from rag.embeddings import EmbeddingService

class VectorStore:
    """
    Base de datos vectorial para el conocimiento de Hannah.
    Almacena documentos como vectores en ChromaDB y permite buscar
    los más similares a una query dada.
    Flujo de datos:
    1. add_documents("Hannah tiene 360M params")
       → EmbeddingService convierte a vector [0.02, -0.15, ...]
       → ChromaDB guarda el vector + texto original + metadata
    2. search("¿Cuántos parámetros?")
       → EmbeddingService convierte la query a vector
       → ChromaDB busca los vectores más cercanos (HNSW)
       → Devuelve los textos originales + distancias
    Persistencia:
    Los datos se guardan en ./hannah_vectordb/ 
    Si cierras Python y lo abres de nuevo, los datos siguen ahí.
    """

    def __init__(self, db_path: str = "./hannah_vectordb"):
        """
        Inicializa la conexión a ChromaDB.
        Args:
            db_path: Carpeta donde se guardará la base de datos.
                     Si no existe, se crea automáticamente.
                     Si ya existe, carga los datos previos.
        """
        # PersistentClient = los datos sobreviven al reinicio de Python, si solo es para trabajar test usar chromadb.Clieent() que es en memoria y más rápido pero se pierden al cerrar Python
        self.client = chromadb.PersistentClient(path=db_path)

        # Servicio de embeddings compartido
        self.embedder = EmbeddingService()

        # Crear o cargar la colección "knowledge_base"
        # ─────────────────────────────────────────────
        # ChromaDB puede tener múltiples colecciones (como tablas en SQL).
        # Trabajamos con: "knowledge_base".
        # get_or_create_collection:
        #   - Si "knowledge_base" ya existe → la carga
        #   - Si no existe → la crea nueva
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"} #Importante para usar distancia coseno con embeddings normalizados
        )
        print(f"[VectorStore] Colección 'knowledge_base' lista. "
              f"Documentos actuales: {self.collection.count()}")

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """
        Añade documentos a la base de datos vectorial.
        Proceso interno:
        1. Convierte cada documento a un vector de 384 dims (via EmbeddingService)
        2. Guarda en ChromaDB: vector + texto original + metadata + id
        Args:
            documents: Lista de textos a indexar.
                       Ejemplo: ["Hannah tiene 360M params", "El RAG usa ChromaDB"]
            metadatas: Lista de diccionarios con info extra por documento.
                       Ejemplo: [{"source": "docs"}, {"source": "manual"}]
                       → Esto permite filtrar después: "dame solo docs del manual"
            ids: Lista de IDs únicos por documento.
                 Ejemplo: ["doc_001", "doc_002"]
                 → Deben ser únicos. Si repites un ID, ChromaDB da error.
                 → Tip: usa f"doc_{hash(texto)}" para IDs automáticos
        EJEMPLO COMPLETO:
            db.add_documents(
                documents=["Hannah es un modelo de 360M de parámetros"],
                metadatas=[{"source": "arquitectura.pdf", "page": 1}],
                ids=["arch_001"]
            )
        """
        # Convertir textos a vectores en batch (más rápido que uno por uno)
        embeddings = self.embedder.get_embeddings_batch(documents)

        # Insertar en ChromaDB
        # ChromaDB guarda: embedding (vector) + document (texto) + metadata + id
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[VectorStore] {len(documents)} documentos insertados. "
              f"Total en BD: {self.collection.count()}")

    def search(self, query: str, n_results: int = 3) -> dict:
        """
        Busca los documentos más similares a una query.
        Proceso interno:
        1. Convierte la query a vector (384 dims)
        2. ChromaDB busca los n_results vectores más cercanos (HNSW)
        3. Devuelve los textos originales + metadatas + distancias
        Args:
            query: Pregunta o texto de búsqueda.
                   Ejemplo: "¿Cuántos parámetros tiene Hannah?"
            n_results: Cuántos documentos devolver (los N más relevantes).
                       Para modo Simplified (Fast): usar 3
                       Para modo Extended (Slow): usar 10
        Returns:
            Diccionario con formato ChromaDB:
            {
                "documents": [["texto1", "texto2", "texto3"]],  # Lista anidada
                "metadatas": [[{meta1}, {meta2}, {meta3}]],      # Lista anidada
                "distances": [[0.15, 0.25, 0.40]],               # Distancia coseno
                "ids": [["id1", "id2", "id3"]]
            }
            Con cosine distance, los valores van de 0 a 2:
              - 0.0 = idénticos (mismo significado)
              - 1.0 = no relacionados (ortogonales)
              - 2.0 = opuestos
            Para convertir a similitud: similitud = 1 - distancia
        """
        # Convertir la query a vector
        query_embedding = self.embedder.get_embedding(query)

        # Buscar en ChromaDB los N más cercanos
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    # Añade estos métodos a tu clase VectorStore
    def get_all_documents(self, limit: int = 100, include_embeddings: bool = False):
        """
        Obtener todos los documentos almacenados usando la API oficial.
        
        Args:
            limit: Número máximo de documentos a recuperar
            include_embeddings: Si incluir los vectores (True) o no (False)
        
        Returns:
            Diccionario con documentos, metadatos e IDs
        """
        # Para .get(), solo estos include son válidos
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes.append("embeddings")
        
        # Obtener documentos de la colección
        results = self.collection.get(
            limit=limit,
            include=includes
        )
        
        return results

    def view_database_summary(self):
        """
        Mostrar un resumen de lo que hay en la base de datos vectorial.
        """
        total = self.collection.count()
        print(f"\n{'='*50}")
        print(f"  RESÚMEN DE LA BASE DE DATOS VECTORIAL")
        print(f"{'='*50}")
        print(f"Total de documentos: {total}")
        
        if total > 0:
            # Obtener algunos documentos de muestra
            muestras = self.collection.get(limit=5, include=["documents", "metadatas"])
            
            print(f"\n--- MUESTRA DE DOCUMENTOS (primeros {len(muestras['documents'])}) ---")
            for i, (doc_id, texto, metadata) in enumerate(zip(
                muestras['ids'], 
                muestras['documents'], 
                muestras['metadatas']
            )):
                print(f"\n[{i+1}] ID: {doc_id}")
                print(f"    Texto: {texto[:150]}..." if len(texto) > 150 else f"    Texto: {texto}")
                print(f"    Metadatos: {metadata}")
        
        return total
