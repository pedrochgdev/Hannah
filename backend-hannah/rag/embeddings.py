# embeddings.py
# ============================================================================
# MÓDULO 1 DE 6: Servicio de Embeddings
# ============================================================================
# Archivo: rag_standalone/embeddings.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
#
# Descripción:
# ========================
# Convierte texto humano en vectores numéricos (listas de 384 números decimales).
# Esto es necesario porque las computadoras no entienden palabras, pero sí
# entienden números. Al convertir texto a vectores, podemos medir qué tan
# "parecidos" son dos textos usando matemáticas (similitud coseno).
#
# EJEMPLO VISUAL:
#   "Hannah es una IA"  →  [0.023, -0.15, 0.87, ..., 0.42]  (384 números)
#   "Hannah es un bot"  →  [0.021, -0.14, 0.85, ..., 0.41]  (384 números)
#   "El cielo es azul"  →  [0.91, 0.32, -0.05, ..., -0.23]  (384 números)
#
#   Los vectores de las primeras dos frases son MUY parecidos (similitud ~0.95)
#   El tercer vector es MUY diferente (similitud ~0.12)
#   → Así ChromaDB sabe que "Hannah es una IA" y "Hannah es un bot" hablan
#     de lo mismo, pero "El cielo es azul" es un tema distinto.
#
# MODELO USADO:
# =============
# all-MiniLM-L6-v2 (de Sentence Transformers / HuggingFace)
# - Tamaño: ~80MB (liviano, se descarga la primera vez que corres el código)
# - Dimensiones: 384 (cada texto se convierte en 384 números)
# - Velocidad: ~14,000 textos/segundo en CPU
# - Calidad: Top-tier para su tamaño. Entrenado con 1B+ pares de oraciones
#
# ¿Por que normalize_embeddings=True?
# =====================================
# Normalizar significa que todos los vectores tienen longitud (norma) = 1.
# Cuando los vectores están normalizados:
#   - El DOT PRODUCT (multiplicación punto) = SIMILITUD COSENO
#   - Esto es importante porque ChromaDB usa cosine distance
#   - Sin normalización, el dot product no equivale a coseno y los resultados
#     de búsqueda serían menos precisos
#
# DEPENDENCIAS:
# =============
# pip install sentence-transformers  (instala también torch y transformers)
# ============================================================================

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Servicio de embeddings para el RAG de Hannah.
    Convierte texto en vectores numéricos de 384 dimensiones usando
    el modelo all-MiniLM-L6-v2 de Sentence Transformers.
    Este módulo es usado por:
    - VectorStore: para convertir documentos antes de guardarlos en ChromaDB
    - VectorStore: para convertir queries antes de buscar en ChromaDB
    - SemanticCache: para comparar queries por similitud
    - ContextHandler: para reranking de chunks (modo Extended)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo de embeddings.
        Args:
            model_name: Nombre del modelo en HuggingFace. Default "all-MiniLM-L6-v2".
                        La primera vez que se ejecuta, descarga ~80MB.
                        Después usa la versión cacheada en ~/.cache/torch/
        """
        self.model = SentenceTransformer(model_name, device="cpu")
        print(f"[EmbeddingService] Modelo '{model_name}' cargado en CPU")

    def get_embedding(self, text: str) -> list[float]:
        """
        Convierte un texto en un vector de 384 dimensiones.
        Args:
            text: Cualquier texto en español o inglés.
                  Ejemplo: "¿Cuántos parámetros tiene Hannah?"
        Returns:
            Lista de 384 floats. Ejemplo: [0.023, -0.15, 0.87, ..., 0.42]
        DETALLES TÉCNICOS:
        - normalize_embeddings=True → vector con norma 1 → dot product = coseno
        - .tolist() convierte de numpy array a lista de Python (ChromaDB lo necesita)
        - Tiempo: ~1ms por texto en CPU (muy rápido)
        """
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convierte muchos textos en vectores de una sola vez (batch processing).
        Es mucho más rápido que llamar get_embedding() uno por uno porque el
        modelo procesa varios textos en paralelo internamente.
        Args:
            texts: Lista de textos. Ejemplo: ["texto1", "texto2", "texto3"]
        Returns:
            Lista de listas de 384 floats. Una lista por cada texto.
        PARÁMETROS EXPLICADOS:
        - normalize_embeddings=True: misma razón que en get_embedding()
        - show_progress_bar=True: muestra barra de progreso en consola
          (útil cuando procesas cientos de documentos para ver que no se colgó)
        - batch_size=32: procesa 32 textos a la vez internamente.
          → Si tienes poca RAM (<8GB): bajar a 16
          → Si tienes mucha RAM (>32GB): subir a 64 o 128
          → 32 es un buen balance para la mayoría de computadoras
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        ).tolist()
