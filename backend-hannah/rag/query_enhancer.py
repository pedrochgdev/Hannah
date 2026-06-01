# query_enhancer.py
# ============================================================================
# MÓDULO 4 DE 6: Mejora de Queries (QueryEnhancer)
# ============================================================================
# Archivo: rag_standalone/query_enhancer.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui (Arquitecto de Solución)
# ============================================================================
# Descripción:
# ========================
# Mejora las queries del usuario antes de buscar en ChromaDB.
# El problema es que la pregunta del usuario y los documentos en la BD
# pueden estar escritos de formas muy diferentes.
#
# EJEMPLO DEL PROBLEMA:
#   Query del usuario: "¿Cuántos parámetros tiene Hannah?"
#   Documento en BD:   "Hannah 360M utiliza la arquitectura OLMo de AllenAI,
#                       instanciada con parámetros de dimensión reducidos a 360M."
#   → La query dice "cuántos parámetros" pero el documento dice "dimensión
#     reducidos a 360M". Un match puramente por palabras fallaría.
# SOLUCIÓN: Generar MÚLTIPLES variantes de la query para cubrir más
# formas de decir lo mismo y encontrar mejores matches en la BD.
#
# Técnicas implementadas:
# =======================
# 1. QUERY EXPANSION (Multi-Query) — Solo modo Extended
#    Genera variantes de la pregunta original:
#    - Pregunta → Afirmación: "¿Qué es NLP?" → "NLP"
#    - Extracción de keywords: "¿Cómo entreno el DPO?" → "entrenar DPO"
#    - Contextualización: "¿Qué es NLP?" → "Hannah chatbot ¿Qué es NLP?"
#
# 2. HyDE (Hypothetical Document Embeddings) — Solo modo Extended
#    ¿QUÉ ES HyDE?
#    En vez de buscar con la pregunta, generamos un "documento hipotético"
#    que RESPONDERÍA la pregunta, y buscamos con ese texto.
#    El embedding de un documento hipotético está más cerca del embedding
#    de los documentos reales que el embedding de una pregunta corta.
#    EJEMPLO:
#      Query: "¿Qué es NLP?"
#      HyDE genera: "El procesamiento de lenguaje natural (NLP) se refiere
#                    a NLP en el contexto de IA. NLP es una técnica que
#                    permite a los sistemas computacionales..."
#      → El embedding de este párrafo está más cerca de documentos sobre NLP
#        que el embedding de la pregunta "¿Qué es NLP?" (4 palabras)
#    NUESTRA IMPLEMENTACIÓN:
#    HyDE original usa un LLM para generar el documento hipotético.
#    Nosotros usamos TEMPLATES porque:
#    a) Nuestro RAG es standalone (no hay LLM corriendo)
#    b) Los templates capturan ~70% del beneficio sin costo computacional
#    c) Se pueden mejorar después cuando se integre con Hannah/Qwen
#
# MODOS DE OPERACIÓN:
# =============================================================================
# - Simplified (Fast): Solo limpieza básica. Sin HyDE ni expansión.
#   → Razón: el Fast Model (Hannah 360M) necesita latencia mínima (<100ms)
#
# - Extended (Slow): Query Expansion + HyDE + reranking posterior.
#   → Razón: el Slow Model (Qwen2.5-14B-Instruct) puede esperar por mejor contexto
#
# DEPENDENCIAS:
# =============
# Solo usa la librería estándar de Python (re). No requiere pip install.
# ============================================================================
import re

class QueryEnhancer:
    """
    Mejora queries según el modo de operación del RAG.
    Uso:
        enhancer = QueryEnhancer()
        result = enhancer.enhance("¿Qué es Hannah?", mode="simplified")
        # result["search_queries"] = ["Qué es Hannah"]  (solo 1 query limpia)
        result = enhancer.enhance("¿Qué es Hannah?", mode="extended")
        # result["search_queries"] = [
        #     "Qué es Hannah",                          (original limpia)
        #     "Hannah",                                  (afirmación)
        #     "hannah",                                  (keywords)
        #     "Hannah chatbot Qué es Hannah",            (contextual)
        #     "El concepto de hannah se refiere a..."    (HyDE)
        # ]
    """
    # ─── TEMPLATES PARA HyDE ───
    # Cada template genera un "documento hipotético" según el tipo de pregunta.
    # El {topic} se reemplaza con las palabras clave de la query.
    HYDE_TEMPLATES = {
        "definition": (
            "El concepto de {topic} se refiere a {topic} en el contexto de inteligencia "
            "artificial y procesamiento de lenguaje natural. {topic} es una técnica que "
            "permite a los sistemas computacionales comprender y generar lenguaje humano."
        ),
        "how_to": (
            "Para {topic}, se siguen los siguientes pasos: primero se preparan los datos, "
            "luego se configura el modelo, se entrena con los parámetros adecuados, "
            "y finalmente se evalúa el rendimiento del sistema."
        ),
        "comparison": (
            "{topic} tiene varias características distintivas cuando se compara con "
            "alternativas. Las ventajas incluyen mejor rendimiento y eficiencia, "
            "mientras que las limitaciones pueden incluir requisitos de recursos."
        ),
        "factual": (
            "Según la documentación y fuentes oficiales, {topic}. Esta información "
            "ha sido verificada y documentada en el contexto del proyecto Hannah, "
            "un chatbot conversacional de 360 millones de parámetros."
        ),
        "default": (
            "{topic}. Este tema está relacionado con el procesamiento de lenguaje "
            "natural y los modelos de lenguaje conversacionales. En el contexto de "
            "Hannah, esto se aplica para mejorar la calidad de las respuestas."
        )
    }

    # ─── PATRONES PARA CLASIFICAR PREGUNTAS ───
    # Regex que detectan el tipo de pregunta para elegir el template HyDE correcto
    QUESTION_PATTERNS = {
        "definition": r"(?i)(qué es|qué son|define|definición|what is|what are|significa)",
        "how_to": r"(?i)(cómo|how to|pasos|steps|proceso|tutorial|guide)",
        "comparison": r"(?i)(diferencia|vs|versus|comparar|mejor|peor|compare|difference)",
        "factual": r"(?i)(cuánto|cuándo|dónde|quién|how many|when|where|who)"
    }

    def __init__(self):
        """Inicializa el QueryEnhancer. No requiere modelos ni conexiones."""
        pass

    def enhance(self, query: str, mode: str = "simplified") -> dict:
        """
        Punto de entrada principal. Mejora la query según el modo.

        Args:
            query: Query original del usuario.
                   Ejemplo: "¿Qué es el procesamiento de lenguaje natural?"

            mode: "simplified" (para Fast Model) o "extended" (para Slow Model)

        Returns:
            dict con:
            {
                "original": "¿Qué es el procesamiento de lenguaje natural?",
                "cleaned": "Qué es el procesamiento de lenguaje natural",
                "search_queries": ["query1", "query2", ...],  # Lista para buscar
                "hyde_doc": "documento hipotético..." | None,  # Solo en extended
                "mode": "simplified" | "extended"
            }

        MODO SIMPLIFIED (Fast):
            - Solo limpia la query (quita caracteres raros, normaliza espacios)
            - Devuelve 1 sola query de búsqueda
            - Latencia: ~0ms (solo regex)

        MODO EXTENDED (Slow):
            - Limpia + genera variantes + genera HyDE
            - Devuelve 4-5 queries de búsqueda
            - Latencia: ~1ms (solo regex y templates, no hay IA)
        """
        cleaned = self._clean_query(query)

        if mode == "simplified":
            # ─── MODO FAST: Solo limpieza, búsqueda directa ───
            return {
                "original": query,
                "cleaned": cleaned,
                "search_queries": [cleaned],
                "hyde_doc": None,
                "mode": "simplified"
            }

        elif mode == "extended":
            # ─── MODO SLOW: Query Expansion + HyDE ───
            # 1. Generar variantes de la pregunta
            expanded = self._expand_query(cleaned)

            # 2. Generar documento hipotético (HyDE)
            hyde_doc = self._generate_hyde(cleaned)

            # 3. Combinar: original + expansiones + HyDE
            # Todas estas queries se buscarán en ChromaDB y los resultados
            # se fusionan (merge) eliminando duplicados en rag_component.py
            search_queries = [cleaned] + expanded + [hyde_doc]

            return {
                "original": query,
                "cleaned": cleaned,
                "search_queries": search_queries,
                "hyde_doc": hyde_doc,
                "mode": "extended"
            }
        else:
            raise ValueError(f"Modo '{mode}' no reconocido. Usar 'simplified' o 'extended'.")

    # ─────────────────────────────────────────────
    # MÉTODOS PRIVADOS (helpers internos)
    # ─────────────────────────────────────────────
    def _clean_query(self, query: str) -> str:
        """
        Limpieza básica de la query. Usado en AMBOS modos.
        - Elimina espacios múltiples: "hola   mundo" → "hola mundo"
        - Elimina caracteres especiales innecesarios: emojis, @, #, etc.
        - Mantiene: letras, números, signos de pregunta, acentos, puntuación básica
        """
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[^\w\s¿?¡!áéíóúñÁÉÍÓÚÑ.,;:\-]', '', query)
        return query
    def _expand_query(self, query: str) -> list[str]:
        """
        Query Expansion: genera variantes de la query original.
        Solo se usa en modo Extended.
        Estrategias:
        1. Reformulación: Pregunta → Afirmación declarativa
           "¿Qué es NLP?" → "NLP"
           → Los documentos suelen ser afirmativos, no interrogativos
        2. Keywords: Solo palabras clave sin stopwords
           "¿Cómo entreno el modelo DPO?" → "entrenar modelo DPO"
           → Busca por contenido sin ruido gramatical
        3. Contextualización: Añadir "Hannah chatbot" al inicio
           → Sesga la búsqueda hacia documentos del dominio del proyecto
        """
        variants = []

        # Variante 1: Pregunta → Afirmación
        declarative = self._question_to_statement(query)
        if declarative != query:
            variants.append(declarative)

        # Variante 2: Solo keywords
        keywords = self._extract_keywords(query)
        if keywords:
            variants.append(keywords)

        # Variante 3: Contextualizada al proyecto
        contextual = f"Hannah chatbot {query}"
        variants.append(contextual)

        return variants

    def _generate_hyde(self, query: str) -> str:
        """
        HyDE: Genera un documento hipotético basado en templates.
        Proceso:
        1. Clasifica el tipo de pregunta (definición, how-to, comparación, factual)
        2. Extrae el tema principal de la query
        3. Rellena el template correspondiente con el tema
        En una implementación completa, aquí se usaría un LLM
        (como Qwen2.5-14B o incluso Hannah) para generar el documento.
        Nuestro approach con templates es un MVP que funciona sin LLM.
        """
        question_type = self._classify_question(query)
        topic = self._extract_topic(query)
        template = self.HYDE_TEMPLATES.get(question_type, self.HYDE_TEMPLATES["default"])
        hyde_doc = template.format(topic=topic)
        return hyde_doc

    def _classify_question(self, query: str) -> str:
        """
        Clasifica el tipo de pregunta usando regex.
        Retorna: "definition", "how_to", "comparison", "factual", o "default"
        """
        for qtype, pattern in self.QUESTION_PATTERNS.items():
            if re.search(pattern, query):
                return qtype
        return "default"

    def _extract_topic(self, query: str) -> str:
        """
        Extrae el tema principal eliminando palabras interrogativas y stopwords.
        "¿Qué es el procesamiento de lenguaje natural?" → "procesamiento lenguaje natural"
        """
        stopwords = [
            # Español
            "qué", "que", "cómo", "como", "cuál", "cual", "cuánto", "cuanto",
            "cuándo", "cuando", "dónde", "donde", "quién", "quien", "por qué",
            "es", "son", "está", "están", "tiene", "tienen", "puede", "pueden",
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "de", "del", "en", "con", "para", "por", "a", "al",
            # Inglés
            "what", "how", "when", "where", "who", "which", "is", "are", "the",
            # Pronombres
            "me", "se", "le", "lo", "nos"
        ]
        words = query.lower().replace("¿", "").replace("?", "").split()
        topic_words = [w for w in words if w not in stopwords]
        return " ".join(topic_words) if topic_words else query

    def _question_to_statement(self, query: str) -> str:
        """
        Convierte pregunta a afirmación eliminando signos y palabras interrogativas.
        "¿Qué es NLP?" → "NLP"
        "¿Cómo funciona el DPO?" → "funciona el DPO"
        """
        statement = query.replace("¿", "").replace("?", "").strip()
        prefixes = [
            "qué es ", "qué son ", "cómo ", "cuál es ", "cuánto ",
            "cuándo ", "dónde ", "quién ", "por qué "
        ]
        lower = statement.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                statement = statement[len(prefix):]
                break
        return statement.strip()

    def _extract_keywords(self, query: str) -> str:
        """
        Extrae solo palabras significativas (no stopwords).
        "¿Cómo funciona el entrenamiento DPO en Hannah?" → "funciona entrenamiento DPO Hannah"
        """
        stopwords = {
            "qué", "que", "cómo", "como", "cuál", "cual", "es", "son",
            "el", "la", "los", "las", "un", "una", "de", "del", "en",
            "con", "para", "por", "a", "al", "y", "o", "pero", "si",
            "no", "más", "menos", "muy", "se", "le", "lo", "me", "te",
            "nos", "su", "sus", "mi", "tu", "está", "están", "tiene",
            "tienen", "puede", "pueden", "hay"
        }
        words = re.findall(r'\w+', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(keywords)


# ============================================================================
# PRUEBA RÁPIDA
# ============================================================================
# Ejecutar: python query_enhancer.py
# Resultado esperado:
#   - Simplified: 1 query de búsqueda (solo limpieza)
#   - Extended: 4-5 queries (original + expansiones + HyDE)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Test: QueryEnhancer")
    print("=" * 60)

    enhancer = QueryEnhancer()

    # Test 1: Modo Simplified (Fast)
    print("\n--- Test 1: MODO SIMPLIFIED (FAST) ---")
    result = enhancer.enhance("¿Qué es el procesamiento de lenguaje natural?", mode="simplified")
    print(f"  Original:       {result['original']}")
    print(f"  Cleaned:        {result['cleaned']}")
    print(f"  Search queries: {result['search_queries']}")
    print(f"  HyDE:           {result['hyde_doc']}")
    assert len(result['search_queries']) == 1, "Simplified debe tener 1 query"

    # Test 2: Modo Extended (Slow)
    print("\n--- Test 2: MODO EXTENDED (SLOW) ---")
    result2 = enhancer.enhance("¿Cómo funciona el entrenamiento DPO en Hannah?", mode="extended")
    print(f"  Original:       {result2['original']}")
    print(f"  Cleaned:        {result2['cleaned']}")
    print(f"  Search queries ({len(result2['search_queries'])}):")
    for i, q in enumerate(result2['search_queries']):
        label = ["original", "declarativa", "keywords", "contextual", "HyDE"][i] if i < 5 else f"extra_{i}"
        print(f"    [{i}] ({label}) {q[:80]}{'...' if len(q) > 80 else ''}")
    print(f"  HyDE doc:       {result2['hyde_doc'][:100]}...")
    assert len(result2['search_queries']) >= 3, "Extended debe tener múltiples queries"
    assert result2['hyde_doc'] is not None, "Extended debe tener HyDE"

    # Test 3: Clasificación de preguntas
    print("\n--- Test 3: Clasificación de preguntas ---")
    tests = [
        ("¿Qué es NLP?", "definition"),
        ("¿Cómo entreno el modelo?", "how_to"),
        ("¿Cuál es la diferencia entre SFT y DPO?", "comparison"),
        ("¿Cuántos parámetros tiene?", "factual"),
    ]
    for query, expected in tests:
        got = enhancer._classify_question(query)
        status = "✓" if got == expected else "✗"
        print(f"  {status} '{query}' → {got} (esperado: {expected})")

    print("\nTodos los tests completados.")
