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
# Enhances user queries before searching in ChromaDB.
# The problem is that user questions and documents in the DB
# may be written in very different ways.
#
# EXAMPLE OF THE PROBLEM:
#   User query: "How many parameters does Hannah have?"
#   Document in DB: "Hannah 360M uses the OLMo architecture from AllenAI,
#                   instantiated with reduced dimension parameters to 360M."
#   → Query says "how many parameters" but document says "reduced dimension to 360M"
#     A pure word match would fail.
# SOLUTION: Generate MULTIPLE query variants to cover more ways
# of saying the same thing and find better matches in the DB.
#
# Techniques implemented:
# =======================
# 1. QUERY EXPANSION (Multi-Query) — Extended mode only
#    Generates variants of the original question:
#    - Question → Statement: "What is NLP?" → "NLP is a field"
#    - Keyword extraction: "How do I train DPO?" → "train DPO"
#    - Contextualization: "What is NLP?" → "Hannah chatbot what is NLP"
#
# 2. HyDE (Hypothetical Document Embeddings) — Extended mode only
#    WHAT IS HyDE?
#    Instead of searching with the question, we generate a "hypothetical document"
#    that would ANSWER the question, and search with that text.
#    The embedding of a hypothetical document is closer to the embedding
#    of real documents than the embedding of a short question.
#    EXAMPLE:
#      Query: "What is NLP?"
#      HyDE generates: "Natural Language Processing (NLP) refers to...
#                       NLP is a technique that enables computer systems..."
#      → The embedding of this paragraph is closer to documents about NLP
#        than the embedding of the question "What is NLP?" (4 words)
#
# MODES OF OPERATION:
# =============================================================================
# - Simplified (Fast): Basic cleaning only. No HyDE or expansion.
#   → Reason: Fast Model (Hannah 360M) needs minimal latency (<100ms)
#
# - Extended (Slow): Query Expansion + HyDE + subsequent reranking.
#   → Reason: Slow Model (Qwen2.5-14B-Instruct) can wait for better context
#
# DEPENDENCIES:
# =============
# Only uses standard library (re). No pip install required.
# ============================================================================
import re

class QueryEnhancer:
    """
    Enhances queries according to RAG operation mode.
    Usage:
        enhancer = QueryEnhancer()
        result = enhancer.enhance("What is Hannah?", mode="simplified")
        # result["search_queries"] = ["What is Hannah"]  (only 1 cleaned query)
        result = enhancer.enhance("What is Hannah?", mode="extended")
        # result["search_queries"] = [
        #     "What is Hannah",                          (original cleaned)
        #     "Hannah is an AI model",                   (statement)
        #     "Hannah",                                  (keywords)
        #     "Hannah chatbot what is Hannah",           (contextual)
        #     "The concept of Hannah refers to..."       (HyDE)
        # ]
    """
    # ─── TEMPLATES FOR HyDE ───
    # Each template generates a "hypothetical document" based on question type.
    # {topic} is replaced with keywords from the query.
    HYDE_TEMPLATES = {
        "definition": (
            "The concept of {topic} refers to {topic} in the context of artificial "
            "intelligence and natural language processing. {topic} is a technique that "
            "enables computer systems to understand and generate human language."
        ),
        "how_to": (
            "To perform {topic}, the following steps are followed: first, prepare the data, "
            "then configure the model, train it with appropriate parameters, "
            "and finally evaluate the system's performance."
        ),
        "comparison": (
            "{topic} has several distinguishing characteristics when compared with "
            "alternatives. The advantages include better performance and efficiency, "
            "while limitations may include resource requirements."
        ),
        "factual": (
            "According to documentation and official sources, {topic}. This information "
            "has been verified and documented in the context of the Hannah project, "
            "a conversational chatbot with 360 million parameters."
        ),
        "default": (
            "{topic}. This topic is related to natural language processing "
            "and conversational language models. In the context of "
            "Hannah, this applies to improving response quality."
        )
    }

    # ─── PATTERNS FOR CLASSIFYING QUESTIONS ───
    # Regex that detect question type to choose the correct HyDE template
    QUESTION_PATTERNS = {
        "definition": r"(?i)(what is|what are|define|definition|meaning of|what's)",
        "how_to": r"(?i)(how to|how do|steps to|process of|tutorial|guide|way to)",
        "comparison": r"(?i)(difference|vs|versus|compare|better|worse|similarities|differences between|compare to)",
        "factual": r"(?i)(how many|how much|when|where|who|which|why|what time|what date)"
    }

    # ─── STOPWORDS FOR ENGLISH ───
    STOPWORDS = {
        # Question words
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        # Articles
        "a", "an", "the",
        # Prepositions
        "of", "in", "to", "for", "with", "on", "at", "by", "from", "up", "down",
        "off", "over", "under", "again", "further", "then", "once",
        # Conjunctions
        "and", "or", "but", "so", "yet", "for", "nor",
        # Verbs (common)
        "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could",
        # Pronouns
        "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us",
        "them", "my", "your", "his", "her", "its", "our", "their",
        # Other common
        "not", "no", "very", "too", "just", "but", "like", "so", "than", "then",
        "now", "only", "own", "same", "such", "than", "that", "these", "those",
        "this", "those", "through", "until"
    }

    def __init__(self):
        """Initializes QueryEnhancer. No models or connections required."""
        pass

    def enhance(self, query: str, mode: str = "simplified") -> dict:
        """
        Main entry point. Enhances the query according to mode.

        Args:
            query: Original user query.
                   Example: "What is natural language processing?"

            mode: "simplified" (for Fast Model) or "extended" (for Slow Model)

        Returns:
            dict with:
            {
                "original": "What is natural language processing?",
                "cleaned": "What is natural language processing",
                "search_queries": ["query1", "query2", ...],  # List for searching
                "hyde_doc": "hypothetical document..." | None,  # Only in extended
                "mode": "simplified" | "extended"
            }

        SIMPLIFIED MODE (Fast):
            - Only cleans the query (removes special chars, normalizes spaces)
            - Returns 1 search query
            - Latency: ~0ms (regex only)

        EXTENDED MODE (Slow):
            - Clean + generate variants + generate HyDE
            - Returns 4-5 search queries
            - Latency: ~1ms (regex and templates only, no AI)
        """
        cleaned = self._clean_query(query)

        if mode == "simplified":
            # ─── FAST MODE: Only cleaning, direct search ───
            return {
                "original": query,
                "cleaned": cleaned,
                "search_queries": [cleaned],
                "hyde_doc": None,
                "mode": "simplified"
            }

        elif mode == "extended":
            # ─── SLOW MODE: Query Expansion + HyDE ───
            # 1. Generate question variants
            expanded = self._expand_query(cleaned)

            # 2. Generate hypothetical document (HyDE)
            hyde_doc = self._generate_hyde(cleaned)

            # 3. Combine: original + expansions + HyDE
            # All these queries will be searched in ChromaDB and results
            # will be merged (deduplicated) in rag_component.py
            search_queries = [cleaned] + expanded + [hyde_doc]

            return {
                "original": query,
                "cleaned": cleaned,
                "search_queries": search_queries,
                "hyde_doc": hyde_doc,
                "mode": "extended"
            }
        else:
            raise ValueError(f"Mode '{mode}' not recognized. Use 'simplified' or 'extended'.")

    # ─────────────────────────────────────────────
    # PRIVATE METHODS (internal helpers)
    # ─────────────────────────────────────────────
    def _clean_query(self, query: str) -> str:
        """
        Basic query cleaning. Used in BOTH modes.
        - Removes multiple spaces: "hello   world" → "hello world"
        - Removes unnecessary special characters: emojis, @, #, etc.
        - Keeps: letters, numbers, question marks, basic punctuation
        """
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        # Keep alphanumeric, basic punctuation, and spaces
        query = re.sub(r'[^\w\s\?\!\.,;:\-\']', '', query)
        return query

    def _expand_query(self, query: str) -> list[str]:
        """
        Query Expansion: generates variants of the original query.
        Only used in Extended mode.
        Strategies:
        1. Reformulation: Question → Declarative statement
           "What is NLP?" → "NLP is natural language processing"
           → Documents are usually declarative, not interrogative
        2. Keywords: Only keywords without stopwords
           "How do I train the DPO model?" → "train DPO model"
           → Searches by content without grammatical noise
        3. Contextualization: Add "Hannah chatbot" at the beginning
           → Biases search toward project domain documents
        """
        variants = []

        # Variant 1: Question → Statement
        declarative = self._question_to_statement(query)
        if declarative and declarative != query:
            variants.append(declarative)

        # Variant 2: Only keywords
        keywords = self._extract_keywords(query)
        if keywords and keywords != query.lower():
            variants.append(keywords)

        # Variant 3: Contextualized to project
        contextual = f"Hannah chatbot {query}"
        variants.append(contextual)

        return variants

    def _generate_hyde(self, query: str) -> str:
        """
        HyDE: Generates a hypothetical document based on templates.
        Process:
        1. Classifies question type (definition, how_to, comparison, factual)
        2. Extracts main topic from query
        3. Fills corresponding template with the topic
        In a full implementation, an LLM (like Qwen2.5-14B or even Hannah)
        would be used here to generate the document.
        Our template approach is an MVP that works without an LLM.
        """
        question_type = self._classify_question(query)
        topic = self._extract_topic(query)
        template = self.HYDE_TEMPLATES.get(question_type, self.HYDE_TEMPLATES["default"])
        hyde_doc = template.format(topic=topic)
        return hyde_doc

    def _classify_question(self, query: str) -> str:
        """
        Classifies question type using regex.
        Returns: "definition", "how_to", "comparison", "factual", or "default"
        """
        for qtype, pattern in self.QUESTION_PATTERNS.items():
            if re.search(pattern, query):
                return qtype
        return "default"

    def _extract_topic(self, query: str) -> str:
        """
        Extracts main topic by removing question words and stopwords.
        "What is natural language processing?" → "natural language processing"
        """
        # Remove question marks and normalize
        cleaned = query.replace("?", "").replace("!", "")
        
        # Split into words
        words = cleaned.lower().split()
        
        # Remove stopwords (including question words)
        topic_words = [w for w in words if w not in self.STOPWORDS]
        
        # If nothing left, use first few words or original
        if not topic_words:
            topic_words = words[:3] if words else [cleaned]
        
        return " ".join(topic_words)

    def _question_to_statement(self, query: str) -> str:
        """
        Converts question to declarative statement.
        "What is NLP?" → "NLP is natural language processing"
        "How does DPO work?" → "DPO works by..."
        """
        # Remove question marks
        statement = query.replace("?", "").replace("!", "").strip()
        
        # Common question patterns and their statement conversions
        patterns = [
            (r"(?i)^what is\s+(.+)$", r"\1 is a concept related to"),
            (r"(?i)^what are\s+(.+)$", r"\1 are concepts related to"),
            (r"(?i)^how does\s+(.+?)\s+work$", r"\1 works by following specific processes"),
            (r"(?i)^how do\s+(.+)$", r"to \1, one should follow established procedures"),
            (r"(?i)^why does\s+(.+)$", r"\1 occurs due to underlying mechanisms"),
            (r"(?i)^when does\s+(.+)$", r"\1 happens under specific conditions"),
            (r"(?i)^what('s| is)\s+(.+)$", r"\2 is a term that refers to"),
        ]
        
        for pattern, replacement in patterns:
            match = re.match(pattern, statement)
            if match:
                topic = match.group(1) if len(match.groups()) == 1 else match.group(2)
                return replacement.format(topic)
        
        # If no pattern matches, just capitalize first letter
        if statement and statement[0].islower():
            statement = statement[0].upper() + statement[1:]
        
        return statement if statement != query else ""

    def _extract_keywords(self, query: str) -> str:
        """
        Extracts only meaningful keywords (no stopwords).
        "How does DPO training work in Hannah?" → "DPO training work Hannah"
        """
        # Remove punctuation and question marks
        cleaned = re.sub(r'[^\w\s]', '', query.lower())
        words = cleaned.split()
        
        # Filter out stopwords and short words (len <= 2)
        keywords = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        
        return " ".join(keywords)