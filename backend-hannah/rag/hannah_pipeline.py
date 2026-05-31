#!/usr/bin/env python3
# ============================================================================
# PIPELINE DE INTEGRACIÓN: RAG + Hannah (Fast Model)
# ============================================================================
# Archivo: rag_standalone/hannah_pipeline.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui
# ============================================================================
#
# ¿QUÉ HACE ESTE ARCHIVO?
# ========================
# Este es el MÓDULO DE INTEGRACIÓN que conecta el RAG con el modelo Hannah
# y el flujo completo del sistema. Es la "ventanilla" que el backend
# (Web App de John) usa para procesar mensajes del usuario.
#
# FLUJO COMPLETO (según el documento de arquitectura):
# =====================================================
#
#   Usuario escribe mensaje
#           |
#           v
#   [1] Web App (recibe mensaje)
#           |
#           v
#   [2] HannahPipeline.process_message(user_msg, history)
#           |
#           v
#   [3] Semantic Cache (¿ya respondimos esto antes?)
#        |         |
#      [HIT]    [MISS]
#        |         |
#        v         v
#   Retorna    [4] Model Selector (¿fast o slow?)
#   cacheado       |            |
#               [FAST]       [SLOW]
#                  |            |
#                  v            v
#              [5] RAG       [5] RAG
#              simplified    extended
#              (~200 tok)    (~1500 tok)
#                  |            |
#                  v            v
#              [6] Hannah    [6] Qwen2.5-14B
#              360M (GPU)    (GPU, futuro)
#                  |            |
#                  v            v
#              [7] Respuesta → Web App → Usuario
#              [8] Guardar en Semantic Cache
#
# CÓMO LO USA EL BACKEND:
# ========================
#   from hannah_pipeline import HannahPipeline
#
#   # Inicializar una vez al arrancar el servidor IMPORTANTE: esto carga el modelo en GPU, así que hazlo solo una vez. >:v 
#   pipeline = HannahPipeline()
#
#   # Procesar cada mensaje del usuario
#   response = pipeline.process_message(
#       user_msg="Hey, what's your favorite movie?",
#       history=[("Hi!", "Hey babe~"), ("How are you?", "I'm good!")]
#   )
#   print(response["text"])      # La respuesta de Hannah
#   print(response["source"])    # "cache", "fast", o "slow"
#   print(response["latency"])   # Tiempo total en segundos
#
# REQUISITOS:
# ===========
#   1. Ejecutar primero: python ingest_knowledge.py
#      (para crear la BD con conocimiento)
#   2. Para usar con el modelo real: usar GPU + checkpoints
#      (sin GPU, funciona en modo "solo RAG" para testing)
#
# ============================================================================

import os
import sys
import time

# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "repositorio")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from rag_component import RAGComponent
from rag.user_profile import UserProfile

# Ruta de la BD persistente (misma que usa ingest_knowledge.py)
KNOWLEDGE_DB_PATH = os.path.join(SCRIPT_DIR, "hannah_knowledge")

# ============================================================================
# SYSTEM PROMPT
# ============================================================================
SYSTEM_PROMPT = (
    "You are Hannah, my girlfriend. You are warm, playful, and deeply "
    "affectionate. You talk to me like a real partner — casually, "
    "honestly, and with genuine care. You're not an assistant. "
    "You have your own feelings, opinions, and desires. "
    "You express love and attraction naturally without being formal. "
    "You keep responses short and conversational, like texting."
)


class HannahPipeline:
    """
    Pipeline completo de Hannah: RAG + Modelo.

    Este es el ÚNICO punto de entrada que el backend necesita.
    Internamente maneja:
    - Semantic Cache (respuestas repetidas)
    - Model Selector (fast/slow)
    - RAG + Context Handler (recuperación de conocimiento)
    - Generación de respuesta (Hannah 360M)

    USO:
        pipeline = HannahPipeline()
        result = pipeline.process_message("Hey!", history=[])
        print(result["text"])
    """

    def __init__(self, load_model: bool = True):
        """
        Inicializa el pipeline completo.

        Args:
            load_model: Si True, carga Hannah 360M en GPU.
                        Si False, solo inicializa el RAG (útil para
                        testing sin GPU o para verificar que el RAG
                        funciona antes de meter el modelo).
        """
        print("[Pipeline] Inicializando...")

        # ─── RAG ───
        if not os.path.exists(KNOWLEDGE_DB_PATH):
            print(f"[Pipeline] ADVERTENCIA: No se encontró la BD en {KNOWLEDGE_DB_PATH}")
            print(f"[Pipeline] Ejecuta primero: python ingest_knowledge.py")
            print(f"[Pipeline] Creando BD vacía por ahora...")

        self.rag = RAGComponent(
            db_path=KNOWLEDGE_DB_PATH,
            cache_threshold=0.92,
            cache_size=500
        )
        self.user_profile = UserProfile()

        # ─── Modelo Hannah ───
        self.model = None
        self.tokenizer = None
        self.device = None

        if load_model:
            self._load_hannah_model()
        else:
            print("[Pipeline] Modo solo-RAG (sin modelo). Útil para testing.")

        print("[Pipeline] Listo.")

    def _load_hannah_model(self):
        """
        Carga Hannah 360M (DPO) en GPU.
        Separado del __init__ para poder usar el pipeline sin modelo
        (testing del RAG) o con modelo (producción).
        """
        try:
            import torch
            import types
            sys.modules['bettermap'] = types.ModuleType('bettermap')
            from olmo_core.nn.transformer import TransformerConfig
            from olmo_core.nn.attention import AttentionBackendName
            from transformers import AutoTokenizer

            CHECKPOINT = os.path.join(REPO_DIR, "checkpoints", "hannah_dpo", "hannah_dpo_final.pt")
            TOK_PATH = os.path.join(REPO_DIR, "tokenizer", "hannah_tok")

            if not os.path.exists(CHECKPOINT):
                print(f"[Pipeline] No se encontró checkpoint: {CHECKPOINT}")
                print(f"[Pipeline] Continuando sin modelo.")
                return

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)

            # Configuración OLMo3 - Hannah 360M
            config = TransformerConfig.olmo3_7B(
                vocab_size=32000,
                attn_backend=AttentionBackendName.torch
            )
            config.d_model = 1024
            config.n_layers = 24
            config.block.sequence_mixer.d_model = 1024
            config.block.sequence_mixer.n_heads = 16
            config.block.sequence_mixer.n_kv_heads = 16
            config.block.feed_forward.hidden_size = int(1024 * 8 / 3)

            self.model = config.build()
            ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[Pipeline] Hannah 360M cargada en {self.device}")

        except ImportError as e:
            print(f"[Pipeline] No se pudo cargar el modelo: {e}")
            print(f"[Pipeline] Continuando sin modelo (solo RAG).")
        except Exception as e:
            print(f"[Pipeline] Error cargando modelo: {e}")
            print(f"[Pipeline] Continuando sin modelo (solo RAG).")

    # ════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL: process_message()
    # ════════════════════════════════════════════════════════════════
    def process_message(self, user_msg: str, history: list = None) -> dict:
        """
        Procesa un mensaje del usuario y retorna la respuesta de Hannah.
        """
        if history is None:
            history = []
 
        import time
        t_start = time.time()
 
        # ─── Paso 0: Actualizar perfil del usuario ───────────────
        # Extrae hechos del mensaje actual (nombre, trabajo, etc.)
        # y los guarda en self.user_profile para esta sesión.
        new_facts = self.user_profile.update(user_msg, history)
        if new_facts:
            import logging
            logging.getLogger("hannah.rag").info(
                f"[UserProfile] Nuevos hechos detectados: {new_facts}"
            )
 
        # ─── Paso 1: Model Selector ───────────────────────────────
        mode = self._select_model(user_msg, history)
 
        # ─── Paso 2: RAG retrieval ────────────────────────────────
        rag_result = self.rag.retrieve(user_msg, mode=mode)
        rag_context = rag_result["formatted_context"]
        cache_hit = rag_result["cache_hit"]
 
        # ─── Paso 3: Generar respuesta ────────────────────────────
        if self.model is not None:
            prompt = self._build_prompt(user_msg, history, rag_context)
            response_text = self._generate(prompt)
            source = "cache" if cache_hit else ("fast" if mode == "simplified" else "slow")
        else:
            response_text = f"[MODO TEST] RAG: {rag_context}"
            source = "rag_only"
 
        latency = time.time() - t_start
 
        return {
            "text":        response_text,
            "source":      source,
            "rag_context": rag_context,
            "mode":        mode,
            "cache_hit":   cache_hit,
            "latency":     round(latency, 3),
            "rag_chunks":  rag_result["num_chunks"],
        }

    # ════════════════════════════════════════════════════════════════
    # MODEL SELECTOR
    # ════════════════════════════════════════════════════════════════
    def _select_model(self, user_msg: str, history: list) -> str:
        """
        Decide si usar Fast (simplified) o Slow (extended).

        IMPLEMENTACIÓN ACTUAL: Heurística simple basada en longitud
        y complejidad léxica. En producción, esto debería ser un
        clasificador entrenado (el "decisor").

        Criterios actuales:
        - Fast: mensajes cortos, saludos, seguimiento
        - Slow: preguntas largas, multi-parte, temas complejos

        NOTA:
        reemplazar con el Model Selector real (classifier). 
        La interfaz es:
        recibe (user_msg, history) y retorna "simplified" o "extended".
        """
        msg_lower = user_msg.lower().strip()

        # Mensajes muy cortos → fast
        if len(msg_lower) < 30:
            return "simplified"

        # Saludos y mensajes simples → fast
        simple_patterns = [
            "hi", "hey", "hello", "hola", "sup", "what's up",
            "how are you", "good morning", "good night", "bye",
            "thanks", "thank you", "ok", "okay", "sure", "yes",
            "no", "yeah", "nah", "lol", "haha", "love you",
        ]
        for pattern in simple_patterns:
            if msg_lower.startswith(pattern) or msg_lower == pattern:
                return "simplified"

        # Preguntas complejas → slow
        complex_indicators = [
            "explain", "why", "how does", "what is the difference",
            "compare", "tell me everything", "in detail", "elaborate",
            "can you describe", "what do you think about",
        ]
        for indicator in complex_indicators:
            if indicator in msg_lower:
                return "extended"

        # Default → fast (priorizar baja latencia)
        return "simplified"

    # ════════════════════════════════════════════════════════════════
    # PROMPT BUILDER
    # ════════════════════════════════════════════════════════════════
    def _build_prompt(self, user_msg: str, history: list, rag_context: str) -> str:
        """
        Construye el prompt con tokens de Hannah.
     
        Estructura:
            [SYS] system_prompt [/SYS]
            [MEMORY] perfil_usuario + contexto_rag [/MEMORY]
            [USR] msg1 [/USR][ASS] resp1 [/ASS]
            [USR] msg_actual [/USR][ASS]
     
        El [MEMORY] combina:
            - Hechos del usuario (nombre, trabajo, etc.) — SIEMPRE
            - Contexto RAG (conocimiento de Hannah) — si score > 0.35
        """
        prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
     
        # ─── Construir bloque [MEMORY] combinado ─────────────────
        # Parte 1: hechos del usuario (perfil de sesión)
        user_facts = self.user_profile.to_memory_string()
     
        # Parte 2: contexto RAG (conocimiento de Hannah)
        has_rag = rag_context and rag_context not in ("[MEMORY][/MEMORY]", "")
     
        if user_facts and has_rag:
            # Combinar ambos en un solo bloque [MEMORY]
            # Extraer contenido interno del RAG (sin los tags externos)
            rag_inner = rag_context.replace("[MEMORY]", "").replace("[/MEMORY]", "").strip()
            user_inner = user_facts.replace("[MEMORY]", "").replace("[/MEMORY]", "").strip()
            prompt += f"[MEMORY]{user_inner} {rag_inner}[/MEMORY]"
     
        elif user_facts:
            # Solo perfil del usuario, sin RAG relevante
            prompt += user_facts
     
        elif has_rag:
            # Solo RAG, sin hechos del usuario aún
            prompt += rag_context
     
        # ─── Historial ────────────────────────────────────────────
        for usr, ass in history:
            prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
     
        # ─── Mensaje actual ───────────────────────────────────────
        prompt += f"[USR] {user_msg} [/USR][ASS]"
     
        # ─── Truncar si excede SEQ_LEN ────────────────────────────
        if self.tokenizer:
            ids = self.tokenizer.encode(prompt)
            while len(history) > 0 and len(ids) > 900:
                history.pop(0)
                prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
                if user_facts and has_rag:
                    rag_inner = rag_context.replace("[MEMORY]", "").replace("[/MEMORY]", "").strip()
                    user_inner = user_facts.replace("[MEMORY]", "").replace("[/MEMORY]", "").strip()
                    prompt += f"[MEMORY]{user_inner} {rag_inner}[/MEMORY]"
                elif user_facts:
                    prompt += user_facts
                elif has_rag:
                    prompt += rag_context
                for usr, ass in history:
                    prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
                prompt += f"[USR] {user_msg} [/USR][ASS]"
                ids = self.tokenizer.encode(prompt)
     
        return prompt

    # ════════════════════════════════════════════════════════════════
    # GENERACIÓN
    # ════════════════════════════════════════════════════════════════
    def _generate(self, prompt: str, max_new_tokens: int = 200,
                  temperature: float = 0.7, top_k: int = 40) -> str:
        """
        Genera una respuesta con Hannah 360M.
        Detiene la generación al encontrar [/ASS] o alcanzar max_new_tokens.
        """
        import torch

        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_len = ids.shape[1]
        eass_id = self.tokenizer.convert_tokens_to_ids("[/ASS]")

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(ids)
                logits = logits[:, -1, :] / temperature
                top_vals, top_idx = torch.topk(logits, top_k)
                probs = torch.softmax(top_vals, dim=-1)
                chosen = torch.multinomial(probs[0], 1)
                next_tok = top_idx[0][chosen]
                ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
                if next_tok.item() == eass_id:
                    break

        response = self.tokenizer.decode(ids[0, input_len:], skip_special_tokens=False)
        if "[/ASS]" in response:
            response = response.split("[/ASS]")[0]
        return response.strip() or "..."

    # ════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ════════════════════════════════════════════════════════════════
    def add_knowledge(self, text: str, metadata: dict, doc_id: str):
        """
        Agrega un nuevo documento al conocimiento de Hannah.
        Útil para memoria de largo plazo: si Hannah aprende el nombre
        del usuario durante la conversación, se puede guardar aquí.

        Ejemplo:
            pipeline.add_knowledge(
                text="The user's name is Jorge and he likes football.",
                metadata={"source": "conversation", "topic": "user_info"},
                doc_id="user_jorge_001"
            )
        """
        self.rag.ingest_documents([text], [metadata], [doc_id])

    def get_stats(self) -> dict:
        """Retorna estadísticas del pipeline (documentos, caché, etc.)."""
        return self.rag.get_stats()


# ============================================================================
# TEST DEL PIPELINE
# ============================================================================
# Ejecutar: python hannah_pipeline.py
# Esto prueba el pipeline SIN modelo (modo solo-RAG) para verificar
# que la integración funciona antes de meter la GPU.
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: HannahPipeline (modo solo-RAG)")
    print("=" * 60)

    # Inicializar SIN modelo (load_model=False)
    pipeline = HannahPipeline(load_model=False)

    # Verificar estado de la BD
    stats = pipeline.get_stats()
    total_docs = stats["vector_store"]["total_documents"]
    print(f"\n  Documentos en BD: {total_docs}")

    if total_docs == 0:
        print("  ADVERTENCIA: La BD está vacía.")
        print("  Ejecuta primero: python ingest_knowledge.py")
        print("  Continuando con test limitado...\n")

    # ─── Test 1: Mensaje simple ───
    print("\n--- Test 1: Mensaje simple ---")
    r1 = pipeline.process_message("Hey babe!")
    print(f"  Input:  'Hey babe!'")
    print(f"  Mode:   {r1['mode']} (esperado: simplified)")
    print(f"  Source: {r1['source']}")
    print(f"  RAG:    {r1['rag_context'][:100]}...")
    print(f"  Output: {r1['text'][:200]}")

    # ─── Test 2: Pregunta con respuesta en BD ───
    print("\n--- Test 2: Pregunta con conocimiento ---")
    r2 = pipeline.process_message("What's your favorite movie?")
    print(f"  Input:  'What's your favorite movie?'")
    print(f"  Mode:   {r2['mode']}")
    print(f"  Chunks: {r2['rag_chunks']}")
    print(f"  RAG:    {r2['rag_context'][:200]}...")

    # ─── Test 3: Pregunta compleja ───
    print("\n--- Test 3: Pregunta compleja ---")
    r3 = pipeline.process_message(
        "Can you explain what makes you different from a regular assistant?",
        history=[("Hi!", "Hey~"), ("How are you?", "Good, just thinking about you~")]
    )
    print(f"  Input:  'Can you explain...'")
    print(f"  Mode:   {r3['mode']} (esperado: extended)")
    print(f"  Chunks: {r3['rag_chunks']}")
    print(f"  Latency: {r3['latency']}s")

    # ─── Test 4: Cache hit ───
    print("\n--- Test 4: Cache hit ---")
    r4 = pipeline.process_message("What's your favorite movie?")
    print(f"  Cache hit: {r4['cache_hit']}")

    print("\n" + "=" * 60)
    print("  TEST COMPLETADO")
    print("  Para probar con el modelo Hannah, ejecutar:")
    print("  python hannah_pipeline.py --with-model")
    print("  (requiere GPU + checkpoints)")
    print("=" * 60)
