#!/usr/bin/env python3
# ============================================================================
# PIPELINE DE INTEGRACIÓN: RAG + Fast Hannah + Slow Hannah
# ============================================================================
# Archivo: rag_standalone/hannah_pipeline.py
# Proyecto: Hannah AI Companion - RAG Pipeline
# Autor: Luis Miranda Mallqui
# Revisiones: arreglos de acople backend (ver README_integration.md)
# ============================================================================
#
# ¿QUÉ HACE ESTE ARCHIVO?
# ========================
# Módulo de integración que conecta el RAG con los modelos de generación.
# Es la "ventanilla" que el backend usa para procesar mensajes del usuario.
#
# FLUJO COMPLETO:
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
#   [3] Caché de respuestas completas (¿ya respondimos ESTO antes?)
#        |         |
#      [HIT]    [MISS]
#        |         |
#        v         v
#   Retorna    [4] Classifier Model (¿fast o slow?)
#   directo        |            |
#               [FAST]       [SLOW]
#                  |            |
#                  v            v
#              [5] RAG       [5] RAG
#              simplified    extended
#              (~200 tok)    (~1500 tok)
#                  |            |
#                  v            v
#              [6] Fast      [6] Slow
#              Hannah 360M   Hannah Qwen2.5-14B
#                  |            |
#                  v            v
#              [7] Respuesta → Guardar en caché → Web App → Usuario
#
# DIFERENCIA CON LA VERSIÓN ANTERIOR:
# =====================================
# - El Semantic Cache ahora está al nivel del pipeline y cachea la
#   RESPUESTA COMPLETA (texto + metadata), no solo el contexto RAG.
#   Así, en un HIT, se saltea tanto el RAG como la generación LLM.
# - Se agrega soporte estructural para Slow Hannah (Qwen2.5-14B).
# - Se corrige mutación del historial en _build_prompt.
# - El Classifier Model tiene una interfaz clara para ser reemplazado.
#
# CÓMO LO USA EL BACKEND:
# ========================
#   from hannah_pipeline import HannahPipeline
#
#   # Inicializar UNA SOLA VEZ al arrancar el servidor
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
#   2. Para generación real: GPU + checkpoints de Fast y/o Slow Hannah
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
REPO_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "repositorio")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from rag_component  import RAGComponent
from semantic_cache import SemanticCache   # caché de respuestas completas

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

# ============================================================================
# PATHS A LOS MODELOS
# ============================================================================
# Fast Hannah (360M — OLMo3)
FAST_CHECKPOINT = os.path.join(REPO_DIR, "checkpoints", "hannah_dpo", "hannah_dpo_final.pt")
FAST_TOK_PATH   = os.path.join(REPO_DIR, "tokenizer",   "hannah_tok")

# Slow Hannah (Qwen2.5-14B-Instruct)
# ─── AJUSTAR estas rutas cuando el equipo de Slow Hannah entregue el checkpoint ───
SLOW_CHECKPOINT = os.path.join(REPO_DIR, "checkpoints", "slow_hannah")
SLOW_TOK_PATH   = os.path.join(REPO_DIR, "tokenizer",   "slow_tok")


class HannahPipeline:
    """
    Pipeline completo de Hannah: Caché → Classifier → RAG → LLM.

    Este es el ÚNICO punto de entrada que el backend necesita.
    Internamente maneja:
    - Caché de respuestas completas (nivel pipeline)
    - Classifier Model (fast/slow)
    - RAG + Context Handler (recuperación de conocimiento)
    - Fast Hannah 360M  (modo simplified)
    - Slow Hannah Qwen  (modo extended)

    USO:
        pipeline = HannahPipeline()
        result = pipeline.process_message("Hey!", history=[])
        print(result["text"])
    """

    def __init__(self, load_fast_model: bool = True,
                       load_slow_model: bool = True,
                       response_cache_threshold: float = 0.92,
                       response_cache_size:      int   = 500):
        """
        Inicializa el pipeline completo.

        Args:
            load_fast_model:
                Si True, carga Fast Hannah (360M) en GPU.
                Poner False para testing puro del RAG.
            load_slow_model:
                Si True, intenta cargar Slow Hannah (Qwen2.5-14B).
                Si no se encuentra el checkpoint, continúa sin ella.
            response_cache_threshold:
                Umbral de similitud coseno para el caché de respuestas.
                (Igual al threshold del caché interno del RAG: 0.92)
            response_cache_size:
                Máximo de respuestas guardadas en caché.
        """
        print("[Pipeline] Inicializando...")

        # ─── 1. RAG (caché interno a nivel de contexto) ───
        if not os.path.exists(KNOWLEDGE_DB_PATH):
            print(f"[Pipeline] ADVERTENCIA: BD no encontrada en {KNOWLEDGE_DB_PATH}")
            print(f"[Pipeline] Ejecuta primero: python ingest_knowledge.py")

        self.rag = RAGComponent(
            db_path=KNOWLEDGE_DB_PATH,
            cache_threshold=response_cache_threshold,
            cache_size=response_cache_size
        )

        # ─── 2. Caché de RESPUESTAS COMPLETAS (nivel pipeline) ───
        # A diferencia del caché interno del RAG (que solo guarda el
        # contexto ChromaDB), este guarda el dict completo de respuesta:
        # texto generado, source, latency, etc.
        # En un HIT aquí se saltea tanto el RAG como la generación LLM.
        self.response_cache = SemanticCache(
            similarity_threshold=response_cache_threshold,
            max_cache_size=response_cache_size
        )

        # ─── 3. Modelos de generación ───
        self.fast_model     = None
        self.fast_tokenizer = None
        self.fast_device    = None

        self.slow_model     = None
        self.slow_tokenizer = None

        if load_fast_model:
            self._load_fast_model()
        else:
            print("[Pipeline] Fast Hannah desactivada (modo solo-RAG).")

        if load_slow_model:
            self._load_slow_model()
        else:
            print("[Pipeline] Slow Hannah desactivada.")

        print("[Pipeline] Listo.")

    # ════════════════════════════════════════════════════════════════
    # CARGA DE MODELOS
    # ════════════════════════════════════════════════════════════════

    def _load_fast_model(self):
        """
        Carga Fast Hannah 360M (OLMo3, DPO-finetuned) en GPU.
        Si el checkpoint no existe, el pipeline sigue funcionando
        en modo solo-RAG (útil para testing).
        """
        try:
            import torch
            import types
            sys.modules['bettermap'] = types.ModuleType('bettermap')
            from olmo_core.nn.transformer import TransformerConfig
            from olmo_core.nn.attention   import AttentionBackendName
            from transformers              import AutoTokenizer

            if not os.path.exists(FAST_CHECKPOINT):
                print(f"[Pipeline] Fast Hannah: checkpoint no encontrado en {FAST_CHECKPOINT}")
                print(f"[Pipeline] Continuando sin Fast Hannah.")
                return

            self.fast_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.fast_tokenizer = AutoTokenizer.from_pretrained(FAST_TOK_PATH)

            # Configuración OLMo3 — 360M
            config = TransformerConfig.olmo3_7B(
                vocab_size=32000,
                attn_backend=AttentionBackendName.torch
            )
            config.d_model                                  = 1024
            config.n_layers                                 = 24
            config.block.sequence_mixer.d_model             = 1024
            config.block.sequence_mixer.n_heads             = 16
            config.block.sequence_mixer.n_kv_heads          = 16
            config.block.feed_forward.hidden_size           = int(1024 * 8 / 3)

            self.fast_model = config.build()
            ckpt       = torch.load(FAST_CHECKPOINT, map_location=self.fast_device, weights_only=False)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
            self.fast_model.load_state_dict(state_dict)
            self.fast_model.to(self.fast_device)
            self.fast_model.eval()

            print(f"[Pipeline] Fast Hannah 360M cargada en {self.fast_device}")

        except ImportError as e:
            print(f"[Pipeline] Dependencia faltante para Fast Hannah: {e}")
            print(f"[Pipeline] Continuando sin Fast Hannah.")
        except Exception as e:
            print(f"[Pipeline] Error cargando Fast Hannah: {e}")
            print(f"[Pipeline] Continuando sin Fast Hannah.")

    def _load_slow_model(self):
        """
        Carga Slow Hannah (Qwen2.5-14B-Instruct) en GPU.

        Usa device_map="auto" para distribuir capas entre GPUs disponibles.
        Si el checkpoint no existe, el pipeline fallback a Fast Hannah
        incluso cuando el Classifier elige modo "extended".

        ─── NOTA PARA EL EQUIPO ───
        Cuando el checkpoint de Qwen esté listo, asegurarse de que:
        - SLOW_CHECKPOINT apunte a la carpeta con config.json + pesos
        - SLOW_TOK_PATH apunte al tokenizer (o usar el mismo que Qwen HuggingFace)
        - Hay suficiente VRAM: Qwen2.5-14B en bfloat16 ≈ 28 GB
          (o repartido entre 2× GPU de 16 GB con device_map="auto")
        ───────────────────────────
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not os.path.exists(SLOW_CHECKPOINT):
                print(f"[Pipeline] Slow Hannah: checkpoint no encontrado en {SLOW_CHECKPOINT}")
                print(f"[Pipeline] Modo extended usará Fast Hannah como fallback.")
                return

            self.slow_tokenizer = AutoTokenizer.from_pretrained(
                SLOW_TOK_PATH if os.path.exists(SLOW_TOK_PATH) else SLOW_CHECKPOINT
            )
            self.slow_model = AutoModelForCausalLM.from_pretrained(
                SLOW_CHECKPOINT,
                torch_dtype=torch.bfloat16,
                device_map="auto"       # distribuye capas entre GPUs disponibles
            )
            self.slow_model.eval()
            print("[Pipeline] Slow Hannah (Qwen2.5-14B) cargada.")

        except ImportError as e:
            print(f"[Pipeline] Dependencia faltante para Slow Hannah: {e}")
            print(f"[Pipeline] Modo extended usará Fast Hannah como fallback.")
        except Exception as e:
            print(f"[Pipeline] Error cargando Slow Hannah: {e}")
            print(f"[Pipeline] Modo extended usará Fast Hannah como fallback.")

    # ════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL
    # ════════════════════════════════════════════════════════════════

    def process_message(self, user_msg: str, history: list = None) -> dict:
        """
        Procesa un mensaje del usuario y retorna la respuesta de Hannah.
        Este es el ÚNICO método que el backend necesita llamar.

        Args:
            user_msg: El mensaje del usuario (string).
            history:  Lista de tuplas (user_msg, hannah_response) con
                      el historial de la conversación actual.
                      Ejemplo: [("Hi!", "Hey~"), ("How are you?", "Good!")]

        Returns:
            dict con:
                - "text":        La respuesta de Hannah (string)
                - "source":      "cache" | "fast" | "slow" | "fast_fallback" | "rag_only"
                - "rag_context": El contexto RAG usado (para debug)
                - "mode":        "simplified" o "extended"
                - "cache_hit":   True/False
                - "latency":     Tiempo total en segundos
                - "rag_chunks":  Número de chunks recuperados
        """
        if history is None:
            history = []

        t_start = time.time()

        # ─── Paso 1: Caché de respuestas completas ───
        # Si ya respondimos una query semánticamente idéntica (similitud
        # coseno >= 0.92), devolvemos esa respuesta directamente.
        # Esto saltea el RAG Y la generación LLM.
        cached_response = self.response_cache.lookup(user_msg)
        if cached_response is not None:
            # Actualizar latencia para reflejar el tiempo real de este request
            cached_response = dict(cached_response)   # copia — no mutar el caché
            cached_response["latency"]   = round(time.time() - t_start, 3)
            cached_response["cache_hit"] = True
            cached_response["source"]    = "cache"
            return cached_response

        # ─── Paso 2: Classifier Model (decidir fast/slow) ───
        mode = self._select_model(user_msg, history)

        # ─── Paso 3: RAG retrieval ───
        # El RAGComponent tiene su propio caché interno a nivel de contexto.
        # Si el contexto para esta query ya fue calculado, lo reutiliza.
        rag_result  = self.rag.retrieve(user_msg, mode=mode)
        rag_context = rag_result["formatted_context"]

        # ─── Paso 4: Generación de respuesta ───
        response_text, source = self._route_generation(user_msg, history, rag_context, mode)

        latency = time.time() - t_start

        result = {
            "text":        response_text,
            "source":      source,
            "rag_context": rag_context,
            "mode":        mode,
            "cache_hit":   False,
            "latency":     round(latency, 3),
            "rag_chunks":  rag_result["num_chunks"],
        }

        # ─── Paso 5: Guardar respuesta completa en caché ───
        # Solo guardamos respuestas reales (no el modo de testing sin modelo)
        if source not in ("rag_only",):
            self.response_cache.store(user_msg, result)

        return result

    # ════════════════════════════════════════════════════════════════
    # ENRUTADOR DE GENERACIÓN
    # ════════════════════════════════════════════════════════════════

    def _route_generation(self, user_msg: str, history: list,
                          rag_context: str, mode: str) -> tuple[str, str]:
        """
        Elige el modelo correcto según el modo y genera la respuesta.

        Orden de prioridad:
          1. Slow Hannah  (si mode == "extended" y está cargada)
          2. Fast Hannah  (si está cargada — fallback de Slow o modo fast)
          3. Modo test    (sin modelos, retorna el contexto RAG para debug)

        Returns:
            Tupla (texto_de_respuesta, source_label)
            source_label: "fast" | "slow" | "fast_fallback" | "rag_only"
        """
        if mode == "extended" and self.slow_model is not None:
            prompt = self._build_slow_prompt(user_msg, history, rag_context)
            text   = self._generate_slow(prompt)
            return text, "slow"

        if self.fast_model is not None:
            prompt = self._build_fast_prompt(user_msg, history, rag_context)
            text   = self._generate_fast(prompt)
            # Si Slow no estaba disponible pero el modo era extended, lo aclaramos
            source = "fast" if mode == "simplified" else "fast_fallback"
            return text, source

        # Sin ningún modelo cargado → modo testing
        text = f"[MODO TEST - SIN MODELOS] Contexto RAG: {rag_context}"
        return text, "rag_only"

    # ════════════════════════════════════════════════════════════════
    # CLASSIFIER MODEL
    # ════════════════════════════════════════════════════════════════

    def _select_model(self, user_msg: str, history: list) -> str:
        """
        Decide si usar Fast (simplified) o Slow (extended).

        IMPLEMENTACIÓN ACTUAL: heurística simple basada en longitud
        y palabras clave. Funciona como placeholder hasta que el
        equipo del Classifier Model entregue su componente.

        ─── CÓMO REEMPLAZAR CON EL CLASSIFIER REAL ───
        El Classifier Model debe exponerse como un callable con esta firma:

            def classify(user_msg: str, history: list[tuple[str,str]]) -> str:
                # retorna "simplified" o "extended"

        Para conectarlo, pasar la función al constructor:

            from mi_clasificador import MiClasificador
            clf = MiClasificador()
            pipeline = HannahPipeline(classifier_fn=clf.predict)

        O editar directamente este método y reemplazar la heurística.
        ────────────────────────────────────────────────────────────

        Args:
            user_msg: Mensaje actual del usuario.
            history:  Historial de la conversación (no usado en heurística,
                      pero disponible para el Classifier real).

        Returns:
            "simplified" → Fast Hannah (Hannah 360M)
            "extended"   → Slow Hannah (Qwen2.5-14B)
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
    # PROMPT BUILDERS
    # ════════════════════════════════════════════════════════════════

    def _build_fast_prompt(self, user_msg: str, history: list, rag_context: str) -> str:
        """
        Construye el prompt para Fast Hannah con tokens especiales del modelo.

        Estructura:
            [SYS] system_prompt [/SYS]
            [MEMORY] contexto RAG [/MEMORY]
            [USR] msg1 [/USR][ASS] resp1 [/ASS]
            [USR] msg_actual [/USR][ASS]   ← Hannah genera aquí

        Presupuesto de tokens (ventana de 512 tokens de Hannah):
            System   ≈  100 tokens
            Memory   ≈  200 tokens (modo simplified)
            Historia ≈  200 tokens (se trunca si excede)
            Total    ≈  500 tokens  → cabe en 512 con margen

        CORRECCIÓN: se trabaja sobre una copia del historial para no
        mutar la lista original del caller (bug de versión anterior).
        """
        # Copia defensiva — no mutar el historial del caller
        history = list(history)

        prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"

        if rag_context and rag_context != "[MEMORY][/MEMORY]":
            prompt += f"\n{rag_context}"

        for usr, ass in history:
            prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
        prompt += f"[USR] {user_msg} [/USR][ASS]"

        # Truncar historial si el prompt excede 900 tokens
        if self.fast_tokenizer:
            ids = self.fast_tokenizer.encode(prompt)
            while history and len(ids) > 900:
                history.pop(0)
                # Reconstruir sin el turno más antiguo
                prompt = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
                if rag_context and rag_context != "[MEMORY][/MEMORY]":
                    prompt += f"\n{rag_context}"
                for usr, ass in history:
                    prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
                prompt += f"[USR] {user_msg} [/USR][ASS]"
                ids = self.fast_tokenizer.encode(prompt)

        return prompt

    def _build_slow_prompt(self, user_msg: str, history: list, rag_context: str) -> str:
        """
        Construye el prompt para Slow Hannah (Qwen2.5-14B-Instruct).

        Qwen usa el formato ChatML estándar de HuggingFace:
            <|im_start|>system
            {system_prompt}
            {rag_context}
            <|im_end|>
            <|im_start|>user
            {msg}
            <|im_end|>
            <|im_start|>assistant

        El contexto RAG va dentro del bloque system para que el modelo
        lo trate como "información de fondo", no como instrucción del usuario.

        Presupuesto de tokens (ventana de 32K de Qwen):
            System + Memory ≈ 1600 tokens (modo extended)
            Historia        ≈ variable (sin límite práctico)
            No se necesita truncar historial para Qwen.

        CORRECCIÓN: se trabaja sobre una copia del historial.
        """
        # Copia defensiva — no mutar el historial del caller
        history = list(history)

        system_block = SYSTEM_PROMPT
        if rag_context and rag_context != "[MEMORY][/MEMORY]":
            system_block += f"\n\n{rag_context}"

        prompt = f"<|im_start|>system\n{system_block}\n<|im_end|>\n"

        for usr, ass in history:
            prompt += f"<|im_start|>user\n{usr}\n<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{ass}\n<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    # ════════════════════════════════════════════════════════════════
    # GENERACIÓN
    # ════════════════════════════════════════════════════════════════

    def _generate_fast(self, prompt: str,
                       max_new_tokens: int   = 200,
                       temperature:    float = 0.7,
                       top_k:          int   = 40) -> str:
        """
        Genera una respuesta con Fast Hannah 360M (OLMo3).
        Detiene la generación al encontrar [/ASS] o al llegar a max_new_tokens.
        """
        import torch

        ids      = self.fast_tokenizer.encode(prompt, return_tensors="pt").to(self.fast_device)
        input_len = ids.shape[1]
        eass_id  = self.fast_tokenizer.convert_tokens_to_ids("[/ASS]")

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.fast_model(ids)
                logits      = logits[:, -1, :] / temperature
                top_vals, top_idx = torch.topk(logits, top_k)
                probs       = torch.softmax(top_vals, dim=-1)
                chosen      = torch.multinomial(probs[0], 1)
                next_tok    = top_idx[0][chosen]
                ids         = torch.cat([ids, next_tok.view(1, 1)], dim=1)
                if next_tok.item() == eass_id:
                    break

        response = self.fast_tokenizer.decode(ids[0, input_len:], skip_special_tokens=False)
        if "[/ASS]" in response:
            response = response.split("[/ASS]")[0]
        return response.strip() or "..."

    def _generate_slow(self, prompt: str,
                       max_new_tokens: int   = 512,
                       temperature:    float = 0.7,
                       top_p:          float = 0.9) -> str:
        """
        Genera una respuesta con Slow Hannah (Qwen2.5-14B-Instruct).

        Usa top_p (nucleus sampling) en lugar de top_k, que es el
        método recomendado para Qwen y da respuestas más fluidas en
        respuestas largas (modo extended).

        Detiene la generación con los tokens de fin de Qwen:
        <|im_end|> y <|endoftext|>.
        """
        import torch

        stop_tokens = [
            self.slow_tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.slow_tokenizer.eos_token_id,
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]

        inputs    = self.slow_tokenizer(prompt, return_tensors="pt").to(self.slow_model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.slow_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=stop_tokens,
                pad_token_id=self.slow_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_len:]
        response   = self.slow_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Limpiar tags residuales de Qwen si quedaron
        response = response.replace("<|im_end|>", "").strip()
        return response or "..."

    # ════════════════════════════════════════════════════════════════
    # UTILIDADES PÚBLICAS
    # ════════════════════════════════════════════════════════════════

    def add_knowledge(self, text: str, metadata: dict, doc_id: str):
        """
        Agrega un nuevo documento al conocimiento de Hannah en tiempo real.
        Útil para memoria de largo plazo: nombre del usuario, preferencias
        aprendidas durante la conversación, etc.

        Args:
            text:     El texto del nuevo conocimiento.
            metadata: Dict con info extra (source, topic, etc.)
            doc_id:   ID único. Si ya existe, hace upsert (actualiza).

        Ejemplo:
            pipeline.add_knowledge(
                text="The user's name is Jorge and he likes football.",
                metadata={"source": "conversation", "topic": "user_info"},
                doc_id="user_jorge_001"
            )
        """
        self.rag.ingest_documents([text], [metadata], [doc_id])

    def get_stats(self) -> dict:
        """
        Retorna estadísticas del pipeline para monitoreo y debugging.
        Incluye: documentos en ChromaDB, estado del caché RAG,
        estado del caché de respuestas, y qué modelos están cargados.
        """
        rag_stats = self.rag.get_stats()
        return {
            **rag_stats,
            "response_cache": self.response_cache.get_stats(),
            "models": {
                "fast_hannah_loaded": self.fast_model is not None,
                "slow_hannah_loaded": self.slow_model is not None,
                "fast_device":        str(self.fast_device) if self.fast_device else "N/A",
            }
        }