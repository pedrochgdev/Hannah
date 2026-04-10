"""
build_dialogue_corpus.py
========================

DATASETS INCLUIDOS:
  1. allenai/soda            ~1.5 GB  | CC-BY 4.0      | público
  2. allenai/WildChat-4.8M   ~11 GB   | ODC-BY         | público
  3. allenai/prosocial-dialog ~500 MB | Apache 2.0     | público
  4. jihyoung/ConversationChronicles ~1 GB | CC-BY 4.0 | público
  5. Estwld/empathetic_dialogues_llm | CC-BY-NC-SA 4.0 | público
  6. icybee/share_gpt_90k_v1 ~300 MB  | público

USO:
    pip install datasets huggingface_hub tqdm langdetect
    python build_dialogue_corpus.py

    Con token HF (para datasets gated como lmsys/lmsys-chat-1m):
    HF_TOKEN=hf_xxxx python build_dialogue_corpus.py

VARIABLES DE ENTORNO:
    HF_TOKEN          Token de Hugging Face
    OUTPUT_FILE       Ruta al archivo destino 
    TARGET_GB         GB objetivo de datos nuevos a agregar 
    WILDCHAT_4_8M     Si "1", incluye WildChat-4.8M 
    MAX_WORKERS       Hilos para escritura 
    SKIP_LANGDETECT   Si "1", omite filtro de idioma 
"""

import os
import json
import re
import sys
import logging
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

# ─── Configuración ────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", None)
OUTPUT_FILE    = Path(os.environ.get("OUTPUT_FILE", "corpus_final.jsonl"))
TARGET_GB      = float(os.environ.get("TARGET_GB", "20"))
INCLUDE_4_8M   = os.environ.get("WILDCHAT_4_8M", "1") == "1"
SKIP_LANGDETECT= os.environ.get("SKIP_LANGDETECT", "0") == "1"

TARGET_BYTES = int(TARGET_GB * 1024 ** 3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_corpus.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Utilidades de texto ──────────────────────────────────────────────────────

def clean_text(text: str) -> Optional[str]:
    """Limpieza básica: elimina artefactos comunes, normaliza espacios."""
    if not isinstance(text, str):
        return None
    # Eliminar tokens de control unicode
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Colapsar múltiples saltos de línea (>3) en dos
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Colapsar espacios múltiples dentro de línea
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    if len(text) < 50:        # descartar muestras muy cortas
        return None
    if len(text) > 500_000:   # descartar textos anómalamente largos
        return None
    return text


def is_english(text: str) -> bool:
    """Detección rápida de inglés. Si SKIP_LANGDETECT=1 siempre True."""
    if SKIP_LANGDETECT:
        return True
    try:
        from langdetect import detect, LangDetectException
        return detect(text[:2000]) == "en"
    except Exception:
        return True  # en caso de duda, conservar


def dialogue_list_to_text(turns: list, roles: tuple = ("human", "gpt")) -> Optional[str]:
    """
    Convierte una lista de turnos [{from, value}] o [{role, content}]
    en un bloque de texto con el formato:
        Human: ...
        Assistant: ...
    """
    lines = []
    for turn in turns:
        if isinstance(turn, dict):
            role    = turn.get("from") or turn.get("role", "")
            content = turn.get("value") or turn.get("content", "")
        else:
            continue
        if not content or role in ("system",):
            continue
        label = "Human" if role in ("human", "user") else "Assistant"
        content = content.strip()
        if content:
            lines.append(f"{label}: {content}")
    if len(lines) < 2:
        return None
    return "\n".join(lines)


# ─── Escritor de corpus ───────────────────────────────────────────────────────

class CorpusWriter:
    def __init__(self, path: Path):
        self.path = path
        self._written = 0
        self._skipped = 0
        self._file = open(path, "a", encoding="utf-8")

    def write(self, text: str, source: str) -> bool:
        text = clean_text(text)
        if text is None:
            self._skipped += 1
            return False
        if not is_english(text):
            self._skipped += 1
            return False
        record = json.dumps({"text": text, "source": source}, ensure_ascii=False)
        self._file.write(record + "\n")
        self._written += 1
        return True

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def written(self):
        return self._written

    @property
    def skipped(self):
        return self._skipped

    def output_size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


# ─── Procesadores por dataset ─────────────────────────────────────────────────

def load_hf_dataset(name: str, split: str = "train", config: str = None, **kwargs):
    """Carga un dataset de HF con manejo de errores."""
    from datasets import load_dataset
    try:
        kw = dict(split=split, trust_remote_code=True)
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        if config:
            return load_dataset(name, config, **kw, **kwargs)
        return load_dataset(name, **kw, **kwargs)
    except Exception as e:
        log.error(f"No se pudo cargar {name}: {e}")
        return None


def process_soda(writer: CorpusWriter):
    """allenai/soda — diálogos sociales en primera persona, ~1.5 GB"""
    log.info("▶ Procesando allenai/soda ...")
    ds = load_hf_dataset("allenai/soda")
    if ds is None:
        return
    for row in tqdm(ds, desc="soda", unit="conv"):
        dialogue = row.get("dialogue", [])
        if not dialogue:
            continue
        # dialogue es una lista de strings alternando hablantes
        lines = []
        for i, utterance in enumerate(dialogue):
            label = "Human" if i % 2 == 0 else "Assistant"
            lines.append(f"{label}: {utterance.strip()}")
        text = "\n".join(lines)
        writer.write(text, "allenai/soda")
    writer.flush()
    log.info(f"  soda: escritos={writer.written}, omitidos={writer.skipped}")


def process_wildchat_4_8m(writer: CorpusWriter):
    """allenai/WildChat-4.8M — 3.2M conversaciones (filtrado no-tóxico), ~11 GB"""
    log.info("▶ Procesando allenai/WildChat-4.8M (solo inglés) ...")
    ds = load_hf_dataset("allenai/WildChat-4.8M")
    if ds is None:
        return
    before = writer.written
    for row in tqdm(ds, desc="WildChat-4.8M", unit="conv"):
        if row.get("language", "") != "English":
            continue
        turns = row.get("conversation", [])
        text = dialogue_list_to_text(turns)
        if text:
            writer.write(text, "allenai/WildChat-4.8M")
    writer.flush()
    log.info(f"  WildChat-4.8M: nuevos escritos={writer.written - before}")


def process_prosocial(writer: CorpusWriter):
    """allenai/prosocial-dialog — 58K diálogos, ~500 MB"""
    log.info("▶ Procesando allenai/prosocial-dialog ...")
    ds = load_hf_dataset("allenai/prosocial-dialog")
    if ds is None:
        return
    before = writer.written
    for row in tqdm(ds, desc="prosocial-dialog", unit="conv"):
        context = row.get("context", [])
        response = row.get("response", "")
        if not context:
            continue
        lines = []
        for i, utt in enumerate(context):
            label = "Human" if i % 2 == 0 else "Assistant"
            lines.append(f"{label}: {utt.strip()}")
        if response:
            lines.append(f"Assistant: {response.strip()}")
        text = "\n".join(lines)
        writer.write(text, "allenai/prosocial-dialog")
    writer.flush()
    log.info(f"  prosocial-dialog: nuevos escritos={writer.written - before}")


def process_conversation_chronicles(writer: CorpusWriter):
    """jihyoung/ConversationChronicles — 200K episodios con hasta 5 sesiones por episodio.
    Formato real: columnas {first,second,third,fourth,fifth}_session_dialogue (lista de strings)
                           {first,second,third,fourth,fifth}_session_speakers  (lista de strings)
    Cada sesión se escribe como una conversación independiente.
    """
    log.info("▶ Procesando jihyoung/ConversationChronicles ...")
    ds = load_hf_dataset("jihyoung/ConversationChronicles")
    if ds is None:
        return
    before = writer.written

    SESSION_PREFIXES = [
        "first_session",
        "second_session",
        "third_session",
        "fourth_session",
        "fifth_session",
    ]

    for row in tqdm(ds, desc="ConversationChronicles", unit="episode"):
        for prefix in SESSION_PREFIXES:
            dialogue  = row.get(f"{prefix}_dialogue",  []) or []
            speakers  = row.get(f"{prefix}_speakers",  []) or []
            if not dialogue:
                continue

            lines = []
            # Identificar los dos hablantes únicos para asignar Human/Assistant
            unique_speakers = []
            for sp in speakers:
                if sp and sp not in unique_speakers:
                    unique_speakers.append(sp)

            for i, utterance in enumerate(dialogue):
                utterance = (utterance or "").strip()
                if not utterance:
                    continue
                speaker = speakers[i] if i < len(speakers) else ""
                # El primer hablante único → Human, el segundo → Assistant
                if unique_speakers and speaker == unique_speakers[0]:
                    label = "Human"
                else:
                    label = "Assistant"
                lines.append(f"{label}: {utterance}")

            if len(lines) >= 2:
                writer.write("\n".join(lines), "jihyoung/ConversationChronicles")

    writer.flush()
    log.info(f"  ConversationChronicles: nuevos escritos={writer.written - before}")


def process_empathetic_dialogues(writer: CorpusWriter):
    """Estwld/empathetic_dialogues_llm — diálogos empáticos"""
    log.info("▶ Procesando Estwld/empathetic_dialogues_llm ...")
    ds = load_hf_dataset("Estwld/empathetic_dialogues_llm")
    if ds is None:
        return
    before = writer.written
    for row in tqdm(ds, desc="empathetic_dialogues", unit="conv"):
        # Puede tener columna 'conversations' (lista) o columnas individuales
        convs = row.get("conversations", None)
        if convs:
            text = dialogue_list_to_text(convs)
        else:
            # Formato alternativo: columnas utterance_1, utterance_2 ...
            lines = []
            for key in sorted(row.keys()):
                if key.startswith("utterance"):
                    val = row[key]
                    if val:
                        i = int(key.replace("utterance_", ""))
                        label = "Human" if i % 2 == 1 else "Assistant"
                        lines.append(f"{label}: {val.strip()}")
            text = "\n".join(lines) if len(lines) >= 2 else None
        if text:
            writer.write(text, "Estwld/empathetic_dialogues_llm")
    writer.flush()
    log.info(f"  empathetic_dialogues: nuevos escritos={writer.written - before}")


def process_sharegpt(writer: CorpusWriter):
    """icybee/share_gpt_90k_v1 — 90K conversaciones ShareGPT"""
    log.info("▶ Procesando icybee/share_gpt_90k_v1 ...")
    ds = load_hf_dataset("icybee/share_gpt_90k_v1")
    if ds is None:
        return
    before = writer.written
    for row in tqdm(ds, desc="ShareGPT-90k", unit="conv"):
        convs = row.get("conversations", [])
        text = dialogue_list_to_text(convs)
        if text:
            writer.write(text, "ShareGPT-90k")
    writer.flush()
    log.info(f"  ShareGPT-90k: nuevos escritos={writer.written - before}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

PIPELINE = [
    process_soda,
    process_prosocial,
    process_empathetic_dialogues,
    process_conversation_chronicles,
    process_sharegpt,
]

if INCLUDE_4_8M:
    PIPELINE.append(process_wildchat_4_8m)


def check_dependencies():
    missing = []
    for pkg in ["datasets", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not SKIP_LANGDETECT:
        try:
            import langdetect  # noqa
        except ImportError:
            missing.append("langdetect")
    if missing:
        log.error(f"Faltan dependencias: {missing}")
        log.error("Instala con: pip install " + " ".join(missing))
        sys.exit(1)


def main():
    check_dependencies()

    log.info("=" * 60)
    log.info(f"Corpus destino : {OUTPUT_FILE.resolve()}")
    log.info(f"Objetivo       : {TARGET_GB} GB de datos nuevos")
    log.info(f"HF_TOKEN       : {'configurado' if HF_TOKEN else 'no configurado (solo datasets públicos)'}")
    log.info(f"WildChat-4.8M  : {'sí' if INCLUDE_4_8M else 'no'}")
    log.info(f"LangDetect     : {'desactivado' if SKIP_LANGDETECT else 'activo'}")
    log.info("=" * 60)

    initial_size = OUTPUT_FILE.stat().st_size if OUTPUT_FILE.exists() else 0
    log.info(f"Tamaño inicial de {OUTPUT_FILE.name}: {initial_size / 1024**3:.2f} GB")

    writer = CorpusWriter(OUTPUT_FILE)

    try:
        for processor in PIPELINE:
            current_size = writer.output_size_bytes() - initial_size
            log.info(f"\n── Datos nuevos acumulados: {current_size / 1024**3:.2f} GB / {TARGET_GB} GB objetivo ──")
            if current_size >= TARGET_BYTES:
                log.info(f"✅ Objetivo de {TARGET_GB} GB alcanzado. Deteniendo pipeline.")
                break
            try:
                processor(writer)
            except KeyboardInterrupt:
                log.warning("Interrupción de usuario. Guardando progreso...")
                break
            except Exception as e:
                log.error(f"Error en {processor.__name__}: {e}", exc_info=True)
                log.info("Continuando con el siguiente dataset...")
    finally:
        writer.close()

    final_size = OUTPUT_FILE.stat().st_size if OUTPUT_FILE.exists() else 0
    added_bytes = final_size - initial_size

    log.info("\n" + "=" * 60)
    log.info("RESUMEN FINAL")
    log.info(f"  Registros escritos : {writer.written:,}")
    log.info(f"  Registros omitidos : {writer.skipped:,}")
    log.info(f"  Datos agregados    : {added_bytes / 1024**3:.2f} GB")
    log.info(f"  Tamaño total corpus: {final_size / 1024**3:.2f} GB")
    log.info(f"  Archivo            : {OUTPUT_FILE.resolve()}")
    log.info("=" * 60)

    if added_bytes < TARGET_BYTES:
        log.warning(
            f"⚠️  Solo se agregaron {added_bytes/1024**3:.2f} GB de los {TARGET_GB} GB objetivo.\n"
            "   Opciones para obtener más datos:\n"
            "   1. Establece WILDCHAT_4_8M=1 para incluir WildChat-4.8M (~11 GB, solo inglés).\n"
            "   2. Añade lmsys/lmsys-chat-1m (requiere aprobación manual en HF)."
        )


if __name__ == "__main__":
    main()
