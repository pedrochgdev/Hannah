"""
Entrena un tokenizador BPE de 32k vocab sobre corpus_final.jsonl.
Usa sentencepiece como backend y exporta en formato HuggingFace
para compatibilidad directa con el training loop de PyTorch.

Uso:
    python tokenizer/train.py --corpus corpus_final.jsonl --output tokenizer/hannah_tok
"""

import argparse, pathlib, json, itertools, sentencepiece as spm
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing

# ── Tokens especiales del proyecto ───────────────────────────────────────────
# Deben coincidir exactamente con el chat template de model.py
SPECIAL_TOKENS = [
    "<pad>",    # padding
    "<unk>",    # token desconocido
    "<bos>",    # inicio de secuencia
    "<eos>",    # fin de secuencia
    "[SYS]",    # inicio system prompt
    "[/SYS]",   # fin system prompt
    "[USR]",    # turno usuario
    "[/USR]",   # fin turno usuario
    "[ASS]",    # turno asistente
    "[/ASS]",   # fin turno asistente
    "[MEMORY]", # inicio bloque RAG
    "[/MEMORY]",# fin bloque RAG
]

VOCAB_SIZE    = 32_000
CHARACTER_COVERAGE = 0.9995   # cubre prácticamente todo el inglés
MAX_SENTENCEPIECE_LENGTH = 16  # tokens más largos se parten

def extract_text_iterator(corpus_path: pathlib.Path, max_lines: int = None):
    """
    Genera líneas de texto desde el .jsonl.
    SentencePiece necesita un iterador de strings, no el archivo completo.
    """
    count = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)["text"]
                if text and len(text.strip()) > 20:
                    yield text.strip()
                    count += 1
                    if max_lines and count >= max_lines:
                        return
            except (json.JSONDecodeError, KeyError):
                continue

def write_temp_corpus(corpus_path: pathlib.Path,
                      tmp_path: pathlib.Path,
                      max_lines: int = None):
    """
    SentencePiece necesita un archivo de texto plano (una oración por línea).
    Escribe un archivo temporal desde el .jsonl.
    """
    print(f"[1/3] Extrayendo texto desde {corpus_path.name}...")
    written = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for text in extract_text_iterator(corpus_path, max_lines):
            # Partir en oraciones cortas mejora la calidad del BPE
            # SentencePiece funciona mejor con líneas de longitud media (~100-300 chars)
            for chunk in _split_into_sentences(text):
                if chunk.strip():
                    f.write(chunk.strip() + "\n")
                    written += 1

    size_mb = tmp_path.stat().st_size / 1e6
    print(f"    {written:,} oraciones extraídas → {size_mb:.1f} MB")
    return written

def _split_into_sentences(text: str, max_len: int = 500) -> list[str]:
    """
    Parte texto largo en fragmentos manejables.
    No es un sentence splitter perfecto, pero es suficiente para BPE.
    """
    import re
    # Partir en puntuación final
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current = (current + " " + s).strip()
        else:
            if current:
                result.append(current)
            current = s
    if current:
        result.append(current)
    return result

def train_sentencepiece(tmp_corpus: pathlib.Path, output_dir: pathlib.Path):
    """
    Paso 1: entrenar con SentencePiece (más estable para vocab grande).
    Genera tokenizer.model y tokenizer.vocab.
    """
    print(f"\n[2/3] Entrenando SentencePiece BPE ({VOCAB_SIZE:,} vocab)...")
    print(f"      Esto puede tardar 30-90 minutos según el tamaño del corpus.")

    model_prefix = str(output_dir / "tokenizer")

    spm.SentencePieceTrainer.train(
        input            = str(tmp_corpus),
        model_prefix     = model_prefix,
        model_type       = "bpe",               # Byte Pair Encoding
        vocab_size       = VOCAB_SIZE,
        character_coverage = CHARACTER_COVERAGE,
        max_sentencepiece_length = MAX_SENTENCEPIECE_LENGTH,

        # Tokens especiales — el orden importa, define los IDs
        pad_id           = 0,   # <pad>
        unk_id           = 1,   # <unk>
        bos_id           = 2,   # <bos>
        eos_id           = 3,   # <eos>
        pad_piece        = "<pad>",
        unk_piece        = "<unk>",
        bos_piece        = "<bos>",
        eos_piece        = "<eos>",

        # Tokens especiales adicionales (IDs 4-11)
        user_defined_symbols = ",".join(SPECIAL_TOKENS[4:]),

        # Opciones de calidad
        split_digits     = True,    # "123" → "1", "2", "3" (mejor para números)
        byte_fallback    = True,    # bytes como fallback para chars raros
        normalization_rule_name = "nmt_nfkc",  # normalización unicode estándar

        # Rendimiento
        num_threads      = 8,       # ajustar según los cores disponibles
        shuffle_input_sentence = True,
        input_sentence_size = 5_000_000,  # máximo de oraciones usadas para entrenar
                                           # el BPE (5M es más que suficiente)
        # Logging
        train_extremely_large_corpus = True,
    )
    print(f"    Modelo guardado: {model_prefix}.model")

def export_huggingface_format(output_dir: pathlib.Path):
    """
    Paso 2: convertir el modelo SentencePiece al formato HuggingFace tokenizers.
    Esto permite usar el tokenizador directamente con AutoTokenizer.from_pretrained().
    """
    print(f"\n[3/3] Exportando a formato HuggingFace...")

    # Cargar con la librería tokenizers de HF
    tok = SentencePieceBPETokenizer(
        vocab   = str(output_dir / "tokenizer.vocab"),
        merges  = None,   # SentencePiece BPE no usa archivo de merges separado
    )

    # Post-processor: añadir <bos> al inicio y <eos> al final automáticamente
    tok.post_processor = TemplateProcessing(
        single   = "<bos> $A <eos>",
        pair     = "<bos> $A <eos> $B:1 <eos>:1",
        special_tokens = [
            ("<bos>", tok.token_to_id("<bos>")),
            ("<eos>", tok.token_to_id("<eos>")),
        ],
    )

    # Guardar en formato HuggingFace (genera tokenizer.json + tokenizer_config.json)
    tok.save_model(str(output_dir))
    tok.save(str(output_dir / "tokenizer.json"))

    # tokenizer_config.json — necesario para AutoTokenizer.from_pretrained()
    config = {
        "tokenizer_class":   "PreTrainedTokenizerFast",
        "model_max_length":  1024,
        "padding_side":      "right",
        "bos_token":         "<bos>",
        "eos_token":         "<eos>",
        "unk_token":         "<unk>",
        "pad_token":         "<pad>",
        "additional_special_tokens": SPECIAL_TOKENS[4:],
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"    Exportado en: {output_dir}/")

def run(corpus_path: pathlib.Path, output_dir: pathlib.Path,
        max_lines: int = None, keep_tmp: bool = False):

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_corpus = output_dir / "_tmp_corpus.txt"

    try:
        write_temp_corpus(corpus_path, tmp_corpus, max_lines)
        train_sentencepiece(tmp_corpus, output_dir)
        export_huggingface_format(output_dir)
    finally:
        if not keep_tmp and tmp_corpus.exists():
            tmp_corpus.unlink()
            print(f"\n    (corpus temporal eliminado)")

    print(f"\n{'='*50}")
    print(f"  Tokenizador listo en: {output_dir}")
    print(f"  Cargar con:")
    print(f"    from transformers import AutoTokenizer")
    print(f"    tok = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",    type=pathlib.Path, default=pathlib.Path("corpus_final.jsonl"))
    parser.add_argument("--output",   type=pathlib.Path, default=pathlib.Path("tokenizer/hannah_tok"))
    parser.add_argument("--max-lines",type=int,          default=None,
                        help="Limitar oraciones para prueba rápida (ej: 500000)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="No borrar el corpus temporal después de entrenar")
    args = parser.parse_args()
    run(args.corpus, args.output, args.max_lines, args.keep_tmp)