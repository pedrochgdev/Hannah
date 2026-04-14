"""
Entrena un tokenizador BPE de 32k vocab sobre corpus_final.jsonl.
Usa sentencepiece como backend y exporta en formato HuggingFace.
¡Versión optimizada para múltiples núcleos!
"""

import argparse
import pathlib
import json
import re
import os
import multiprocessing as mp
import sentencepiece as spm

# ── Tokens especiales del proyecto ───────────────────────────────────────────
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

def _split_into_sentences(text: str, max_len: int = 500) -> list[str]:
    """Parte texto largo en fragmentos manejables usando Regex."""
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

def process_chunk(args):
    """
    Función para el Multiprocessing.
    Cada núcleo lee un pedazo del JSONL y extrae las oraciones a su propio archivo temporal.
    """
    file_path, start_byte, end_byte, tmp_out_path = args
    written = 0
    
    with open(file_path, 'rb') as f, open(tmp_out_path, 'w', encoding='utf-8') as out_f:
        if start_byte != 0:
            f.seek(start_byte - 1)
            if f.read(1) != b'\n':
                f.readline()
                
        while True:
            if f.tell() >= end_byte: break
            line = f.readline()
            if not line: break
            
            try:
                # Decodificamos y leemos el JSON
                text = json.loads(line.decode('utf-8', errors='ignore')).get("text", "")
                if text and len(text.strip()) > 20:
                    for chunk in _split_into_sentences(text.strip()):
                        if chunk.strip():
                            out_f.write(chunk.strip() + "\n")
                            written += 1
            except (json.JSONDecodeError, KeyError):
                continue
                
    return written, tmp_out_path

def write_temp_corpus_parallel(corpus_path: pathlib.Path, output_dir: pathlib.Path):
    """
    Divide el corpus de 30GB usando los hilos del CPU para extraer el texto en paralelo.
    """
    print(f"[1/3] Extrayendo texto en paralelo desde {corpus_path.name}...")
    
    file_size = os.path.getsize(corpus_path)
    # Dejamos 2 hilos libres para que el PC no se congele
    num_cores = max(1, mp.cpu_count() - 2) 
    chunk_size = file_size // num_cores
    
    chunks = []
    for i in range(num_cores):
        start = i * chunk_size
        end = start + chunk_size if i < num_cores - 1 else file_size
        tmp_out = str(output_dir / f"_tmp_corpus_{i}.txt")
        chunks.append((str(corpus_path), start, end, tmp_out))
        
    print(f"    ⚡ Usando {num_cores} núcleos para procesar los datos...")
    
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)
        
    total_written = sum(r[0] for r in results)
    tmp_files = [r[1] for r in results]
    
    print(f"    ✅ {total_written:,} oraciones extraídas en {len(tmp_files)} archivos temporales.")
    return tmp_files

def train_sentencepiece(tmp_files: list[str], output_dir: pathlib.Path):
    """
    Paso 1: entrenar con SentencePiece.
    """
    print(f"\n[2/3] Entrenando SentencePiece BPE ({VOCAB_SIZE:,} vocab)...")
    model_prefix = str(output_dir / "tokenizer")
    
    # SentencePiece permite pasarle una lista de archivos separados por comas
    input_files_str = ",".join(tmp_files)

    spm.SentencePieceTrainer.train(
        input                    = input_files_str,
        model_prefix             = model_prefix,
        model_type               = "bpe",
        vocab_size               = VOCAB_SIZE,
        character_coverage       = CHARACTER_COVERAGE,
        max_sentencepiece_length = MAX_SENTENCEPIECE_LENGTH,
        max_sentence_length      = 32768,

        # Tokens especiales
        pad_id = 0, unk_id = 1, bos_id = 2, eos_id = 3,
        pad_piece = "<pad>", unk_piece = "<unk>", bos_piece = "<bos>", eos_piece = "<eos>",
        user_defined_symbols = ",".join(SPECIAL_TOKENS[4:]),

        # Opciones de calidad
        split_digits = True,
        byte_fallback = True,
        normalization_rule_name = "nmt_nfkc",

        # Rendimiento
        num_threads = max(1, mp.cpu_count() - 2),
        shuffle_input_sentence = True,
        input_sentence_size = 5_000_000, 
        train_extremely_large_corpus = True,
    )
    print(f"    Modelo guardado: {model_prefix}.model")

def export_huggingface_format(output_dir: pathlib.Path):
    from transformers import AutoTokenizer
    from transformers.convert_slow_tokenizer import convert_slow_tokenizer
    from transformers import LlamaTokenizer
    import json

    model_path = str(output_dir / "tokenizer.model")

    # Cargamos el tokenizador lento
    tok_slow = LlamaTokenizer(
        vocab_file=model_path,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        legacy=False,
    )
    tok_slow.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS[4:]
    })

    # Convertimos directamente al backend rápido (conserva las merges)
    fast_tokenizer = convert_slow_tokenizer(tok_slow)

    # Guardamos el tokenizer.json manualmente
    fast_tokenizer.save(str(output_dir / "tokenizer.json"))

    # Guardamos el resto de la config con el tokenizador lento como base
    tok_slow.save_pretrained(str(output_dir))
    
    print(f"    Exportado exitosamente en: {output_dir}/")

def run(corpus_path: pathlib.Path, output_dir: pathlib.Path, keep_tmp: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_files = []

    try:
        tmp_files = write_temp_corpus_parallel(corpus_path, output_dir)
        train_sentencepiece(tmp_files, output_dir)
        export_huggingface_format(output_dir)
    finally:
        if not keep_tmp:
            for tmp in tmp_files:
                if os.path.exists(tmp):
                    os.remove(tmp)
            print(f"\n    (Archivos temporales eliminados)")

    print(f"\n{'='*50}")
    print(f"  Tokenizador listo en: {output_dir}")
    print(f"  Cargar con:")
    print(f"    from transformers import AutoTokenizer")
    print(f"    tok = AutoTokenizer.from_pretrained(r'{output_dir}')")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path("data_pipeline/corpus_final.jsonl"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("src/tokenizer/hannah_tok"))
    parser.add_argument("--keep-tmp", action="store_true", help="No borrar el corpus temporal después de entrenar")
    args = parser.parse_args()
    
    # Evitar problemas de Multiprocessing en Windows
    mp.freeze_support()
    run(args.corpus, args.output, args.keep_tmp)