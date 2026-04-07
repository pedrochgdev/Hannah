# train_debug.py
"""
Versión de depuración de train.py con opción de corpus sintético
"""

import argparse
import pathlib
import json
import itertools
import sentencepiece as spm
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "[SYS]", "[/SYS]", "[USR]", "[/USR]",
    "[ASS]", "[/ASS]", "[MEMORY]", "[/MEMORY]",
]

VOCAB_SIZE = 2000  # Reducido para pruebas rápidas
CHARACTER_COVERAGE = 0.9995
MAX_SENTENCEPIECE_LENGTH = 16

def generate_synthetic_corpus(output_path, num_examples=5000):
    """Genera corpus sintético para pruebas"""
    templates = [
        "Hello world! This is a test sentence.",
        "I love programming and machine learning.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is fascinating.",
        "How are you doing today?",
        "[SYS] System message [/SYS] [USR] User input [/USR] [ASS] Assistant response [/ASS]",
        "Numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        "Special characters: @ # $ % ^ & * ( )",
        "This is a longer sentence " + "with repetition " * 5,
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            text = templates[i % len(templates)]
            # Añadir variación
            if i % 7 == 0:
                text = f"Example {i}: {text}"
            f.write(json.dumps({"text": text}) + "\n")
    
    return output_path

def extract_text_iterator(corpus_path, max_lines=None):
    """Versión modificada con fallback a corpus sintético"""
    if not pathlib.Path(corpus_path).exists():
        print(f"⚠ Corpus {corpus_path} no existe, generando corpus sintético...")
        corpus_path = generate_synthetic_corpus("synthetic_corpus.jsonl")
    
    count = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
                if text and len(text.strip()) > 5:  # Reducido para pruebas
                    yield text.strip()
                    count += 1
                    if max_lines and count >= max_lines:
                        return
            except (json.JSONDecodeError, KeyError):
                continue

def _split_into_sentences(text, max_len=200):  # Reducido para pruebas
    import re
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

def write_temp_corpus(corpus_path, tmp_path, max_lines=None):
    print(f"[1/3] Extrayendo texto...")
    written = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for text in extract_text_iterator(corpus_path, max_lines or 1000):  # Límite para pruebas
            for chunk in _split_into_sentences(text):
                if chunk.strip():
                    f.write(chunk.strip() + "\n")
                    written += 1
    
    size_mb = tmp_path.stat().st_size / 1e6
    print(f"    {written:,} oraciones → {size_mb:.2f} MB")
    return written

def train_sentencepiece(tmp_corpus, output_dir):
    print(f"\n[2/3] Entrenando SentencePiece BPE ({VOCAB_SIZE:,} vocab)...")
    
    model_prefix = str(output_dir / "tokenizer")
    
    spm.SentencePieceTrainer.train(
        input=str(tmp_corpus),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        max_sentencepiece_length=MAX_SENTENCEPIECE_LENGTH,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=",".join(SPECIAL_TOKENS[4:]),
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        num_threads=4,  # Reducido para pruebas
        shuffle_input_sentence=True,
        input_sentence_size=10000,  # Reducido para pruebas
    )
    print(f"    Modelo guardado: {model_prefix}.model")

def export_huggingface_format(output_dir):
    print(f"\n[3/3] Exportando a formato HuggingFace...")
    
    tok = SentencePieceBPETokenizer(
        vocab=str(output_dir / "tokenizer.vocab"),
        merges=None,
    )
    
    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tok.token_to_id("<bos>")),
            ("<eos>", tok.token_to_id("<eos>")),
        ],
    )
    
    tok.save_model(str(output_dir))
    tok.save(str(output_dir / "tokenizer.json"))
    
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1024,
        "padding_side": "right",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": SPECIAL_TOKENS[4:],
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"    Exportado en: {output_dir}/")

def run(corpus_path, output_dir, max_lines=None, keep_tmp=False, debug=True):
    if debug:
        print("🔧 MODO DEBUG ACTIVADO")
        print(f"   Vocab size reducido: {VOCAB_SIZE}")
        print(f"   Máximo de líneas: {max_lines or 1000}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_corpus = output_dir / "_tmp_corpus.txt"
    
    try:
        write_temp_corpus(corpus_path, tmp_corpus, max_lines)
        train_sentencepiece(tmp_corpus, output_dir)
        export_huggingface_format(output_dir)
    finally:
        if not keep_tmp and tmp_corpus.exists():
            tmp_corpus.unlink()
    
    print(f"\n{'='*50}")
    print(f"  Tokenizador listo en: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path("test_corpus.jsonl"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("test_tokenizer"))
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--no-debug", action="store_true", help="Desactivar modo debug")
    args = parser.parse_args()
    
    run(args.corpus, args.output, args.max_lines, args.keep_tmp, debug=not args.no_debug)