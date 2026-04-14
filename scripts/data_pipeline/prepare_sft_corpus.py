import os
import json
import pathlib
import numpy as np
import multiprocessing as mp
import logging
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

TOKENIZER_PATH = "src/tokenizer/hannah_tok"

global_tok = None
train_lock = None
val_lock = None
train_file_path = None
val_file_path = None


def worker_init(tok_path, t_lock, v_lock, t_path, v_path):
    global global_tok, train_lock, val_lock, train_file_path, val_file_path
    global_tok = AutoTokenizer.from_pretrained(tok_path)
    train_lock = t_lock
    val_lock = v_lock
    train_file_path = t_path
    val_file_path = v_path


def process_chunk_direct(chunk_data):
    train_ids = []
    val_ids = []

    for idx, line in chunk_data:
        try:
            text = json.loads(line).get("text", "")
            if len(text.strip()) > 20:
                ids = global_tok.encode(text, add_special_tokens=False)
                ids.append(global_tok.eos_token_id)
                if idx % 10 == 0:
                    val_ids.extend(ids)
                else:
                    train_ids.extend(ids)
        except Exception:
            continue

    if train_ids:
        arr = np.array(train_ids, dtype=np.uint16)
        with train_lock:
            with open(train_file_path, "ab") as f:
                arr.tofile(f)

    if val_ids:
        arr = np.array(val_ids, dtype=np.uint16)
        with val_lock:
            with open(val_file_path, "ab") as f:
                arr.tofile(f)

    return len(train_ids), len(val_ids)


def chunk_generator(filepath, chunk_size=10000):
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            chunk.append((i, line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def main():
    corpus_path = pathlib.Path("data/finetuning/sft_corpus_clean.jsonl")
    output_dir = pathlib.Path("data/finetuning/sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    t_path = output_dir / "train.bin"
    v_path = output_dir / "val.bin"

    if t_path.exists():
        t_path.unlink()
    if v_path.exists():
        v_path.unlink()

    print("Contando lineas del corpus SFT...")

    def _count_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    with open(corpus_path, "rb") as fp:
        total_lines = sum(buffer.count(b"\n") for buffer in _count_generator(fp.raw.read))

    chunk_size = 10000
    total_chunks = (total_lines // chunk_size) + 1
    num_workers = max(1, mp.cpu_count() - 2)

    m = mp.Manager()
    t_lock = m.Lock()
    v_lock = m.Lock()

    print(f"Iniciando {num_workers} procesos...")

    total_train = 0
    total_val = 0

    with mp.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(TOKENIZER_PATH, t_lock, v_lock, t_path, v_path),
    ) as pool:
        iterator = pool.imap_unordered(process_chunk_direct, chunk_generator(corpus_path, chunk_size))
        for t_count, v_count in tqdm(iterator, total=total_chunks, desc="Procesando"):
            total_train += t_count
            total_val += v_count

    print("\nProcesamiento completado.")
    print(f"Train: {total_train:,} tokens -> {t_path}")
    print(f"Val:   {total_val:,} tokens -> {v_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
