"""
Limpia el corpus SFT eliminando:
- HTML y caracteres de control
- frases de AI generico
- textos demasiado cortos o rotos
- contenido que involucre menores
"""

import json
import re
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

INPUT = Path("data/finetuning/sft_corpus.jsonl")
OUTPUT = Path("data/finetuning/sft_corpus_clean.jsonl")
NUM_WORKERS = max(1, mp.cpu_count() - 2)

MINOR_HARD_FILTERS = [
    r"\b(1[0-7]|[1-9])\s*years?\s*old\b",
    r"\bminor\b",
    r"\bunderage\b",
    r"\bunderaged\b",
    r"\bteen(ager)?\b",
    r"\bhigh\s*school\s*(girl|boy|student|kid)\b",
    r"\bjunior\s*high\b",
    r"\bkid(s)?\b",
    r"\bchild(ren)?\b",
    r"\bloli\b",
    r"\bshota\b",
    r"\byoung(er)?\s*(girl|boy)\b",
    r"\bschool\s*girl\b",
    r"\bschool\s*boy\b",
]

AI_PHRASES = [
    "openai",
    "chatgpt",
    "gpt-3",
    "gpt-4",
    "gpt-4o",
    "language model trained by",
    "i am a large language model",
    "as an ai",
    "as a language model",
    "i'm an ai",
    "i am an ai",
    "i cannot assist",
    "i'm not able to",
]


def contains_minor_content(text: str) -> bool:
    lower = text.lower()
    for pattern in MINOR_HARD_FILTERS:
        if re.search(pattern, lower):
            return True
    return False


def remove_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid(text: str) -> bool:
    if len(text) < 100:
        return False
    if text.count("<") > 3:
        return False
    lower = text.lower()
    if any(p in lower for p in AI_PHRASES):
        return False
    if "[ASS][ASS]" in text or "[USR][USR]" in text:
        return False
    if contains_minor_content(text):
        return False
    return True


def is_english_content(text: str) -> bool:
    content = re.sub(r"\[SYS\].*?\[/SYS\]", "", text)
    content = re.sub(r"\[(USR|ASS|/USR|/ASS)\]", "", content).strip()
    if not content:
        return False
    non_latin = len(re.findall(r"[^\x00-\x7F]", content[:500]))
    return non_latin <= 20


def process_line(line: str):
    if not line.strip():
        return None
    try:
        record = json.loads(line)
        text = remove_html(record.get("text", ""))
        record["text"] = text
        if not is_valid(text):
            return None
        if not is_english_content(text):
            return None
        return record
    except Exception:
        return None


def process_chunk(lines):
    results = []
    for line in lines:
        result = process_line(line)
        if result is not None:
            results.append(result)
    return results


def main():
    print(f"Usando {NUM_WORKERS} nucleos de CPU...")
    lines = INPUT.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    chunk_size = max(1000, total // (NUM_WORKERS * 4))
    chunks = [lines[i : i + chunk_size] for i in range(0, total, chunk_size)]

    written = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        with mp.Pool(NUM_WORKERS) as pool:
            for batch in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Limpiando"):
                for record in batch:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    skipped = total - written
    print(f"Escritos : {written:,}")
    print(f"Omitidos : {skipped:,}")
    print(f"Output   : {OUTPUT}")


if __name__ == "__main__":
    main()
