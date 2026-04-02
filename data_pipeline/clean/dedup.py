# deduplicación MinHash

from datasketch import MinHash, MinHashLSH
import json, pathlib, re

# Similitud de Jaccard: dos documentos se consideran duplicados si > 85%
THRESHOLD  = 0.85
NUM_PERM   = 128   # permutaciones MinHash — más = más preciso pero más lento

def shingle(text: str, k: int = 5) -> set:
    """Genera k-shingles de caracteres."""
    text = re.sub(r"\s+", " ", text.lower())
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def deduplicate(input_path: pathlib.Path, output_path: pathlib.Path):
    lsh   = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    seen  = 0
    dupes = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            text   = record["text"]

            # Construir MinHash para este documento
            mh = MinHash(num_perm=NUM_PERM)
            for s in shingle(text):
                mh.update(s.encode("utf8"))

            # Buscar vecinos en el índice LSH
            result = lsh.query(mh)
            if result:
                dupes += 1
                continue   # es duplicado: saltar

            # No es duplicado: insertar y escribir
            lsh.insert(str(i), mh)
            fout.write(json.dumps(record) + "\n")
            seen += 1

    total = seen + dupes
    print(f"[DEDUP] {total:,} entradas → {seen:,} únicas, {dupes:,} duplicadas ({dupes/total*100:.1f}%)")
    return seen

if __name__ == "__main__":
    deduplicate(
        pathlib.Path("corpus_merged.jsonl"),
        pathlib.Path("corpus_deduped.jsonl"),
    )