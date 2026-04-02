import json, pathlib, collections
from clean.filters import clean
from clean.dedup   import deduplicate

RAW_DIR    = pathlib.Path("raw")
MERGED     = pathlib.Path("corpus_merged.jsonl")
DEDUPED    = pathlib.Path("corpus_deduped.jsonl")
FINAL      = pathlib.Path("corpus_final.jsonl")

# Cuántos tokens aproximados tiene un archivo .jsonl de N bytes
# Rule of thumb: 1 token ≈ 4 bytes de texto inglés
def estimate_tokens(path: pathlib.Path) -> int:
    return path.stat().st_size // 4

def merge_and_clean():
    """Paso 1: leer todos los .jsonl raw, limpiar y escribir corpus_merged.jsonl"""
    stats = collections.Counter()
    total_written = 0

    with open(MERGED, "w", encoding="utf-8") as fout:
        for jsonl_file in RAW_DIR.rglob("*.jsonl"):
            source_count = 0
            with open(jsonl_file, encoding="utf-8") as fin:
                for line in fin:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["parse_error"] += 1
                        continue

                    cleaned = clean(record.get("text", ""))
                    if cleaned is None:
                        stats["filtered"] += 1
                        continue

                    record["text"] = cleaned
                    fout.write(json.dumps(record) + "\n")
                    stats["written"] += 1
                    source_count += 1
                    total_written += 1

            print(f"  {jsonl_file.name}: {source_count:,} documentos aceptados")

    print(f"\n[MERGE] Total: {total_written:,} documentos")
    print(f"        Filtrados: {stats['filtered']:,}")
    print(f"        Errores de parse: {stats['parse_error']:,}")

def finalize():
    """Paso 3: renombrar el resultado y reportar stats finales."""
    DEDUPED.rename(FINAL)
    size_gb   = FINAL.stat().st_size / 1e9
    est_tok   = estimate_tokens(FINAL)

    print(f"\n{'-'*50}")
    print(f"  Corpus final: {FINAL}")
    print(f"  Tamaño:       {size_gb:.2f} GB")
    print(f"  Tokens est.:  {est_tok/1e9:.1f}B")
    print(f"{'-'*50}")

    if size_gb < 10:
        print("  el corpus es menor a 10 GB.")
        print("  Considerar añadir más datos de C4 o BookCorpus.")

if __name__ == "__main__":
    print("Limpieza")
    merge_and_clean()

    print("\nDeduplicación MinHash ===")
    deduplicate(MERGED, DEDUPED)

    print("\nFin")
    finalize()