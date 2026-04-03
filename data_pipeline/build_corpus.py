import json, pathlib, collections
import faulthandler    # <-- AÑADIR
faulthandler.enable()  # <-- AÑADIR
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
            # AVISO: Saber en qué archivo estamos ANTES de que se congele
            print(f"\n-> Procesando: {jsonl_file.name}...") 
            
            source_count = 0
            file_filtered = 0
            file_errors = 0
            
            with open(jsonl_file, encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    if len(line) > 5_000_000:
                        stats["filtered"] += 1
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["parse_error"] += 1
                        file_errors += 1
                        continue

                    # Atrapar cualquier error durante la limpieza
                    try:
                        cleaned = clean(record.get("text", ""))
                    except Exception as e:
                        print(f" [!] Error en línea {i}: {e}", flush=True)
                        file_errors += 1
                        continue

                    if cleaned is None:
                        stats["filtered"] += 1
                        file_filtered += 1
                        continue

                    record["text"] = cleaned
                    fout.write(json.dumps(record) + "\n")
                    stats["written"] += 1
                    source_count += 1
                    total_written += 1
                    
                    # Forzar impresión en consola con flush=True
                    if (i + 1) % 50000 == 0:
                        print(f"    ... {i + 1:,} líneas procesadas...", flush=True)

            # También forzar el print final del archivo
            print(f"  [OK] {jsonl_file.name}: {source_count:,} aceptados | {file_filtered:,} filtrados | {file_errors:,} errores", flush=True)

    print(f"\n[MERGE] Total: {total_written:,} documentos escritos")
    print(f"        Filtrados totales: {stats['filtered']:,}")
    print(f"        Errores de parse totales: {stats['parse_error']:,}")

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
    # print("Limpieza")
    # merge_and_clean()

    print("\nDeduplicación MinHash ===")
    deduplicate(MERGED, DEDUPED)

    print("\nFin")
    finalize()