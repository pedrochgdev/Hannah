"""
Pipeline principal. Usa multiprocessing para aprovechar todos los cores
mientras la GPU no está en uso (mes 1).

Uso:
    python -m clean.pipeline \
        --input  raw/ \
        --output corpus_merged.jsonl \
        --workers 8
"""

import json, pathlib, argparse, collections, time
import multiprocessing as mp
from clean.filters import clean_with_reason

# ── Worker: procesa un chunk de líneas ───────────────────────────────────────
def _worker(args):
    """Ejecutado en cada proceso hijo."""
    lines, source_hint = args
    results  = []
    rejected = collections.Counter()

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            rejected["parse_error"] += 1
            continue

        text = record.get("text", "")
        cleaned, reason = clean_with_reason(text)

        if cleaned is None:
            rejected[reason] += 1
        else:
            record["text"] = cleaned
            results.append(json.dumps(record))

    return results, rejected

# ── Chunker ───────────────────────────────────────────────────────────────────
def _chunk_lines(lines: list, n: int):
    """Divide una lista de líneas en n chunks aproximadamente iguales."""
    size = max(1, len(lines) // n)
    for i in range(0, len(lines), size):
        yield lines[i:i + size]

# ── Pipeline principal ────────────────────────────────────────────────────────
def run(input_dir: pathlib.Path,
        output_path: pathlib.Path,
        workers: int = 8,
        log_every: int = 100_000):

    all_files = sorted(input_dir.rglob("*.jsonl"))
    print(f"[PIPELINE] {len(all_files)} archivos fuente, {workers} workers\n")

    total_stats = collections.Counter()
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for file in all_files:
            print(f"  Procesando {file.name}...")
            lines = file.read_text(encoding="utf-8").splitlines()

            # Distribuir líneas entre workers
            chunks = list(_chunk_lines(lines, workers))
            tasks  = [(chunk, file.stem) for chunk in chunks]

            with mp.Pool(workers) as pool:
                for results, rejected in pool.imap_unordered(_worker, tasks):
                    for record_str in results:
                        fout.write(record_str + "\n")
                    total_stats["written"]  += len(results)
                    total_stats["rejected"] += sum(rejected.values())
                    for reason, count in rejected.items():
                        total_stats[f"rej_{reason}"] += count

            total_processed = total_stats["written"] + total_stats["rejected"]
            pct_kept = total_stats["written"] / max(total_processed, 1) * 100
            print(f"    {len(lines):>10,} líneas → "
                  f"{total_stats['written']:,} aceptadas ({pct_kept:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Output:          {output_path}")
    print(f"  Tiempo total:    {elapsed/60:.1f} min")
    print(f"  Escritos:        {total_stats['written']:,}")
    print(f"  Rechazados:      {total_stats['rejected']:,}")
    print(f"\n  Rechazos por regla:")

    reject_rules = {k: v for k, v in total_stats.items() if k.startswith("rej_")}
    for rule, count in sorted(reject_rules.items(), key=lambda x: -x[1]):
        print(f"    {rule:<30} {count:>8,}")
    print(f"{'='*55}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=pathlib.Path, default=pathlib.Path("raw"))
    parser.add_argument("--output",  type=pathlib.Path, default=pathlib.Path("corpus_merged.jsonl"))
    parser.add_argument("--workers", type=int,          default=8)
    args = parser.parse_args()
    run(args.input, args.output, args.workers)