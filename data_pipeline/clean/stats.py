"""
Genera un perfil estadístico de cualquier .jsonl antes y después de limpiar.
Útil para el informe académico (distribución de longitudes, fuentes, etc.)
"""

import json, pathlib, statistics, collections
from typing import Iterator

def iter_texts(path: pathlib.Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def profile(path: pathlib.Path) -> dict:
    lengths = []
    sources = collections.Counter()
    word_counts = []

    for record in iter_texts(path):
        text = record.get("text", "")
        src  = record.get("source", "unknown")
        sources[src] += 1
        lengths.append(len(text))
        word_counts.append(len(text.split()))

    if not lengths:
        return {"error": "corpus vacío"}

    total_chars = sum(lengths)
    return {
        "total_docs":      len(lengths),
        "total_chars":     total_chars,
        "total_tokens_est": total_chars // 4,      # ~4 bytes por token en inglés
        "size_gb":         total_chars / 1e9,
        "char_len": {
            "min":    min(lengths),
            "max":    max(lengths),
            "mean":   round(statistics.mean(lengths)),
            "median": round(statistics.median(lengths)),
            "p10":    round(sorted(lengths)[len(lengths) // 10]),
            "p90":    round(sorted(lengths)[int(len(lengths) * 0.9)]),
        },
        "word_count": {
            "mean":   round(statistics.mean(word_counts)),
            "median": round(statistics.median(word_counts)),
        },
        "sources": dict(sources.most_common()),
    }

def print_profile(path: pathlib.Path):
    p = profile(path)
    print(f"\n{'─'*50}")
    print(f"  Corpus: {path.name}")
    print(f"  Documentos:   {p['total_docs']:>12,}")
    print(f"  Tamaño:       {p['size_gb']:>11.2f} GB")
    print(f"  Tokens est.:  {p['total_tokens_est']:>12,}")
    print(f"  Longitud media: {p['char_len']['mean']:>9,} chars")
    print(f"  Longitud p10/p90: {p['char_len']['p10']:,} / {p['char_len']['p90']:,}")
    print(f"\n  Documentos por fuente:")
    for src, count in p["sources"].items():
        pct = count / p["total_docs"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {src:<25} {count:>8,}  {pct:5.1f}%  {bar}")
    print(f"{'─'*50}\n")