"""
Valida el corpus final antes de pasarlo al tokenizador.
Si algún check falla, el entrenamiento no debe iniciar.

Uso:
    python -m clean.validate corpus_final.jsonl
"""

import json, sys, pathlib
from clean.stats import profile

# Umbrales mínimos para considerar el corpus listo
REQUIREMENTS = {
    "min_docs":       500_000,
    "min_size_gb":    10.0,
    "min_tokens_est": 7_000_000_000,   # 7B tokens
    "max_sources":    3,               # al menos 3 fuentes distintas
}

def validate(path: pathlib.Path) -> bool:
    print(f"Validando {path}...")
    p = profile(path)

    checks = []

    def check(name, value, target, mode="min"):
        passed = (value >= target) if mode == "min" else (value <= target)
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {value:,}  (req: {'≥' if mode=='min' else '≤'} {target:,})")
        checks.append(passed)

    check("Total documentos",  p["total_docs"],       REQUIREMENTS["min_docs"])
    check("Tamaño GB",         int(p["size_gb"]*1e9), int(REQUIREMENTS["min_size_gb"]*1e9))
    check("Tokens estimados",  p["total_tokens_est"], REQUIREMENTS["min_tokens_est"])
    check("Fuentes distintas", len(p["sources"]),     REQUIREMENTS["max_sources"])

    # Check adicional: ninguna fuente domina más del 60% del corpus
    total = p["total_docs"]
    dominant = max(p["sources"].values()) / total
    dom_ok = dominant < 0.60
    print(f"  {'✓' if dom_ok else '✗'} Fuente dominante: {dominant*100:.1f}%  (req: < 60%)")
    checks.append(dom_ok)

    all_passed = all(checks)
    print(f"\n  {'LISTO PARA ENTRENAR' if all_passed else 'CORPUS INSUFICIENTE — revisar fuentes'}")
    return all_passed

if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("corpus_final.jsonl")
    ok   = validate(path)
    sys.exit(0 if ok else 1)