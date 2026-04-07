"""
Verifica que el tokenizador entrenado funciona correctamente
y calcula métricas de calidad para el informe académico.

Uso:
    python tokenizer/validate.py --tok tokenizer/hannah_tok
"""

import argparse, pathlib, json
from transformers import AutoTokenizer

# Frases de prueba — cubren los casos de uso del proyecto
TEST_SENTENCES = [
    # Diálogo romántico (core del proyecto)
    "I've been thinking about you all day, and I couldn't help but smile.",
    "You always know exactly what to say when I need to hear it most.",
    "Tell me something you've never told anyone else.",

    # Tokens especiales del chat template
    "[SYS] You are a warm and caring companion. [/SYS] [USR] How are you? [/USR] [ASS]",

    # Texto literario (del corpus de Gutenberg)
    "It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife.",

    # Texto informativo (Wikipedia)
    "The water cycle, also known as the hydrological cycle, describes the continuous movement of water on Earth.",

    # Números y caracteres especiales
    "I was born on March 15, 1998, and today is April 2, 2026.",

    # Texto corto
    "Hello!",
    "I love you.",
]

def validate(tok_path: pathlib.Path):
    print(f"Cargando tokenizador desde {tok_path}...\n")
    tok = AutoTokenizer.from_pretrained(str(tok_path))

    print(f"{'─'*60}")
    print(f"  Vocab size:     {tok.vocab_size:,}")
    print(f"  Tokens especiales:")
    for name, token in [
        ("pad", tok.pad_token), ("unk", tok.unk_token),
        ("bos", tok.bos_token), ("eos", tok.eos_token),
    ]:
        tid = tok.convert_tokens_to_ids(token)
        print(f"    {name:<6} = '{token}' (id={tid})")
    print(f"{'─'*60}\n")

    # ── Test 1: tokens especiales del proyecto ───────────────────────────────
    print("Test 1: tokens especiales del chat template")
    for special in ["[SYS]", "[/SYS]", "[USR]", "[/USR]", "[ASS]", "[/ASS]",
                    "[MEMORY]", "[/MEMORY]"]:
        tid = tok.convert_tokens_to_ids(special)
        status = "OK" if tid != tok.unk_token_id else "FALLO — es <unk>"
        print(f"  {special:<12} id={tid:<6} {status}")

    # ── Test 2: encode/decode roundtrip ─────────────────────────────────────
    print("\nTest 2: encode → decode (debe recuperar el texto original)")
    all_ok = True
    for sentence in TEST_SENTENCES[:4]:
        ids      = tok.encode(sentence)
        decoded  = tok.decode(ids, skip_special_tokens=False)
        # Comparar ignorando espacios extra alrededor de tokens especiales
        ok = sentence.strip() in decoded.replace("  ", " ").strip()
        status = "OK" if ok else "DIFF"
        if not ok:
            all_ok = False
        print(f"  [{status}] '{sentence[:50]}...' → {len(ids)} tokens")
        if not ok:
            print(f"       decoded: '{decoded[:80]}'")

    # ── Test 3: token efficiency (métrica académica) ─────────────────────────
    print("\nTest 3: token efficiency (tokens por carácter — menor es mejor)")
    print("  Referencia: GPT-2 tokenizer ≈ 0.28 tok/char en inglés")
    efficiencies = []
    for sentence in TEST_SENTENCES:
        ids  = tok.encode(sentence, add_special_tokens=False)
        tpc  = len(ids) / max(len(sentence), 1)
        efficiencies.append(tpc)
        print(f"  {tpc:.3f} tok/char — '{sentence[:55]}'")

    avg_eff = sum(efficiencies) / len(efficiencies)
    print(f"\n  Promedio: {avg_eff:.3f} tok/char")
    if avg_eff < 0.32:
        print("  RESULTADO: buena compresión (comparable a GPT-2 o mejor)")
    elif avg_eff < 0.40:
        print("  RESULTADO: compresión aceptable")
    else:
        print("  RESULTADO: compresión baja — revisar el corpus de entrenamiento")

    # ── Test 4: context window real ──────────────────────────────────────────
    print("\nTest 4: cuántos caracteres caben en 1024 tokens (promedio)")
    chars_per_1024 = int(1024 / avg_eff)
    print(f"  ~{chars_per_1024:,} caracteres = ~{chars_per_1024//5} palabras por contexto")

    print(f"\n{'─'*60}")
    print(f"  {'VALIDACIÓN COMPLETA' if all_ok else 'HAY PROBLEMAS — revisar arriba'}")
    print(f"{'─'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok", type=pathlib.Path,
                        default=pathlib.Path("tokenizer/hannah_tok"))
    args = parser.parse_args()
    validate(args.tok)