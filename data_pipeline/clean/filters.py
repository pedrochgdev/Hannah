# reglas de Limpieza
import re, unicodedata

# ── Constantes ────────────────────────────────────────────────────────────────
MIN_CHARS      = 20      # descartar textos demasiado cortos
MAX_CHARS      = 8000    # descartar textos anómalamente largos (posible spam)
MAX_DIGIT_RATIO = 0.15   # máximo 15% de dígitos (filtra tablas, código, etc.)
MAX_UPPER_RATIO = 0.40   # máximo 40% de mayúsculas (filtra ALL-CAPS spam)
MIN_WORD_LEN   = 3.5     # longitud media mínima de palabra (filtra basura)

# Detectar URLs, emails, caracteres de control
RE_URL     = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL   = re.compile(r"\S+@\S+\.\S+")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_MULTI_SPACE = re.compile(r" {2,}")
RE_MULTI_NL    = re.compile(r"\n{3,}")

def normalize(text: str) -> str:
    """Normalización unicode + limpieza de espacios."""
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_URL.sub("", text)
    text = RE_EMAIL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NL.sub("\n\n", text)
    return text.strip()

def passes_quality(text: str) -> bool:
    """
    Devuelve True si el texto pasa todos los filtros de calidad.
    Devuelve False + razón para logging si falla alguno.
    """
    n = len(text)
    if n < MIN_CHARS:
        return False, "too_short"
    if n > MAX_CHARS:
        return False, "too_long"

    # Ratio de dígitos
    digit_ratio = sum(c.isdigit() for c in text) / n
    if digit_ratio > MAX_DIGIT_RATIO:
        return False, "too_many_digits"

    # Ratio de mayúsculas
    alpha = [c for c in text if c.isalpha()]
    if alpha:
        upper_ratio = sum(c.isupper() for c in alpha) / len(alpha)
        if upper_ratio > MAX_UPPER_RATIO:
            return False, "too_many_uppercase"

    # Longitud media de palabras (filtra tokens sin sentido)
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < MIN_WORD_LEN:
            return False, "avg_word_too_short"

    return True, "ok"

def clean(text: str) -> str | None:
    """Pipeline completo: normalizar → filtrar → devolver o None."""
    text = normalize(text)
    passed, reason = passes_quality(text)
    if not passed:
        return None
    return text