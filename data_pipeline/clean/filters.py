# reglas de Limpieza
import re, unicodedata, dataclasses
from typing import Optional


@dataclasses.dataclass
class FilterConfig:
    min_chars:       int   = 80
    max_chars:       int   = 8_000
    max_digit_ratio: float = 0.15
    max_upper_ratio: float = 0.40
    min_avg_word:    float = 3.5
    max_avg_word:    float = 12.0   # filtra URLs pegadas sin espacios
    min_alpha_ratio: float = 0.60   # al menos 60% caracteres alfabéticos
    max_rep_ratio:   float = 0.20   # máximo 20% líneas repetidas (boilerplate)

CFG = FilterConfig()

# Patrones compilados una sola vez
RE_URL      = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL    = re.compile(r"\S+@\S+\.\S+")
RE_CONTROL  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_SPACES   = re.compile(r" {2,}")
RE_NEWLINES = re.compile(r"\n{3,}")
RE_HTML_TAG = re.compile(r"<[^>]{1,60}>")         # tags HTML residuales
RE_BULLETS  = re.compile(r"^[\s\-\*\•]{1,4}\s",   # listas de bullets
               re.MULTILINE)

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    text = RE_URL.sub("[URL]", text)    # reemplazar, no borrar (mantiene estructura)
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_SPACES.sub(" ", text)
    text = RE_NEWLINES.sub("\n\n", text)
    return text.strip()

def _digit_ratio(text: str) -> float:
    return sum(c.isdigit() for c in text) / max(len(text), 1)

def _upper_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(c.isupper() for c in alpha) / len(alpha)

def _alpha_ratio(text: str) -> float:
    return sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)

def _avg_word_len(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)

def _rep_ratio(text: str) -> float:
    """Ratio de líneas duplicadas — detecta boilerplate repetido."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return 1 - (len(set(lines)) / len(lines))

def passes_quality(text: str) -> tuple[bool, str]:
    """
    Devuelve (True, 'ok') o (False, 'razon_del_rechazo').
    El motivo de rechazo permite generar estadísticas por regla.
    """
    n = len(text)

    if n < CFG.min_chars:                     return False, "too_short"
    if n > CFG.max_chars:                     return False, "too_long"
    if _digit_ratio(text) > CFG.max_digit_ratio:
                                              return False, "digit_heavy"
    if _upper_ratio(text) > CFG.max_upper_ratio:
                                              return False, "upper_heavy"
    if _alpha_ratio(text) < CFG.min_alpha_ratio:
                                              return False, "low_alpha"

    awl = _avg_word_len(text)
    if awl < CFG.min_avg_word:               return False, "word_too_short"
    if awl > CFG.max_avg_word:               return False, "word_too_long"
    if _rep_ratio(text) > CFG.max_rep_ratio: return False, "repetitive"

    return True, "ok"

def clean(text: str) -> Optional[str]:
    """Normalizar → filtrar → devolver texto limpio o None."""
    text = normalize(text)
    passed, _ = passes_quality(text)
    return text if passed else None

def clean_with_reason(text: str) -> tuple[Optional[str], str]:
    """Igual que clean() pero devuelve el motivo de rechazo para logging."""
    text = normalize(text)
    passed, reason = passes_quality(text)
    return (text if passed else None), reason