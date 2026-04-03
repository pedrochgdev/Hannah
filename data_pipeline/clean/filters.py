# reglas de Limpieza
import re, unicodedata, dataclasses
from typing import Optional


@dataclasses.dataclass
class FilterConfig:
    min_chars:       int   = 80
    max_chars:       int   = 5_000_000
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
    sample = text[:20000]
    # map(str.isdigit) se ejecuta en C puro. sum() cuenta los True como 1.
    digits = sum(map(str.isdigit, sample))
    return digits / max(len(sample), 1)

def _upper_ratio(text: str) -> float:
    sample = text[:20000]
    alphas = sum(map(str.isalpha, sample))
    if not alphas:
        return 0.0
    uppers = sum(map(str.isupper, sample))
    return uppers / alphas

def _alpha_ratio(text: str) -> float:
    sample = text[:20000]
    # Un carácter no puede ser letra y espacio a la vez, sumar ambos es matemáticamente exacto
    valid_chars = sum(map(str.isalpha, sample)) + sum(map(str.isspace, sample))
    return valid_chars / max(len(sample), 1)

def _avg_word_len(text: str) -> float:
    sample = text[:20000]
    words = sample.split()
    if not words:
        return 0.0
    # Aquí sum() usa un generador, pero sobre "palabras", no "letras" (es 100 veces menos carga)
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
    # 1. Filtro de longitud extrema
    if len(text) > CFG.max_chars * 3: 
        return None
        
    # 2. DEFENSA ANTI-SEGFAULT: Prevenir que textos gigantes sin espacios
    # colapsen las expresiones regulares (como RE_URL y RE_EMAIL)
    if len(text) > 1000 and text.count(" ") < (len(text) / 50):
        return None
        
    text = normalize(text)
    passed, _ = passes_quality(text)
    return text if passed else None

def clean_with_reason(text: str) -> tuple[Optional[str], str]:
    if len(text) > CFG.max_chars * 3:
        return None, "too_long_pre_regex"

    # La misma defensa aquí
    if len(text) > 1000 and text.count(" ") < (len(text) / 50):
        return None, "regex_hazard"

    text = normalize(text)
    passed, reason = passes_quality(text)
    return (text if passed else None), reason