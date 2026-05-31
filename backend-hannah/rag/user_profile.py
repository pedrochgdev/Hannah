# rag/user_profile.py
"""
UserProfile — Perfil de sesión del usuario.

Captura hechos que el usuario menciona en la conversación
(nombre, trabajo, hobbies, etc.) y los inyecta siempre en [MEMORY],
sin depender del score del RAG vectorial.

El RAG vectorial busca conocimiento de Hannah.
Este módulo guarda conocimiento del usuario.

Uso:
    profile = UserProfile()
    profile.update("My name is Gabriel, remember it", [])
    profile.update("I work as an engineer", [])
    memory_str = profile.to_memory_string()
    # → "[MEMORY]User's name is Gabriel. User works as an engineer.[/MEMORY]"
"""

import re
from typing import Optional

# ── Patrones de extracción ─────────────────────────────────────────────
# Cada patrón tiene: regex, clave en el perfil, y cómo formatear el hecho

NAME_PATTERNS = [
    r"my name is ([A-Za-z]+)",
    r"call me ([A-Za-z]+)",
    r"i(?:'m| am) ([A-Za-z]+),?\s+(?:nice to meet|remember|that's me)",
    r"name(?:'s| is) ([A-Za-z]+)",
    r"people call me ([A-Za-z]+)",
    r"(?:friends?|everyone) calls? me ([A-Za-z]+)",
    r"i go by ([A-Za-z]+)",
]

JOB_PATTERNS = [
    r"i(?:'m| am) (?:a |an )?([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"i work (?:as |as a |as an )?([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"my (?:job|profession|occupation) is ([a-z][\w\s]+?)(?:\.|,|$)",
    r"i study ([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"i(?:'m| am) studying ([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"i(?:'m| am) a student (?:of )?([a-z][\w\s]+?)(?:\.|,| and| but|$)",
]

HOBBY_PATTERNS = [
    r"i (?:love|like|enjoy|really like|really love) ([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"my hobby is ([a-z][\w\s]+?)(?:\.|,|$)",
    r"i(?:'m| am) into ([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"i(?:'m| am) passionate about ([a-z][\w\s]+?)(?:\.|,| and| but|$)",
    r"i spend (?:a lot of )?time ([a-z][\w\s]+?ing)(?:\.|,| and| but|$)",
]

LOCATION_PATTERNS = [
    r"i(?:'m| am) from ([A-Za-z][\w\s,]+?)(?:\.|,| and| but|$)",
    r"i live in ([A-Za-z][\w\s,]+?)(?:\.|,| and| but|$)",
    r"i(?:'m| am) based in ([A-Za-z][\w\s,]+?)(?:\.|,| and| but|$)",
    r"i(?:'m| am) in ([A-Za-z][\w\s,]+?) (?:right now|currently|atm)",
]

AGE_PATTERNS = [
    r"i(?:'m| am) (\d{1,2}) years? old",
    r"my age is (\d{1,2})",
    r"i(?:'m| am) (\d{1,2}),? (?:years old|yo)",
    r"turned (\d{1,2}) (?:last|this)",
]

# Palabras que NO son trabajos reales — evitar falsos positivos de "I'm tired"
JOB_BLACKLIST = {
    "tired", "bored", "fine", "okay", "ok", "good", "bad", "sad",
    "happy", "here", "home", "back", "out", "away", "busy", "free",
    "ready", "done", "lost", "stuck", "confused", "excited", "nervous",
    "a little", "not sure", "just", "gonna", "going",
}

# Palabras que NO son hobbies reales
HOBBY_BLACKLIST = {
    "you", "her", "him", "them", "it", "this", "that", "the", "a",
    "an", "your", "my", "our", "their", "what", "how", "when", "where",
    "to", "of", "in", "on", "at", "for", "with", "about",
}


class UserProfile:
    """
    Perfil de sesión del usuario. Se crea uno por conversación.
    Captura hechos mencionados y los inyecta en [MEMORY].
    """

    def __init__(self):
        self._facts: dict[str, str] = {}
        # Orden de inserción para el string final
        self._order: list[str] = []

    def update(self, user_msg: str, history: list) -> list[str]:
        """
        Analiza el mensaje del usuario y extrae hechos nuevos.

        Args:
            user_msg: Mensaje actual del usuario.
            history:  Lista de turnos anteriores (no se usa actualmente,
                      reservado para extracción multi-turno futura).

        Returns:
            Lista de claves nuevas que se extrajeron (vacía si nada nuevo).
        """
        msg_lower = user_msg.lower().strip()
        new_keys = []

        # ── Nombre ────────────────────────────────────────────────────
        name = self._extract_name(msg_lower)
        if name and "name" not in self._facts:
            self._set("name", f"User's name is {name.capitalize()}")
            new_keys.append("name")

        # ── Trabajo / estudio ─────────────────────────────────────────
        job = self._extract_job(msg_lower)
        if job and "job" not in self._facts:
            self._set("job", f"User works as {job}")
            new_keys.append("job")

        # ── Hobby / interés ───────────────────────────────────────────
        hobby = self._extract_hobby(msg_lower)
        if hobby and "hobby" not in self._facts:
            self._set("hobby", f"User enjoys {hobby}")
            new_keys.append("hobby")

        # ── Ubicación ─────────────────────────────────────────────────
        location = self._extract_location(msg_lower)
        if location and "location" not in self._facts:
            self._set("location", f"User is from {location.title()}")
            new_keys.append("location")

        # ── Edad ──────────────────────────────────────────────────────
        age = self._extract_age(msg_lower)
        if age and "age" not in self._facts:
            self._set("age", f"User is {age} years old")
            new_keys.append("age")

        return new_keys

    def to_memory_string(self) -> str:
        """
        Retorna los hechos del usuario formateados para inyectar en [MEMORY].

        Returns:
            String vacío si no hay hechos.
            "[MEMORY]User's name is Gabriel. User works as engineer.[/MEMORY]"
            si hay hechos.
        """
        if not self._facts:
            return ""

        facts_text = " ".join(self._facts[k] + "." for k in self._order if k in self._facts)
        return f"[MEMORY]{facts_text}[/MEMORY]"

    def get(self, key: str) -> Optional[str]:
        """Retorna el valor de un hecho específico o None."""
        return self._facts.get(key)

    def has_facts(self) -> bool:
        """True si hay al menos un hecho almacenado."""
        return len(self._facts) > 0

    def clear(self):
        """Limpia el perfil (nueva sesión)."""
        self._facts.clear()
        self._order.clear()

    # ── Internos ──────────────────────────────────────────────────────

    def _set(self, key: str, value: str):
        if key not in self._facts:
            self._order.append(key)
        self._facts[key] = value

    def _extract_name(self, msg: str) -> Optional[str]:
        for pattern in NAME_PATTERNS:
            m = re.search(pattern, msg)
            if m:
                candidate = m.group(1).strip()
                # Filtrar palabras que no son nombres propios
                if len(candidate) >= 2 and candidate.isalpha():
                    # Ignorar palabras funcionales
                    if candidate.lower() not in {"the", "a", "an", "my", "your"}:
                        return candidate
        return None

    def _extract_job(self, msg: str) -> Optional[str]:
        for pattern in JOB_PATTERNS:
            m = re.search(pattern, msg)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                words = candidate.split()
                if words and words[0] not in JOB_BLACKLIST and len(candidate) > 3:
                    return candidate
        return None

    def _extract_hobby(self, msg: str) -> Optional[str]:
        for pattern in HOBBY_PATTERNS:
            m = re.search(pattern, msg)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                words = candidate.split()
                if words and words[0] not in HOBBY_BLACKLIST and len(candidate) > 2:
                    return candidate
        return None

    def _extract_location(self, msg: str) -> Optional[str]:
        for pattern in LOCATION_PATTERNS:
            m = re.search(pattern, msg)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                if len(candidate) > 2:
                    return candidate
        return None

    def _extract_age(self, msg: str) -> Optional[str]:
        for pattern in AGE_PATTERNS:
            m = re.search(pattern, msg)
            if m:
                age = int(m.group(1))
                if 5 <= age <= 120:  # rango razonable
                    return str(age)
        return None

    def __repr__(self):
        return f"UserProfile({self._facts})"
