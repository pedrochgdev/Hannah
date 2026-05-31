"""
core/token_handler.py
---------------------
Token Sequence Handler — respaldado por Redis para soporte multi-sesion.
Cada sesion se almacena como JSON en Redis con TTL configurable.
Interfaz publica identica a la version en memoria original.
"""
from __future__ import annotations

import json

import redis as _redis

from config import settings


class TokenHandler:
    """
    Gestiona historiales de conversacion por sesion usando Redis.
    """

    def __init__(self) -> None:
        self.max_history_turns = settings.max_history_turns
        self.max_context_tokens = settings.max_context_tokens
        self._redis = _redis.from_url(
            settings.redis_url or "redis://localhost:6379",
            decode_responses=True,
        )
        self._ttl = settings.session_ttl_seconds

    # ── Helpers ───────────────────────────────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:turns"

    def _load(self, session_id: str) -> list[dict]:
        raw = self._redis.get(self._key(session_id))
        return json.loads(raw) if raw else []

    def _save(self, session_id: str, turns: list[dict]) -> None:
        self._redis.set(self._key(session_id), json.dumps(turns), ex=self._ttl)

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, int(len(text) / settings.chars_per_token))

    # ── Public API ────────────────────────────────────────────────────

    def build_context(self, session_id: str, current_prompt: str) -> dict:
        turns = self._load(session_id)
        # Renovar TTL en cada acceso activo
        if turns:
            self._redis.expire(self._key(session_id), self._ttl)

        candidate = turns[-self.max_history_turns:] if len(turns) > self.max_history_turns else turns
        prompt_tokens = self._approx_tokens(current_prompt)
        history_budget = int(self.max_context_tokens * 0.8) - prompt_tokens

        included: list[dict] = []
        used_tokens = 0
        for turn in reversed(candidate):
            t = self._approx_tokens(turn.get("user", "")) + self._approx_tokens(turn.get("assistant", ""))
            if used_tokens + t > history_budget:
                break
            included.append(turn)
            used_tokens += t
        included.reverse()

        return {
            "prompt":      current_prompt,
            "history":     included,
            "token_count": prompt_tokens + used_tokens,
            "turns_used":  len(included),
        }

    def record_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        turns = self._load(session_id)
        turns.append({"user": user_message, "assistant": assistant_message})
        if len(turns) > self.max_history_turns:
            turns = turns[-self.max_history_turns:]
        self._save(session_id, turns)

    def get_history(self, session_id: str) -> list[dict]:
        return self._load(session_id)

    def clear_session(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def session_count(self) -> int:
        keys = self._redis.keys("session:*:turns")
        return len(keys)
