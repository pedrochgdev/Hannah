"""
core/token_handler.py
---------------------
Token Sequence Handler

Responsibilities:
  1. Maintain per-session conversation history (in memory or external store).
  2. Build the context string that gets sent to downstream models.
  3. Truncate history when the context would exceed the token budget.

Design notes:
  - Sessions are keyed by a string session_id (UUID from the web app).
  - History is a list of Turn objects ordered oldest → newest.
  - Token counting is approximate (chars / chars_per_token) to avoid
    loading a full tokenizer in the backend. Exact counting can be
    plugged in later via the count_tokens() override.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from config import settings


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class Turn:
    """A single user/assistant exchange."""
    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """All state associated with one conversation session."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append(Turn(user=user, assistant=assistant))
        self.last_active = time.time()

    def recent_turns(self, n: int) -> list[Turn]:
        """Return the n most recent turns."""
        return self.turns[-n:] if len(self.turns) >= n else self.turns[:]


# ── Token Handler ─────────────────────────────────────────────────────

class TokenHandler:
    """
    Manages conversation sessions and builds truncated context windows.

    Usage:
        handler = TokenHandler()
        context = handler.build_context(session_id, new_user_prompt)
        # ... get response from model ...
        handler.record_turn(session_id, new_user_prompt, model_response)
    """

    def __init__(
        self,
        max_history_turns: int | None = None,
        max_context_tokens: int | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_history_turns:
            Hard cap on how many turns to consider (oldest are dropped first).
        max_context_tokens:
            Soft cap on total tokens in the assembled context.
        count_tokens:
            Optional callable(text) -> int. Defaults to the chars/token
            approximation defined in settings. Swap in a real tokenizer here.
        """
        self._sessions: dict[str, Session] = {}

        self.max_history_turns = max_history_turns or settings.max_history_turns
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens

        # Token counting strategy — default is a fast approximation
        self._count_tokens: Callable[[str], int] = count_tokens or self._approx_tokens

    # ── Public API ────────────────────────────────────────────────────

    def get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def build_context(self, session_id: str, current_prompt: str) -> dict:
        """
        Assemble the context that will be sent to a model.

        Returns a dict with:
          - "prompt"       : the current user message
          - "history"      : list of {"user": ..., "assistant": ...} dicts
                             (truncated to fit within token budget)
          - "token_count"  : estimated total tokens
          - "turns_used"   : how many history turns were included
        """
        session = self.get_or_create_session(session_id)
        candidate_turns = session.recent_turns(self.max_history_turns)

        # Budget: reserve tokens for the current prompt + a response headroom
        prompt_tokens = self._count_tokens(current_prompt)
        # Reserve 20 % of the budget for the model's response
        history_budget = int(self.max_context_tokens * 0.8) - prompt_tokens

        # Walk history from most-recent to oldest, include as many as fit
        included: list[Turn] = []
        used_tokens = 0

        for turn in reversed(candidate_turns):
            turn_tokens = (
                self._count_tokens(turn.user)
                + self._count_tokens(turn.assistant)
            )
            if used_tokens + turn_tokens > history_budget:
                break
            included.append(turn)
            used_tokens += turn_tokens

        # Restore chronological order
        included.reverse()

        history_dicts = [
            {"user": t.user, "assistant": t.assistant}
            for t in included
        ]

        return {
            "prompt": current_prompt,
            "history": history_dicts,
            "token_count": prompt_tokens + used_tokens,
            "turns_used": len(included),
        }

    def record_turn(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Persist a completed turn into session history."""
        session = self.get_or_create_session(session_id)
        session.add_turn(user=user_message, assistant=assistant_message)

    def get_history(self, session_id: str) -> list[dict]:
        """Return full history for a session as a list of dicts."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [{"user": t.user, "assistant": t.assistant} for t in session.turns]

    def clear_session(self, session_id: str) -> None:
        """Remove all history for a session."""
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """
        Fast approximation: 1 token ~ settings.chars_per_token characters.
        Accurate enough for budget decisions without loading a tokenizer.
        """
        return max(1, int(len(text) / settings.chars_per_token))
