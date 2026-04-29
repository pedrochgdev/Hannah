"""
core/semantic_cache.py
----------------------
Semantic Cache — similarity-based response lookup.

How it works:
  1. On each new query, compute its embedding vector.
  2. Compute cosine similarity against all stored embeddings.
  3. If the best match exceeds the threshold, return the cached response (HIT).
  4. Otherwise signal a MISS and, after the model responds, store the new
     (embedding, response) pair.

Storage backends:
  - InMemoryStore  : default, fast, process-local, lost on restart.
  - RedisStore     : persistent across restarts, suitable for production.
    Activated automatically when settings.redis_url is set.

The two backends share the CacheStore interface so they are interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings


# ── Cache entry ───────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: list[float]      # stored as plain list for serialisation
    created_at: float
    hit_count: int = 0


# ── Result objects ────────────────────────────────────────────────────

@dataclass
class CacheResult:
    hit: bool
    response: str | None = None
    similarity: float = 0.0
    matched_query: str | None = None


# ── Storage interface ─────────────────────────────────────────────────

class CacheStore(ABC):
    """Minimal key-value interface used by SemanticCache."""

    @abstractmethod
    def get_all(self) -> list[CacheEntry]:
        """Return all stored entries."""

    @abstractmethod
    def put(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """Persist an entry under key."""

    @abstractmethod
    def update_hit_count(self, key: str) -> None:
        """Increment the hit counter for an existing entry."""

    @abstractmethod
    def size(self) -> int:
        """Number of entries currently stored."""


# ── In-memory backend (default) ───────────────────────────────────────

class InMemoryStore(CacheStore):
    """
    Dict-based store with TTL and FIFO eviction.

    - TTL is enforced lazily on get_all() — expired entries are filtered out.
    - FIFO eviction: when the store reaches max_entries, the oldest key
      (insertion order, guaranteed by Python 3.7+ dicts) is removed before
      the new entry is inserted.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._store: dict[str, tuple[CacheEntry, float]] = {}
        # tuple is (entry, expires_at); expires_at == 0 means never expires
        self._max_entries = max_entries or settings.cache_max_entries

    def get_all(self) -> list[CacheEntry]:
        now = time.time()
        valid = {
            k: (e, exp)
            for k, (e, exp) in self._store.items()
            if exp == 0 or exp > now
        }
        self._store = valid
        return [e for e, _ in valid.values()]

    def put(self, key: str, entry: CacheEntry, ttl: int) -> None:
        # FIFO eviction: drop the oldest key when at capacity
        if len(self._store) >= self._max_entries and key not in self._store:
            oldest_key = next(iter(self._store))
            self._store.pop(oldest_key)

        expires_at = (time.time() + ttl) if ttl > 0 else 0
        self._store[key] = (entry, expires_at)

    def update_hit_count(self, key: str) -> None:
        if key in self._store:
            entry, exp = self._store[key]
            entry.hit_count += 1

    def size(self) -> int:
        return len(self._store)


# ── Redis backend (optional) ──────────────────────────────────────────

class RedisStore(CacheStore):
    """
    Redis-backed store. Entries are serialised as JSON.
    Requires: pip install redis
    """

    _PREFIX = "hannah:cache:"

    def __init__(self, redis_url: str) -> None:
        import redis as _redis
        self._client = _redis.from_url(redis_url, decode_responses=True)

    def get_all(self) -> list[CacheEntry]:
        keys = self._client.keys(f"{self._PREFIX}*")
        entries = []
        for key in keys:
            raw = self._client.get(key)
            if raw:
                data = json.loads(raw)
                entries.append(CacheEntry(**data))
        return entries

    def put(self, key: str, entry: CacheEntry, ttl: int) -> None:
        full_key = f"{self._PREFIX}{key}"
        payload = json.dumps({
            "query": entry.query,
            "response": entry.response,
            "embedding": entry.embedding,
            "created_at": entry.created_at,
            "hit_count": entry.hit_count,
        })
        if ttl > 0:
            self._client.setex(full_key, ttl, payload)
        else:
            self._client.set(full_key, payload)

    def update_hit_count(self, key: str) -> None:
        full_key = f"{self._PREFIX}{key}"
        raw = self._client.get(full_key)
        if raw:
            data = json.loads(raw)
            data["hit_count"] = data.get("hit_count", 0) + 1
            ttl = self._client.ttl(full_key)
            self._client.setex(full_key, max(ttl, 1), json.dumps(data))

    def size(self) -> int:
        return len(self._client.keys(f"{self._PREFIX}*"))


# ── Semantic Cache ────────────────────────────────────────────────────

class SemanticCache:
    """
    Main cache component.

    Usage:
        cache = SemanticCache()

        result = cache.lookup(user_query)
        if result.hit:
            return result.response

        response = call_model(user_query)
        cache.store(user_query, response)
    """

    def __init__(
        self,
        threshold: float | None = None,
        store: CacheStore | None = None,
    ) -> None:
        self.threshold = threshold or settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds

        # Embedding model — loaded once at startup
        self._encoder = SentenceTransformer(settings.embedding_model)

        # Storage backend — Redis if configured, otherwise in-memory
        if store is not None:
            self._store = store
        elif settings.redis_url:
            self._store = RedisStore(settings.redis_url)
        else:
            self._store = InMemoryStore()

    # ── Public API ────────────────────────────────────────────────────

    def lookup(self, query: str) -> CacheResult:
        """
        Check whether a semantically similar query has been answered before.

        Returns CacheResult with hit=True and the cached response if found,
        or hit=False if no sufficiently similar entry exists.
        """
        entries = self._store.get_all()
        if not entries:
            return CacheResult(hit=False)

        query_vec = self._embed(query)

        # Compute cosine similarities against all stored embeddings
        stored_vecs = np.array([e.embedding for e in entries])
        similarities = self._cosine_similarity_batch(query_vec, stored_vecs)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            best_entry = entries[best_idx]
            # Update hit counter asynchronously (fire-and-forget)
            cache_key = self._make_key(best_entry.query)
            self._store.update_hit_count(cache_key)

            return CacheResult(
                hit=True,
                response=best_entry.response,
                similarity=best_sim,
                matched_query=best_entry.query,
            )

        return CacheResult(hit=False, similarity=best_sim)

    def store(self, query: str, response: str) -> None:
        """Persist a query-response pair for future lookups."""
        embedding = self._embed(query).tolist()
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
        )
        key = self._make_key(query)
        self._store.put(key, entry, ttl=self._ttl)

    def size(self) -> int:
        return self._store.size()

    # ── Internal helpers ──────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return a normalised embedding vector for the given text."""
        vec = self._encoder.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _cosine_similarity_batch(
        query_vec: np.ndarray, stored_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query vector and a matrix of
        stored vectors. Both are assumed to be unit-normalised already.
        """
        return stored_vecs @ query_vec

    @staticmethod
    def _make_key(query: str) -> str:
        """Deterministic cache key from a query string."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
