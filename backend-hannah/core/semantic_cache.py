"""
core/semantic_cache.py
----------------------
Semantic Cache con soporte de tenant_id para aislamiento multi-sesion.
Backends: InMemoryStore (default) o RedisStore (si REDIS_URL esta configurado).
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


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: list[float]
    created_at: float
    hit_count: int = 0


@dataclass
class CacheResult:
    hit: bool
    response: str | None = None
    similarity: float = 0.0
    matched_query: str | None = None


class CacheStore(ABC):
    @abstractmethod
    def get_all(self, tenant_id: str = "local") -> list[CacheEntry]: ...
    @abstractmethod
    def put(self, key: str, entry: CacheEntry, ttl: int, tenant_id: str = "local") -> None: ...
    @abstractmethod
    def update_hit_count(self, key: str, tenant_id: str = "local") -> None: ...
    @abstractmethod
    def size(self, tenant_id: str = "local") -> int: ...


class InMemoryStore(CacheStore):
    def __init__(self, max_entries: int | None = None) -> None:
        self._store: dict[str, tuple[CacheEntry, float]] = {}
        self._max_entries = max_entries or settings.cache_max_entries

    def get_all(self, tenant_id: str = "local") -> list[CacheEntry]:
        now = time.time()
        valid = {k: (e, exp) for k, (e, exp) in self._store.items() if exp == 0 or exp > now}
        self._store = valid
        return [e for e, _ in valid.values()]

    def put(self, key: str, entry: CacheEntry, ttl: int, tenant_id: str = "local") -> None:
        if len(self._store) >= self._max_entries and key not in self._store:
            self._store.pop(next(iter(self._store)))
        expires_at = (time.time() + ttl) if ttl > 0 else 0
        self._store[key] = (entry, expires_at)

    def update_hit_count(self, key: str, tenant_id: str = "local") -> None:
        if key in self._store:
            self._store[key][0].hit_count += 1

    def size(self, tenant_id: str = "local") -> int:
        return len(self._store)


class RedisStore(CacheStore):
    _PREFIX = "hannah:cache:"

    def __init__(self, redis_url: str) -> None:
        import redis as _redis
        self._client = _redis.from_url(redis_url, decode_responses=True)

    def _full_key(self, tenant_id: str, key: str) -> str:
        return f"{self._PREFIX}{tenant_id}:{key}"

    def get_all(self, tenant_id: str = "local") -> list[CacheEntry]:
        keys = self._client.keys(f"{self._PREFIX}{tenant_id}:*")
        entries = []
        for key in keys:
            raw = self._client.get(key)
            if raw:
                entries.append(CacheEntry(**json.loads(raw)))
        return entries

    def put(self, key: str, entry: CacheEntry, ttl: int, tenant_id: str = "local") -> None:
        full_key = self._full_key(tenant_id, key)
        payload = json.dumps({
            "query": entry.query, "response": entry.response,
            "embedding": entry.embedding, "created_at": entry.created_at,
            "hit_count": entry.hit_count,
        })
        if ttl > 0:
            self._client.setex(full_key, ttl, payload)
        else:
            self._client.set(full_key, payload)

    def update_hit_count(self, key: str, tenant_id: str = "local") -> None:
        full_key = self._full_key(tenant_id, key)
        raw = self._client.get(full_key)
        if raw:
            data = json.loads(raw)
            data["hit_count"] = data.get("hit_count", 0) + 1
            ttl = self._client.ttl(full_key)
            self._client.setex(full_key, max(ttl, 1), json.dumps(data))

    def size(self, tenant_id: str = "local") -> int:
        return len(self._client.keys(f"{self._PREFIX}{tenant_id}:*"))


class SemanticCache:
    def __init__(self, threshold: float | None = None, store: CacheStore | None = None) -> None:
        self.threshold = threshold or settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds
        self._encoder = SentenceTransformer(settings.embedding_model)
        if store is not None:
            self._store = store
        elif settings.redis_url:
            self._store = RedisStore(settings.redis_url)
        else:
            self._store = InMemoryStore()

    def lookup(self, query: str, tenant_id: str = "local") -> CacheResult:
        entries = self._store.get_all(tenant_id)
        if not entries:
            return CacheResult(hit=False)
        query_vec = self._embed(query)
        stored_vecs = np.array([e.embedding for e in entries])
        similarities = self._cosine_similarity_batch(query_vec, stored_vecs)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim >= self.threshold:
            best_entry = entries[best_idx]
            self._store.update_hit_count(self._make_key(best_entry.query), tenant_id)
            return CacheResult(hit=True, response=best_entry.response,
                               similarity=best_sim, matched_query=best_entry.query)
        return CacheResult(hit=False, similarity=best_sim)

    def store(self, query: str, response: str, tenant_id: str = "local") -> None:
        embedding = self._embed(query).tolist()
        entry = CacheEntry(query=query, response=response,
                           embedding=embedding, created_at=time.time())
        self._store.put(self._make_key(query), entry, ttl=self._ttl, tenant_id=tenant_id)

    def size(self, tenant_id: str = "local") -> int:
        return self._store.size(tenant_id)

    def _embed(self, text: str) -> np.ndarray:
        vec = self._encoder.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _cosine_similarity_batch(query_vec: np.ndarray, stored_vecs: np.ndarray) -> np.ndarray:
        return stored_vecs @ query_vec

    @staticmethod
    def _make_key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]
