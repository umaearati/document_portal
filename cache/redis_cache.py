"""
Redis caching layer for Document Portal.

Caches:
  1. LLM responses     — identical question + session → skip the chain entirely
  2. FAISS embeddings  — reuse computed embeddings across restarts (future)
  3. Session metadata  — fast session lookup without hitting PostgreSQL

Cache key format:
    portal:query:{session_id}:{question_hash}   → cached answer (str)
    portal:session:{session_id}                 → session metadata (JSON)

TTLs (configurable via env vars):
    REDIS_QUERY_TTL_SECONDS    default 3600  (1 hour)
    REDIS_SESSION_TTL_SECONDS  default 86400 (24 hours)

Environment variables:
    REDIS_URL   — e.g. redis://default:password@redis-12345.c1.eu-west-1-2.ec2.cloud.redislabs.com:12345
                  defaults to redis://localhost:6379

Graceful degradation: if Redis is unreachable, every method is a no-op
and the app continues as before (just without caching).

Usage:
    from cache.redis_cache import get_cache

    cache = get_cache()

    # Before calling RAG chain:
    cached = cache.get_query(session_id, question)
    if cached:
        return cached

    # After getting answer:
    cache.set_query(session_id, question, answer)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from logger import GLOBAL_LOGGER as log


def _connect_redis():
    try:
        import redis  # type: ignore
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(url, decode_responses=True, socket_timeout=2)
        client.ping()
        log.info("Redis connected", url=url.split("@")[-1])  # hide password
        return client
    except ImportError:
        log.warning("redis package not installed — caching disabled")
        return None
    except Exception as exc:
        log.warning("Redis connection failed — caching disabled", error=str(exc))
        return None


class RedisCache:
    """
    Thin wrapper around Redis with automatic serialisation and TTL management.
    All methods degrade gracefully if Redis is unavailable.
    """

    QUERY_PREFIX   = "portal:query"
    SESSION_PREFIX = "portal:session"

    def __init__(self):
        self._client = _connect_redis()
        self._query_ttl   = int(os.getenv("REDIS_QUERY_TTL_SECONDS",   "3600"))
        self._session_ttl = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "86400"))

    # ------------------------------------------------------------------
    # Query cache  (most impactful — saves full LLM round-trip)
    # ------------------------------------------------------------------

    def get_query(self, session_id: str, question: str) -> Optional[str]:
        """Return cached answer or None."""
        if not self._client:
            return None
        try:
            key = self._query_key(session_id, question)
            value = self._client.get(key)
            if value:
                log.info("Redis cache hit", session_id=session_id)
            return value
        except Exception as exc:
            log.warning("Redis get_query failed", error=str(exc))
            return None

    def set_query(self, session_id: str, question: str, answer: str) -> None:
        """Store answer with TTL."""
        if not self._client:
            return
        try:
            key = self._query_key(session_id, question)
            self._client.setex(key, self._query_ttl, answer)
            log.info("Redis cache set", session_id=session_id, ttl=self._query_ttl)
        except Exception as exc:
            log.warning("Redis set_query failed", error=str(exc))

    def invalidate_session(self, session_id: str) -> None:
        """Delete all cache entries for a session (call when re-indexing)."""
        if not self._client:
            return
        try:
            pattern = f"{self.QUERY_PREFIX}:{session_id}:*"
            keys = list(self._client.scan_iter(pattern))
            if keys:
                self._client.delete(*keys)
            log.info("Redis session cache invalidated", session_id=session_id, keys_deleted=len(keys))
        except Exception as exc:
            log.warning("Redis invalidate_session failed", error=str(exc))

    # ------------------------------------------------------------------
    # Session metadata cache  (avoids PostgreSQL hit on every request)
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return cached session metadata dict or None."""
        if not self._client:
            return None
        try:
            key = f"{self.SESSION_PREFIX}:{session_id}"
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("Redis get_session failed", error=str(exc))
            return None

    def set_session(self, session_id: str, metadata: dict) -> None:
        """Store session metadata dict with TTL."""
        if not self._client:
            return
        try:
            key = f"{self.SESSION_PREFIX}:{session_id}"
            self._client.setex(key, self._session_ttl, json.dumps(metadata))
        except Exception as exc:
            log.warning("Redis set_session failed", error=str(exc))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if not self._client:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_key(self, session_id: str, question: str) -> str:
        q_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
        return f"{self.QUERY_PREFIX}:{session_id}:{q_hash}"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
