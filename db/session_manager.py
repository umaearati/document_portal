"""
DB-backed session manager.

Replaces the bare `rag_instances: Dict[str, ConversationalRAG]` dict in
api/main.py with a class that:

  1. Keeps a process-local LRU cache of live RAG instances (fast path).
  2. Persists every session to PostgreSQL (audit / restart recovery).
  3. Records every chat turn to chat_history.
  4. Writes a row to query_audit_log after each query completes.

Usage (from api/main.py):
    from db.session_manager import SessionManager
    session_mgr = SessionManager()

    # on /chat/index
    session_mgr.create_session(session_id, file_names, chunk_size, chunk_overlap, k)

    # on /chat/query — replaces manual rag_instances cache check
    rag = session_mgr.get_or_load_rag(session_id, index_dir, k, index_name)
    answer = await rag.invoke(question, chat_history=[])
    session_mgr.record_query(session_id, question, answer, latency_ms, pii_count)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.exc import IntegrityError

from db.models import ChatMessage, ChatSession, QueryAuditLog, SessionLocal
from logger import GLOBAL_LOGGER as log


# Lazy import to avoid circular: ConversationalRAG → retrieval → model_loader
def _load_rag_class():
    from src.document_chat.retrieval import ConversationalRAG  # noqa
    return ConversationalRAG


_MAX_CACHE = 50   # evict LRU after this many live instances


class SessionManager:
    """
    Thread-safe (GIL-protected) manager for RAG session lifecycle.

    The in-process LRU cache keeps hot sessions fast; PostgreSQL keeps the
    durable record so you can audit, restart, and report across deploys.
    """

    def __init__(self, max_cache: int = _MAX_CACHE):
        # OrderedDict used as an LRU: most-recently-used moves to end
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_cache = max_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        file_names: list[str],
        chunk_size: int,
        chunk_overlap: int,
        k: int,
    ) -> None:
        """Persist a new chat session to PostgreSQL (idempotent on conflict)."""
        with SessionLocal() as db:
            try:
                row = ChatSession(
                    session_id=session_id,
                    file_names=",".join(file_names),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                )
                db.add(row)
                db.commit()
                log.info("Chat session persisted to DB", session_id=session_id)
            except IntegrityError:
                db.rollback()
                log.info("Session already exists in DB (idempotent)", session_id=session_id)

    def get_or_load_rag(
        self,
        session_id: str,
        index_dir: str,
        k: int,
        index_name: str = "index",
    ) -> object:
        """
        Return a live ConversationalRAG instance.

        Hot path: return from process-local LRU cache.
        Cold path: load FAISS from disk, build chain, cache it.
        """
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            log.info("RAG cache hit", session_id=session_id)
            return self._cache[session_id]

        ConversationalRAG = _load_rag_class()
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=index_name)

        self._cache[session_id] = rag
        self._cache.move_to_end(session_id)

        # Evict oldest if over capacity
        while len(self._cache) > self._max_cache:
            evicted, _ = self._cache.popitem(last=False)
            log.info("RAG instance evicted from cache (LRU)", evicted=evicted)
            self._mark_expired(evicted)

        log.info("RAG instance loaded and cached", session_id=session_id)
        return self._cache[session_id]

    def record_message(self, session_id: str, role: str, content: str) -> None:
        """Append a chat turn (user or assistant) to chat_history."""
        with SessionLocal() as db:
            db.add(ChatMessage(session_id=session_id, role=role, content=content))
            db.commit()

    def record_query(
        self,
        session_id: str,
        question: str,
        answer: str,
        latency_ms: float,
        pii_count: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        k_used: Optional[int] = None,
    ) -> None:
        """Write a row to query_audit_log and update session.last_active."""
        with SessionLocal() as db:
            db.add(
                QueryAuditLog(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    k_used=k_used,
                    pii_redacted=pii_count,
                )
            )
            # Touch last_active on the session row
            session_row = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if session_row:
                session_row.last_active = datetime.now(timezone.utc)
            db.commit()

    def get_chat_history(self, session_id: str) -> list[Dict]:
        """Return all messages for a session ordered by creation time."""
        with SessionLocal() as db:
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            return [{"role": r.role, "content": r.content} for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_expired(self, session_id: str) -> None:
        with SessionLocal() as db:
            row = (
                db.query(ChatSession)
                .filter(ChatSession.session_id == session_id)
                .first()
            )
            if row:
                row.status = "expired"
                db.commit()
