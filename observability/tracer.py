"""
Langfuse observability wrapper for Document Portal.

Traces every RAG query with:
  - Input question + session context
  - Retrieved context (as a span)
  - LLM response
  - Token usage (prompt + completion)
  - End-to-end latency
  - PII redaction count (custom metadata)

LangSmith is kept as-is; this runs alongside it.

Environment variables required:
    LANGFUSE_PUBLIC_KEY   — from your Langfuse project settings
    LANGFUSE_SECRET_KEY   — from your Langfuse project settings
    LANGFUSE_HOST         — defaults to https://cloud.langfuse.com

Usage:
    from observability.tracer import get_tracer
    tracer = get_tracer()

    with tracer.trace_query(session_id, question, k) as span:
        answer = await rag.invoke(question)
        span.record_answer(answer, prompt_tokens=50, completion_tokens=120)
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator, Optional

from logger import GLOBAL_LOGGER as log

# ---------------------------------------------------------------------------
# Lazy Langfuse import — app still boots if the package isn't installed yet
# (it will be added to requirements.txt below)
# ---------------------------------------------------------------------------

def _get_langfuse():
    try:
        from langfuse import Langfuse  # type: ignore
        return Langfuse
    except ImportError:
        log.warning("langfuse package not installed — observability disabled")
        return None


# ---------------------------------------------------------------------------
# QuerySpan — context manager returned by trace_query()
# ---------------------------------------------------------------------------

class QuerySpan:
    """
    Wraps a single RAG query inside a Langfuse trace.

    Records:
        - generation span with the LLM answer and token counts
        - custom metadata: session_id, k, pii_count, latency_ms
    """

    def __init__(self, trace, session_id: str, question: str, k: int):
        self._trace = trace
        self._session_id = session_id
        self._question = question
        self._k = k
        self._start = time.perf_counter()
        self._generation = None

        if self._trace:
            self._generation = self._trace.generation(
                name="rag-query",
                input=question,
                metadata={"session_id": session_id, "k": k},
            )

    def record_answer(
        self,
        answer: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        pii_count: int = 0,
    ) -> float:
        """
        Finalise the generation span with the LLM output.

        Returns latency_ms so the caller can pass it to session_manager.record_query().
        """
        latency_ms = (time.perf_counter() - self._start) * 1000

        if self._generation:
            usage = {}
            if prompt_tokens is not None:
                usage["promptTokens"] = prompt_tokens
            if completion_tokens is not None:
                usage["completionTokens"] = completion_tokens

            self._generation.end(
                output=answer,
                usage=usage or None,
                metadata={
                    "session_id": self._session_id,
                    "latency_ms": round(latency_ms, 2),
                    "pii_count": pii_count,
                    "k": self._k,
                },
            )

        log.info(
            "Query traced",
            session_id=self._session_id,
            latency_ms=round(latency_ms, 2),
            pii_count=pii_count,
        )
        return latency_ms

    def record_error(self, error: Exception) -> None:
        if self._generation:
            self._generation.end(
                level="ERROR",
                status_message=str(error),
            )


# ---------------------------------------------------------------------------
# Tracer — singleton, initialised once at import time
# ---------------------------------------------------------------------------

class PortalTracer:
    """
    Thin wrapper around the Langfuse client.

    Degrades gracefully: if LANGFUSE_PUBLIC_KEY is absent or the package
    isn't installed, every method becomes a no-op so the app still runs.
    """

    def __init__(self):
        Langfuse = _get_langfuse()
        self._client = None

        if Langfuse is None:
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            log.warning(
                "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
                "Langfuse tracing disabled"
            )
            return

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        log.info("Langfuse tracer initialised")

    @contextmanager
    def trace_query(
        self,
        session_id: str,
        question: str,
        k: int,
    ) -> Generator[QuerySpan, None, None]:
        """
        Context manager that wraps a RAG query in a Langfuse trace.

        Example:
            with tracer.trace_query(session_id, question, k) as span:
                answer = await rag.invoke(question)
                latency = span.record_answer(answer, pii_count=n)
        """
        trace = None
        if self._client:
            trace = self._client.trace(
                name="document-portal-rag",
                session_id=session_id,
                input={"question": question},
                metadata={"k": k},
            )

        span = QuerySpan(trace, session_id, question, k)
        try:
            yield span
        except Exception as exc:
            span.record_error(exc)
            raise
        finally:
            if self._client:
                self._client.flush()

    def trace_ingestion(self, session_id: str, file_count: int, chunk_count: int) -> None:
        """Fire-and-forget event for document ingestion."""
        if not self._client:
            return
        self._client.event(
            name="document-ingestion",
            session_id=session_id,
            metadata={"file_count": file_count, "chunk_count": chunk_count},
        )
        self._client.flush()

    def trace_analysis(self, session_id: str, doc_type: str) -> None:
        """Fire-and-forget event for document analysis."""
        if not self._client:
            return
        self._client.event(
            name="document-analysis",
            session_id=session_id,
            metadata={"doc_type": doc_type},
        )
        self._client.flush()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracer: Optional[PortalTracer] = None


def get_tracer() -> PortalTracer:
    global _tracer
    if _tracer is None:
        _tracer = PortalTracer()
    return _tracer
