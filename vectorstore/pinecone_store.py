"""
Pinecone vector store for Document Portal.

Replaces FAISS as the vector store for document chat.

Why Pinecone over FAISS:
  - Persistent across restarts (no re-indexing needed)
  - Scales beyond memory limits
  - Managed — no disk I/O, no save_local()
  - Metadata filtering built in

Environment variables:
    PINECONE_API_KEY     — from console.pinecone.io → API Keys
    PINECONE_INDEX_NAME  — name of your index (default: document-portal)
    PINECONE_CLOUD       — aws | gcp | azure  (default: aws)
    PINECONE_REGION      — e.g. us-east-1     (default: us-east-1)
    VECTOR_STORE         — pinecone | faiss   (default: faiss)
                           set to "pinecone" to activate

The VECTOR_STORE env var is the master switch.
If set to "faiss" (or missing), the existing FAISS pipeline is used.
If set to "pinecone", this module takes over.

Usage:
    from vectorstore.pinecone_store import get_pinecone_retriever

    retriever = get_pinecone_retriever(session_id, docs, k=3)
    # returns a LangChain-compatible retriever
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain.schema import Document
from logger import GLOBAL_LOGGER as log


def _is_pinecone_enabled() -> bool:
    return os.getenv("VECTOR_STORE", "faiss").lower() == "pinecone"


def _get_pinecone_client():
    try:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=api_key)
        return pc, ServerlessSpec
    except ImportError:
        log.error("pinecone-client package not installed")
        raise
    except Exception as exc:
        log.error("Pinecone client init failed", error=str(exc))
        raise


def _ensure_index(pc, spec, index_name: str, dimension: int = 1536) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,        # text-embedding-3-small = 1536
            metric="cosine",
            spec=spec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
        log.info("Pinecone index created", index=index_name)
    else:
        log.info("Pinecone index already exists", index=index_name)


class PineconeStore:
    """
    Manages document ingestion and retrieval via Pinecone.

    Namespace = session_id so each user's documents are isolated
    within the same Pinecone index.
    """

    def __init__(self):
        self._index_name = os.getenv("PINECONE_INDEX_NAME", "document-portal")
        pc, ServerlessSpec = _get_pinecone_client()
        _ensure_index(pc, ServerlessSpec, self._index_name)
        self._index = pc.Index(self._index_name)
        self._embeddings = self._load_embeddings()
        log.info("PineconeStore ready", index=self._index_name)

    def ingest(self, docs: List[Document], session_id: str) -> int:
        """
        Embed and upsert documents into Pinecone under the session namespace.
        Returns number of vectors upserted.
        """
        from langchain_pinecone import PineconeVectorStore  # type: ignore

        vs = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=self._embeddings,
            index_name=self._index_name,
            namespace=session_id,
        )
        count = len(docs)
        log.info("Documents upserted to Pinecone", count=count, session_id=session_id)
        return count

    def get_retriever(self, session_id: str, k: int = 3):
        """
        Return a LangChain retriever scoped to the session namespace.
        """
        from langchain_pinecone import PineconeVectorStore  # type: ignore

        vs = PineconeVectorStore(
            index=self._index,
            embedding=self._embeddings,
            namespace=session_id,
        )
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        log.info("Pinecone retriever created", session_id=session_id, k=k)
        return retriever

    def delete_session(self, session_id: str) -> None:
        """Delete all vectors for a session namespace (cleanup)."""
        try:
            self._index.delete(delete_all=True, namespace=session_id)
            log.info("Pinecone namespace deleted", session_id=session_id)
        except Exception as exc:
            log.warning("Pinecone delete_session failed", error=str(exc))

    def _load_embeddings(self):
        from utils.model_loader import ModelLoader  # noqa
        return ModelLoader().load_embeddings()


# ---------------------------------------------------------------------------
# Convenience function used by ChatIngestor (swap-in for FAISS)
# ---------------------------------------------------------------------------

_store: Optional[PineconeStore] = None


def get_pinecone_store() -> PineconeStore:
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


def get_pinecone_retriever(session_id: str, docs: List[Document], k: int = 3):
    """
    Ingest docs into Pinecone and return a retriever for the session.
    Called from ChatIngestor when VECTOR_STORE=pinecone.
    """
    store = get_pinecone_store()
    store.ingest(docs, session_id)
    return store.get_retriever(session_id, k=k)


def is_pinecone_enabled() -> bool:
    return _is_pinecone_enabled()
