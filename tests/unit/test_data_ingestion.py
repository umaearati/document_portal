import pathlib
import pytest
from langchain.schema import Document

from src.document_ingestion.data_ingestion import (
    generate_session_id,
    ChatIngestor,
    FaissManager,
)


def test_generate_session_id_format_and_uniqueness():
    a = generate_session_id()
    b = generate_session_id()
    assert a != b
    assert a.startswith("session_") and b.startswith("session_")
    # Rough pattern check: session_YYYYMMDD_HHMMSS_XXXXXXXX -> 4 parts
    assert len(a.split("_")) == 4


def test_chat_ingestor_resolve_dir_uses_session_dirs(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    assert ing.session_id
    assert str(ing.temp_dir).endswith(ing.session_id)
    assert str(ing.faiss_dir).endswith(ing.session_id)


def test_split_chunks_respect_size_and_overlap(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    docs = [Document(page_content="A" * 1200, metadata={"source": "x.txt"})]
    chunks = ing._split(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    # spot check boundaries
    assert len(chunks[0].page_content) <= 500


def test_faiss_manager_add_documents_idempotent(tmp_dirs, stub_model_loader):
    fm = FaissManager(index_dir=pathlib.Path("faiss_index/test"))
    fm.load_or_create(texts=["hello", "world"], metadatas=[{"source": "a"}, {"source": "b"}])
    docs = [Document(page_content="hello", metadata={"source": "a"})]
    first = fm.add_documents(docs)
    second = fm.add_documents(docs)
    assert first >= 0
    assert second == 0
    
# If FAISS was not mocked properly,
# what kind of problem might occur?

#     Call OpenAI / Google API
# Require real API key
# Fail in CI
# Cost money
# Be slow

    # To ensure tests are deterministic, fast, and independent of external services. Mocking prevents real API calls and avoids writing to production storage.
    
    
#  “Do you test embedding models?”

# No, I mock embedding models in unit tests because they are external dependencies. I instead test integration logic, ensuring embeddings are generated, stored in FAISS, and retrieved correctly.   


#  What you are testing  What to use 
#  FastAPI route         TestClient 
#  Class                Import class 
#  Function             Import function
