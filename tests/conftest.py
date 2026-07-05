import os
import io
import types
import json
import shutil
import pathlib
import sys
import pytest

os.environ.setdefault("PYTHONPATH", str(pathlib.Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "google")
# In tests, I set dummy environment variables to prevent real API calls and ensure configuration-dependent components initialise safely.
from fastapi.testclient import TestClient

# Ensure repository root is importable for `import main`
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# I ensure the project root is added to Python’s module search path so imports work consistently when running pytest, especially in non-installed local development environments.
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)

# TestClient simulates HTTP requests to your FastAPI app without starting a real server.
@pytest.fixture
def clear_sessions():
    app.SESSIONS.clear()
    yield
    app.SESSIONS.clear()
# Chat memory
# Session history
# LangGraph state

# Clean state before test
# Clean state after test

@pytest.fixture
def tmp_dirs(tmp_path: pathlib.Path): 
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        # Point working directories used by app code to tmp ones by chdir
        os.chdir(tmp_path)
        yield {"data": data_dir, "faiss": faiss_dir}
    finally:
        os.chdir(cwd)
# Isolation per test

class _StubEmbeddings:
    def embed_query(self, text: str):
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.0, 0.1, 0.2] for _ in texts]

    def __call__(self, text: str):
        return [0.0, 0.1, 0.2]


class _StubLLM:
    def invoke(self, input):
        return "stubbed answer"


@pytest.fixture
def stub_model_loader(monkeypatch):
    # Patch both module paths to cover imports via `utils.model_loader` and `document_portal.utils.model_loader`
    import utils.model_loader as ml_mod
    from utils import model_loader as ml_mod2

    class FakeApiKeyMgr:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x", "GOOGLE_API_KEY": "y"}

        def get(self, key: str) -> str:
            return self.api_keys[key]

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyMgr()
            self.config = {
                "embedding_model": {"model_name": "fake-embed"},
                "llm": {
                    "google": {
                        "provider": "google",
                        "model_name": "fake-llm",
                        "temperature": 0.0,
                        "max_output_tokens": 128,
                    }
                },
            }

        def load_embeddings(self):
            return _StubEmbeddings()

        def load_llm(self):
            return _StubLLM()

    monkeypatch.setattr(ml_mod, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(ml_mod2, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod2, "ModelLoader", FakeModelLoader)

    # Also patch the already-imported symbols used in modules under test
    import src.document_ingestion.data_ingestion as di
    import src.document_chat.retrieval as r
    monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(r, "ModelLoader", FakeModelLoader)
    yield FakeModelLoader


@pytest.fixture
def stub_ingestor(monkeypatch):
    import src.document_ingestion.data_ingestion as di

    class FakeIngestor:
        def __init__(self, use_session_dirs=True, **kwargs):
            self.use_session = use_session_dirs
            self.session_id = "sess_test"

        def built_retriver(self, uploaded_files, **kwargs):
            return None

    monkeypatch.setattr(di, "ChatIngestor", FakeIngestor)
    monkeypatch.setattr(app, "ChatIngestor", FakeIngestor)
    yield FakeIngestor


@pytest.fixture
def stub_rag(monkeypatch):
    import src.document_chat.retrieval as r

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
            self.retriever = retriever

        def load_retriever_from_faiss(self, index_path, **kwargs):
            return None

        def invoke(self, user_input, chat_history=None):
            return "stubbed answer"

    monkeypatch.setattr(r, "ConversationalRAG", FakeRAG)
    monkeypatch.setattr(app, "ConversationalRAG", FakeRAG)
    yield FakeRAG
    # I isolate external dependencies by mocking the model loader, embedding models, and RAG pipeline. I use stub classes to replace LLM and retriever behaviour. I also use temporary directories for FAISS index and clear in-memory sessions between tests to avoid cross-test contamination.