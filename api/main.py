import os
import time
from typing import List, Optional, Any, Dict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from logger import GLOBAL_LOGGER as log

from db.models import create_all_tables
from db.session_manager import SessionManager
from observability.tracer import get_tracer
from pii.redactor import get_redactor
from cache.redis_cache import get_cache
from security.guardrails import get_guardrail
from security.auth import (
    RegisterRequest, LoginRequest, TokenResponse, UserOut,
    register_user, login_user, get_current_user, create_user_table,
)


# ── CONFIG ──────────────────────────────────────────────────────────────────

FAISS_BASE       = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE      = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title="Document Portal API", version="0.4")

BASE_DIR      = Path(__file__).resolve().parent.parent
static_dir    = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(templates_dir))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SINGLETONS  (initialised once at startup) ────────────────────────────────

session_mgr = SessionManager()
tracer      = get_tracer()
redactor    = get_redactor()
cache       = get_cache()
guardrail   = get_guardrail()


@app.on_event("startup")
def on_startup():
    """Create DB tables on first boot (idempotent)."""
    try:
        create_all_tables()
        create_user_table()
        log.info("PostgreSQL tables verified / created")
    except Exception as exc:
        log.error("DB startup failed — app continues without persistence", error=str(exc))


# ── ROOT + HEALTH ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserOut, status_code=201)
def register(req: RegisterRequest):
    """
    Register a new user account.
    Password is bcrypt-hashed before storage — never stored in plaintext.

    Request body: { "email": "...", "password": "..." }
    Returns: { "email": "...", "is_active": true }
    """
    return register_user(req.email, req.password)


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    Login with email + password.
    Returns a JWT access token valid for 30 minutes (configurable via JWT_EXPIRE_MINUTES).

    Request body: { "email": "...", "password": "..." }
    Returns: { "access_token": "...", "token_type": "bearer", "expires_in": 1800 }

    Use the token in subsequent requests:
        Authorization: Bearer <access_token>
    """
    return login_user(req.email, req.password)


@app.get("/auth/me", response_model=UserOut)
def me(current_user: UserOut = Depends(get_current_user)):
    """
    Return the currently authenticated user's profile.
    Requires: Authorization: Bearer <access_token>
    """
    return current_user


# ── ANALYZE ──────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")

        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)

        return JSONResponse(content=result)

    except Exception as e:
        log.exception("Document analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── COMPARE ──────────────────────────────────────────────────────────────────

@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...),
    actual: UploadFile = File(...),
) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference),
            FastAPIFileAdapter(actual),
        )

        combined_text = dc.combine_documents()

        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)

        return {
            "rows": df.to_dict(orient="records"),
            "session_id": dc.session_id,
        }

    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── CHAT: BUILD INDEX ─────────────────────────────────────────────────────────

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(400),
    chunk_overlap: int = Form(50),
    k: int = Form(3),
    current_user: UserOut = Depends(get_current_user),
) -> Any:
    try:
        log.info(f"Indexing session: {session_id}")

        wrapped = [FastAPIFileAdapter(f) for f in files]

        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )

        ci.built_retriver(
            wrapped,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )

        # ── FIX: Invalidate Redis cache when a new index is built for an
        #         existing session — prevents stale answers being returned
        #         after a user re-uploads a different document.
        if session_id:
            cache.invalidate_session(session_id)
            log.info("Redis cache invalidated for session", session_id=session_id)

        # ── Persist session to PostgreSQL ──
        file_names = [f.filename for f in files]
        session_mgr.create_session(
            session_id=ci.session_id,
            file_names=file_names,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )

        # ── Langfuse: trace ingestion event ──
        tracer.trace_ingestion(
            session_id=ci.session_id,
            file_count=len(file_names),
            chunk_count=0,
        )

        return {
            "session_id": ci.session_id,
            "k": k,
            "chunk_size": chunk_size,
        }

    except Exception as e:
        log.exception("Index building failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── CHAT: STREAM QUERY ───────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Streaming version of /chat/query.
    Returns tokens as they are generated using Server-Sent Events (text/event-stream).
    The browser receives partial responses in real time instead of waiting for
    the full answer — significantly better UX for long responses.

    Frontend usage:
        const resp = await fetch('/chat/stream', { method: 'POST', body: formData });
        const reader = resp.body.getReader();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            displayChunk(new TextDecoder().decode(value));
        }
    """
    if use_session_dirs and not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    index_dir = (
        os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
    )

    if not os.path.isdir(index_dir):
        raise HTTPException(status_code=404, detail="FAISS index not found")

    # Guardrail + PII (same as /chat/query)
    guard_result = guardrail.check(question)
    if not guard_result.is_safe:
        raise HTTPException(status_code=400, detail=guard_result.reason)
    question = guardrail.sanitise(question)

    clean_question, pii_count = redactor.redact(question)

    # Fetch history + persist user message
    raw_history = session_mgr.get_chat_history(session_id)
    chat_history = []
    for msg in raw_history:
        if msg["role"] == "user":
            from langchain_core.messages import HumanMessage
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            from langchain_core.messages import AIMessage
            chat_history.append(AIMessage(content=msg["content"]))

    session_mgr.record_message(session_id, role="user", content=clean_question)

    rag = session_mgr.get_or_load_rag(session_id, index_dir, k, FAISS_INDEX_NAME)

    async def token_generator():
        """Yield tokens as they stream from the LLM chain."""
        full_response = []
        async for chunk in rag.stream(clean_question, chat_history=chat_history):
            full_response.append(chunk)
            yield chunk
        # After streaming completes, persist the full answer
        complete_answer = "".join(full_response)
        session_mgr.record_message(session_id, role="assistant", content=complete_answer)
        cache.set_query(session_id, clean_question, complete_answer)
        log.info("Stream complete, answer persisted", session_id=session_id)

    return StreamingResponse(token_generator(), media_type="text/plain")


# ── CHAT: QUERY ───────────────────────────────────────────────────────────────

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    current_user: UserOut = Depends(get_current_user),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        index_dir = (
            os.path.join(FAISS_BASE, session_id)
            if use_session_dirs
            else FAISS_BASE
        )

        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail="FAISS index not found")

        # ── Guardrail: block prompt injection before anything else ──
        guard_result = guardrail.check(question)
        if not guard_result.is_safe:
            raise HTTPException(status_code=400, detail=guard_result.reason)
        question = guardrail.sanitise(question)

        # ── PII: redact the incoming question before it hits the chain ──
        clean_question, pii_count = redactor.redact(question)
        if pii_count:
            log.info("PII redacted from question", count=pii_count, session_id=session_id)

        # ── Redis: return cached answer immediately if available ──
        cached_answer = cache.get_query(session_id, clean_question)
        if cached_answer:
            return {
                "answer": cached_answer,
                "session_id": session_id,
                "k": k,
                "engine": "LCEL-RAG",
                "cache": "hit",
                "pii_redacted": pii_count,
            }

        # ── FIX: Fetch real chat history from PostgreSQL and convert to
        #         LangChain message objects so the contextualize prompt can
        #         rewrite follow-up questions correctly.
        raw_history = session_mgr.get_chat_history(session_id)
        chat_history = []
        for msg in raw_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # ── Persist user message AFTER fetching history (don't include
        #    the current question in the history passed to the chain) ──
        session_mgr.record_message(session_id, role="user", content=clean_question)

        # ── Get (or load) RAG instance via session manager ──
        rag = session_mgr.get_or_load_rag(
            session_id, index_dir, k, FAISS_INDEX_NAME
        )

        # ── Langfuse: trace the full query ──
        with tracer.trace_query(session_id, clean_question, k) as span:
            response = await rag.invoke(clean_question, chat_history=chat_history)
            latency_ms = span.record_answer(response, pii_count=pii_count)

        # ── Persist assistant message + audit log row ──
        session_mgr.record_message(session_id, role="assistant", content=response)
        session_mgr.record_query(
            session_id=session_id,
            question=clean_question,
            answer=response,
            latency_ms=latency_ms,
            pii_count=pii_count,
            k_used=k,
        )

        # ── Redis: cache the answer for next time ──
        cache.set_query(session_id, clean_question, response)

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
            "cache": "miss",
            "pii_redacted": pii_count,
        }

    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=str(e))
