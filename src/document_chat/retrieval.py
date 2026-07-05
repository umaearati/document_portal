import sys
import os
from operator import itemgetter
from typing import AsyncGenerator, List, Optional, Dict, Any

from langchain_core.language_models import LLM
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.
    """

    # ---------- NEW: Token control settings ----------
    MAX_CONTEXT_CHARS = 4000
    MAX_CHAT_HISTORY = 5

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 3,  #  reduced default
        index_name: str = "index",
        search_type: str = "mmr",
        lambda_mult: float = 0.3,  #  better diversity
        fetch_k: int = 10,  #  reduced fetch size
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )

            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
                search_type=search_type,
                lambda_mult=lambda_mult if search_type == "mmr" else None,
                fetch_k=fetch_k if search_type == "mmr" else None,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    async def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().",
                    sys,
                )

            #  NEW: Chat history windowing
            chat_history = (chat_history or [])[-self.MAX_CHAT_HISTORY :]

            #  NEW: Dynamic k tuning
            dynamic_k = self._dynamic_k(user_input)
            if hasattr(self.retriever, "search_kwargs"):
                self.retriever.search_kwargs["k"] = dynamic_k

            payload = {"input": user_input, "chat_history": chat_history}
            answer = await self.chain.ainvoke(payload)

            if not answer:
                log.warning(
                    "No answer generated",
                    user_input=user_input,
                    session_id=self.session_id,
                )
                return "no answer generated."

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                dynamic_k=dynamic_k,
                answer_preview=str(answer)[:150],
            )

            return answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    async def stream(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the LLM response token by token using LangChain's .astream().
        Yields string chunks as they arrive — the caller wraps these in
        FastAPI's StreamingResponse for real-time delivery to the browser.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() first.", sys
                )

            chat_history = (chat_history or [])[-self.MAX_CHAT_HISTORY:]
            dynamic_k = self._dynamic_k(user_input)
            if hasattr(self.retriever, "search_kwargs"):
                self.retriever.search_kwargs["k"] = dynamic_k

            payload = {"input": user_input, "chat_history": chat_history}

            async for chunk in self.chain.astream(payload):
                if chunk:
                    yield chunk

            log.info("Streaming complete", session_id=self.session_id, dynamic_k=dynamic_k)

        except Exception as e:
            log.error("Failed to stream ConversationalRAG", error=str(e))
            raise DocumentPortalException("Streaming error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    def _dynamic_k(self, user_input: str) -> int:
        """
        Adjust retrieval k based on question complexity.
        - Short / simple questions  → k=2  (fast, cheap)
        - Medium questions          → k=3
        - Long / complex questions  → k=5  (broader context)
        """
        word_count = len(user_input.split())
        if word_count <= 6:
            return 2
        elif word_count <= 15:
            return 3
        else:
            return 5

    def _format_docs(self, docs) -> str:
        """
        Controlled context size to reduce token cost.
        """
        combined = ""
        for doc in docs:
            content = getattr(doc, "page_content", str(doc))
            if len(combined) + len(content) > self.MAX_CONTEXT_CHARS:
                break
            combined += content + "\n\n"
        return combined.strip()

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = (
                question_rewriter
                | self.retriever
                | self._format_docs
            )

            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)

        except Exception as e:
            log.error(
                "Failed to build LCEL chain",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException("Failed to build LCEL chain", sys)

