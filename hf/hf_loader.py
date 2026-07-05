"""
Hugging Face integration for Document Portal.

Adds HuggingFace as a fourth LLM/embedding provider alongside
OpenAI, Google, and Groq.

Two modes:
  1. HF Inference API  — hosted models, uses your HF token, free tier available
  2. Local pipeline    — downloads model weights, runs on CPU/GPU, fully free

Supported via config.yaml:
    llm:
      huggingface:
        provider: huggingface
        model_name: mistralai/Mistral-7B-Instruct-v0.3
        mode: api          # api | local
        max_new_tokens: 512
        temperature: 0.2

Environment variables:
    HUGGINGFACE_API_TOKEN  — from huggingface.co → Settings → Access Tokens

Usage (via ModelLoader — automatic when LLM_PROVIDER=huggingface):
    export LLM_PROVIDER=huggingface
    # or in docker-compose.yml: LLM_PROVIDER: huggingface

Direct usage:
    from hf.hf_loader import get_hf_llm, get_hf_embeddings

    llm = get_hf_llm()
    embeddings = get_hf_embeddings()
"""

from __future__ import annotations

import os
from typing import Optional

from logger import GLOBAL_LOGGER as log

# ---------------------------------------------------------------------------
# Recommended free models
# ---------------------------------------------------------------------------
# LLM (Inference API, free tier):
#   mistralai/Mistral-7B-Instruct-v0.3   — strong general purpose
#   HuggingFaceH4/zephyr-7b-beta         — instruction-tuned
#   google/flan-t5-large                 — lighter, faster
#
# Embeddings (free, local):
#   sentence-transformers/all-MiniLM-L6-v2   — fast, 384 dim
#   sentence-transformers/all-mpnet-base-v2  — better quality, 768 dim
# ---------------------------------------------------------------------------

_DEFAULT_LLM_MODEL        = "mistralai/Mistral-7B-Instruct-v0.3"
_DEFAULT_EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def get_hf_llm(
    model_name: Optional[str] = None,
    mode: str = "api",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
):
    """
    Return a LangChain-compatible LLM backed by HuggingFace.

    mode="api"   → HuggingFaceEndpoint (calls HF Inference API, needs token)
    mode="local" → HuggingFacePipeline (downloads weights, runs locally)
    """
    model = model_name or os.getenv("HF_LLM_MODEL", _DEFAULT_LLM_MODEL)
    token = os.getenv("HUGGINGFACE_API_TOKEN")

    if mode == "api":
        return _api_llm(model, token, max_new_tokens, temperature)
    elif mode == "local":
        return _local_llm(model, max_new_tokens, temperature)
    else:
        raise ValueError(f"Unknown HF mode: {mode}. Use 'api' or 'local'.")


def get_hf_embeddings(model_name: Optional[str] = None):
    """
    Return a LangChain-compatible embedding model from HuggingFace.
    Runs locally — no API token needed, completely free.
    """
    model = model_name or os.getenv("HF_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        embeddings = HuggingFaceEmbeddings(model_name=model)
        log.info("HuggingFace embeddings loaded", model=model)
        return embeddings
    except ImportError:
        log.error("langchain-huggingface not installed. Run: pip install langchain-huggingface")
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_llm(model: str, token: Optional[str], max_new_tokens: int, temperature: float):
    """HuggingFace Inference API — hosted, uses free tier."""
    try:
        from langchain_huggingface import HuggingFaceEndpoint  # type: ignore

        if not token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not set. "
                "Get yours at huggingface.co → Settings → Access Tokens"
            )

        llm = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        log.info("HuggingFace Inference API LLM loaded", model=model)
        return llm

    except ImportError:
        log.error("langchain-huggingface not installed. Run: pip install langchain-huggingface")
        raise


def _local_llm(model: str, max_new_tokens: int, temperature: float):
    """
    Local HuggingFace pipeline — downloads weights on first run.
    Warning: requires ~4GB+ disk space and a decent CPU/GPU.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        from langchain_huggingface import HuggingFacePipeline  # type: ignore

        log.info("Loading HF model locally (first run downloads weights)", model=model)

        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(model)

        pipe = pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        log.info("HuggingFace local LLM loaded", model=model)
        return llm

    except ImportError:
        log.error("transformers / langchain-huggingface not installed")
        raise
