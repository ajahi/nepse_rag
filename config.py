#!/usr/bin/env python3
"""Shared configuration + a single place to build the LLM and embeddings.

Keys/model match app28.py (your Gemini-Pro project): one Groq LLM shared by the
SQL agent and the RAG path, Cohere for embeddings. Everything degrades
gracefully — if a key is missing the app still boots and the affected path
returns an honest "I'm not sure" instead of crashing.
"""
import os
import functools

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- LLM (Groq) — same model app28.py uses ---
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Embeddings (Cohere) — same model app28.py uses ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-v4.0")

# --- RAG index folder (FAISS load_local expects a directory) ---
DEFAULT_INDEX = os.environ.get("DEFAULT_INDEX", "nepse_kb")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))

# --- Database (analytical + prediction) ---
PG_DSN = os.environ.get("PG_DSN")

HAS_LLM = bool(GROQ_API_KEY)
HAS_COHERE = bool(COHERE_API_KEY)
HAS_DB = bool(PG_DSN)

# Confidence below this triggers the "I'm not sure" hedge everywhere.
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.45"))

UNSURE = "I'm not sure about certain things with this query."


@functools.lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0, max_tokens: int = 1000):
    """Return a cached ChatGroq, or None if no key is configured."""
    if not HAS_LLM:
        return None
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
        max_tokens=max_tokens,
    )


@functools.lru_cache(maxsize=None)
def get_embeddings():
    """Return cached Cohere embeddings (embed-v4.0), or None if no key."""
    if not HAS_COHERE:
        return None
    from langchain_cohere import CohereEmbeddings
    return CohereEmbeddings(
        model=COHERE_EMBED_MODEL,
        cohere_api_key=COHERE_API_KEY,
        max_retries=3,
    )
