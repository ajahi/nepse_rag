#!/usr/bin/env python3
"""Conversational / knowledge path: RAG over a FAISS index.

Ported from app28.py (your Gemini-Pro project): Cohere embed-v4.0 embeddings,
FAISS dense retrieval with source attribution from the docstore metadata, and
Groq for answering. The knowledge base is small on purpose — enough for basic
NEPSE questions. Build/refresh it with:  python build_index.py

Anti-hallucination: answers come ONLY from retrieved context. Weak/no retrieval
-> we say we're not sure rather than invent facts.
"""
import os
from dataclasses import dataclass

from config import (get_llm, get_embeddings, DEFAULT_INDEX, RETRIEVAL_K,
                    UNSURE)

HERE = os.path.dirname(os.path.abspath(__file__))
_retriever = None
_load_error = None


@dataclass
class RagResult:
    answer: str
    confidence: float
    sources: list


def _index_path() -> str:
    # DEFAULT_INDEX may be an absolute path or a folder next to this file.
    return DEFAULT_INDEX if os.path.isabs(DEFAULT_INDEX) else os.path.join(
        HERE, DEFAULT_INDEX)


def _get_retriever():
    """Lazy-load the FAISS retriever once. None if it can't be loaded."""
    global _retriever, _load_error
    if _retriever is not None or _load_error is not None:
        return _retriever
    embeddings = get_embeddings()
    if embeddings is None:
        _load_error = "no COHERE_API_KEY"
        return None
    try:
        from langchain_community.vectorstores import FAISS
        store = FAISS.load_local(
            _index_path(), embeddings, allow_dangerous_deserialization=True)
        _retriever = store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    except Exception as e:  # missing index / dim mismatch / network
        _load_error = str(e)
        _retriever = None
    return _retriever


def _format_source(doc) -> str:
    m = getattr(doc, "metadata", None) or {}
    src = m.get("source", "knowledge base")
    if "line_start" in m and "line_end" in m:
        return f"{src} lines {m['line_start']}-{m['line_end']}"
    if "page" in m:
        return f"{src} page {m['page']}"
    return str(src)


def _collect_sources(docs):
    seen = list(dict.fromkeys(_format_source(d) for d in docs))
    return [{"source": s} for s in seen]


_ANSWER_PROMPT = """You are a NEPSE (Nepal Stock Exchange) assistant. Answer the
user's question using ONLY the context below. Keep it concise and factual.
If the context does not contain the answer, reply EXACTLY: "{unsure}"
Do not use outside knowledge, and never invent figures, tickers or dates.

Context:
{context}

Question: {q}
Answer:"""


def answer(query: str) -> RagResult:
    retriever = _get_retriever()
    llm = get_llm(temperature=0.3)

    # No index available -> only answer safe, generic concepts, and hedge.
    if retriever is None:
        if llm is None:
            return RagResult(UNSURE, 0.2, [])
        prompt = (f"You are a finance assistant. Answer briefly ONLY if you are "
                  f"confident about a general (non-NEPSE-specific) finance "
                  f"concept. If the question needs specific NEPSE data you don't "
                  f"have, reply EXACTLY: \"{UNSURE}\"\n\nQuestion: {query}")
        try:
            out = llm.invoke(prompt).content.strip()
        except Exception:
            return RagResult(UNSURE, 0.2, [])
        return RagResult(out, 0.35 if out == UNSURE else 0.5, [])

    # Retrieval available.
    try:
        docs = retriever.invoke(query)
    except Exception:
        return RagResult(UNSURE, 0.3, [])
    if not docs:
        return RagResult(UNSURE, 0.3, [])

    context = "\n\n".join(d.page_content for d in docs)[:6000]
    sources = _collect_sources(docs)

    if llm is None:
        # Return the most relevant chunk verbatim rather than fabricating prose.
        return RagResult(docs[0].page_content.strip(), 0.4, sources)

    try:
        out = llm.invoke(_ANSWER_PROMPT.format(
            unsure=UNSURE, context=context, q=query)).content.strip()
    except Exception:
        return RagResult(UNSURE, 0.3, sources)

    conf = 0.3 if out == UNSURE else 0.8
    return RagResult(out, conf, sources)
