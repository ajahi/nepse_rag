#!/usr/bin/env python3
"""Live web-search path: for real-time facts the KB and DB can't cover.

Use cases: "latest NEPSE index today", "current news on X", anything time-
sensitive. Keyless by design — uses DuckDuckGo (ddgs) so there's no extra API
key to manage. Results are synthesized by Groq, grounded ONLY in the snippets
returned, with the source URLs attached.

Anti-hallucination: if search returns nothing usable, or the LLM can't ground an
answer in the snippets, we say we're not sure instead of guessing.
"""
from dataclasses import dataclass, field

from config import get_llm, UNSURE

MAX_RESULTS = 5


@dataclass
class WebResult:
    answer: str
    confidence: float
    sources: list = field(default_factory=list)


def _search(query: str):
    """Return [{title, body, href}] or [] if search is unavailable."""
    try:
        from ddgs import DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS  # older package name
        except Exception:
            return None  # library not installed
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=MAX_RESULTS))
    except Exception:
        return []


_PROMPT = """You are a NEPSE assistant answering a time-sensitive question using
live web search results. Use ONLY the snippets below. Be concise. Cite nothing
that isn't in the snippets. If the snippets don't clearly answer the question,
reply EXACTLY: "{unsure}"

Search results:
{results}

Question: {q}
Answer:"""


def answer(query: str) -> WebResult:
    hits = _search(query)

    if hits is None:
        return WebResult(
            f"{UNSURE} Live web search isn't available (the ddgs package isn't "
            f"installed).", 0.2)
    if not hits:
        return WebResult(
            f"{UNSURE} I couldn't find current web results for that.", 0.3)

    sources = [{"source": h.get("title") or h.get("href", "web"),
                "url": h.get("href")} for h in hits]

    llm = get_llm(temperature=0.2)
    if llm is None:
        # No LLM: surface the top snippets directly instead of inventing prose.
        top = hits[0]
        text = f"{top.get('title', '')}\n{top.get('body', '')}".strip()
        return WebResult(text or UNSURE, 0.4, sources)

    blob = "\n\n".join(
        f"[{i+1}] {h.get('title','')}\n{h.get('body','')}\n{h.get('href','')}"
        for i, h in enumerate(hits))
    try:
        out = llm.invoke(_PROMPT.format(
            unsure=UNSURE, results=blob, q=query)).content.strip()
    except Exception:
        return WebResult(UNSURE, 0.3, sources)

    conf = 0.3 if out == UNSURE else 0.7  # live web is useful but not verified
    return WebResult(out, conf, sources)
