#!/usr/bin/env python3
"""Orchestrator: route a message, dispatch to the right path, gate on confidence.

Flow:
    classify -> analytical | prediction | websearch | conversational
             -> run that path
             -> if confidence < MIN_CONFIDENCE, prepend the honest hedge
             -> return a uniform dict the API/UI can render.

The verifier already blocks fabricated numbers on the analytical path; this layer
adds the softer "I'm not sure" hedge for every path when confidence is low, so we
never present a shaky answer as if it were certain.
"""
from config import MIN_CONFIDENCE, UNSURE, get_llm
import router
import history


_CONDENSE = """Given the conversation so far and a follow-up message, rewrite the
follow-up as a standalone question that makes sense on its own (resolve pronouns
like "it"/"that" and implicit references using the history). If it is already
standalone, return it unchanged. Return ONLY the question, nothing else.

Conversation:
{history}

Follow-up: {q}
Standalone question:"""


def _standalone(q: str, history_text: str) -> str:
    """Rewrite a follow-up into a self-contained question using chat history."""
    if not history_text:
        return q
    llm = get_llm()
    if llm is None:
        return q
    try:
        out = llm.invoke(_CONDENSE.format(history=history_text, q=q)).content.strip()
        return out or q
    except Exception:
        return q


def answer(query: str, session_id: str = "default") -> dict:
    q = (query or "").strip()
    if not q:
        return _pack("Ask me anything about NEPSE stocks.",
                     route="conversational", confidence=0.2, meta={})

    # Use recent history to turn a follow-up into a standalone question, so
    # routing, retrieval and SQL all see the full intent.
    hist = history.as_text(session_id)
    standalone = _standalone(q, hist)

    r = router.classify(standalone)

    if r.route == router.ANALYTICAL:
        import analytical
        res = analytical.answer(standalone)
        meta = {"verified": res.verified, "verification": res.verification,
                "blocked": res.blocked, "sources": res.sources}
        result = _pack(res.answer, r.route, res.confidence, meta, r, standalone, q)

    elif r.route == router.PREDICTION:
        import forecast
        res = forecast.forecast(standalone)
        meta = {"symbol": res.symbol, "horizon_days": res.horizon_days,
                "projection": res.projection}
        result = _pack(res.answer, r.route, res.confidence, meta, r, standalone, q)

    elif r.route == router.WEBSEARCH:
        import websearch
        res = websearch.answer(standalone)
        meta = {"sources": res.sources}
        result = _pack(res.answer, r.route, res.confidence, meta, r, standalone, q)

    else:
        import rag
        res = rag.answer(standalone)
        meta = {"sources": res.sources}
        result = _pack(res.answer, r.route, res.confidence, meta, r, standalone, q)

    # Remember this turn for the next follow-up.
    history.append(session_id, q, result["answer"])
    return result


def _pack(text, route, confidence, meta, route_obj=None,
          standalone=None, original=None):
    hedged = text
    # Don't double-hedge if the path already said it plainly.
    if confidence < MIN_CONFIDENCE and UNSURE not in text and not text.startswith(
            "I can't provide this answer"):
        hedged = f"{UNSURE}\n\n{text}"
    return {
        "answer": hedged,
        "route": route,
        "confidence": round(float(confidence), 2),
        "low_confidence": confidence < MIN_CONFIDENCE,
        "routing_reason": route_obj.reason if route_obj else "",
        # Surface the rewritten question only when it actually changed.
        "rewritten_query": standalone if standalone and standalone != original else None,
        "meta": meta,
    }


if __name__ == "__main__":
    import sys
    import json
    q = " ".join(sys.argv[1:]) or "What is a stock market index?"
    print(json.dumps(answer(q), indent=2, default=str))
