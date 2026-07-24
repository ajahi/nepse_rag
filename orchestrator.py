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
from config import MIN_CONFIDENCE, UNSURE
import router


def answer(query: str) -> dict:
    q = (query or "").strip()
    if not q:
        return _pack("Ask me anything about NEPSE stocks.",
                     route="conversational", confidence=0.2, meta={})

    r = router.classify(q)

    if r.route == router.ANALYTICAL:
        import analytical
        res = analytical.answer(q)
        meta = {"verified": res.verified, "verification": res.verification,
                "blocked": res.blocked, "sources": res.sources}
        return _pack(res.answer, r.route, res.confidence, meta, r)

    if r.route == router.PREDICTION:
        import forecast
        res = forecast.forecast(q)
        meta = {"symbol": res.symbol, "horizon_days": res.horizon_days,
                "projection": res.projection}
        return _pack(res.answer, r.route, res.confidence, meta, r)

    if r.route == router.WEBSEARCH:
        import websearch
        res = websearch.answer(q)
        meta = {"sources": res.sources}
        return _pack(res.answer, r.route, res.confidence, meta, r)

    import rag
    res = rag.answer(q)
    meta = {"sources": res.sources}
    return _pack(res.answer, r.route, res.confidence, meta, r)


def _pack(text, route, confidence, meta, route_obj=None):
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
        "meta": meta,
    }


if __name__ == "__main__":
    import sys
    import json
    q = " ".join(sys.argv[1:]) or "What is a stock market index?"
    print(json.dumps(answer(q), indent=2, default=str))
