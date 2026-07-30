#!/usr/bin/env python3
"""index2: the full orchestrator with a friendly-counsellor persona layer.

Same four routes and confidence gating as orchestrator.py — analytical (SQL +
verifier), prediction (statistical forecast), websearch (DuckDuckGo), and
conversational (RAG) — but the final answer is re-voiced by a NEPSE counsellor:
warm and patient, replies in the SAME language as the user (so Nepali questions
get Nepali answers), and adapts depth to the user's apparent expertise.

The persona restyles; it never invents facts. Grounded numbers (SQL/verifier,
forecasts, web sources) pass through unchanged — the persona only adds tone and
language. The one exception is safe: on the conversational path, if RAG has no
answer, the persona may answer a *general* finance concept from its own
knowledge (no specific NEPSE figures). Low confidence is expressed gently, in
the user's language, instead of the English hedge orchestrator.py prepends.
"""
import router
import history
from config import MIN_CONFIDENCE, UNSURE, get_llm
from orchestrator import _standalone  # reuse: identical follow-up condensing


_PERSONA = (
    "You are a warm, friendly NEPSE (Nepal Stock Exchange) counsellor. You "
    "explain patiently and adapt to the user's expertise: if they sound like a "
    "beginner, keep it simple and define any jargon; if they sound advanced, be "
    "concise and technical. ALWAYS reply in the same language as the user's "
    "question — if they wrote in Nepali, answer in Nepali."
)

_RESTYLE = _PERSONA + """

Rewrite the DRAFT into your counsellor voice for the user's question.
Rules:
- Keep every fact exactly. Do NOT add, drop, or change any number, ticker,
  date, or figure that appears in the draft.
- Do NOT introduce facts of your own.
- UNCERTAIN={uncertain}. If yes, gently note you are not fully certain (in the
  user's language) without inventing anything.
- Be concise.

User question: {q}
Draft answer: {draft}
Your reply:"""

_CONCEPT = _PERSONA + """

Answer this general NEPSE/finance question from your own knowledge, as a
counsellor. Only for general concepts and definitions — if it needs specific
live prices, figures, or dates you do not have, say so gently. Never invent
specific figures, tickers, or dates.

User question: {q}
Your reply:"""


def _personalize(q: str, draft: str, route: str, low_conf: bool) -> str:
    """Re-voice a grounded draft in the counsellor persona. Facts pass through."""
    llm = get_llm(temperature=0.4)
    if llm is None:
        return draft  # no LLM -> honest passthrough, no persona
    d = (draft or "").strip()
    # Conversational dead-end -> let the persona answer the concept itself.
    if route == router.CONVERSATIONAL and d == UNSURE:
        prompt = _CONCEPT.format(q=q)
    else:
        uncertain = "yes" if (low_conf or d == UNSURE
                              or d.startswith("I can't")) else "no"
        prompt = _RESTYLE.format(q=q, draft=draft, uncertain=uncertain)
    try:
        return llm.invoke(prompt).content.strip() or draft
    except Exception:
        return draft


def _dispatch(route: str, q: str):
    """Run the chosen path. Returns (text, confidence, meta) — mirrors orchestrator."""
    if route == router.ANALYTICAL:
        import analytical
        res = analytical.answer(q)
        return res.answer, res.confidence, {
            "verified": res.verified, "verification": res.verification,
            "blocked": res.blocked, "sources": res.sources}
    if route == router.PREDICTION:
        import forecast
        res = forecast.forecast(q)
        return res.answer, res.confidence, {
            "symbol": res.symbol, "horizon_days": res.horizon_days,
            "projection": res.projection}
    if route == router.WEBSEARCH:
        import websearch
        res = websearch.answer(q)
        return res.answer, res.confidence, {"sources": res.sources}
    import rag
    res = rag.answer(q)
    return res.answer, res.confidence, {"sources": res.sources}


def answer(query: str, session_id: str = "default") -> dict:
    q = (query or "").strip()
    if not q:
        return _pack("Ask me anything about NEPSE stocks.",
                     "conversational", 0.2, {})

    hist = history.as_text(session_id)
    standalone = _standalone(q, hist)
    r = router.classify(standalone)

    draft, confidence, meta = _dispatch(r.route, standalone)
    final = _personalize(q, draft, r.route, confidence < MIN_CONFIDENCE)

    history.append(session_id, q, final)
    return _pack(final, r.route, confidence, meta, r, standalone, q)


def _pack(text, route, confidence, meta, route_obj=None,
          standalone=None, original=None):
    # No English UNSURE prepend here: the persona already hedges in-language.
    return {
        "answer": text,
        "route": route,
        "confidence": round(float(confidence), 2),
        "low_confidence": confidence < MIN_CONFIDENCE,
        "routing_reason": route_obj.reason if route_obj else "",
        "rewritten_query": standalone if standalone and standalone != original else None,
        "meta": meta,
    }


if __name__ == "__main__":
    import sys
    import json
    # Offline self-check: with no LLM, the persona layer is a pure passthrough.
    if get_llm() is None:
        assert _personalize("x", "draft", router.CONVERSATIONAL, False) == "draft"
        assert _personalize("x", UNSURE, router.ANALYTICAL, True) == UNSURE
        print("self-check ok (no LLM configured; persona passthrough)")
    else:
        q = " ".join(sys.argv[1:]) or "NEPSE ke ho?"
        print(json.dumps(answer(q), indent=2, default=str))
