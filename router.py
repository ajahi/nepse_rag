#!/usr/bin/env python3
"""Query router: decide which skill/path answers a user message.

Four routes:
  - analytical    -> text2SQL agent over NEPSE prices (facts, aggregates, rankings)
  - prediction    -> statistical forecast from historical data
  - websearch     -> live internet search for real-time / current facts
  - conversational-> RAG over the knowledge base / general finance Q&A

Strategy (cheap first): keyword rules give a fast, confident answer for the
obvious cases. Only when rules are ambiguous do we spend an LLM call. The LLM
returns strict JSON; if that fails we fall back to "conversational" with low
confidence so the orchestrator can hedge.
"""
import json
import re
from dataclasses import dataclass

from config import get_llm

ANALYTICAL = "analytical"
PREDICTION = "prediction"
WEBSEARCH = "websearch"
CONVERSATIONAL = "conversational"
ROUTES = (ANALYTICAL, PREDICTION, WEBSEARCH, CONVERSATIONAL)

# --- keyword signals ------------------------------------------------------
_PREDICTION_KW = re.compile(
    r"\b(predict|forecast|projection|will\s+\w+\s+(rise|fall|go|be)|"
    r"next\s+(week|month|quarter|year|day|days)|future|expect(ed)?|"
    r"outlook|estimate.*(future|next)|going\s+to\s+(rise|fall|be))\b", re.I)

# Real-time / current-fact intent -> go to the internet, not the historical DB.
_WEBSEARCH_KW = re.compile(
    r"\b(latest|current(ly)?|today|todays|right\s+now|as\s+of\s+now|this\s+week|"
    r"recent(ly)?|news|headline|breaking|live|real[\s-]?time|up[\s-]?to[\s-]?date|"
    r"nowadays|these\s+days|at\s+the\s+moment|announced|just\s+happened)\b", re.I)

_ANALYTICAL_KW = re.compile(
    r"\b(highest|lowest|most|least|top|bottom|average|avg|mean|median|sum|total|"
    r"how\s+many|how\s+much|count|gain|gained|loss|lost|return|volume|turnover|"
    r"market\s*cap|compare|versus|vs\.?|between|rank|list|which\s+stock|"
    r"close|open|high|low|price\s+on|traded)\b", re.I)

_CONVERSATIONAL_KW = re.compile(
    r"\b(what\s+is|what\s+are|explain|define|definition|how\s+do(es)?|"
    r"why|meaning|tell\s+me\s+about|difference\s+between|hello|hi|hey|thanks|"
    r"who\s+are\s+you|help)\b", re.I)


@dataclass
class Route:
    route: str
    confidence: float
    reason: str

    def as_dict(self):
        return {"route": self.route, "confidence": self.confidence,
                "reason": self.reason}


def _rule_scores(q: str) -> dict:
    return {
        PREDICTION: len(_PREDICTION_KW.findall(q)) * 1.0,
        WEBSEARCH: len(_WEBSEARCH_KW.findall(q)) * 1.0,
        ANALYTICAL: len(_ANALYTICAL_KW.findall(q)) * 1.0,
        CONVERSATIONAL: len(_CONVERSATIONAL_KW.findall(q)) * 0.8,
    }


_LLM_PROMPT = """Classify the user's NEPSE question into exactly one route.

Routes:
- analytical: needs facts/aggregates from the historical price database
  (rankings, averages, gains, volumes, "which stock...", specific past prices).
- prediction: asks about the future (forecast, "will X rise", next month/year).
- websearch: needs current/real-time info not in a historical DB (today's index,
  latest news, "current" anything, recent announcements).
- conversational: general finance concepts, definitions, greetings.

Return ONLY JSON: {{"route": "...", "confidence": 0.0-1.0}}
Question: {q}
JSON:"""


def _llm_route(q: str) -> Route:
    llm = get_llm()
    if llm is None:
        return Route(CONVERSATIONAL, 0.3, "no rules matched; LLM unavailable")
    try:
        raw = llm.invoke(_LLM_PROMPT.format(q=q)).content
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e + 1])
        route = data["route"] if data.get("route") in ROUTES else CONVERSATIONAL
        return Route(route, float(data.get("confidence", 0.6)), "LLM classifier")
    except Exception:
        return Route(CONVERSATIONAL, 0.3, "LLM classification failed")


def classify(q: str) -> Route:
    """Return the chosen Route. Rules first, LLM only when ambiguous."""
    q = (q or "").strip()
    if not q:
        return Route(CONVERSATIONAL, 0.2, "empty query")

    scores = _rule_scores(q)

    # Real-time intent wins even against analytical keywords: "current price" and
    # "latest gainers" belong on the web, not the historical DB.
    if scores[WEBSEARCH] >= 1:
        return Route(WEBSEARCH, min(0.9, 0.6 + 0.1 * scores[WEBSEARCH]),
                     "real-time keywords")

    # Prediction intent is a strong, specific signal — trust it even alone.
    if scores[PREDICTION] >= 1 and scores[PREDICTION] >= scores[ANALYTICAL]:
        return Route(PREDICTION, min(0.9, 0.6 + 0.1 * scores[PREDICTION]),
                     "prediction keywords")

    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    if scores[best] >= 0.8 and margin >= 0.8:
        return Route(best, min(0.9, 0.55 + 0.12 * scores[best]), "keyword rules")

    # Ambiguous -> ask the LLM.
    return _llm_route(q)
