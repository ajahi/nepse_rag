#!/usr/bin/env python3
"""Analytical path: reuse the existing text2SQL agent + deterministic verifier.

This is a thin wrapper over nepse_agent.py so we don't duplicate the SQL agent,
the data_coverage guard, or the numeric verification layer you already built.
The verifier re-checks every number against the DB; a failed answer is blocked.
"""
from dataclasses import dataclass, field

from config import HAS_LLM, HAS_DB, UNSURE


@dataclass
class AnalyticalResult:
    answer: str
    confidence: float
    verified: bool = False
    verification: str = ""
    blocked: bool = False
    sources: list = field(default_factory=list)


def answer(query: str) -> AnalyticalResult:
    if not HAS_LLM:
        return AnalyticalResult(
            f"{UNSURE} The analytical path needs the LLM (GROQ_API_KEY) to turn "
            f"your question into SQL.", 0.2)
    if not HAS_DB:
        return AnalyticalResult(
            f"{UNSURE} I can't run analytical queries without a live database "
            f"(PG_DSN is not configured).", 0.2)

    try:
        import nepse_agent as na
    except Exception as e:
        return AnalyticalResult(f"{UNSURE} SQL agent unavailable: {e}", 0.2)

    try:
        agent, llm = na.build_agent("sql")
        raw = na.ask(agent, query, verbose=False)
    except Exception as e:
        return AnalyticalResult(f"{UNSURE} The query failed to run: {e}", 0.25)

    if not raw:
        return AnalyticalResult(UNSURE, 0.3)

    # Deterministic verification + human-in-the-loop gating (existing logic).
    try:
        final, report = na.verify_and_gate(llm, "sql", query, raw)
        passed = report.passed
        summary = report.summary()
    except Exception as e:
        # If verification itself errors, return the answer but flag low confidence.
        return AnalyticalResult(raw, 0.4, verified=False,
                                verification=f"verification error: {e}")

    if not passed:
        return AnalyticalResult(final, 0.2, verified=False,
                                verification=summary, blocked=True)

    return AnalyticalResult(final, 0.85, verified=True, verification=summary)
