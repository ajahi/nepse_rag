#!/usr/bin/env python3
"""Append-only conversation logger.

One JSON line per turn -> logs/conversations.jsonl (override with CHAT_LOG_FILE).
Kept dead simple: no DB dependency, always works, easy to grep or load into
Postgres later. Logging never breaks a chat — any failure is swallowed.

Read it back later, e.g.:
    cat logs/conversations.jsonl | jq 'select(.route=="analytical")'
"""
import os
import json
import threading
import datetime as dt

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.environ.get("CHAT_LOG_FILE",
                          os.path.join(_LOG_DIR, "conversations.jsonl"))
FEEDBACK_FILE = os.environ.get("FEEDBACK_FILE",
                               os.path.join(_LOG_DIR, "feedback.jsonl"))

_lock = threading.Lock()


def _append(path: str, entry: dict) -> None:
    """Append one JSON line. Never raises — logging must not break the app."""
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **entry}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_turn(session_id: str, message: str, result: dict) -> None:
    """Append one turn. `result` is the orchestrator's response dict."""
    meta = result.get("meta", {}) or {}
    _append(LOG_FILE, {
        "session_id": session_id or "default",
        "message": message,
        "route": result.get("route"),
        "confidence": result.get("confidence"),
        "low_confidence": result.get("low_confidence"),
        "routing_reason": result.get("routing_reason"),
        "confidence_reason": result.get("confidence_reason"),
        "draft": result.get("draft"),
        "verified": meta.get("verified"),
        "blocked": meta.get("blocked"),
        "sources": meta.get("sources"),
        "answer": result.get("answer"),
    })


def log_feedback(session_id: str, feedback: str, context=None) -> None:
    """Append one user feedback entry, tied to the session."""
    _append(FEEDBACK_FILE, {
        "session_id": session_id or "default",
        "feedback": feedback,
        "context": context,
    })
