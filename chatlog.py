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

LOG_FILE = os.environ.get(
    "CHAT_LOG_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "logs", "conversations.jsonl"))

_lock = threading.Lock()


def log_turn(session_id: str, message: str, result: dict) -> None:
    """Append one turn. `result` is the orchestrator's response dict."""
    meta = result.get("meta", {}) or {}
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session_id": session_id or "default",
        "message": message,
        "route": result.get("route"),
        "confidence": result.get("confidence"),
        "low_confidence": result.get("low_confidence"),
        "routing_reason": result.get("routing_reason"),
        "verified": meta.get("verified"),
        "blocked": meta.get("blocked"),
        "sources": meta.get("sources"),
        "answer": result.get("answer"),
    }
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # logging must never break the chat
