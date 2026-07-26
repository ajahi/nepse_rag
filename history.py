#!/usr/bin/env python3
"""Per-session conversation memory (in-process).

Keeps the last few turns per session_id so follow-up questions can be understood
in context. In-memory by design — simple and fast. Note: with multiple uvicorn
workers this isn't shared across processes; for that, back it with Redis or a DB.
The durable record of conversations is chatlog.py (JSONL); this is just the
short working memory the LLM sees.
"""
import os
import threading
from collections import defaultdict, deque

MAX_TURNS = int(os.environ.get("HISTORY_TURNS", "6"))

_store = defaultdict(lambda: deque(maxlen=MAX_TURNS))
_lock = threading.Lock()


def get(session_id: str) -> list:
    with _lock:
        return list(_store[session_id or "default"])


def append(session_id: str, user: str, bot: str) -> None:
    with _lock:
        _store[session_id or "default"].append({"user": user, "bot": bot})


def as_text(session_id: str, max_turns: int = 4) -> str:
    """Recent turns as a compact transcript for prompting."""
    turns = get(session_id)[-max_turns:]
    return "\n".join(f"User: {t['user']}\nBot: {t['bot']}" for t in turns)
