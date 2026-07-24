#!/usr/bin/env python3
"""FastAPI backend for the NEPSE chatbot.

Serves the static chat UI and a single /api/chat endpoint that runs the
orchestrator (router -> path -> verifier/confidence gate).

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import orchestrator
from config import HAS_LLM, HAS_DB

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="NEPSE Chat")


class ChatIn(BaseModel):
    message: str


@app.get("/api/health")
def health():
    return {"ok": True, "llm": HAS_LLM, "db": HAS_DB}


@app.post("/api/chat")
def chat(body: ChatIn):
    try:
        return orchestrator.answer(body.message)
    except Exception as e:
        return {"answer": f"Something went wrong: {e}", "route": "error",
                "confidence": 0.0, "low_confidence": True,
                "routing_reason": "", "meta": {}}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
