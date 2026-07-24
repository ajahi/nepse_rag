# NEPSE Chat

A routed AI chatbot over NEPSE stock data. One chat box, four skills, a
verification layer, and an honest "I'm not sure" when confidence is low.

## How it works

```
user message
    │
    ▼
router.py ──── keyword rules first, LLM classifier when ambiguous
    │
    ├── analytical    → analytical.py → nepse_agent (text2SQL) → verifier.py  (numbers re-checked vs DB)
    ├── prediction    → forecast.py   → trend + moving-average projection with a wide uncertainty band
    ├── websearch     → websearch.py  → live DuckDuckGo results synthesized by Groq (real-time facts)
    └── conversational→ rag.py        → Cohere-embedded FAISS knowledge base, answers only from context
    │
    ▼
orchestrator.py ── if confidence < MIN_CONFIDENCE, prepend the honest hedge
    │
    ▼
app.py (FastAPI) → static/index.html chat UI  (route + confidence + verification badges)
```

Same stack as your `app28.py`: **Groq `openai/gpt-oss-120b`** for the LLM (shared
by the SQL agent, RAG, and web search) and **Cohere `embed-v4.0`** for embeddings.

Anti-hallucination is layered:
- Analytical answers are machine-verified — every number is re-queried against
  the database (`verifier.py`); a mismatch **blocks** the answer.
- RAG answers come only from retrieved context; weak retrieval → "I'm not sure".
- Web search answers are grounded only in the returned snippets, with source URLs.
- Prediction is framed as a statistical extrapolation, reporting a range not a point.
- Any low-confidence answer is hedged by the orchestrator.

## Run

```bash
cd nepse_chat_app
pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY, COHERE_API_KEY, PG_DSN
python build_index.py         # embed knowledge_base.md -> nepse_kb/  (needs Cohere key)
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

The app **boots even with nothing configured** — it just answers "I'm not sure"
on paths whose dependencies (LLM key / Cohere index / database) are missing.
Check `/api/health` to see what's wired.

## Run with Docker (recommended)

Brings up the app + a TimescaleDB database (your `index.sql` dump) in one command.

```bash
# 1. keys — create .env (NOT in the repo) with your Groq + Cohere keys
cp .env.example .env && nano .env      # GROQ_API_KEY, COHERE_API_KEY

# 2. the DB dump is NOT in git (too big) — put index.sql next to docker-compose.yml
#    (scp it from wherever it lives)

# 3. build + run
docker compose up -d --build
# app: http://<server-ip>:8002   ·   logs: docker compose logs -f
```

First start restores the ~64 MB TimescaleDB dump, which takes a few minutes; the
`web` container waits for the DB to be healthy before it starts.

Two things that WILL trip you up on a fresh clone:
- **`.env` is gitignored** — it won't exist after `git clone`. Create it.
- **`index.sql` is gitignored** — it won't exist after `git clone` either. Copy it
  in manually. Without it the DB starts empty and analytical/prediction answer
  "I'm not sure" (RAG + web search still work).

No database handy? Run just the app and point it at any Postgres:
`docker compose up -d --build web` with `PG_DSN` set in `.env`.

## Editing the knowledge base

It's deliberately small — enough for basic NEPSE questions. Edit
`knowledge_base.md`, then re-run `python build_index.py` to rebuild `nepse_kb/`.

## Files

| File | Role |
|------|------|
| `router.py` | classify → analytical / prediction / websearch / conversational |
| `analytical.py` | wraps `nepse_agent.py` text2SQL + `verifier.py` gating |
| `forecast.py` | simple statistical forecast with uncertainty band |
| `rag.py` | Cohere + FAISS retrieval, grounded answering |
| `websearch.py` | keyless DuckDuckGo search + Groq synthesis |
| `build_index.py` | embed `knowledge_base.md` into `nepse_kb/` with Cohere |
| `orchestrator.py` | routing + confidence hedge, single `answer()` entry |
| `app.py` | FastAPI: `/api/chat`, `/api/health`, serves the UI |
| `static/index.html` | single-file chat frontend |
| `nepse_agent.py`, `verifier.py`, `notifier_26.py` | your existing MVP, reused |

Neo4j is intentionally not wired in — SQL is the analytical backend.
