#!/usr/bin/env python3
"""
Multi-hop Q&A agent over NEPSE data — Neo4j (Cypher) vs Postgres (SQL) —
now with a deterministic VERIFICATION layer (verifier.py) + HITL notification.

What changed vs the old version (the ANLB fix):
  1. data_coverage tool — agent must check a symbol's REAL first/last trade
     date before answering about any period. Per-ticker coverage varies;
     the old prompt's global "2023-01-01..2026-03-29" made the LLM assume
     every ticker existed on 2023-01-01 and fabricate baselines.
  2. Hardened system prompt — out-of-coverage periods must be disclosed,
     never interpolated.
  3. Post-answer verification — every numeric claim in the final answer is
     extracted and re-queried against the DB. Any failure -> answer BLOCKED
     and an email alert goes out via notifier_26 (human-in-the-loop).

Usage:
    python nepse_agent.py --backend neo4j "Which stock gained the most in 2024?"
    python nepse_agent.py --backend sql   --chat
    python nepse_agent.py --backend both  "..."
    python nepse_agent.py --no-verify ... # old behaviour (not recommended)

Deps: pip install langgraph langchain-groq neo4j psycopg2-binary

Env vars: GROQ_API_KEY; NEO4J_URI/USERNAME/PASSWORD or PG_DSN;
          SMTP_USER/SMTP_PASS/ALERT_TO for alerts (see notifier_26.py).
"""
import os
import time
import json
import asyncio
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import verifier
from notifier_26 import EmailNotifier, NotificationManager

MAX_RESULT_CHARS = 4000

# ==================== SCHEMA DESCRIPTIONS ====================
COVERAGE_WARNING = """
CRITICAL — per-ticker coverage varies:
- The date range above is the OVERALL dataset range. Individual tickers were
  listed at different times; many have NO data before some later date.
- BEFORE reporting any figure for a specific symbol over a period, call the
  data_coverage tool for that symbol. If the requested period starts before
  the symbol's first trade date, SAY SO explicitly and report only from the
  actual first date. NEVER supply a value for a date the data doesn't cover.
- The dataset contains nothing before 2023-01-01. Questions about 2022 or
  earlier cannot be answered — say so.
"""

NEO4J_SCHEMA = """
Graph schema:
  (t:Ticker {symbol: string})-[:HAS_PRICE]->(p:Price {
      date: date, open: float, high: float, low: float, close: float,
      volume: int, turnover: float, num_trades: int, market_cap: float})
Data: 406 NEPSE tickers, daily OHLCV, overall range 2023-01-01 .. 2026-03-29. NPR.
Notes:
- Filter years with: p.date >= date('2024-01-01') AND p.date < date('2025-01-01')
- Yearly gain of a ticker = (last close - first close) / first close within the year.
- Always LIMIT results (<= 25 rows).
""" + COVERAGE_WARNING

SQL_SCHEMA = """
Postgres table public.daily_prices:
  symbol varchar, trade_date date, open numeric, high numeric, low numeric,
  close numeric, volume int, turnover numeric, num_trades int, market_cap numeric
Data: 406 NEPSE tickers, daily OHLCV, overall range 2023-01-01 .. 2026-03-29. NPR.
Notes:
- Yearly gain: DISTINCT ON or window functions per symbol per year.
- Always LIMIT results (<= 25 rows). Read-only: SELECT statements only.
""" + COVERAGE_WARNING

SYSTEM_TEMPLATE = """You are a NEPSE stock-data analyst agent.

Today's date is {today}. The database is historical and only goes up to its last
loaded trade date, which may be BEFORE today.
- "today's / latest / current / most recent price" with no explicit date means
  the most recent row available, i.e. the row with MAX(trade_date) for that
  symbol. Query it directly (e.g. ORDER BY trade_date DESC LIMIT 1); do NOT guess
  a calendar date, and do NOT filter by today's date literally.
- Always report the actual trade_date you used, and if it is well before today
  (e.g. weeks/months), say the market data isn't more recent than that date.

{schema}

Approach:
1. Think about whether the question needs one query or several dependent steps.
2. For any question naming or resolving to specific symbols over a time period,
   call data_coverage for those symbols FIRST and respect their real windows.
3. For multi-step questions, run a query, LOOK at the result, then decide the
   next query using those concrete values.
4. Numbers must come from query results — never estimate, interpolate or invent.
   Your final answer is machine-verified against the database; any number that
   doesn't match a real row gets the answer rejected.
5. If a query errors, read the error and fix the query (max 3 retries).
6. Final answer: concise; name ticker(s), exact figures, exact dates used
   (e.g. "from 2023-05-02 (first available) to 2025-12-28"), and any coverage
   gaps versus what was asked. If data can't answer it, say so.
"""

# ==================== CONNECTIONS ====================
_neo4j_driver = None
_pg_conn = None


def neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )
    return _neo4j_driver


def pg_conn():
    global _pg_conn
    import psycopg2
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(os.environ["PG_DSN"])
        _pg_conn.set_session(readonly=True, autocommit=True)
    return _pg_conn


def _truncate(rows) -> str:
    s = json.dumps(rows, default=str)
    if len(s) > MAX_RESULT_CHARS:
        s = s[:MAX_RESULT_CHARS] + f'... [truncated, {len(rows)} rows total — use LIMIT or aggregate]'
    return s


# ==================== TOOLS ====================
_ACTIVE_BACKEND = "neo4j"  # set by build_agent; used by data_coverage


@tool
def run_cypher(query: str) -> str:
    """Run a read-only Cypher query against the NEPSE Neo4j graph and return rows as JSON."""
    try:
        with neo4j_driver().session() as s:
            rows = [r.data() for r in s.run(query)]
        return _truncate(rows)
    except Exception as e:
        return f"CYPHER ERROR: {e}"


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL SELECT against the NEPSE Postgres daily_prices table, rows as JSON."""
    q = query.strip().rstrip(";")
    if not q.lower().startswith(("select", "with")):
        return "SQL ERROR: only SELECT/WITH queries are allowed."
    try:
        import psycopg2.extras
        with pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(q)
            rows = cur.fetchall()
        return _truncate([dict(r) for r in rows])
    except Exception as e:
        return f"SQL ERROR: {e}"


@tool
def data_coverage(symbol: str) -> str:
    """Return the REAL first/last trade date and row count for one ticker symbol.
    ALWAYS call this before reporting figures for a symbol over a time period —
    many tickers were listed after the overall dataset start date."""
    try:
        db = get_verify_db(_ACTIVE_BACKEND)
        cov = db.fetch_coverage(symbol.upper().strip())
        if cov is None:
            return f"{symbol}: NOT IN DATABASE — do not answer about this symbol."
        first, last, n = cov
        return f"{symbol}: {n} rows, first trade date {first}, last trade date {last}."
    except Exception as e:
        return f"COVERAGE ERROR: {e}"


# ==================== VERIFICATION + NOTIFICATION ====================
def get_verify_db(backend: str) -> verifier.BaseDB:
    if backend == "neo4j":
        return verifier.Neo4jDB(neo4j_driver())
    return verifier.PostgresDB(pg_conn())


_notify_manager = None


def notify_manager() -> NotificationManager:
    global _notify_manager
    if _notify_manager is None:
        _notify_manager = NotificationManager([EmailNotifier()])
    return _notify_manager


BLOCKED_ANSWER = (
    "I can't provide this answer: automated verification found that one or more "
    "figures did not match the database, so it has been withheld and flagged for "
    "human review.\n\nVerification report:\n{report}"
)


def verify_and_gate(llm, backend: str, question: str, answer: str) -> tuple:
    """Returns (final_answer, report). Blocks + notifies on failure."""
    report = verifier.verify_answer(llm, get_verify_db(backend), question, answer)
    if report.passed:
        return answer, report
    body = (f"Question: {question}\n\nBlocked answer:\n{answer}\n\n"
            f"Verification report:\n{report.summary()}")
    try:
        asyncio.run(notify_manager().notify(
            "[NEPSE RAG] Hallucination blocked — human review needed", body))
    except Exception as e:
        print(f"  (alert send failed: {e})")
    return BLOCKED_ANSWER.format(report=report.summary()), report


# ==================== AGENT ====================
def build_agent(backend: str):
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )
    if backend == "neo4j":
        tools, schema = [run_cypher, data_coverage], NEO4J_SCHEMA
    else:
        tools, schema = [run_sql, data_coverage], SQL_SCHEMA
    from datetime import date
    agent = create_react_agent(
        llm, tools,
        prompt=SYSTEM_TEMPLATE.format(schema=schema, today=date.today().isoformat()))
    return agent, llm


def ask(agent, question: str, verbose: bool = True) -> str:
    final = ""
    for step in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        msg = step["messages"][-1]
        if verbose:
            kind = getattr(msg, "type", "?")
            if kind == "ai" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    arg = tc["args"].get("query") or tc["args"].get("symbol") or ""
                    print(f"  -> {tc['name']}: {str(arg)[:200]}")
            elif kind == "tool":
                print(f"  <- result: {str(msg.content)[:200]}")
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            final = msg.content
    return final


def run_question(question: str, backend: str, verify: bool = True):
    backends = ["neo4j", "sql"] if backend == "both" else [backend]
    for b in backends:
        print(f"\n{'=' * 60}\nBACKEND: {b}\n{'=' * 60}")
        agent, llm = build_agent(b)
        t0 = time.perf_counter()
        answer = ask(agent, question)
        if verify and answer:
            print("\n  [verifying answer against database...]")
            answer, report = verify_and_gate(llm, b, question, answer)
            print(f"  [verification: {'PASS' if report.passed else 'FAIL — answer blocked, alert sent'}]")
            if report.results:
                for line in report.summary().splitlines():
                    print(f"    {line}")
        dt = time.perf_counter() - t0
        print(f"\nAnswer ({dt:.1f}s):\n{answer}")


def main():
    p = argparse.ArgumentParser(description="NEPSE multi-hop agent with verification")
    p.add_argument("question", nargs="?", help="question to answer")
    p.add_argument("--backend", choices=["neo4j", "sql", "both"], default="neo4j")
    p.add_argument("--chat", action="store_true", help="interactive mode")
    p.add_argument("--no-verify", action="store_true", help="skip answer verification")
    args = p.parse_args()

    if args.chat:
        print(f"NEPSE agent ready (backend={args.backend}, verify={not args.no_verify}). 'quit' to exit.\n")
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            run_question(q, args.backend, verify=not args.no_verify)
            print()
    elif args.question:
        run_question(args.question, args.backend, verify=not args.no_verify)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
