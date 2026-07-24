#!/usr/bin/env python3
"""
Post-answer verification layer for the NEPSE GraphRAG agent.

Catches the exact failure you hit: the agent claimed ANLB was worth 436 on
2023-01-01, but ANLB's first real row is 2023-05-02 — the number was fabricated.

Pipeline:
    1. extract_claims(): LLM turns the final answer into structured claims
       (symbol, date/period, metric, value). Extraction only — no new numbers.
    2. verify(): every claim is re-checked DETERMINISTICALLY against the DB:
         a. symbol exists?
         b. claimed date/period inside that symbol's actual coverage window?
            (this alone catches the ANLB case)
         c. claimed price within tolerance of the real close on the nearest
            trading day?
         d. claimed gain % recomputed from real first/last closes of the period?
    3. Caller blocks the answer + notifies a human if anything fails.

The LLM is never trusted for numbers here — only for reading its own answer.
"""
import json
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger("nepse.verifier")

PRICE_TOLERANCE = 0.02      # 2% relative diff allowed (rounding in answers)
GAIN_TOLERANCE_PP = 3.0     # gain % may differ by 3 percentage points
NEAREST_DAY_WINDOW = 7      # market holidays: match nearest trading day +/- 7d


# ==================== DB adapters (deterministic side) ====================
class BaseDB:
    """fetch_coverage / fetch_close for one backend. No LLM involved."""

    def fetch_coverage(self, symbol: str) -> Optional[tuple]:
        """(first_date, last_date, n_rows) or None if symbol unknown."""
        raise NotImplementedError

    def fetch_close(self, symbol: str, date: dt.date) -> Optional[tuple]:
        """(actual_date, close) for nearest trading day within window, else None."""
        raise NotImplementedError


class PostgresDB(BaseDB):
    def __init__(self, conn):
        self.conn = conn

    def _one(self, q, params):
        with self.conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchone()

    def fetch_coverage(self, symbol):
        r = self._one(
            "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) "
            "FROM daily_prices WHERE symbol = %s", (symbol,))
        return None if not r or r[2] == 0 else (r[0], r[1], r[2])

    def fetch_close(self, symbol, date):
        r = self._one(
            "SELECT trade_date, close FROM daily_prices "
            "WHERE symbol = %s AND trade_date BETWEEN %s AND %s "
            "ORDER BY ABS(trade_date - %s) LIMIT 1",
            (symbol, date - dt.timedelta(days=NEAREST_DAY_WINDOW),
             date + dt.timedelta(days=NEAREST_DAY_WINDOW), date))
        return (r[0], float(r[1])) if r else None


class Neo4jDB(BaseDB):
    def __init__(self, driver):
        self.driver = driver

    def _one(self, q, **params):
        with self.driver.session() as s:
            rec = s.run(q, **params).single()
            return rec

    def fetch_coverage(self, symbol):
        r = self._one(
            "MATCH (:Ticker {symbol:$sym})-[:HAS_PRICE]->(p:Price) "
            "RETURN min(p.date) AS d1, max(p.date) AS d2, count(p) AS n",
            sym=symbol)
        if not r or not r["n"]:
            return None
        return (r["d1"].to_native(), r["d2"].to_native(), r["n"])

    def fetch_close(self, symbol, date):
        r = self._one(
            "MATCH (:Ticker {symbol:$sym})-[:HAS_PRICE]->(p:Price) "
            "WHERE p.date >= date($lo) AND p.date <= date($hi) "
            "RETURN p.date AS d, p.close AS c "
            "ORDER BY abs(duration.inDays(p.date, date($target)).days) LIMIT 1",
            sym=symbol,
            lo=str(date - dt.timedelta(days=NEAREST_DAY_WINDOW)),
            hi=str(date + dt.timedelta(days=NEAREST_DAY_WINDOW)),
            target=str(date))
        return (r["d"].to_native(), float(r["c"])) if r else None


# ==================== claims ====================
@dataclass
class Claim:
    symbol: str
    kind: str                       # "price" | "gain"
    value: float
    date: Optional[str] = None      # for price claims, YYYY-MM-DD
    period_start: Optional[str] = None  # for gain claims
    period_end: Optional[str] = None


@dataclass
class CheckResult:
    claim: Claim
    ok: bool
    detail: str


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.ok]

    def summary(self) -> str:
        lines = [f"{'PASS' if r.ok else 'FAIL'}  {r.claim.symbol} {r.claim.kind} "
                 f"{r.claim.value} ({r.claim.date or f'{r.claim.period_start}..{r.claim.period_end}'})"
                 f" -> {r.detail}" for r in self.results]
        return "\n".join(lines) or "no verifiable claims found"


EXTRACT_PROMPT = """Extract every factual numeric claim about a stock from the answer below.
Output ONLY a JSON array, no prose. Each element:
  {{"symbol": "TICKER", "kind": "price"|"gain", "value": number,
   "date": "YYYY-MM-DD" or null, "period_start": "YYYY-MM-DD" or null,
   "period_end": "YYYY-MM-DD" or null}}
Rules:
- "price": a specific price at (or near) a specific date. Set "date".
  If the answer says a value "in 2022" with no date, use "YYYY-01-01" of that year.
  A phrase like "gained from 436 to 6110 between 2022 and 2025" yields TWO price
  claims (start value at period start, end value at period end) plus one gain claim
  if a percentage is stated.
- "gain": a percentage change over a period. Set period_start/period_end
  (use Jan 1 / Dec 31 when only years are given). value = the percent number.
- Skip claims with no number. Empty array if none.

Question: {question}
Answer: {answer}
JSON:"""


def extract_claims(llm, question: str, answer: str) -> List[Claim]:
    raw = llm.invoke(EXTRACT_PROMPT.format(question=question, answer=answer)).content
    # tolerate code fences / stray prose around the array
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        logger.warning(f"Claim extraction returned no JSON: {raw[:200]}")
        return []
    claims = []
    try:
        for c in json.loads(raw[start:end + 1]):
            claims.append(Claim(
                symbol=str(c["symbol"]).upper().strip(),
                kind=c["kind"],
                value=float(c["value"]),
                date=c.get("date"),
                period_start=c.get("period_start"),
                period_end=c.get("period_end"),
            ))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        logger.warning(f"Bad claim JSON: {e}")
    return claims


# ==================== verification ====================
def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _check_price(db: BaseDB, claim: Claim, cov) -> CheckResult:
    date = _d(claim.date)
    first, last, _ = cov
    if date < first or date > last:
        return CheckResult(claim, False,
                           f"OUT OF COVERAGE: {claim.symbol} has data only "
                           f"{first}..{last}, claim is about {date}")
    row = db.fetch_close(claim.symbol, date)
    if row is None:
        return CheckResult(claim, False, f"no trading day within "
                           f"{NEAREST_DAY_WINDOW}d of {date}")
    actual_date, actual_close = row
    rel = abs(claim.value - actual_close) / actual_close if actual_close else 1.0
    if rel <= PRICE_TOLERANCE:
        return CheckResult(claim, True, f"close on {actual_date} = {actual_close}")
    return CheckResult(claim, False,
                       f"VALUE MISMATCH: claimed {claim.value}, actual close on "
                       f"{actual_date} = {actual_close} ({rel:.1%} off)")


def _check_gain(db: BaseDB, claim: Claim, cov) -> CheckResult:
    first, last, _ = cov
    ps, pe = _d(claim.period_start), _d(claim.period_end)
    if ps < first:
        return CheckResult(claim, False,
                           f"PERIOD STARTS BEFORE COVERAGE: {claim.symbol} first "
                           f"data {first}, claimed period starts {ps}")
    start_row = db.fetch_close(claim.symbol, ps)
    end_row = db.fetch_close(claim.symbol, min(pe, last))
    if not start_row or not end_row:
        return CheckResult(claim, False, "cannot resolve period endpoints in data")
    (_, c1), (d2, c2) = start_row, end_row
    actual_gain = (c2 - c1) / c1 * 100 if c1 else 0.0
    if abs(actual_gain - claim.value) <= max(GAIN_TOLERANCE_PP,
                                             abs(actual_gain) * PRICE_TOLERANCE):
        return CheckResult(claim, True, f"recomputed gain {actual_gain:.1f}% "
                           f"({c1} -> {c2} by {d2})")
    return CheckResult(claim, False,
                       f"GAIN MISMATCH: claimed {claim.value}%, recomputed "
                       f"{actual_gain:.1f}% ({c1} -> {c2})")


def verify(db: BaseDB, claims: List[Claim]) -> Report:
    report = Report()
    for claim in claims:
        try:
            cov = db.fetch_coverage(claim.symbol)
            if cov is None:
                report.results.append(CheckResult(
                    claim, False, f"UNKNOWN SYMBOL: {claim.symbol} not in database"))
                continue
            if claim.kind == "price" and claim.date:
                report.results.append(_check_price(db, claim, cov))
            elif claim.kind == "gain" and claim.period_start and claim.period_end:
                report.results.append(_check_gain(db, claim, cov))
            else:
                report.results.append(CheckResult(
                    claim, False, "claim missing date/period — cannot verify"))
        except Exception as e:
            report.results.append(CheckResult(claim, False, f"verify error: {e}"))
    return report


def verify_answer(llm, db: BaseDB, question: str, answer: str) -> Report:
    """One-call entry point: extract claims from the answer, check them all."""
    return verify(db, extract_claims(llm, question, answer))
