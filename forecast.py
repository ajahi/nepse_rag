#!/usr/bin/env python3
"""Prediction path: a deliberately simple, honest statistical forecast.

We do NOT pretend to know the future. We fit a linear trend to recent closes,
blend it with a moving-average level, and project a horizon with an uncertainty
band derived from the data's own volatility. The band is wide on purpose and the
answer always states that this is an extrapolation, not a guarantee.

If there's no database, no matching symbol, or too little history, we return the
honest "I'm not sure" instead of inventing a number.
"""
import os
import re
import datetime as dt
from dataclasses import dataclass, field

from config import PG_DSN, UNSURE

_conn = None
_STOPWORDS = {"THE", "WILL", "WHAT", "WHEN", "NEXT", "WEEK", "YEAR", "PRICE",
              "STOCK", "SHARE", "NEPSE", "AND", "FOR", "HOW", "MUCH", "GO",
              "RISE", "FALL", "MONTH", "DAY", "DAYS", "FUTURE", "PREDICT"}


@dataclass
class ForecastResult:
    answer: str
    confidence: float
    symbol: str = ""
    horizon_days: int = 0
    projection: list = field(default_factory=list)  # [{date, mean, low, high}]


def _pg():
    global _conn
    if not PG_DSN:
        return None
    try:
        import psycopg2
        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(PG_DSN)
            _conn.set_session(readonly=True, autocommit=True)
        return _conn
    except Exception:
        # psycopg2 missing, or DB unreachable -> degrade instead of crashing.
        return None


def _candidate_symbols(q: str):
    # NEPSE tickers are short uppercase tokens; take them in appearance order.
    toks = re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", q)
    return [t for t in toks if t not in _STOPWORDS]


def _resolve_symbol(conn, q: str):
    for sym in _candidate_symbols(q):
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM daily_prices WHERE symbol=%s LIMIT 1",
                        (sym,))
            if cur.fetchone():
                return sym
    return None


def _recent_closes(conn, symbol: str, lookback: int = 120):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, close FROM daily_prices WHERE symbol=%s "
            "ORDER BY trade_date DESC LIMIT %s", (symbol, lookback))
        rows = cur.fetchall()
    rows.reverse()  # chronological
    return [(r[0], float(r[1])) for r in rows]


def _horizon(q: str) -> int:
    m = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", q, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return n * (7 if "week" in unit else 30 if "month" in unit else 1)
    if re.search(r"month", q, re.I):
        return 30
    if re.search(r"week", q, re.I):
        return 7
    return 30


def forecast(query: str) -> ForecastResult:
    conn = _pg()
    if conn is None:
        return ForecastResult(
            f"{UNSURE} I can't forecast without a live price database "
            f"(PG_DSN is not configured).", 0.2)

    symbol = _resolve_symbol(conn, query)
    if not symbol:
        return ForecastResult(
            f"{UNSURE} I couldn't identify a NEPSE symbol in your question to "
            f"forecast. Try including the ticker, e.g. \"forecast NABIL next month\".",
            0.25)

    series = _recent_closes(conn, symbol)
    if len(series) < 20:
        return ForecastResult(
            f"{UNSURE} There isn't enough recent history for {symbol} "
            f"({len(series)} points) to make even a rough projection.",
            0.25, symbol=symbol)

    import numpy as np
    closes = np.array([c for _, c in series], dtype=float)
    x = np.arange(len(closes))

    slope, intercept = np.polyfit(x, closes, 1)
    fitted = slope * x + intercept
    resid_std = float(np.std(closes - fitted))
    ma = float(np.mean(closes[-10:]))            # short moving-average level
    last_close = float(closes[-1])
    last_date = series[-1][0]

    horizon = _horizon(query)
    # Blend trend extrapolation with the MA level so a noisy slope can't run away.
    proj = []
    for h in range(1, horizon + 1):
        trend_val = slope * (len(closes) - 1 + h) + intercept
        mean = 0.6 * trend_val + 0.4 * ma
        band = resid_std * (1 + (h / horizon))   # widen with the horizon
        d = last_date + dt.timedelta(days=h)
        proj.append({"date": d.isoformat(),
                     "mean": round(mean, 2),
                     "low": round(mean - 1.96 * band, 2),
                     "high": round(mean + 1.96 * band, 2)})

    end = proj[-1]
    pct = (end["mean"] - last_close) / last_close * 100 if last_close else 0.0
    direction = "higher" if pct > 1 else "lower" if pct < -1 else "roughly flat"
    vol = resid_std / last_close * 100 if last_close else 0.0

    # Confidence is intentionally modest and shrinks with volatility / horizon.
    conf = max(0.25, min(0.55, 0.55 - vol / 100 - horizon / 400))

    answer = (
        f"Simple statistical projection for {symbol} (NOT a guarantee — markets "
        f"are not reliably predictable):\n"
        f"- Last close: {last_close:.2f} on {last_date.isoformat()}\n"
        f"- ~{horizon}-day trend points {direction}: central estimate "
        f"{end['mean']:.2f} ({pct:+.1f}%)\n"
        f"- 95% uncertainty range by then: {end['low']:.2f} to {end['high']:.2f}\n"
        f"- Recent daily volatility ≈ {vol:.1f}% of price\n"
        f"This is a trend + moving-average extrapolation of the last "
        f"{len(closes)} closes, not a market prediction. Treat the range, not "
        f"the point, as the takeaway.")

    return ForecastResult(answer, round(conf, 2), symbol=symbol,
                          horizon_days=horizon, projection=proj)
