"""HTTP data bridge for the my-ai-agent stock analyst (Nuxt) app.

WHY THIS EXISTS
---------------
The Nuxt agent is TypeScript and cannot import this project's Python data layer.
The A-share support in particular (src/tools/cn_data.py) is akshare-backed and has
no JavaScript equivalent. So instead of re-implementing data fetching in TS, we
expose the *existing* Python functions over a tiny local HTTP endpoint that the
Nuxt server-side tool calls.

Routing is automatic: src/tools/api.py already sends 6-digit A-share tickers
(600519, 000001, 688981, optionally .SH/.SZ) to akshare and everything else to
financialdatasets.ai. This bridge just calls those functions and flattens the
result into compact, LLM-friendly JSON.

RUN IT
------
    poetry run python scripts/agent_bridge.py
        (or)
    poetry run uvicorn scripts.agent_bridge:app --port 8077

Then: GET http://localhost:8077/financials?ticker=AAPL
      GET http://localhost:8077/financials?ticker=600519   (Kweichow Moutai, A-share)
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

# Make `import src...` work no matter where this script is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

# Load the project's .env so FINANCIAL_DATASETS_API_KEY is available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv should be installed
    pass

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_market_cap,
)

app = FastAPI(title="my-ai-agent data bridge")

# Shared-secret lock. When BRIDGE_TOKEN is set (it always should be in
# production), every /financials call must present a matching X-Bridge-Token
# header. This stops strangers who discover the public URL from spending your
# financialdatasets.ai quota — only the Nuxt app, which knows the token, can call.
_BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "").strip()


def _require_token(x_bridge_token: str | None) -> None:
    if _BRIDGE_TOKEN and x_bridge_token != _BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Bridge-Token.")

# The subset of FinancialMetrics fields that actually move an investment thesis.
# Keeping this list short keeps the JSON (and the LLM's token bill) small.
_KEY_METRICS = [
    "price_to_earnings_ratio",
    "price_to_book_ratio",
    "price_to_sales_ratio",
    "peg_ratio",
    "enterprise_value_to_ebitda_ratio",
    "free_cash_flow_yield",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_equity",
    "return_on_invested_capital",
    "return_on_assets",
    "debt_to_equity",
    "current_ratio",
    "interest_coverage",
    "revenue_growth",
    "earnings_growth",
    "free_cash_flow_growth",
    "earnings_per_share",
    "free_cash_flow_per_share",
    "book_value_per_share",
]

# A few series we surface as a short trend so the model can see direction, not
# just a snapshot.
_TREND_METRICS = ["revenue_growth", "earnings_growth", "net_margin", "return_on_equity"]


# Accept HEAD as well as GET: UptimeRobot (and most uptime pingers) probe with a
# HEAD request by default. FastAPI's @app.get registers ONLY GET, so a HEAD probe
# was getting 405 Method Not Allowed -> the monitor reported the bridge as DOWN
# even though it was healthy. api_route with both verbs fixes that.
@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {"ok": True}


@app.get("/financials")
def financials(
    ticker: str = Query(..., description="US/foreign symbol (AAPL) or A-share code (600519)"),
    end_date: str | None = Query(None, description="As-of date YYYY-MM-DD; defaults to today"),
    x_bridge_token: str | None = Header(None),
) -> JSONResponse:
    _require_token(x_bridge_token)
    ticker = ticker.strip().upper()
    end_date = end_date or _dt.date.today().isoformat()

    metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=8)
    if not metrics:
        return JSONResponse(
            status_code=404,
            content={"error": f"No financial data found for '{ticker}'. "
                              "Check the symbol (A-shares are 6 digits, e.g. 600519)."},
        )

    latest = metrics[0]
    snapshot = {k: getattr(latest, k, None) for k in _KEY_METRICS}

    # Short trend: oldest -> newest for a handful of series.
    ordered = list(reversed(metrics))  # api returns newest-first
    trend = {
        m: [
            {"period": getattr(x, "report_period", None), "value": getattr(x, m, None)}
            for x in ordered
        ]
        for m in _TREND_METRICS
    }

    market_cap = get_market_cap(ticker, end_date)

    headlines: list[str] = []
    try:
        news = get_company_news(ticker, end_date, limit=5) or []
        headlines = [getattr(n, "title", None) for n in news if getattr(n, "title", None)][:5]
    except Exception:
        headlines = []  # news is best-effort; never fail the whole request over it

    return JSONResponse(
        content={
            "ticker": latest.ticker,
            "currency": latest.currency,
            "as_of": latest.report_period,
            "market_cap": market_cap,
            "metrics_ttm": snapshot,
            "trend": trend,
            "recent_news": headlines,
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Bind to 0.0.0.0 (all interfaces) so the container is reachable from outside,
    # and honor $PORT, which hosts like Render inject at runtime. Falls back to
    # 8077 for local dev so the existing localhost workflow is unchanged.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8077"))
    uvicorn.run(app, host=host, port=port)
