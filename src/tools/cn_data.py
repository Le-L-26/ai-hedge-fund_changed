"""China A-share data adapter (akshare-backed).

financialdatasets.ai has no China coverage, so A-share tickers (600519, 000001,
688981, optionally with a .SH/.SZ suffix) are routed here instead. This module
reimplements the handful of data functions the agents depend on against akshare,
mapping everything into the same Pydantic models used for US names so the rest of
the pipeline is unchanged. Currency is CNY throughout.

Data sources (all free, no key):
  * prices            -> ak.stock_zh_a_hist (qfq-adjusted)
  * income/balance/cf -> ak.stock_financial_report_sina  (YTD-cumulative, raw CNY)
  * ratios            -> ak.stock_financial_analysis_indicator
  * sector/profile    -> ak.stock_profile_cninfo
  * news              -> ak.stock_news_em
  * insider trades    -> not a meaningful A-share concept -> []

Two correctness points worth calling out:
  * Sina statements are *cumulative within the fiscal year* (Q1=3m, H1=6m,
    9M=9m, FY=12m). "annual" therefore means the 1231 columns; "ttm" is computed
    as cum(latest) + FY(prev year) - cum(same period prev year).
  * A-shares have a ¥1 par value, so 实收资本(或股本) (share capital, in yuan)
    equals the share count. Market cap = shares * latest *unadjusted* close.
"""

from __future__ import annotations

import datetime
import functools
import logging
import re
import time

import pandas as pd

from src.data.models import (
    CompanyFacts,
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

logger = logging.getLogger(__name__)

_CN_TICKER_RE = re.compile(r"^(\d{6})(?:\.(SH|SS|SZ|SHA|SZA))?$", re.IGNORECASE)


def is_cn_ticker(ticker: str) -> bool:
    """True for Shanghai/Shenzhen/STAR A-share codes (6 digits, optional suffix)."""
    return bool(_CN_TICKER_RE.match(ticker.strip()))


def _retry(fn, *args, attempts: int = 3, **kwargs):
    """Call an akshare function, retrying on transient connection drops.

    akshare's Eastmoney-backed endpoints intermittently RemoteDisconnect; a short
    backoff usually clears it. Returns None if every attempt fails.
    """
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - akshare raises requests errors
            if i == attempts - 1:
                logger.warning("%s failed after %d attempts: %s", getattr(fn, "__name__", fn), attempts, e)
                return None
            time.sleep(1.5 * (i + 1))
    return None


def _code(ticker: str) -> str:
    """Bare 6-digit code, e.g. '600519.SH' -> '600519'."""
    m = _CN_TICKER_RE.match(ticker.strip())
    return m.group(1) if m else ticker


def _sina_symbol(code: str) -> str:
    """Sina-style symbol, e.g. '600519' -> 'sh600519', '000001' -> 'sz000001'."""
    if code.startswith(("5", "6", "9")):  # 6xx main board, 688 STAR, 9xx B
        return f"sh{code}"
    return f"sz{code}"  # 0xx / 3xx (ChiNext) main + growth boards


# ---------------------------------------------------------------------------
# Cached raw pulls (akshare scrapes, so memoize per process to avoid re-fetching
# the same statement table for every agent in a run).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _statement(code: str, kind: str) -> pd.DataFrame | None:
    """Sina financial statement keyed by 报告日 (report date 'YYYYMMDD')."""
    import akshare as ak

    name = {"income": "利润表", "balance": "资产负债表", "cashflow": "现金流量表"}[kind]
    df = _retry(ak.stock_financial_report_sina, stock=_sina_symbol(code), symbol=name)
    if df is None or df.empty or "报告日" not in df.columns:
        return None
    df = df.copy()
    df["报告日"] = df["报告日"].astype(str).str.replace("-", "", regex=False).str[:8]
    df = df.set_index("报告日")
    return df


@functools.lru_cache(maxsize=64)
def _indicators(code: str) -> pd.DataFrame | None:
    """Per-period financial ratios, indexed by report date 'YYYYMMDD'."""
    import akshare as ak

    start_year = str(datetime.date.today().year - 6)
    df = _retry(ak.stock_financial_analysis_indicator, symbol=code, start_year=start_year)
    if df is None or df.empty or "日期" not in df.columns:
        return None
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y%m%d")
    return df.dropna(subset=["日期"]).set_index("日期")


@functools.lru_cache(maxsize=64)
def _profile(code: str) -> dict | None:
    """Company profile (name, CSRC industry, listing date) via cninfo."""
    import akshare as ak

    df = _retry(ak.stock_profile_cninfo, symbol=code)
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


@functools.lru_cache(maxsize=256)
def _price_df(code: str, start: str, end: str, adjust: str) -> pd.DataFrame | None:
    """Daily OHLCV via Sina (reliable; Eastmoney's stock_zh_a_hist drops
    connections). adjust='qfq' for analysis, '' for raw (market-cap) prices.

    Columns: date, open, high, low, close, volume (+ outstanding_share).
    """
    import akshare as ak

    df = _retry(
        ak.stock_zh_a_daily,
        symbol=_sina_symbol(code), start_date=start, end_date=end, adjust=adjust,
    )
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame | None, date: str, col: str) -> float | None:
    if df is None or col not in df.columns or date not in df.index:
        return None
    val = pd.to_numeric(pd.Series([df.at[date, col]]), errors="coerce").iloc[0]
    return None if pd.isna(val) else float(val)


def _report_dates(code: str, end_date: str, *, annual_only: bool) -> list[str]:
    """Report dates (newest first) on/before end_date, from the balance sheet."""
    bs = _statement(code, "balance")
    if bs is None:
        return []
    end = end_date.replace("-", "")
    dates = [d for d in bs.index if d <= end]
    if annual_only:
        dates = [d for d in dates if d.endswith("1231")]
    return sorted(set(dates), reverse=True)


def _ttm_flow(series: dict[str, float], date: str) -> float | None:
    """Trailing-twelve-month value for a cumulative-YTD flow item at `date`."""
    if date.endswith("1231"):
        return series.get(date)
    year, mmdd = int(date[:4]), date[4:]
    cur, prev_fy, prev_same = series.get(date), series.get(f"{year-1}1231"), series.get(f"{year-1}{mmdd}")
    if cur is None or prev_fy is None or prev_same is None:
        return None
    return cur + prev_fy - prev_same


# ---------------------------------------------------------------------------
# Line items
# ---------------------------------------------------------------------------
# name -> (statement, chinese column). Flows are summed to TTM when period='ttm';
# balances are point-in-time snapshots. Items requiring arithmetic are handled
# in _line_item_value below rather than via this direct map.
_DIRECT: dict[str, tuple[str, str]] = {
    "revenue": ("income", "营业总收入"),
    "net_income": ("income", "归属于母公司所有者的净利润"),
    "operating_income": ("income", "营业利润"),
    "interest_expense": ("income", "利息费用"),
    "research_and_development": ("income", "研发费用"),
    "income_tax_expense": ("income", "所得税费用"),
    "earnings_per_share": ("income", "基本每股收益"),
    "total_assets": ("balance", "资产总计"),
    "current_assets": ("balance", "流动资产合计"),
    "current_liabilities": ("balance", "流动负债合计"),
    "total_liabilities": ("balance", "负债合计"),
    "cash_and_equivalents": ("balance", "货币资金"),
    "shareholders_equity": ("balance", "归属于母公司股东权益合计"),
    "outstanding_shares": ("balance", "实收资本(或股本)"),
    "goodwill": ("balance", "商誉"),
    "intangible_assets": ("balance", "无形资产"),
    "operating_cash_flow": ("cashflow", "经营活动产生的现金流量净额"),
}
_FLOW_STATEMENTS = {"income", "cashflow"}


def _series(code: str, statement: str, col: str) -> dict[str, float]:
    df = _statement(code, statement)
    if df is None or col not in df.columns:
        return {}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return {str(k): float(v) for k, v in s.items()}


def _value(code: str, statement: str, col: str, date: str, ttm: bool) -> float | None:
    if statement in _FLOW_STATEMENTS and ttm:
        return _ttm_flow(_series(code, statement, col), date)
    return _num(_statement(code, statement), date, col)


def _capex(code: str, date: str, ttm: bool) -> float | None:
    # Stored negative to match the financialdatasets convention (cash outflow).
    v = _value(code, "cashflow", "购建固定资产、无形资产和其他长期资产所支付的现金", date, ttm)
    return None if v is None else -abs(v)


def _total_debt(code: str, date: str) -> float | None:
    bs = _statement(code, "balance")
    parts = [
        _num(bs, date, c)
        for c in ("短期借款", "一年内到期的非流动负债", "长期借款", "应付债券")
    ]
    vals = [p for p in parts if p is not None]
    return sum(vals) if vals else None


def _line_item_value(code: str, name: str, date: str, ttm: bool) -> float | None:
    """Resolve one requested line item to a number (or None if unavailable)."""
    if name in _DIRECT:
        st, col = _DIRECT[name]
        return _value(code, st, col, date, ttm)

    if name in ("capital_expenditure",):
        return _capex(code, date, ttm)
    if name in ("depreciation_and_amortization", "ebitda"):
        return None  # not exposed cleanly by Sina statements
    if name == "free_cash_flow":
        ocf = _value(code, "cashflow", "经营活动产生的现金流量净额", date, ttm)
        capex = _capex(code, date, ttm)
        return None if ocf is None or capex is None else ocf + capex
    if name in ("dividends_and_other_cash_distributions",):
        v = _value(code, "cashflow", "分配股利、利润或偿付利息所支付的现金", date, ttm)
        return None if v is None else -abs(v)
    if name == "gross_profit":
        rev = _value(code, "income", "营业收入", date, ttm)
        cogs = _value(code, "income", "营业成本", date, ttm)
        return None if rev is None or cogs is None else rev - cogs
    if name == "operating_expense":
        parts = [_value(code, "income", c, date, ttm) for c in ("销售费用", "管理费用")]
        vals = [p for p in parts if p is not None]
        return sum(vals) if vals else None
    if name == "ebit":
        # Pre-tax profit + interest expense ≈ EBIT.
        op = _value(code, "income", "利润总额", date, ttm)
        interest = _value(code, "income", "利息费用", date, ttm) or 0.0
        return None if op is None else op + interest
    if name == "total_debt":
        return _total_debt(code, date)
    if name == "working_capital":
        ca = _num(_statement(code, "balance"), date, "流动资产合计")
        cl = _num(_statement(code, "balance"), date, "流动负债合计")
        return None if ca is None or cl is None else ca - cl
    if name == "goodwill_and_intangible_assets":
        bs = _statement(code, "balance")
        parts = [_num(bs, date, c) for c in ("商誉", "无形资产")]
        vals = [p for p in parts if p is not None]
        return sum(vals) if vals else None
    if name == "book_value_per_share":
        eq = _num(_statement(code, "balance"), date, "归属于母公司股东权益合计")
        sh = _num(_statement(code, "balance"), date, "实收资本(或股本)")
        return None if not eq or not sh else eq / sh
    if name in ("gross_margin", "operating_margin", "net_margin"):
        rev = _value(code, "income", "营业总收入", date, ttm)
        if not rev:
            return None
        if name == "net_margin":
            ni = _value(code, "income", "归属于母公司所有者的净利润", date, ttm)
            return None if ni is None else ni / rev
        if name == "operating_margin":
            oi = _value(code, "income", "营业利润", date, ttm)
            return None if oi is None else oi / rev
        gp = _line_item_value(code, "gross_profit", date, ttm)
        return None if gp is None else gp / rev
    if name in ("debt_to_equity",):
        td = _total_debt(code, date)
        eq = _num(_statement(code, "balance"), date, "归属于母公司股东权益合计")
        return None if td is None or not eq else td / eq
    return None  # unknown / unsupported -> let downstream treat as missing


def get_company_news(ticker: str, end_date: str, start_date=None, limit: int = 1000, **_) -> list[CompanyNews]:
    import akshare as ak

    code = _code(ticker)
    df = _retry(ak.stock_news_em, symbol=code)
    if df is None or df.empty:
        return []
    out: list[CompanyNews] = []
    for _, row in df.iterrows():
        date = str(row.get("发布时间", ""))[:10]
        if date and date > end_date:
            continue
        if start_date and date and date < start_date:
            continue
        out.append(
            CompanyNews(
                ticker=ticker,
                title=str(row.get("新闻标题", "")),
                author=None,
                source=str(row.get("文章来源", "")) or "eastmoney",
                date=date or end_date,
                url=str(row.get("新闻链接", "")),
                sentiment=None,
            )
        )
    return out[:limit]


def get_insider_trades(ticker: str, end_date: str, start_date=None, limit: int = 1000, **_) -> list[InsiderTrade]:
    # A-share disclosure (高管持股变动) is structured very differently and not a
    # 1:1 analog of US Form-4 insider trades; return empty rather than mislead.
    return []


def _shares(code: str, date: str) -> float | None:
    """Total shares outstanding (= 实收资本 in yuan for ¥1-par A-shares)."""
    bs = _statement(code, "balance")
    if bs is None:
        return None
    cands = [d for d in bs.index if d <= date.replace("-", "")]
    if not cands:
        return None
    return _num(bs, max(cands), "实收资本(或股本)")


def get_market_cap(ticker: str, end_date: str, **_) -> float | None:
    code = _code(ticker)
    shares = _shares(code, end_date)
    if not shares:
        return None
    # Unadjusted close at/just before end_date (qfq would distort historical cap).
    end = end_date.replace("-", "")
    start = (datetime.datetime.strptime(end, "%Y%m%d") - datetime.timedelta(days=14)).strftime("%Y%m%d")
    df = _price_df(code, start, end, "")
    if df is None or df.empty:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    return None if close.empty else float(close.iloc[-1]) * shares


def get_prices(ticker: str, start_date: str, end_date: str, **_) -> list[Price]:
    code = _code(ticker)
    df = _price_df(code, start_date.replace("-", ""), end_date.replace("-", ""), "qfq")
    if df is None or df.empty:
        return []
    out: list[Price] = []
    for _, r in df.iterrows():
        try:
            out.append(
                Price(
                    open=float(r["open"]),
                    close=float(r["close"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    volume=int(r["volume"]),
                    time=str(r["date"]),
                )
            )
        except (ValueError, TypeError, KeyError):
            continue
    return out


def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, **_) -> list[LineItem]:
    code = _code(ticker)
    ttm = period.lower() == "ttm"
    dates = _report_dates(code, end_date, annual_only=not ttm)
    if not dates:
        return []
    results: list[LineItem] = []
    for date in dates[:limit]:
        payload = {
            "ticker": ticker,
            "report_period": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "period": period,
            "currency": "CNY",
        }
        for name in line_items:
            payload[name] = _line_item_value(code, name, date, ttm)
        results.append(LineItem(**payload))
    return results


def _pct(df: pd.DataFrame | None, date: str, col: str) -> float | None:
    """Indicator value stored as a percent -> fraction (e.g. 8.34 -> 0.0834)."""
    v = _num(df, date, col)
    return None if v is None else v / 100.0


def _prior_year(date: str) -> str:
    return f"{int(date[:4]) - 1}{date[4:]}"


def _ratio(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den else None


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10, **_) -> list[FinancialMetrics]:
    code = _code(ticker)
    ttm = period.lower() == "ttm"
    dates = _report_dates(code, end_date, annual_only=not ttm)
    if not dates:
        return []
    ind = _indicators(code)
    mcap = get_market_cap(ticker, end_date)

    out: list[FinancialMetrics] = []
    for date in dates[:limit]:
        # Profitability/growth ratios are computed from the (TTM or annual) flows
        # rather than read from the as-reported indicator table, because for an
        # interim report (Q1/H1/9M) the table's 净资产收益率/margins are cumulative
        # YTD, not trailing-twelve-month — which understated ROE etc. for analysts
        # that always request period="ttm".
        rev = _value(code, "income", "营业总收入", date, ttm)
        ni = _value(code, "income", "归属于母公司所有者的净利润", date, ttm)
        oi = _value(code, "income", "营业利润", date, ttm)
        cogs = _value(code, "income", "营业成本", date, ttm)
        rev_main = _value(code, "income", "营业收入", date, ttm)
        equity = _num(_statement(code, "balance"), date, "归属于母公司股东权益合计")
        assets = _num(_statement(code, "balance"), date, "资产总计")

        py = _prior_year(date)
        rev_py = _value(code, "income", "营业总收入", py, ttm)
        ni_py = _value(code, "income", "归属于母公司所有者的净利润", py, ttm)
        equity_py = _num(_statement(code, "balance"), py, "归属于母公司股东权益合计")

        net_margin = _ratio(ni, rev)
        operating_margin = _ratio(oi, rev)
        gross_margin = _ratio(rev_main - cogs, rev_main) if rev_main and cogs is not None else None
        roe = _ratio(ni, equity)
        roa = _ratio(ni, assets)
        revenue_growth = (rev / rev_py - 1) if rev and rev_py else None
        earnings_growth = (ni / ni_py - 1) if ni and ni_py else None
        book_value_growth = (equity / equity_py - 1) if equity and equity_py else None

        # Only the most-recent row gets a live market cap (price is "now"); older
        # periods leave valuation ratios None rather than mixing a stale price.
        cap = mcap if date == dates[0] else None
        out.append(
            FinancialMetrics(
                ticker=ticker,
                report_period=f"{date[:4]}-{date[4:6]}-{date[6:]}",
                period=period,
                currency="CNY",
                market_cap=cap,
                enterprise_value=None,
                price_to_earnings_ratio=(cap / ni) if cap and ni else None,
                price_to_book_ratio=(cap / equity) if cap and equity else None,
                price_to_sales_ratio=(cap / rev) if cap and rev else None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=None,
                asset_turnover=_num(ind, date, "总资产周转率(次)"),
                inventory_turnover=_num(ind, date, "存货周转率(次)"),
                receivables_turnover=_num(ind, date, "应收账款周转率(次)"),
                days_sales_outstanding=_num(ind, date, "应收账款周转天数(天)"),
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=_num(ind, date, "流动比率"),
                quick_ratio=_num(ind, date, "速动比率"),
                cash_ratio=_pct(ind, date, "现金比率(%)"),
                operating_cash_flow_ratio=None,
                debt_to_equity=_pct(ind, date, "负债与所有者权益比率(%)"),
                debt_to_assets=_pct(ind, date, "资产负债率(%)"),
                interest_coverage=_num(ind, date, "利息支付倍数"),
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                book_value_growth=book_value_growth,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=_pct(ind, date, "股息发放率(%)"),
                earnings_per_share=_ratio(ni, _shares(code, date)),
                book_value_per_share=_ratio(equity, _shares(code, date)),
                free_cash_flow_per_share=None,
            )
        )
    return out


@functools.lru_cache(maxsize=8)
def _index_close(sina_symbol: str) -> pd.Series | None:
    """Full daily close history for an index via Sina (reliable; Eastmoney's
    index endpoints are flaky). Indexed by date (Timestamp)."""
    import akshare as ak

    df = _retry(ak.stock_zh_index_daily, symbol=sina_symbol)
    if df is None or df.empty or "close" not in df.columns:
        return None
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s.dropna()


def get_benchmark_returns(start_date: str, end_date: str, **_) -> pd.Series | None:
    """Daily returns of the CSI 300 (沪深300, Sina symbol sh000300) — the A-share
    market proxy used in place of SPY for beta and relative strength."""
    close = _index_close("sh000300")
    if close is None:
        return None
    window = close[(close.index >= pd.Timestamp(start_date)) & (close.index <= pd.Timestamp(end_date))]
    if len(window) < 2:
        return None
    return window.pct_change().dropna()


def get_company_facts(ticker: str, **_) -> CompanyFacts | None:
    code = _code(ticker)
    prof = _profile(code)
    name = ticker
    industry = sector = None
    listing = None
    if prof:
        name = prof.get("英文名称") or prof.get("公司名称") or ticker
        industry = prof.get("所属行业")
        sector = industry  # CSRC industry string; normalize_sector handles it
        listing = prof.get("上市日期")
    return CompanyFacts(
        ticker=ticker,
        name=str(name),
        industry=str(industry) if industry else None,
        sector=str(sector) if sector else None,
        exchange="SSE" if _sina_symbol(code).startswith("sh") else "SZSE",
        is_active=True,
        listing_date=str(listing) if listing else None,
        location="China",
        market_cap=get_market_cap(ticker, datetime.datetime.now().strftime("%Y-%m-%d")),
    )
