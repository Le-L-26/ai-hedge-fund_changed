"""Benchmark and beta helpers shared by the valuation and technical agents.

The base pipeline used a flat beta of 1.0 for every stock (so a utility and a
biotech got the same cost of equity) and computed momentum in absolute terms
(no comparison to the market). Both are fixed here by deriving a benchmark
return series and an estimated beta from the price history we already fetch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tools import cn_data
from src.tools.api import get_prices, prices_to_df

# Broad-market proxy. Override via the env if desired later.
BENCHMARK_TICKER = "SPY"


def get_benchmark_returns(
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    tickers: list[str] | None = None,
) -> pd.Series | None:
    """Daily close-to-close returns for the benchmark over the window, or None.

    Picks the benchmark from the universe being analyzed: A-share tickers use the
    CSI 300 index, everything else uses SPY. (A run is normally single-market; if
    mixed, A-shares present at all switch the shared benchmark to CSI 300.)
    """
    if tickers and any(cn_data.is_cn_ticker(t) for t in tickers):
        return cn_data.get_benchmark_returns(start_date, end_date)

    prices = get_prices(BENCHMARK_TICKER, start_date, end_date, api_key=api_key)
    if not prices:
        return None
    df = prices_to_df(prices)
    if df.empty or len(df) < 2:
        return None
    return df["close"].pct_change().dropna()


def estimate_beta(
    stock_returns: pd.Series | None,
    benchmark_returns: pd.Series | None,
    default: float = 1.0,
    lo: float = 0.3,
    hi: float = 2.5,
) -> float:
    """Estimate beta = cov(stock, mkt) / var(mkt), aligned on common dates.

    Falls back to ``default`` (and clamps to a sane [lo, hi] range) when there is
    not enough overlapping data to regress reliably.
    """
    if stock_returns is None or benchmark_returns is None:
        return default

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 30:  # need a meaningful overlap
        return default

    stock = aligned.iloc[:, 0].to_numpy()
    bench = aligned.iloc[:, 1].to_numpy()
    var = np.var(bench)
    if var == 0:
        return default

    beta = float(np.cov(stock, bench)[0][1] / var)
    if not np.isfinite(beta):
        return default
    return max(lo, min(hi, beta))
