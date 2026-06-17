"""Sector-aware classification and thresholds for fundamental analysis.

The base pipeline judged every company with a single set of thresholds tuned for
large-cap growth tech (ROE > 15%, net margin > 20%, P/B < 3, D/E < 0.5, ...).
Those yardsticks misclassify the stocks that make up most of the market: banks
and utilities are levered by design, REITs are valued on FFO not EPS, energy and
staples run thin margins, etc.

This module maps a company's reported sector/industry onto a small set of
buckets and returns thresholds (and applicability flags) appropriate to each.
Anything we don't recognize falls back to DEFAULT (the original tech-ish bars).
"""

from __future__ import annotations

# Canonical buckets we branch on.
DEFAULT = "default"
FINANCIALS = "financials"
REAL_ESTATE = "real_estate"
UTILITIES = "utilities"
ENERGY = "energy"
CONSUMER_STAPLES = "consumer_staples"


# Substring matchers against the raw sector/industry strings from company facts.
# Order matters: first match wins, so put the most specific buckets first.
_SECTOR_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (REAL_ESTATE, ("reit", "real estate", "mortgage real estate")),
    (FINANCIALS, ("financ", "bank", "insurance", "capital markets", "asset management",
                  "credit", "thrift", "brokerage")),
    (UTILITIES, ("utilit", "electric", "water utilit", "gas utilit", "power")),
    (ENERGY, ("energy", "oil", "gas", "petroleum", "drilling", "pipeline", "coal")),
    (CONSUMER_STAPLES, ("consumer staples", "consumer defensive", "food", "beverage",
                        "household", "tobacco", "grocery")),
]


def normalize_sector(sector: str | None, industry: str | None = None) -> str:
    """Collapse raw sector/industry text into one of our buckets."""
    haystack = " ".join(s for s in (sector, industry) if s).lower()
    if not haystack:
        return DEFAULT
    for bucket, keywords in _SECTOR_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return bucket
    return DEFAULT


# Per-sector thresholds. Each block mirrors what fundamentals.py checks:
#   profitability: (return_on_equity, net_margin, operating_margin)
#   growth:        (revenue_growth, earnings_growth, book_value_growth)
#   price_ratios:  (price_to_earnings, price_to_book, price_to_sales)
# Price ratios are "rich above this" cutoffs (higher = more expensive = bearish).
_THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    DEFAULT: {
        "profitability": {"roe": 0.15, "net_margin": 0.20, "operating_margin": 0.15},
        "growth": {"revenue": 0.10, "earnings": 0.10, "book_value": 0.10},
        "price": {"pe": 25, "pb": 3, "ps": 5},
    },
    FINANCIALS: {
        # Banks: ROE/ROA matter, margins are not comparable, book value is real.
        "profitability": {"roe": 0.10, "net_margin": 0.15, "operating_margin": 0.20},
        "growth": {"revenue": 0.05, "earnings": 0.07, "book_value": 0.07},
        "price": {"pe": 15, "pb": 1.5, "ps": 4},
    },
    REAL_ESTATE: {
        # REITs trade on high P/FFO multiples; net income understates cash earnings.
        "profitability": {"roe": 0.06, "net_margin": 0.15, "operating_margin": 0.25},
        "growth": {"revenue": 0.04, "earnings": 0.05, "book_value": 0.03},
        "price": {"pe": 40, "pb": 2.5, "ps": 8},
    },
    UTILITIES: {
        # Regulated, slow-growing, levered; modest returns are healthy.
        "profitability": {"roe": 0.08, "net_margin": 0.08, "operating_margin": 0.15},
        "growth": {"revenue": 0.03, "earnings": 0.04, "book_value": 0.04},
        "price": {"pe": 22, "pb": 2, "ps": 3},
    },
    ENERGY: {
        # Cyclical; thin margins at trough, book value meaningful (asset heavy).
        "profitability": {"roe": 0.10, "net_margin": 0.08, "operating_margin": 0.12},
        "growth": {"revenue": 0.05, "earnings": 0.05, "book_value": 0.05},
        "price": {"pe": 15, "pb": 2, "ps": 2},
    },
    CONSUMER_STAPLES: {
        # Low-margin, low-growth, stable; do not demand tech-style numbers.
        "profitability": {"roe": 0.12, "net_margin": 0.06, "operating_margin": 0.10},
        "growth": {"revenue": 0.04, "earnings": 0.05, "book_value": 0.05},
        "price": {"pe": 25, "pb": 4, "ps": 3},
    },
}


def get_thresholds(bucket: str) -> dict[str, dict[str, float]]:
    """Return the threshold block for a bucket (DEFAULT if unknown)."""
    return _THRESHOLDS.get(bucket, _THRESHOLDS[DEFAULT])


def is_leveraged_by_design(bucket: str) -> bool:
    """Sectors where high debt is normal, so a low-D/E bonus should be skipped."""
    return bucket in (FINANCIALS, UTILITIES, REAL_ESTATE)


def fcf_meaningful(bucket: str) -> bool:
    """Whether free-cash-flow conversion is a meaningful health check.

    For banks/insurers FCF is not well defined; for REITs reported FCF/EPS misses
    the FFO economics. Skip the FCF-conversion bonus for these.
    """
    return bucket not in (FINANCIALS, REAL_ESTATE)
