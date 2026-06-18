from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress
import json

from src.tools.api import get_financial_metrics, get_sector_bucket
from src.utils.sectors import get_thresholds, is_leveraged_by_design, fcf_meaningful
from src.utils.quant import signed_score, aggregate


##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Resolve the company's sector so thresholds adapt to the business type
        # (banks/utilities/REITs are not judged by tech-company yardsticks).
        sector = get_sector_bucket(ticker, api_key=api_key)
        sector_thresholds = get_thresholds(sector)

        reasoning = {}
        # Each pillar produces a continuous signed score in (-1, 1) via tanh around
        # the sector threshold (the old binary "metric > threshold" cliffs threw
        # away magnitude). Pillar verdicts then aggregate into the overall read,
        # whose conviction is the magnitude of the net score, not a vote count.
        pillar_scores: list[float | None] = []
        pillar_weights: list[float] = []

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        # 1. Profitability Analysis (sector-adjusted)
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        prof_t = sector_thresholds["profitability"]
        prof = aggregate([
            signed_score(return_on_equity, prof_t["roe"]),
            signed_score(net_margin, prof_t["net_margin"]),
            signed_score(operating_margin, prof_t["operating_margin"]),
        ])
        pillar_scores.append(prof.score if prof.coverage else None)
        pillar_weights.append(1.0)
        reasoning["profitability_signal"] = {
            "signal": prof.signal,
            "score": round(prof.score, 3),
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing growth")
        # 2. Growth Analysis (sector-adjusted)
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        growth_t = sector_thresholds["growth"]
        growth = aggregate([
            signed_score(revenue_growth, growth_t["revenue"]),
            signed_score(earnings_growth, growth_t["earnings"]),
            signed_score(book_value_growth, growth_t["book_value"]),
        ])
        pillar_scores.append(growth.score if growth.coverage else None)
        pillar_weights.append(1.0)
        reasoning["growth_signal"] = {
            "signal": growth.signal,
            "score": round(growth.score, 3),
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing financial health")
        # 3. Financial Health (sector-adjusted). Checks that don't apply to the
        # sector are left out (weight 0) rather than scored, mirroring the prior
        # leverage/FCF applicability logic.
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_scores = [signed_score(current_ratio, 1.5, scale=0.75)]
        health_weights = [1.0]

        # Low-debt is only meaningful where low leverage is a virtue. Banks,
        # utilities and REITs are levered by design, so the check is dropped.
        health_scores.append(signed_score(debt_to_equity, 0.5, scale=0.4, higher_is_better=False))
        health_weights.append(0.0 if is_leveraged_by_design(sector) else 1.0)

        # FCF-to-earnings conversion is not meaningful for financials/REITs.
        fcf_conv = (free_cash_flow_per_share / earnings_per_share) if (free_cash_flow_per_share and earnings_per_share) else None
        health_scores.append(signed_score(fcf_conv, 0.8, scale=0.4))
        health_weights.append(1.0 if fcf_meaningful(sector) else 0.0)

        health = aggregate(health_scores, health_weights)
        pillar_scores.append(health.score if health.coverage else None)
        pillar_weights.append(1.0)
        reasoning["financial_health_signal"] = {
            "signal": health.signal,
            "score": round(health.score, 3),
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A") + (" (leverage check skipped: levered-by-design sector)" if is_leveraged_by_design(sector) else ""),
        }

        progress.update_status(agent_id, ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios (sector-adjusted). Richer-than-threshold is bearish,
        # so these score with higher_is_better=False.
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        price_t = sector_thresholds["price"]
        price = aggregate([
            signed_score(pe_ratio, price_t["pe"], higher_is_better=False),
            signed_score(pb_ratio, price_t["pb"], higher_is_better=False),
            signed_score(ps_ratio, price_t["ps"], higher_is_better=False),
        ])
        pillar_scores.append(price.score if price.coverage else None)
        pillar_weights.append(1.0)
        reasoning["price_ratios_signal"] = {
            "signal": price.signal,
            "score": round(price.score, 3),
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status(agent_id, ticker, "Calculating final signal")
        # Overall: weighted mean of the available pillar scores. Sign is the
        # signal; magnitude is conviction; coverage scales confidence down when
        # whole pillars were missing.
        overall = aggregate(pillar_scores, pillar_weights)
        confidence = round(overall.confidence * 100)

        reasoning["data_coverage"] = {
            "sector": sector,
            "pillars_available": sum(s is not None for s in pillar_scores),
            "pillars_total": len(pillar_scores),
            "coverage": f"{overall.coverage:.0%}",
            "net_score": round(overall.score, 3),
        }

        fundamental_analysis[ticker] = {
            "signal": overall.signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = fundamental_analysis

    progress.update_status(agent_id, None, "Done")
    
    return {
        "messages": [message],
        "data": data,
    }
