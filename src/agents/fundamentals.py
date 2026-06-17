from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress
import json

from src.tools.api import get_financial_metrics, get_sector_bucket
from src.utils.sectors import get_thresholds, is_leveraged_by_design, fcf_meaningful


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

        # Track how many underlying data points were actually available so we can
        # scale confidence down for thin-data names instead of presenting an
        # incomplete read with the same conviction as a fully-covered one.
        available_fields = 0
        total_fields = 0

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        # 1. Profitability Analysis (sector-adjusted)
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        prof_t = sector_thresholds["profitability"]
        thresholds = [
            (return_on_equity, prof_t["roe"]),
            (net_margin, prof_t["net_margin"]),
            (operating_margin, prof_t["operating_margin"]),
        ]
        total_fields += len(thresholds)
        available_fields += sum(metric is not None for metric, _ in thresholds)
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing growth")
        # 2. Growth Analysis (sector-adjusted)
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        growth_t = sector_thresholds["growth"]
        thresholds = [
            (revenue_growth, growth_t["revenue"]),
            (earnings_growth, growth_t["earnings"]),
            (book_value_growth, growth_t["book_value"]),
        ]
        total_fields += len(thresholds)
        available_fields += sum(metric is not None for metric, _ in thresholds)
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing financial health")
        # 3. Financial Health (sector-adjusted)
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        # Count the checks that actually apply to this sector so the score can be
        # judged against the right denominator.
        health_checks = 0
        health_score = 0
        if current_ratio is not None:
            total_fields += 1
            available_fields += 1
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        health_checks += 1

        # Low-debt bonus only for sectors where low leverage is meaningful.
        # Banks, utilities, and REITs are levered by design; penalizing them here
        # would be wrong, so the check is skipped for them.
        if not is_leveraged_by_design(sector):
            health_checks += 1
            if debt_to_equity is not None:
                total_fields += 1
                available_fields += 1
            if debt_to_equity is not None and debt_to_equity < 0.5:  # Conservative debt
                health_score += 1

        # FCF conversion is not a meaningful check for financials/REITs.
        if fcf_meaningful(sector):
            health_checks += 1
            if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:
                health_score += 1

        # Require a majority of the *applicable* checks to call it bullish.
        bullish_health_cut = max(2, (health_checks // 2) + 1)
        health_signal = "bullish" if health_score >= bullish_health_cut else "bearish" if health_score == 0 else "neutral"
        signals.append(health_signal)
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A") + (" (leverage check skipped: levered-by-design sector)" if is_leveraged_by_design(sector) else ""),
        }

        progress.update_status(agent_id, ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios (sector-adjusted)
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        price_t = sector_thresholds["price"]
        thresholds = [
            (pe_ratio, price_t["pe"]),
            (pb_ratio, price_t["pb"]),
            (ps_ratio, price_t["ps"]),
        ]
        total_fields += len(thresholds)
        available_fields += sum(metric is not None for metric, _ in thresholds)
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status(agent_id, ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level, then scale it by data coverage so a read
        # built on partial data is presented with lower conviction.
        total_signals = len(signals)
        base_confidence = max(bullish_signals, bearish_signals) / total_signals
        coverage = (available_fields / total_fields) if total_fields else 1.0
        confidence = round(base_confidence * coverage * 100)

        reasoning["data_coverage"] = {
            "sector": sector,
            "fields_available": available_fields,
            "fields_total": total_fields,
            "coverage": f"{coverage:.0%}",
        }

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
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
