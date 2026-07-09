"""Rolling-window execution for V1 strategies."""

from __future__ import annotations

import copy

import pandas as pd

from .backtest_engine import infer_periods_per_year
from .backtest_event_driven import calculate_portfolio_performance, run_rebalance_backtest
from .metrics import WindowMetrics, make_window_metrics


def run_strategy_rolling(
    symbol: str,
    df: pd.DataFrame,
    strategy,
    strategy_name: str,
    window_days: int,
    step_days: int,
    initial_capital: float = 100.0,
    reserve: float = 20.0,
    fee_rate: float = 0.001,
    timeframe: str = "1d",
    warmup_bars: int = 200,
    execution_mode: str = "next_open",
    artifact_sink: dict[str, list[dict]] | None = None,
    collect_equity_curve: bool = True,
) -> list[WindowMetrics]:
    """Run one strategy over rolling evaluation windows for one symbol.

    Warmup bars are available to indicators and signal state, but are excluded
    from performance metrics, action artifacts, and equity artifacts.
    """
    ppy = infer_periods_per_year(timeframe)
    results: list[WindowMetrics] = []
    total_bars = len(df)

    if "ema168" not in df.columns:
        df = strategy.compute_indicators(df)

    i = 0
    while i + window_days + warmup_bars <= total_bars:
        eval_start = i + warmup_bars
        eval_end = i + window_days + warmup_bars
        backtest_start = eval_start - 1 if execution_mode != "same_close" else eval_start

        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)
        ts_start = window_df["timestamp"].iloc[0]
        ts_end = window_df["timestamp"].iloc[-1]
        window_label = f"{ts_start.date()}~{ts_end.date()}"

        window_strategy = copy.deepcopy(strategy)
        result_df = run_rebalance_backtest(
            {symbol: backtest_df},
            window_strategy,
            initial_capital=initial_capital,
            reserve=reserve,
            fee_rate=fee_rate,
            execution_mode=execution_mode,
        )

        full_action_log = result_df.attrs.get("action_log")
        full_sleeve_events = result_df.attrs.get("sleeve_events")
        full_sleeve_daily = result_df.attrs.get("sleeve_daily")
        result_df = result_df[result_df["timestamp"] >= ts_start].reset_index(drop=True)
        action_log = pd.DataFrame() if full_action_log is None else full_action_log
        if not action_log.empty:
            action_log = action_log[action_log["timestamp"] >= ts_start].reset_index(drop=True)
        sleeve_events = pd.DataFrame() if full_sleeve_events is None else full_sleeve_events
        if not sleeve_events.empty:
            sleeve_events = sleeve_events[sleeve_events["timestamp"] >= ts_start].reset_index(drop=True)
        sleeve_daily = pd.DataFrame() if full_sleeve_daily is None else full_sleeve_daily
        if not sleeve_daily.empty:
            sleeve_daily = sleeve_daily[sleeve_daily["timestamp"] >= ts_start].reset_index(drop=True)
        result_df.attrs["action_log"] = action_log
        result_df.attrs["execution_mode"] = execution_mode

        if artifact_sink is not None:
            meta = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "window_name": f"{window_days}d",
                "window_label": window_label,
                "window_start": str(ts_start),
                "window_end": str(ts_end),
            }
            if collect_equity_curve:
                for row in result_df.to_dict("records"):
                    artifact_sink.setdefault("equity_curves", []).append({**meta, **row})
            if not action_log.empty:
                for row in action_log.to_dict("records"):
                    artifact_sink.setdefault("action_logs", []).append({**meta, **row})
            if not sleeve_events.empty:
                for row in sleeve_events.to_dict("records"):
                    artifact_sink.setdefault("sleeve_events", []).append({**meta, **row})
            if not sleeve_daily.empty:
                for row in sleeve_daily.to_dict("records"):
                    artifact_sink.setdefault("sleeve_daily", []).append({**meta, **row})

        perf = calculate_portfolio_performance(
            result_df,
            initial_capital,
            ppy,
            candle_df=window_df,
            fee_rate=fee_rate,
            benchmark_entry_col="open" if execution_mode == "next_open" else "close",
        )
        results.append(
            make_window_metrics(
                strategy_name=strategy_name,
                symbol=symbol,
                window_label=window_label,
                window_start=str(ts_start),
                window_end=str(ts_end),
                result_df=result_df,
                perf=perf,
                initial_capital=initial_capital,
                window_days=window_days,
            )
        )
        i += step_days

    return results
