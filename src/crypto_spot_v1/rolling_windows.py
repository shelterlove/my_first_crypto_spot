"""滚动窗口回测运行器。"""

from __future__ import annotations

import copy

import pandas as pd

from .backtest_event_driven import (
    run_rebalance_backtest,
    calculate_portfolio_performance,
)
from .backtest_engine import infer_periods_per_year
from .metrics import make_window_metrics, WindowMetrics


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
    """对一个策略在单个币种上运行滚动窗口回测。

    每个滚动子窗口使用策略实例的 deepcopy，确保状态不跨窗口污染。
    策略实例只要有 TARGET_ALLOC 和 compute_indicators 即可。

    Parameters
    ----------
    symbol : str
        币种名 (如 BTC/USDT)
    df : pd.DataFrame
        完整 OHLCV 数据（需要含 timestamp 列）
    strategy : PortfolioStrategyBase
        策略实例 (会被 deepcopy 用于每个子窗口)
    strategy_name : str
        策略名 (用于 WindowMetrics 标识)
    window_days : int
        窗口大小（天，1d 下等于 bar 数）
    step_days : int
        步长（天）
    initial_capital, reserve, fee_rate : float
        资金和费率参数
    timeframe : str
        K 线周期 (用于 ppy 推算)
    warmup_bars : int
        预热 bar 数（指标计算用）

    Returns
    -------
    list[WindowMetrics]
        每个滚动窗口的标准化指标
    """
    ppy = infer_periods_per_year(timeframe)
    results: list[WindowMetrics] = []
    i = 0
    total_bars = len(df)

    # 先预计算一次 indicators，供 calculate_portfolio_performance 的牛熊分析使用
    if "ema168" not in df.columns:
        df = strategy.compute_indicators(df)

    while i + window_days + warmup_bars <= total_bars:
        eval_start = i + warmup_bars
        eval_end = i + window_days + warmup_bars
        backtest_start = eval_start - 1 if execution_mode != "same_close" else eval_start
        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)
        ts_start = window_df["timestamp"].iloc[0]
        ts_end = window_df["timestamp"].iloc[-1]
        window_label = f"{ts_start.date()}~{ts_end.date()}"

        # 每个子窗口用 fresh deepcopy，防止状态跨窗口污染
        window_strategy = copy.deepcopy(strategy)

        candle_dfs = {symbol: backtest_df}
        pd_result = run_rebalance_backtest(
            candle_dfs, window_strategy,
            initial_capital=initial_capital,
            reserve=reserve,
            fee_rate=fee_rate,
            execution_mode=execution_mode,
        )
        full_action_log = pd_result.attrs.get("action_log")
        pd_result = pd_result[pd_result["timestamp"] >= ts_start].reset_index(drop=True)
        action_log = full_action_log
        if action_log is None:
            action_log = pd.DataFrame()
        elif not action_log.empty:
            action_log = action_log[action_log["timestamp"] >= ts_start].reset_index(drop=True)
        pd_result.attrs["action_log"] = action_log
        pd_result.attrs["execution_mode"] = execution_mode
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
                for row in pd_result.to_dict("records"):
                    artifact_sink.setdefault("equity_curves", []).append({
                        **meta,
                        **row,
                    })
            if action_log is not None and not action_log.empty:
                for row in action_log.to_dict("records"):
                    artifact_sink.setdefault("action_logs", []).append({
                        **meta,
                        **row,
                    })

        perf = calculate_portfolio_performance(
            pd_result,
            initial_capital,
            ppy,
            candle_df=window_df,
            fee_rate=fee_rate,
            benchmark_entry_col="open" if execution_mode == "next_open" else "close",
        )

        wm = make_window_metrics(
            strategy_name=strategy_name,
            symbol=symbol,
            window_label=window_label,
            window_start=str(ts_start),
            window_end=str(ts_end),
            result_df=pd_result,
            perf=perf,
            initial_capital=initial_capital,
            window_days=window_days,
        )
        results.append(wm)
        i += step_days

    return results
