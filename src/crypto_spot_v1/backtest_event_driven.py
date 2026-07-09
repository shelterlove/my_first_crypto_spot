"""Event-driven portfolio backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_rebalance import (
    Action,
    PortfolioState,
    PortfolioStrategyBase,
    PositionState,
)
from .backtest_engine import (
    calculate_annual_return,
    calculate_annual_volatility,
    calculate_max_drawdown,
    calculate_sharpe,
)
from .decision import parse_action_reason
from .v47 import position_book as v47_position_book


def align_timestamps(candle_dfs: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Return the sorted inner-join timestamp set across all symbols."""
    common = None
    for df in candle_dfs.values():
        ts = set(df["timestamp"].dropna().unique())
        common = ts if common is None else common & ts
    if not common:
        raise ValueError("No common timestamps across all symbols.")
    return sorted(common)


def execute_action_in_backtest(
    action: Action,
    portfolio: PortfolioState,
    fee_rate: float,
) -> None:
    """Execute one action in place."""
    pos = portfolio.positions.setdefault(action.symbol, PositionState())
    cost = action.quantity * action.price
    fee = cost * fee_rate

    if action.side == "buy":
        total_cost = pos.avg_cost * pos.quantity + cost
        pos.quantity += action.quantity
        pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0.0
        portfolio.cash -= cost + fee
    else:
        proceeds = cost - fee
        portfolio.cash += proceeds
        pos.quantity -= action.quantity
        if pos.quantity <= 1e-12:
            pos.quantity = 0.0
            pos.avg_cost = 0.0


def _build_execution_transform_actions(
    strategy: PortfolioStrategyBase,
    decision_portfolio: PortfolioState,
    execution_portfolio: PortfolioState,
    candles_by_symbol: dict[str, pd.DataFrame],
    execution_prices: dict[str, float],
    fee_rate: float,
    raw_actions: list[Action] | None = None,
) -> list[Action]:
    return v47_position_book.build_rebalance_actions(
        strategy=strategy,
        decision_portfolio=decision_portfolio,
        execution_portfolio=execution_portfolio,
        candles_by_symbol=candles_by_symbol,
        execution_prices=execution_prices,
        fee_rate=fee_rate,
        raw_actions=raw_actions,
    )


def _record_execution_transform_audit(
    strategy: PortfolioStrategyBase,
    timestamp: pd.Timestamp,
    decision_portfolio: PortfolioState,
    portfolio: PortfolioState,
    candles_by_symbol: dict[str, pd.DataFrame],
    mark_prices: dict[str, float],
    cumulative_financing: float,
    financing_cost_today: float,
) -> list[dict]:
    return v47_position_book.audit_rows(
        strategy=strategy,
        timestamp=timestamp,
        decision_portfolio=decision_portfolio,
        portfolio=portfolio,
        candles_by_symbol=candles_by_symbol,
        mark_prices=mark_prices,
        cumulative_financing=cumulative_financing,
        financing_cost_today_value=financing_cost_today,
    )


def _build_execution_transform_liquidation_actions(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    mark_prices: dict[str, float],
    fee_rate: float,
) -> list[Action]:
    return v47_position_book.build_liquidation_actions(
        strategy=strategy,
        portfolio=portfolio,
        mark_prices=mark_prices,
        fee_rate=fee_rate,
    )


def _build_execution_transform_intraday_shock_actions(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    intraday_rows: dict[str, pd.Series],
    fee_rate: float,
) -> list[Action]:
    return v47_position_book.build_intraday_shock_actions(
        strategy=strategy,
        portfolio=portfolio,
        intraday_rows=intraday_rows,
        fee_rate=fee_rate,
    )


def run_rebalance_backtest(
    candle_dfs: dict[str, pd.DataFrame],
    strategy: PortfolioStrategyBase,
    initial_capital: float = 100.0,
    reserve: float = 20.0,
    fee_rate: float = 0.001,
    execution_mode: str = "next_open",
) -> pd.DataFrame:
    """Run an event-driven backtest.

    execution_mode:
      - same_close: signal uses the current bar and fills at the current close.
      - next_open: signal uses the previous bar and fills at the current open.
      - next_close: signal uses the previous bar and fills at the current close.

    Portfolio snapshots are always marked to the current close.
    """
    if execution_mode not in {"same_close", "next_open", "next_close"}:
        raise ValueError("execution_mode must be one of: same_close, next_open, next_close")

    for symbol, df in candle_dfs.items():
        if "ema168" not in df.columns:
            candle_dfs[symbol] = strategy.compute_indicators(df)

    timestamps = align_timestamps(candle_dfs)
    ts_positions: dict[str, np.ndarray] = {}
    for symbol, df in candle_dfs.items():
        ts_positions[symbol] = df["timestamp"].searchsorted(timestamps, side="right")

    portfolio = PortfolioState(cash=initial_capital)
    use_execution_transform = hasattr(strategy, "execution_target_for_symbol")
    decision_portfolio = PortfolioState(cash=initial_capital) if use_execution_transform else portfolio
    for symbol in strategy.TARGET_ALLOC:
        portfolio.positions[symbol] = PositionState()
        if use_execution_transform:
            decision_portfolio.positions[symbol] = PositionState()

    history: list[dict] = []
    action_records: list[dict] = []
    execution_transform_audit: list[dict] = []
    cumulative_fees = 0.0
    cumulative_financing = 0.0

    for idx, ts in enumerate(timestamps):
        truncated: dict[str, pd.DataFrame] = {}
        execution_prices: dict[str, float] = {}
        mark_prices: dict[str, float] = {}
        signal_timestamps: dict[str, pd.Timestamp] = {}
        indicator_timestamps: dict[str, pd.Timestamp] = {}
        btc_regime_timestamps: dict[str, pd.Timestamp] = {}
        intraday_rows: dict[str, pd.Series] = {}

        for symbol, df in candle_dfs.items():
            pos = int(ts_positions[symbol][idx])
            if pos > 0:
                current_row = df.iloc[pos - 1]
                mark_prices[symbol] = float(current_row["close"])
                intraday_rows[symbol] = current_row

            if execution_mode == "same_close":
                signal_pos = pos
                price_row_pos = pos - 1
                price_col = "close"
            else:
                signal_pos = pos - 1
                price_row_pos = pos - 1
                price_col = "open" if execution_mode == "next_open" else "close"

            if signal_pos > 0 and price_row_pos >= 0:
                truncated[symbol] = df.iloc[:signal_pos]
                price_row = df.iloc[price_row_pos]
                signal_row = df.iloc[signal_pos - 1]
                execution_prices[symbol] = float(price_row[price_col])
                signal_timestamps[symbol] = signal_row["timestamp"]
                indicator_timestamps[symbol] = signal_row["timestamp"]
                btc_regime_timestamps[symbol] = signal_row.get("btc_regime_timestamp")
            else:
                truncated[symbol] = df.iloc[:0]

        raw_actions = (
            strategy.compute_actions(truncated, decision_portfolio, execution_prices)
            if execution_prices else []
        )

        if use_execution_transform:
            for action in raw_actions:
                if action.side == "sell":
                    execute_action_in_backtest(action, decision_portfolio, fee_rate)
            for action in raw_actions:
                if action.side == "buy":
                    execute_action_in_backtest(action, decision_portfolio, fee_rate)
            actions = _build_execution_transform_actions(
                strategy=strategy,
                decision_portfolio=decision_portfolio,
                execution_portfolio=portfolio,
                candles_by_symbol=truncated,
                execution_prices=execution_prices,
                fee_rate=fee_rate,
                raw_actions=raw_actions,
            )
        else:
            actions = raw_actions

        for action in actions:
            if action.side == "sell":
                execute_action_in_backtest(action, portfolio, fee_rate)
        for action in actions:
            if action.side == "buy":
                execute_action_in_backtest(action, portfolio, fee_rate)

        financing_cost_today = 0.0
        if use_execution_transform:
            intraday_shock_actions = _build_execution_transform_intraday_shock_actions(
                strategy=strategy,
                portfolio=portfolio,
                intraday_rows=intraday_rows,
                fee_rate=fee_rate,
            )
            for action in intraday_shock_actions:
                execute_action_in_backtest(action, portfolio, fee_rate)
            actions.extend(intraday_shock_actions)

            financing_cost_today = v47_position_book.financing_cost_today(strategy, portfolio)
            if financing_cost_today > 0.0:
                portfolio.cash -= financing_cost_today
                cumulative_financing += financing_cost_today
            liquidation_actions = _build_execution_transform_liquidation_actions(
                strategy=strategy,
                portfolio=portfolio,
                mark_prices=mark_prices,
                fee_rate=fee_rate,
            )
            for action in liquidation_actions:
                execute_action_in_backtest(action, portfolio, fee_rate)
            actions.extend(liquidation_actions)

        for action in actions:
            notional = action.quantity * action.price
            fee = notional * fee_rate
            cumulative_fees += fee
            action_records.append({
                "timestamp": ts,
                "signal_timestamp": signal_timestamps.get(action.symbol),
                "indicator_timestamp": indicator_timestamps.get(action.symbol),
                "btc_regime_timestamp": btc_regime_timestamps.get(action.symbol),
                "execution_mode": execution_mode,
                "symbol": action.symbol,
                "side": action.side,
                "quantity": action.quantity,
                "price": action.price,
                "notional": notional,
                "fee": fee,
                "reason": action.reason,
                **_decision_fields(action.reason),
                **getattr(action, "diagnostics", {}),
            })

        snapshot = {
            "timestamp": ts,
            "cash": portfolio.cash,
            "cumulative_fees": cumulative_fees,
            "cumulative_financing": cumulative_financing,
        }
        total_value = portfolio.cash
        for symbol in strategy.TARGET_ALLOC:
            pos_state = portfolio.positions.get(symbol, PositionState())
            price = mark_prices.get(symbol, 0.0)
            pos_value = pos_state.quantity * price
            total_value += pos_value
            snapshot[f"{symbol}_price"] = price
            snapshot[f"{symbol}_qty"] = pos_state.quantity
            snapshot[f"{symbol}_avg_cost"] = pos_state.avg_cost
            snapshot[f"{symbol}_value"] = pos_value

        snapshot["total_value"] = total_value
        snapshot["action_count"] = len(actions)
        snapshot["action_summary"] = "; ".join(
            f"{a.side[:1]}/{a.symbol[:3]}/{a.reason}" for a in actions
        )
        history.append(snapshot)

        if use_execution_transform:
            execution_transform_audit.extend(_record_execution_transform_audit(
                strategy=strategy,
                timestamp=ts,
                decision_portfolio=decision_portfolio,
                portfolio=portfolio,
                candles_by_symbol=truncated,
                mark_prices=mark_prices,
                cumulative_financing=cumulative_financing,
                financing_cost_today=financing_cost_today,
            ))

    result = pd.DataFrame(history)
    result.attrs["action_log"] = pd.DataFrame(action_records)
    result.attrs["execution_mode"] = execution_mode
    if execution_transform_audit:
        result.attrs["execution_transform_audit"] = pd.DataFrame(execution_transform_audit)
    if hasattr(strategy, "strategy_diagnostics"):
        result.attrs["strategy_diagnostics"] = strategy.strategy_diagnostics()
    if hasattr(strategy, "strategy_defense_episodes"):
        result.attrs["strategy_defense_episodes"] = pd.DataFrame(strategy.strategy_defense_episodes())
    if hasattr(strategy, "strategy_risk_cycles"):
        result.attrs["strategy_risk_cycles"] = pd.DataFrame(strategy.strategy_risk_cycles())
    if hasattr(strategy, "strategy_sleeve_events"):
        result.attrs["sleeve_events"] = pd.DataFrame(strategy.strategy_sleeve_events())
    if hasattr(strategy, "strategy_sleeve_daily"):
        result.attrs["sleeve_daily"] = pd.DataFrame(strategy.strategy_sleeve_daily())
    if hasattr(strategy, "strategy_base_lot_events"):
        result.attrs["base_lot_events"] = pd.DataFrame(strategy.strategy_base_lot_events())
    if hasattr(strategy, "strategy_base_deferred_candidates"):
        result.attrs["base_deferred_candidates"] = pd.DataFrame(strategy.strategy_base_deferred_candidates())
    if hasattr(strategy, "strategy_decision_trace"):
        result.attrs["decision_trace"] = pd.DataFrame(strategy.strategy_decision_trace())
    if hasattr(strategy, "strategy_candidate_orders"):
        result.attrs["candidate_orders"] = pd.DataFrame(strategy.strategy_candidate_orders())
    if hasattr(strategy, "strategy_risk_assessment_shadow"):
        result.attrs["risk_assessment_shadow"] = pd.DataFrame(strategy.strategy_risk_assessment_shadow())
    if hasattr(strategy, "strategy_intent_plan_shadow"):
        result.attrs["intent_plan_shadow"] = pd.DataFrame(strategy.strategy_intent_plan_shadow())
    if hasattr(strategy, "strategy_target_vector_shadow"):
        result.attrs["target_vector_shadow"] = pd.DataFrame(strategy.strategy_target_vector_shadow())
    if hasattr(strategy, "strategy_budget_ledger_shadow"):
        result.attrs["budget_ledger_shadow"] = pd.DataFrame(strategy.strategy_budget_ledger_shadow())
    if hasattr(strategy, "strategy_order_arbiter_shadow"):
        result.attrs["order_arbiter_shadow"] = pd.DataFrame(strategy.strategy_order_arbiter_shadow())
    if hasattr(strategy, "strategy_symbol_policy_shadow"):
        result.attrs["symbol_policy_shadow"] = pd.DataFrame(strategy.strategy_symbol_policy_shadow())
    if hasattr(strategy, "strategy_recovery_state_machine_shadow"):
        result.attrs["recovery_state_machine_shadow"] = pd.DataFrame(strategy.strategy_recovery_state_machine_shadow())
    if hasattr(strategy, "strategy_lifecycle_state_shadow"):
        result.attrs["lifecycle_state_shadow"] = pd.DataFrame(strategy.strategy_lifecycle_state_shadow())
    if hasattr(strategy, "strategy_recovery_credit_events"):
        result.attrs["recovery_credit_events"] = pd.DataFrame(strategy.strategy_recovery_credit_events())
    if hasattr(strategy, "strategy_recovery_credit_checks"):
        result.attrs["recovery_credit_checks"] = pd.DataFrame(strategy.strategy_recovery_credit_checks())
    if hasattr(strategy, "strategy_protected_recovery_events"):
        result.attrs["protected_recovery_events"] = pd.DataFrame(strategy.strategy_protected_recovery_events())
    if hasattr(strategy, "strategy_outer_overlay_events"):
        result.attrs["outer_overlay_events"] = pd.DataFrame(strategy.strategy_outer_overlay_events())
    return result


def _decision_fields(reason: str) -> dict:
    parsed = parse_action_reason(reason)
    return {
        "setup": parsed.get("setup", ""),
        "risk_score": parsed.get("risk_score"),
        "trend_risk": parsed.get("trend_risk"),
        "drawdown_risk": parsed.get("drawdown_risk"),
        "raw_state": parsed.get("raw_state", ""),
        "confirmed_state": parsed.get("confirmed_state", ""),
        "target_pct": parsed.get("target_pct"),
        "guards": parsed.get("guards", ""),
    }


def calculate_portfolio_performance(
    result_df: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 365 * 24,
    candle_df: pd.DataFrame | None = None,
    fee_rate: float = 0.0,
    benchmark_entry_col: str = "close",
) -> dict:
    """Calculate portfolio metrics for the supplied evaluation rows."""
    if result_df.empty:
        return {}

    df = result_df.copy()
    df["return"] = df["total_value"].pct_change().fillna(0)
    if not df.empty and initial_capital > 0:
        df.loc[df.index[0], "return"] = df["total_value"].iloc[0] / initial_capital - 1

    total_ret = (1 + df["return"]).prod() - 1
    final_value = float(df["total_value"].iloc[-1])
    periods = len(df)

    alloc = {}
    for col in df.columns:
        if col.endswith("_value"):
            symbol = col.replace("_value", "")
            alloc[symbol] = float(df[col].iloc[-1]) / final_value if final_value > 0 else 0.0

    action_log = result_df.attrs.get("action_log")
    if action_log is not None and not action_log.empty and "fee" in action_log.columns:
        fee_cost = float(action_log["fee"].sum())
    elif "cumulative_fees" in df.columns:
        fee_cost = float(df["cumulative_fees"].iloc[-1])
    else:
        fee_cost = 0.0

    metrics = {
        "periods": periods,
        "periods_per_year": periods_per_year,
        "start_time": df["timestamp"].iloc[0],
        "end_time": df["timestamp"].iloc[-1],
        "initial_capital": initial_capital,
        "final_equity": final_value,
        "total_return": total_ret,
        "annual_return": calculate_annual_return(total_ret, periods, periods_per_year),
        "annual_volatility": calculate_annual_volatility(df["return"], periods_per_year),
        "sharpe": calculate_sharpe(df["return"], periods_per_year),
        "max_drawdown": calculate_max_drawdown(df["return"]),
        "final_cash": float(df["cash"].iloc[-1]),
        "final_allocation": alloc,
        "avg_action_per_bar": float(df["action_count"].mean()),
        "trade_count": int(df["action_count"].sum()),
        "total_fee_cost": fee_cost,
    }

    if candle_df is not None:
        price_cols = ["timestamp", "close"]
        if benchmark_entry_col in candle_df.columns and benchmark_entry_col != "close":
            price_cols.append(benchmark_entry_col)
        close_merge = pd.merge(
            df[["timestamp"]],
            candle_df[price_cols],
            on="timestamp",
            how="inner",
        )
        if not close_merge.empty:
            entry_col = benchmark_entry_col if benchmark_entry_col in close_merge.columns else "close"
            entry_price = float(close_merge[entry_col].iloc[0])
            if entry_price > 0:
                bh_equity = close_merge["close"] / (entry_price * (1 + fee_rate))
                metrics["bh_total_return"] = float(bh_equity.iloc[-1] - 1)
                bh_ret_series = bh_equity.pct_change().fillna(bh_equity.iloc[0] - 1)
            else:
                metrics["bh_total_return"] = 0.0
                bh_ret_series = close_merge["close"].pct_change().fillna(0)
            metrics["bh_max_drawdown"] = calculate_max_drawdown(bh_ret_series)
            metrics["bh_annual_return"] = calculate_annual_return(
                metrics["bh_total_return"], len(close_merge), periods_per_year)
            metrics["bh_annual_volatility"] = calculate_annual_volatility(
                bh_ret_series, periods_per_year)
            metrics["bh_sharpe"] = calculate_sharpe(
                bh_ret_series, periods_per_year)
    else:
        metrics["bh_total_return"] = 0.0
        metrics["bh_max_drawdown"] = 0.0
        metrics["bh_annual_return"] = 0.0
        metrics["bh_annual_volatility"] = 0.0
        metrics["bh_sharpe"] = 0.0

    if candle_df is not None and "ema168" in candle_df.columns:
        merged = pd.merge(
            df[["timestamp", "return"]],
            candle_df[["timestamp", "close", "ema168"]],
            on="timestamp",
            how="inner",
        )
        if not merged.empty:
            merged["bh_return"] = merged["close"].pct_change().fillna(0)
            merged["is_bull"] = merged["close"] > merged["ema168"]

            bull = merged[merged["is_bull"]]
            bear = merged[~merged["is_bull"]]

            strat_bull = bull["return"].sum()
            bh_bull = bull["bh_return"].sum()
            strat_bear = bear["return"].sum()
            bh_bear = bear["bh_return"].sum()

            metrics["bull_capture_ratio"] = (
                strat_bull / bh_bull if bh_bull > 0.001 else float("nan")
            )
            metrics["bear_protection"] = (
                1 - strat_bear / bh_bear if bh_bear < -0.001 else float("nan")
            )

    return metrics
