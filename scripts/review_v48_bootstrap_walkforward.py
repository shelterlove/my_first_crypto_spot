#!/usr/bin/env python3
"""Review V4.8 with deployment bootstrap alignment over rolling windows.

This simulates a fresh deployment:
- replay the native strategy through a warmup period to reconstruct state;
- on the first evaluation day, align the live spot sleeve up to a capped
  initial position;
- after that, replay native transformed actions with spot-only cash/position
  constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_spot_v1.backtest_engine import (  # noqa: E402
    calculate_annual_return,
    calculate_annual_volatility,
    calculate_max_drawdown,
    calculate_sharpe,
)
from crypto_spot_v1.backtest_event_driven import run_rebalance_backtest  # noqa: E402
from crypto_spot_v1.benchmark import build_strategy  # noqa: E402
from scripts.generate_daily_signal import _load_data_with_btc_regime  # noqa: E402


@dataclass(frozen=True)
class WindowSpec:
    symbol: str
    window_days: int
    start_idx: int
    end_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest_v1.json")
    parser.add_argument("--strategy", default="v4_8_eth_bnb")
    parser.add_argument("--symbols", default="ETH/USDT,BNB/USDT")
    parser.add_argument("--warmup-bars", type=int, default=220)
    parser.add_argument("--bootstrap-cap", type=float, default=0.35)
    parser.add_argument("--window-days", default="365,730,1095")
    parser.add_argument("--step-days", type=int, default=180)
    parser.add_argument("--max-windows-per-size", type=int, default=8)
    parser.add_argument("--output-dir", default="results/strategy_review/v48_bootstrap_walkforward")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    window_days_values = [int(part.strip()) for part in args.window_days.split(",") if part.strip()]
    all_dfs = _load_data_with_btc_regime({**config, "symbols": symbols})

    out_dir = PROJECT_ROOT / args.output_dir / pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        df = all_dfs[symbol].reset_index(drop=True)
        for window_days in window_days_values:
            specs = build_window_specs(
                symbol=symbol,
                df=df,
                warmup_bars=args.warmup_bars,
                window_days=window_days,
                step_days=args.step_days,
                max_windows=args.max_windows_per_size,
            )
            for spec in specs:
                rows.append(run_window(
                    spec=spec,
                    df=df,
                    strategy_name=args.strategy,
                    config=config,
                    warmup_bars=args.warmup_bars,
                    bootstrap_cap=args.bootstrap_cap,
                ))

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "window_metrics.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "strategy": args.strategy,
        "symbols": symbols,
        "warmup_bars": args.warmup_bars,
        "bootstrap_cap": args.bootstrap_cap,
        "window_days": window_days_values,
        "step_days": args.step_days,
        "max_windows_per_size": args.max_windows_per_size,
    }, indent=2), encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(summary.to_string(index=False))


def build_window_specs(
    *,
    symbol: str,
    df: pd.DataFrame,
    warmup_bars: int,
    window_days: int,
    step_days: int,
    max_windows: int,
) -> list[WindowSpec]:
    specs = []
    start = warmup_bars
    while start + window_days <= len(df):
        specs.append(WindowSpec(symbol=symbol, window_days=window_days, start_idx=start, end_idx=start + window_days))
        start += step_days
    if max_windows > 0 and len(specs) > max_windows:
        idxs = sorted({round(i * (len(specs) - 1) / (max_windows - 1)) for i in range(max_windows)})
        specs = [specs[i] for i in idxs]
    return specs


def run_window(
    *,
    spec: WindowSpec,
    df: pd.DataFrame,
    strategy_name: str,
    config: dict[str, Any],
    warmup_bars: int,
    bootstrap_cap: float,
) -> dict[str, Any]:
    start = spec.start_idx
    end = spec.end_idx
    warmup_start = start - warmup_bars
    backtest_df = df.iloc[warmup_start:end].reset_index(drop=True)
    eval_df = df.iloc[start:end].reset_index(drop=True)
    local_eval_start = warmup_bars

    capital = float(config["capital"]["initial"])
    reserve = float(config["capital"]["reserve"])
    fee_rate = float(config["cost"]["fee_rate"])
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")

    strategy = build_strategy(strategy_name, capital, reserve, fee_rate, min_notional=min_notional)
    setattr(strategy, "TARGET_ALLOC", {spec.symbol: 1.0})
    native = run_rebalance_backtest(
        {spec.symbol: backtest_df},
        strategy,
        initial_capital=capital,
        reserve=reserve,
        fee_rate=fee_rate,
        execution_mode=execution_mode,
    )
    action_log = native.attrs.get("action_log")
    action_log = pd.DataFrame() if action_log is None else action_log.copy()

    replay = replay_spot_bootstrap(
        symbol=spec.symbol,
        native=native,
        action_log=action_log,
        df=backtest_df,
        eval_start=local_eval_start,
        capital=capital,
        fee_rate=fee_rate,
        bootstrap_cap=bootstrap_cap,
    )
    metrics = compute_metrics(replay, capital)
    bh_metrics = compute_buyhold_metrics(eval_df, capital, fee_rate)

    return {
        "symbol": spec.symbol,
        "window_days": spec.window_days,
        "window_start": str(eval_df["timestamp"].iloc[0]),
        "window_end": str(eval_df["timestamp"].iloc[-1]),
        "strategy_total_return": metrics["total_return"],
        "strategy_annual_return": metrics["annual_return"],
        "strategy_max_drawdown": metrics["max_drawdown"],
        "strategy_sharpe": metrics["sharpe"],
        "strategy_annual_volatility": metrics["annual_volatility"],
        "buyhold_total_return": bh_metrics["total_return"],
        "buyhold_annual_return": bh_metrics["annual_return"],
        "buyhold_max_drawdown": bh_metrics["max_drawdown"],
        "buyhold_sharpe": bh_metrics["sharpe"],
        "excess_annual_return": metrics["annual_return"] - bh_metrics["annual_return"],
        "drawdown_reduction": abs(bh_metrics["max_drawdown"]) - abs(metrics["max_drawdown"]),
        "trade_count": metrics["trade_count"],
        "bootstrap_pct": replay.attrs.get("bootstrap_pct", 0.0),
        "final_position_pct": replay.attrs.get("final_position_pct", 0.0),
    }


def replay_spot_bootstrap(
    *,
    symbol: str,
    native: pd.DataFrame,
    action_log: pd.DataFrame,
    df: pd.DataFrame,
    eval_start: int,
    capital: float,
    fee_rate: float,
    bootstrap_cap: float,
) -> pd.DataFrame:
    cash = capital
    qty = 0.0
    avg_cost = 0.0
    fee_total = 0.0
    trade_count = 0
    rows = []

    actions_by_ts = {}
    if not action_log.empty:
        action_log["timestamp"] = pd.to_datetime(action_log["timestamp"], utc=True)
        for ts, group in action_log.groupby("timestamp"):
            actions_by_ts[ts] = group.to_dict("records")

    native_eval = native.iloc[eval_start:].reset_index(drop=True)
    df_eval = df.iloc[eval_start:].reset_index(drop=True)
    if native_eval.empty:
        return pd.DataFrame()

    bootstrap_pct = 0.0
    for i, row in df_eval.iterrows():
        ts = pd.to_datetime(row["timestamp"], utc=True)
        open_price = float(row["open"])
        close_price = float(row["close"])

        if i == 0:
            nrow = native.iloc[eval_start]
            native_total = float(nrow.get("total_value", capital) or capital)
            native_value = float(nrow.get(f"{symbol}_value", 0.0) or 0.0)
            native_pct = native_value / native_total if native_total > 0 else 0.0
            bootstrap_pct = max(0.0, min(float(bootstrap_cap), native_pct, 1.0))
            buy_notional = min(cash / (1.0 + fee_rate), capital * bootstrap_pct)
            if buy_notional > 1e-9 and open_price > 0:
                fee = buy_notional * fee_rate
                qty = buy_notional / open_price
                avg_cost = open_price
                cash -= buy_notional + fee
                fee_total += fee
                trade_count += 1

        for action in actions_by_ts.get(ts, []):
            side = str(action.get("side", ""))
            price = float(action.get("price", open_price) or open_price)
            if price <= 0:
                continue
            target_pct = action.get("target_pct")
            if target_pct is None or pd.isna(target_pct):
                notional = abs(float(action.get("notional", 0.0) or 0.0))
            else:
                target = max(0.0, min(1.0, float(target_pct)))
                total_before = cash + qty * price
                current_value = qty * price
                desired_value = target * total_before
                notional = abs(desired_value - current_value)

            if side == "buy":
                buy_notional = min(notional, cash / (1.0 + fee_rate))
                if buy_notional <= 1e-9:
                    continue
                fee = buy_notional * fee_rate
                new_qty = buy_notional / price
                avg_cost = ((avg_cost * qty) + buy_notional) / (qty + new_qty) if qty + new_qty > 0 else 0.0
                qty += new_qty
                cash -= buy_notional + fee
                fee_total += fee
                trade_count += 1
            elif side == "sell":
                sell_qty = min(qty, notional / price)
                if sell_qty <= 1e-12:
                    continue
                proceeds = sell_qty * price
                fee = proceeds * fee_rate
                qty -= sell_qty
                cash += proceeds - fee
                fee_total += fee
                trade_count += 1
                if qty <= 1e-12:
                    qty = 0.0
                    avg_cost = 0.0

        value = qty * close_price
        total = cash + value
        rows.append({
            "timestamp": ts,
            "cash": cash,
            f"{symbol}_qty": qty,
            f"{symbol}_avg_cost": avg_cost,
            f"{symbol}_value": value,
            "total_value": total,
            "action_count": trade_count,
            "cumulative_fees": fee_total,
        })

    out = pd.DataFrame(rows)
    final_total = float(out["total_value"].iloc[-1]) if not out.empty else 0.0
    final_value = float(out[f"{symbol}_value"].iloc[-1]) if not out.empty else 0.0
    out.attrs["bootstrap_pct"] = bootstrap_pct
    out.attrs["final_position_pct"] = final_value / final_total if final_total > 0 else 0.0
    out.attrs["trade_count"] = trade_count
    return out


def compute_metrics(result: pd.DataFrame, capital: float) -> dict[str, float]:
    if result.empty:
        return empty_metrics()
    returns = result["total_value"].pct_change().fillna(0.0)
    returns.iloc[0] = result["total_value"].iloc[0] / capital - 1.0
    total_return = float((1.0 + returns).prod() - 1.0)
    return {
        "total_return": total_return,
        "annual_return": calculate_annual_return(total_return, len(result), 365),
        "annual_volatility": calculate_annual_volatility(returns, 365),
        "sharpe": calculate_sharpe(returns, 365),
        "max_drawdown": calculate_max_drawdown(returns),
        "trade_count": int(result.attrs.get("trade_count", 0)),
    }


def compute_buyhold_metrics(df: pd.DataFrame, capital: float, fee_rate: float) -> dict[str, float]:
    if df.empty:
        return empty_metrics()
    entry = float(df["open"].iloc[0])
    if entry <= 0:
        return empty_metrics()
    qty = capital / (entry * (1.0 + fee_rate))
    equity = qty * df["close"].astype(float)
    returns = equity.pct_change().fillna(equity.iloc[0] / capital - 1.0)
    total_return = float((1.0 + returns).prod() - 1.0)
    return {
        "total_return": total_return,
        "annual_return": calculate_annual_return(total_return, len(df), 365),
        "annual_volatility": calculate_annual_volatility(returns, 365),
        "sharpe": calculate_sharpe(returns, 365),
        "max_drawdown": calculate_max_drawdown(returns),
        "trade_count": 1,
    }


def empty_metrics() -> dict[str, float]:
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "annual_volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["symbol", "window_days"]
    for keys, group in detail.groupby(group_cols):
        symbol, window_days = keys
        rows.append({
            "symbol": symbol,
            "window_days": window_days,
            "windows": len(group),
            "mean_strategy_annual": group["strategy_annual_return"].mean(),
            "median_strategy_annual": group["strategy_annual_return"].median(),
            "mean_buyhold_annual": group["buyhold_annual_return"].mean(),
            "mean_excess_annual": group["excess_annual_return"].mean(),
            "win_rate_vs_buyhold": float((group["excess_annual_return"] > 0).mean()),
            "mean_strategy_mdd": group["strategy_max_drawdown"].mean(),
            "mean_buyhold_mdd": group["buyhold_max_drawdown"].mean(),
            "mean_sharpe": group["strategy_sharpe"].replace([math.inf, -math.inf], pd.NA).dropna().mean(),
            "mean_buyhold_sharpe": group["buyhold_sharpe"].replace([math.inf, -math.inf], pd.NA).dropna().mean(),
            "mean_trade_count": group["trade_count"].mean(),
            "mean_bootstrap_pct": group["bootstrap_pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["symbol", "window_days"])


if __name__ == "__main__":
    main()
