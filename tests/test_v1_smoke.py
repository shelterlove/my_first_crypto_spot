#!/usr/bin/env python3
"""Smoke tests for the V1 migration wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import V1SpotStrategy
from crypto_spot_v1.benchmark import BuyHoldStrategy
from crypto_spot_v1.rolling_windows import run_strategy_rolling


def synthetic_daily(n: int = 260) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.25,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1.0,
    })


def main() -> None:
    strategy = V1SpotStrategy()
    strategy.TARGET_ALLOC = {"BTC/USDT": 1.0}

    artifacts: dict[str, list[dict]] = {}
    windows = run_strategy_rolling(
        symbol="BTC/USDT",
        df=synthetic_daily(),
        strategy=strategy,
        strategy_name="v1",
        window_days=40,
        step_days=20,
        initial_capital=100.0,
        reserve=20.0,
        fee_rate=0.001,
        timeframe="1d",
        warmup_bars=200,
        execution_mode="next_open",
        artifact_sink=artifacts,
    )

    assert windows, "expected at least one rolling window"
    actions = pd.DataFrame(artifacts.get("action_logs", []))
    equity = pd.DataFrame(artifacts.get("equity_curves", []))
    assert not equity.empty, "expected equity artifacts"
    assert (pd.to_datetime(equity["timestamp"], utc=True) >= pd.to_datetime(equity["window_start"], utc=True)).all()
    assert equity["cash"].min() >= -1e-8, "cash should not materially go negative"
    value_cols = [col for col in equity.columns if col.endswith("_value") and col != "total_value"]
    reconstructed = equity["cash"] + equity[value_cols].fillna(0.0).sum(axis=1)
    assert (reconstructed - equity["total_value"]).abs().max() < 1e-8
    if not actions.empty:
        ts = pd.to_datetime(actions["timestamp"], utc=True)
        window_start = pd.to_datetime(actions["window_start"], utc=True)
        signal_ts = pd.to_datetime(actions["signal_timestamp"], utc=True)
        assert (ts >= window_start).all()
        assert (signal_ts < ts).all()
        assert set(actions["execution_mode"]) == {"next_open"}
        assert int(equity["action_count"].sum()) == len(actions)

    bh_artifacts: dict[str, list[dict]] = {}
    buy_hold = BuyHoldStrategy(initial_capital=100.0, reserve=20.0, fee_rate=0.001)
    buy_hold.TARGET_ALLOC = {"BTC/USDT": 1.0}
    bh_windows = run_strategy_rolling(
        symbol="BTC/USDT",
        df=synthetic_daily(),
        strategy=buy_hold,
        strategy_name="buy_hold",
        window_days=40,
        step_days=20,
        initial_capital=100.0,
        reserve=20.0,
        fee_rate=0.001,
        timeframe="1d",
        warmup_bars=200,
        execution_mode="next_open",
        artifact_sink=bh_artifacts,
    )
    assert bh_windows, "expected Buy & Hold rolling windows"
    bh_actions = pd.DataFrame(bh_artifacts.get("action_logs", []))
    assert not bh_actions.empty, "expected Buy & Hold action logs"
    assert set(bh_actions["side"]) == {"buy"}
    assert set(bh_actions["execution_mode"]) == {"next_open"}
    spend = bh_actions["notional"] + bh_actions["fee"]
    assert (spend - 100.0).abs().max() < 1e-8, "Buy & Hold should invest all cash after fee adjustment"

    print("V1 smoke tests passed")


if __name__ == "__main__":
    main()
