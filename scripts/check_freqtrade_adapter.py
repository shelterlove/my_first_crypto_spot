#!/usr/bin/env python3
"""Sanity-check the Freqtrade adapter without requiring Freqtrade."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils
from crypto_spot_v1.freqtrade_adapter import build_native_signal_frame, build_target_position_decision


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
    df = strategy_utils.compute_indicators(synthetic_daily())
    decision = build_target_position_decision(
        pair="BTC/USDT",
        dataframe=df,
        current_position_pct=0.0,
    )
    assert decision.action in {"buy", "sell", "hold"}
    assert 0.0 <= decision.current_pct <= 1.0
    assert -1.0 <= decision.delta_pct <= 1.0
    if decision.target_pct is not None:
        assert 0.0 <= decision.target_pct <= 1.0
    signal_frame = build_native_signal_frame(
        pair="BTC/USDT",
        dataframe=df,
        strategy_name="v2_21E",
        startup_candle_count=220,
    )
    assert "btc_regime" in signal_frame.columns
    assert signal_frame["btc_regime"].notna().any()
    print(decision)


if __name__ == "__main__":
    main()
