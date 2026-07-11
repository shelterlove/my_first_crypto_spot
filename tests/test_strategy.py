#!/usr/bin/env python3
"""Official V1 clean strategy regression checks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_v1 import strategy_utils
from futures_v1.benchmark import build_strategy
from futures_v1.rolling_windows import run_strategy_rolling
from futures_v1.strategy_core.strategy import FuturesV1Strategy


EXPECTED_DIGESTS = {
    "ETH/USDT": "4d9f29ab377c6cc0",
    "BNB/USDT": "25aa1eb465bc6190",
}
EXPECTED_ACTION_COUNTS = {"ETH/USDT": 26, "BNB/USDT": 26}


def synthetic_daily(n: int = 520, start: float = 100.0, drift: float = 0.0007, amp: float = 0.06) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = []
    for i in range(n):
        wave = math.sin(i / 17.0) * amp + math.sin(i / 43.0) * amp * 0.7
        shock = -0.18 if 260 <= i < 285 else (0.12 if 330 <= i < 350 else 0.0)
        close.append(max(start * (1 + drift * i + wave + shock), start * 0.35))
    close_s = pd.Series(close, dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close_s.shift(1).fillna(close_s.iloc[0]) * 0.998,
        "high": close_s * 1.018,
        "low": close_s * 0.982,
        "close": close_s,
        "volume": 1000.0,
    })


def with_btc_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    btc_i = strategy_utils.compute_indicators(btc)
    btc_i["btc_regime"] = strategy_utils.compute_btc_regime(btc_i)
    btc_i["btc_price_vs_ema72"] = btc_i["close"] / btc_i["ema72"] - 1.0
    btc_i["btc_price_vs_ema168"] = btc_i["close"] / btc_i["ema168"] - 1.0
    maps = {
        "btc_regime": dict(zip(btc_i["timestamp"], btc_i["btc_regime"])),
        "btc_regime_timestamp": dict(zip(btc_i["timestamp"], btc_i["timestamp"])),
        "btc_price_vs_ema72": dict(zip(btc_i["timestamp"], btc_i["btc_price_vs_ema72"])),
        "btc_price_vs_ema168": dict(zip(btc_i["timestamp"], btc_i["btc_price_vs_ema168"])),
        "btc_ema24_slope": dict(zip(btc_i["timestamp"], btc_i["ema24_slope"])),
        "btc_ema168_slope": dict(zip(btc_i["timestamp"], btc_i["ema168_slope"])),
        "btc_roc_20": dict(zip(btc_i["timestamp"], btc_i["roc_20"])),
    }
    out = df.copy()
    for col, values in maps.items():
        out[col] = out["timestamp"].map(values).ffill()
    return out


def action_digest(symbol: str, start: float) -> str:
    btc = synthetic_daily(start=30000.0, drift=0.0004, amp=0.05)
    df = with_btc_features(synthetic_daily(start=start), btc)
    strategy = build_strategy("eth_bnb_futures_v1", 100.0, 20.0, 0.001, min_notional=0.0)
    strategy.TARGET_ALLOC = {symbol: 1.0}
    artifacts: dict[str, list[dict]] = {}
    windows = run_strategy_rolling(
        symbol=symbol,
        df=df,
        strategy=strategy,
        strategy_name="eth_bnb_futures_v1",
        window_days=180,
        step_days=90,
        initial_capital=100.0,
        reserve=20.0,
        fee_rate=0.001,
        timeframe="1d",
        warmup_bars=220,
        execution_mode="next_open",
        artifact_sink=artifacts,
        collect_equity_curve=True,
    )
    assert len(windows) == 2
    actions = pd.DataFrame(artifacts.get("action_logs", []))
    assert len(actions) == EXPECTED_ACTION_COUNTS[symbol]
    rows = actions[["symbol", "side", "quantity", "price", "reason"]].round(8).to_dict("records")
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16]


def test_strategy_has_clean_mro() -> None:
    assert all(not cls.__module__.endswith("_legacy") for cls in FuturesV1Strategy.mro())


def test_strategy_golden_master() -> None:
    assert action_digest("ETH/USDT", 1800.0) == EXPECTED_DIGESTS["ETH/USDT"]
    assert action_digest("BNB/USDT", 250.0) == EXPECTED_DIGESTS["BNB/USDT"]


def test_research_gross_cap_does_not_change_default() -> None:
    strategy = build_strategy("eth_bnb_futures_v1", 100.0, 20.0, 0.001)
    assert strategy._apply_research_target_gross_cap(2.25) == 2.25
    strategy.RESEARCH_TARGET_GROSS_CAP = 1.5
    assert strategy._apply_research_target_gross_cap(2.25) == 1.5


def main() -> None:
    test_strategy_has_clean_mro()
    test_strategy_golden_master()
    print("Official strategy tests passed")


if __name__ == "__main__":
    main()
