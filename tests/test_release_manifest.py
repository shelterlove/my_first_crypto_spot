from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_strategy_review_chart import build_composite, frame_fingerprint, symbol_release_metrics  # noqa: E402


def test_frame_fingerprint_is_deterministic_and_content_sensitive() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-02", tz="UTC")
    first = frame_fingerprint(frame, start_ts=start, end_ts=end)
    second = frame_fingerprint(frame.copy(), start_ts=start, end_ts=end)
    changed = frame.copy()
    changed.loc[1, "close"] = 102.5
    third = frame_fingerprint(changed, start_ts=start, end_ts=end)
    assert first == second
    assert first["sha256"] != third["sha256"]


def test_symbol_release_metrics_reports_drawdown_and_gross(monkeypatch) -> None:
    monkeypatch.setattr("scripts.render_strategy_review_chart.SYMBOL_ORDER", ["ETH/USDT"])
    equity = pd.DataFrame({
        "symbol": ["ETH/USDT"] * 3,
        "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        "equity_norm": [1.0, 1.2, 0.9],
        "position_pct": [0.5, 1.2, 0.8],
    })
    actions = pd.DataFrame({"symbol": ["ETH/USDT", "ETH/USDT"]})
    metrics = symbol_release_metrics({"equity": equity, "actions": actions})["ETH/USDT"]
    assert round(metrics["total_return"], 8) == -0.1
    assert metrics["max_drawdown"] == -0.25
    assert metrics["max_gross"] == 1.2
    assert metrics["trade_count"] == 2


def test_composite_treats_pre_listing_sleeve_as_cash(monkeypatch) -> None:
    monkeypatch.setattr("scripts.render_strategy_review_chart.SYMBOL_ORDER", ["ETH/USDT", "BNB/USDT"])
    timestamps = pd.date_range("2020-01-01", periods=3, tz="UTC")
    equity = pd.DataFrame({
        "symbol": ["ETH/USDT"] * 3 + ["BNB/USDT"] * 2,
        "timestamp": list(timestamps) + list(timestamps[1:]),
        "equity_norm": [1.0, 1.1, 1.2, 1.0, 0.9],
        "position_pct": [0.5, 0.6, 0.7, 0.0, 0.4],
    })
    prices = pd.DataFrame({
        "symbol": ["ETH/USDT"] * 3 + ["BNB/USDT"] * 2,
        "timestamp": list(timestamps) + list(timestamps[1:]),
        "price_norm": [1.0, 1.1, 1.2, 1.0, 0.9],
    })
    composite = build_composite(equity, prices)
    assert list(composite["timestamp"]) == list(timestamps)
    assert composite.iloc[0]["strategy_equity"] == 1.0
    assert composite.iloc[0]["avg_position_pct"] == 0.25
