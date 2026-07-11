from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import research_btc_regime as research  # noqa: E402


def sample_data() -> dict[str, pd.DataFrame]:
    timestamp = pd.date_range("2026-01-01", periods=3, tz="UTC")
    traded = pd.DataFrame({
        "timestamp": timestamp,
        "btc_regime": ["BULL", "BEAR", "RANGE"],
        "btc_regime_timestamp": timestamp,
        "btc_roc_20": [0.1, -0.2, 0.0],
    })
    return {"ETH/USDT": traded.copy(), "BNB/USDT": traded.copy(), "BTC/USDT": traded.copy()}


def test_neutralize_btc_does_not_mutate_input(monkeypatch) -> None:
    monkeypatch.setattr(research, "SYMBOL_ORDER", ["ETH/USDT", "BNB/USDT"])
    data = sample_data()
    neutral = research.neutralize_btc(data)
    assert list(neutral["ETH/USDT"]["btc_regime"]) == ["RANGE"] * 3
    assert list(data["ETH/USDT"]["btc_regime"]) == ["BULL", "BEAR", "RANGE"]


def test_delay_btc_is_strictly_lagged(monkeypatch) -> None:
    monkeypatch.setattr(research, "SYMBOL_ORDER", ["ETH/USDT", "BNB/USDT"])
    delayed = research.delay_btc(sample_data(), 1)["ETH/USDT"]
    assert list(delayed["btc_regime"]) == ["RANGE", "BULL", "BEAR"]
    assert list(delayed["btc_roc_20"]) == [0.0, 0.1, -0.2]
