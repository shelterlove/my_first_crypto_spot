from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research_symbol_weights import build_weight_study  # noqa: E402


def test_weight_study_preserves_pre_listing_cash() -> None:
    timestamps = pd.date_range("2020-01-01", periods=3, tz="UTC")
    equity = pd.DataFrame({
        "symbol": ["ETH/USDT"] * 3 + ["BNB/USDT"] * 2,
        "timestamp": list(timestamps) + list(timestamps[1:]),
        "equity_norm": [1.0, 1.1, 1.2, 1.0, 0.8],
        "position_pct": [1.0, 1.0, 1.0, 0.0, 1.0],
    })
    study = build_weight_study(equity, [0.0, 0.5, 1.0])
    assert list(study["eth_weight"]) == [0.0, 0.5, 1.0]
    assert round(study.iloc[0]["total_return"], 8) == -0.2
    assert round(study.iloc[2]["total_return"], 8) == 0.2
    assert study.iloc[1]["max_gross"] <= 1.0
