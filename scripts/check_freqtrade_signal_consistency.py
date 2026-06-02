#!/usr/bin/env python3
"""Compare stateless Freqtrade decisions with stateful native signals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils
from crypto_spot_v1.freqtrade_adapter import (
    build_native_signal_frame,
    build_target_position_decision,
)


PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"


def main() -> None:
    rows = []
    for pair in PAIRS:
        df = _load_pair(pair)
        native = build_native_signal_frame(
            pair=pair,
            dataframe=strategy_utils.compute_indicators(df),
            strategy_name="v2_19B",
            min_notional=0.0,
        )
        stateless = _stateless_decisions(pair, native)

        native_buys = _date_set(native, "native_action", "buy")
        native_sells = _date_set(native, "native_action", "sell")
        stateless_buys = _date_set(stateless, "stateless_entry", True)
        stateless_sells = _date_set(stateless, "stateless_exit", True)

        rows.append({
            "pair": pair,
            "native_buys": len(native_buys),
            "stateless_buys": len(stateless_buys),
            "extra_stateless_buys": len(stateless_buys - native_buys),
            "missed_native_buys": len(native_buys - stateless_buys),
            "native_sells": len(native_sells),
            "stateless_sells": len(stateless_sells),
            "extra_stateless_sells": len(stateless_sells - native_sells),
            "missed_native_sells": len(native_sells - stateless_sells),
            "first_extra_entry": _first_date(stateless_buys - native_buys),
            "first_missed_entry": _first_date(native_buys - stateless_buys),
            "first_extra_exit": _first_date(stateless_sells - native_sells),
            "first_missed_exit": _first_date(native_sells - stateless_sells),
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))


def _load_pair(pair: str) -> pd.DataFrame:
    name = pair.replace("/", "_") + "-1d.feather"
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing Freqtrade data file: {path}")
    df = pd.read_feather(path)
    if "timestamp" not in df.columns:
        df["timestamp"] = df["date"] if "date" in df.columns else df.index
    return df


def _stateless_decisions(pair: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe[["timestamp"]].copy()
    out["stateless_entry"] = False
    out["stateless_exit"] = False
    for idx in range(220, len(dataframe)):
        frame = dataframe.iloc[: idx + 1]
        entry = build_target_position_decision(
            pair=pair,
            dataframe=frame,
            current_position_pct=0.0,
            strategy_name="v2_19B",
            min_notional=0.0,
        )
        exit_ = build_target_position_decision(
            pair=pair,
            dataframe=frame,
            current_position_pct=1.0,
            strategy_name="v2_19B",
            min_notional=0.0,
        )
        row_index = out.index[idx]
        out.loc[row_index, "stateless_entry"] = entry.action == "buy" and entry.delta_pct >= 0.02
        out.loc[row_index, "stateless_exit"] = exit_.action == "sell" and abs(exit_.delta_pct) >= 0.02
    return out


def _date_set(df: pd.DataFrame, col: str, value) -> set[pd.Timestamp]:
    mask = df[col] == value
    return set(pd.to_datetime(df.loc[mask, "timestamp"], utc=True))


def _first_date(values: set[pd.Timestamp]) -> str:
    if not values:
        return ""
    return str(min(values).date())


if __name__ == "__main__":
    main()
