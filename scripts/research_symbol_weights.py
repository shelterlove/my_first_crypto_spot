#!/usr/bin/env python3
"""Research static ETH/BNB capital weights from independent sleeve curves."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equity",
        default="results/strategy_review/official_v1_full_20200101_20260518/equity_curves.csv",
    )
    parser.add_argument("--eth-weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--output", default="results/research/symbol_weight_study.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    equity = pd.read_csv(PROJECT_ROOT / args.equity)
    weights = [float(value.strip()) for value in args.eth_weights.split(",") if value.strip()]
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise SystemExit("--eth-weights must contain values between 0 and 1")
    study = build_weight_study(equity, weights)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    study.to_csv(output, index=False)
    print(study.to_string(index=False))
    print(f"symbol_weight_study={output}")


def build_weight_study(equity: pd.DataFrame, eth_weights: list[float]) -> pd.DataFrame:
    required = {"symbol", "timestamp", "equity_norm", "position_pct"}
    if not required.issubset(equity.columns):
        raise ValueError(f"Equity data is missing columns: {sorted(required - set(equity.columns))}")
    frame = equity.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    curves = {}
    for symbol in ("ETH/USDT", "BNB/USDT"):
        rows = frame[frame["symbol"] == symbol][["timestamp", "equity_norm", "position_pct"]].copy()
        if rows.empty:
            raise ValueError(f"Missing sleeve curve for {symbol}")
        prefix = symbol.split("/", 1)[0].lower()
        curves[symbol] = rows.rename(columns={
            "equity_norm": f"{prefix}_equity",
            "position_pct": f"{prefix}_gross",
        })
    aligned = pd.merge(curves["ETH/USDT"], curves["BNB/USDT"], on="timestamp", how="outer")
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)
    for column in ("eth_equity", "bnb_equity"):
        aligned[column] = aligned[column].ffill().fillna(1.0)
    for column in ("eth_gross", "bnb_gross"):
        aligned[column] = aligned[column].ffill().fillna(0.0)

    years = max(
        (aligned["timestamp"].iloc[-1] - aligned["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 24 * 3600),
        1e-9,
    )
    rows = []
    for eth_weight in eth_weights:
        bnb_weight = 1.0 - eth_weight
        curve = eth_weight * aligned["eth_equity"] + bnb_weight * aligned["bnb_equity"]
        gross_notional = (
            eth_weight * aligned["eth_equity"] * aligned["eth_gross"]
            + bnb_weight * aligned["bnb_equity"] * aligned["bnb_gross"]
        )
        gross = gross_notional / curve.where(curve > 0.0)
        returns = curve.pct_change().dropna()
        total_return = float(curve.iloc[-1] - 1.0)
        drawdown = curve / curve.cummax() - 1.0
        std = float(returns.std())
        rows.append({
            "eth_weight": eth_weight,
            "bnb_weight": bnb_weight,
            "annual_return": float((1.0 + total_return) ** (1.0 / years) - 1.0),
            "total_return": total_return,
            "max_drawdown": float(drawdown.min()),
            "sharpe_daily": float(returns.mean() / std * math.sqrt(365.0)) if std > 0.0 else float("nan"),
            "average_gross": float(gross.mean()),
            "max_gross": float(gross.max()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
