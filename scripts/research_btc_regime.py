#!/usr/bin/env python3
"""Ablate and delay BTC reference features without changing strategy code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from futures_v1.benchmark import V1BenchmarkRunner  # noqa: E402
from scripts.render_strategy_review_chart import (  # noqa: E402
    SYMBOL_ORDER,
    _as_utc,
    _load_review_data,
    build_metrics,
    run_full_window,
)


BTC_FEATURE_COLUMNS = (
    "btc_regime",
    "btc_regime_timestamp",
    "btc_price_vs_ema72",
    "btc_price_vs_ema168",
    "btc_ema24_slope",
    "btc_ema168_slope",
    "btc_roc_20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays", default="1,3,7")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--output", default="results/research/btc_regime_study.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delays = [int(value.strip()) for value in args.delays.split(",") if value.strip()]
    if any(value < 1 for value in delays):
        raise SystemExit("--delays must contain positive integers")
    start_ts = _as_utc(args.start)
    end_ts = _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    load_symbols = list(dict.fromkeys(SYMBOL_ORDER + ["BTC/USDT"]))
    base_data = _load_review_data(runner, load_symbols, start_ts, end_ts)

    scenarios = [("baseline", clone_data(base_data)), ("neutral", neutralize_btc(base_data))]
    scenarios.extend((f"delay_{delay}d", delay_btc(base_data, delay)) for delay in delays)
    rows = []
    for name, data in scenarios:
        report = run_full_window("eth_bnb_futures_v1", data, runner, start_ts, end_ts)
        metrics = build_metrics(report, start_ts, end_ts)
        row = {
            "scenario": name,
            "annual_return": metrics["strategy_annual_return"],
            "total_return": metrics["strategy_total_return"],
            "max_drawdown": metrics["strategy_max_drawdown"],
            "sharpe_daily": metrics["strategy_sharpe_daily"],
            "trade_count": metrics["trade_count"],
            "average_gross": metrics["avg_position_pct"],
            "observed_max_gross": metrics["execution_transform_max_gross_position"],
        }
        rows.append(row)
        print(
            f"scenario={name} annual={row['annual_return']:.2%} "
            f"mdd={row['max_drawdown']:.2%} sharpe={row['sharpe_daily']:.3f}",
            flush=True,
        )
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"btc_regime_study={output}")


def clone_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {symbol: frame.copy() for symbol, frame in data.items()}


def neutralize_btc(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = clone_data(data)
    for symbol in SYMBOL_ORDER:
        frame = out[symbol]
        frame["btc_regime"] = "RANGE"
        frame["btc_regime_timestamp"] = frame["timestamp"]
        for column in BTC_FEATURE_COLUMNS:
            if column not in {"btc_regime", "btc_regime_timestamp"}:
                frame[column] = 0.0
    return out


def delay_btc(data: dict[str, pd.DataFrame], days: int) -> dict[str, pd.DataFrame]:
    out = clone_data(data)
    for symbol in SYMBOL_ORDER:
        frame = out[symbol]
        for column in BTC_FEATURE_COLUMNS:
            if column not in frame.columns:
                continue
            shifted = frame[column].shift(days)
            if column == "btc_regime":
                frame[column] = shifted.fillna("RANGE")
            elif column == "btc_regime_timestamp":
                frame[column] = shifted.fillna(frame["timestamp"])
            else:
                frame[column] = pd.to_numeric(shifted, errors="coerce").fillna(0.0)
    return out


if __name__ == "__main__":
    main()
