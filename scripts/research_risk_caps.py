#!/usr/bin/env python3
"""Research gross caps without changing the deployable strategy defaults."""

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
    symbol_release_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--output", default="results/research/risk_cap_study.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caps = [float(value.strip()) for value in args.caps.split(",") if value.strip()]
    if not caps or any(value <= 0.0 for value in caps):
        raise SystemExit("--caps must contain positive numbers")
    start_ts = _as_utc(args.start)
    end_ts = _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    load_symbols = list(dict.fromkeys(SYMBOL_ORDER + ["BTC/USDT"]))
    data = _load_review_data(runner, load_symbols, start_ts, end_ts)

    rows = []
    for cap in caps:
        report = run_full_window(
            "eth_bnb_futures_v1",
            data,
            runner,
            start_ts,
            end_ts,
            target_gross_cap=cap,
        )
        metrics = build_metrics(report, start_ts, end_ts)
        symbol_metrics = symbol_release_metrics(report)
        row = {
            "target_gross_cap": cap,
            "annual_return": metrics["strategy_annual_return"],
            "total_return": metrics["strategy_total_return"],
            "max_drawdown": metrics["strategy_max_drawdown"],
            "sharpe_daily": metrics["strategy_sharpe_daily"],
            "trade_count": metrics["trade_count"],
            "average_gross": metrics["avg_position_pct"],
            "observed_max_gross": metrics["execution_transform_max_gross_position"],
        }
        for symbol, values in symbol_metrics.items():
            prefix = symbol.split("/", 1)[0].lower()
            row[f"{prefix}_total_return"] = values["total_return"]
            row[f"{prefix}_max_drawdown"] = values["max_drawdown"]
            row[f"{prefix}_max_gross"] = values["max_gross"]
        rows.append(row)
        print(
            f"cap={cap:.2f} annual={row['annual_return']:.2%} "
            f"mdd={row['max_drawdown']:.2%} max_gross={row['observed_max_gross']:.3f}",
            flush=True,
        )

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"risk_cap_study={output}")


if __name__ == "__main__":
    main()
