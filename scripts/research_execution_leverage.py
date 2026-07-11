#!/usr/bin/env python3
"""Compare execution-layer leverage variants without changing V1 defaults."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from futures_v1.benchmark import V1BenchmarkRunner  # noqa: E402
from scripts.render_strategy_review_chart import (  # noqa: E402
    SYMBOL_ORDER,
    _as_utc,
    _load_review_data,
    build_metrics,
    run_full_window,
)
from scripts.research_funding_stress import funding_overlay_metrics  # noqa: E402


VARIANTS = {
    "a_current": {
        "low_recovery_mult": 2.60,
        "low_recovery_cap": 2.00,
        "trend_mult": 1.75,
        "trend_cap": 1.75,
    },
    "b_trend_2_0": {
        "low_recovery_mult": 2.60,
        "low_recovery_cap": 2.00,
        "trend_mult": 2.00,
        "trend_cap": 2.00,
    },
    "c_trend_2_2_low_reduced": {
        "low_recovery_mult": 2.40,
        "low_recovery_cap": 1.90,
        "trend_mult": 2.20,
        "trend_cap": 2.30,
    },
    "d_trend_2_3": {
        "low_recovery_mult": 2.60,
        "low_recovery_cap": 2.00,
        "trend_mult": 2.30,
        "trend_cap": 2.30,
    },
    "e_all_2_3": {
        "low_recovery_mult": 3.00,
        "low_recovery_cap": 2.30,
        "trend_mult": 2.30,
        "trend_cap": 2.30,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--target-gross-cap", type=float, default=2.40)
    parser.add_argument("--output-dir", default="results/research/execution_leverage")
    return parser.parse_args()


def calendar_returns(composite: pd.DataFrame) -> dict[int, float]:
    frame = composite[["timestamp", "strategy_equity"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["year"] = frame["timestamp"].dt.year
    return {
        int(year): float(rows["strategy_equity"].iloc[-1] / rows["strategy_equity"].iloc[0] - 1.0)
        for year, rows in frame.groupby("year")
        if len(rows) > 1
    }


def audit_metrics(audit: pd.DataFrame) -> dict[str, float | int]:
    if audit.empty:
        return {
            "gross_p95": 0.0,
            "gross_p99": 0.0,
            "symbol_days_gross_gt_2": 0,
            "trend_confirmed_rows": 0,
            "low_recovery_rows": 0,
        }
    gross = audit["gross_position"].astype(float)
    reasons = audit["transform_reason"].astype(str)
    return {
        "gross_p95": float(gross.quantile(0.95)),
        "gross_p99": float(gross.quantile(0.99)),
        "symbol_days_gross_gt_2": int((gross > 2.0).sum()),
        "trend_confirmed_rows": int(reasons.str.contains("trend_confirmed", regex=False).sum()),
        "low_recovery_rows": int(reasons.str.contains("low_recovery", regex=False).sum()),
    }


def main() -> None:
    args = parse_args()
    start, end = _as_utc(args.start), _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    data = _load_review_data(runner, SYMBOL_ORDER + ["BTC/USDT"], start, end)
    summary_rows = []
    yearly_rows = []
    for name, overrides in VARIANTS.items():
        report = run_full_window(
            "eth_bnb_futures_v1",
            data,
            runner,
            start,
            end,
            target_gross_cap=args.target_gross_cap,
            execution_overrides=overrides,
        )
        metrics = build_metrics(report, start, end)
        funding_1x = funding_overlay_metrics(report, start, end, 1.0)
        funding_2x = funding_overlay_metrics(report, start, end, 2.0)
        years = calendar_returns(report["composite"])
        yearly_rows.extend({"variant": name, "year": year, "return": value} for year, value in years.items())
        row = {
            "variant": name,
            **{key.lower(): value for key, value in overrides.items()},
            "target_gross_cap": args.target_gross_cap,
            "annual_return": metrics["strategy_annual_return"],
            "total_return": metrics["strategy_total_return"],
            "max_drawdown": metrics["strategy_max_drawdown"],
            "sharpe_daily": metrics["strategy_sharpe_daily"],
            "trade_count": metrics["trade_count"],
            "average_gross": metrics["avg_position_pct"],
            "observed_max_gross": metrics["execution_transform_max_gross_position"],
            "worst_calendar_year": min(years.values()),
            "positive_calendar_years": sum(value > 0.0 for value in years.values()),
            "funding_1x_annual_return": funding_1x["annual_return"],
            "funding_1x_max_drawdown": funding_1x["max_drawdown"],
            "funding_1x_annual_drag": funding_1x["approx_annual_funding_drag"],
            "funding_2x_annual_return": funding_2x["annual_return"],
            "funding_2x_max_drawdown": funding_2x["max_drawdown"],
            **audit_metrics(report["execution_transform_audit"]),
        }
        summary_rows.append(row)
        print(
            f"{name}: annual={row['annual_return']:.2%} mdd={row['max_drawdown']:.2%} "
            f"funding1x={row['funding_1x_annual_return']:.2%} maxGross={row['observed_max_gross']:.3f}",
            flush=True,
        )

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(output_dir / "calendar_returns.csv", index=False)
    manifest = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": {"start": str(start.date()), "end": str(end.date())},
        "target_gross_cap": args.target_gross_cap,
        "variants": VARIANTS,
        "assumptions": {
            "execution": "next_open daily",
            "intraday_shock_ladder": False,
            "funding_overlay": "daily closing-exposure approximation; does not rerun sizing after debits",
            "funding_multipliers": [1.0, 2.0],
            "strategy_defaults_changed": False,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"execution_leverage_research={output_dir}")


if __name__ == "__main__":
    main()
