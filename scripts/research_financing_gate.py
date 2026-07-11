#!/usr/bin/env python3
"""Research gates for the financed portion of V1 exposure."""

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
    symbol_release_metrics,
)
from scripts.research_btc_regime import delay_btc  # noqa: E402
from scripts.research_funding_stress import attach_daily_funding  # noqa: E402
from scripts.research_realism_robustness import (  # noqa: E402
    curve_attribution,
    weighted_portfolio_metrics,
)


SCENARIOS = {
    "a_realistic": {"gate": False, "non_strong_cap": 0.0, "strong_cap": 0.0, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "a_stress": {"gate": False, "non_strong_cap": 0.0, "strong_cap": 0.0, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 2.0, "slippage": 10.0, "fill": 0.95, "delay": 0},
    "g1_non_strong_1_0": {"gate": True, "non_strong_cap": 1.0, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g2_non_strong_1_2": {"gate": True, "non_strong_cap": 1.2, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g3_vol_non_strong": {"gate": True, "non_strong_cap": 1.2, "strong_cap": 1.75, "vol_target": 0.60, "funding_q": 0.0, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g4_funding_80q": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.80, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g4_funding_70q": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.70, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g4_funding_90q": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.90, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 0},
    "g4_btc_delay_1d": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.80, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 1},
    "g4_stress": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.80, "funding_mult": 2.0, "slippage": 10.0, "fill": 0.95, "delay": 0},
    "g4_90q_btc_delay_1d": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.90, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 1},
    "g4_90q_stress": {"gate": True, "non_strong_cap": 1.75, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.90, "funding_mult": 2.0, "slippage": 10.0, "fill": 0.95, "delay": 0},
    "g2_btc_delay_1d": {"gate": True, "non_strong_cap": 1.2, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 1.0, "slippage": 5.0, "fill": 1.0, "delay": 1},
    "g2_stress": {"gate": True, "non_strong_cap": 1.2, "strong_cap": 1.75, "vol_target": 0.0, "funding_q": 0.0, "funding_mult": 2.0, "slippage": 10.0, "fill": 0.95, "delay": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--output-dir", default="results/research/financing_gate")
    parser.add_argument("--scenarios", default="", help="Optional comma-separated scenario names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = _as_utc(args.start), _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    base_data = _load_review_data(runner, SYMBOL_ORDER + ["BTC/USDT"], start, end)
    summary, calendar_rows, worst_rows, symbol_rows, weight_rows = [], [], [], [], []

    selected = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    unknown = set(selected) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}")
    scenario_items = [(name, SCENARIOS[name]) for name in selected] if selected else list(SCENARIOS.items())
    for name, scenario in scenario_items:
        data = attach_daily_funding(base_data, start, end, scenario["funding_mult"])
        if scenario["delay"]:
            data = delay_btc(data, scenario["delay"])
        report = run_full_window(
            "eth_bnb_futures_v1",
            data,
            runner,
            start,
            end,
            target_gross_cap=2.0,
            strategy_overrides={
                "RESEARCH_SLIPPAGE_BPS": scenario["slippage"],
                "RESEARCH_FILL_RATIO": scenario["fill"],
                "RESEARCH_ACTUAL_GROSS_HARD_CAP": 2.20,
                "RESEARCH_ACTUAL_GROSS_REDUCE_TO": 2.00,
                "RESEARCH_INTRADAY_LIQUIDATION": True,
                "RESEARCH_MAINTENANCE_MARGIN_RATE": 0.005,
                "RESEARCH_LIQUIDATION_FEE_RATE": 0.01,
                "RESEARCH_FINANCING_GATE_ENABLED": scenario["gate"],
                "RESEARCH_FINANCING_GATE_NON_STRONG_BULL_CAP": scenario["non_strong_cap"],
                "RESEARCH_FINANCING_GATE_STRONG_BULL_CAP": scenario["strong_cap"],
                "RESEARCH_FINANCING_GATE_VOL_TARGET": scenario["vol_target"],
                "RESEARCH_FINANCING_GATE_FUNDING_QUANTILE": scenario["funding_q"],
                "RESEARCH_FINANCING_GATE_FUNDING_CAP": 1.0,
            },
        )
        metrics = build_metrics(report, start, end)
        non_2021, worst, calendar = curve_attribution(report)
        audit = report["execution_transform_audit"]
        reasons = audit["transform_reason"].astype(str)
        summary.append({
            "scenario": name,
            **scenario,
            "annual_return": metrics["strategy_annual_return"],
            "max_drawdown": metrics["strategy_max_drawdown"],
            "sharpe_daily": metrics["strategy_sharpe_daily"],
            "observed_max_gross": metrics["execution_transform_max_gross_position"],
            "trade_count": metrics["trade_count"],
            "gate_rows": int(reasons.str.contains("financing-gate", regex=False).sum()),
            "funding_gate_rows": int(reasons.str.contains("funding-high", regex=False).sum()),
            **{f"ex_2021_{key}": value for key, value in non_2021.items()},
        })
        calendar_rows.extend({"scenario": name, **row} for row in calendar)
        worst_rows.extend({"scenario": name, **row} for row in worst)
        for symbol, values in symbol_release_metrics(report).items():
            symbol_rows.append({"scenario": name, "symbol": symbol, **values})
        weight_rows.extend({"scenario": name, **row} for row in weighted_portfolio_metrics(report))
        print(
            f"{name}: annual={metrics['strategy_annual_return']:.2%} mdd={metrics['strategy_max_drawdown']:.2%} "
            f"sharpe={metrics['strategy_sharpe_daily']:.3f} maxGross={metrics['execution_transform_max_gross_position']:.3f}",
            flush=True,
        )

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in {
        "summary.csv": summary,
        "calendar_returns.csv": calendar_rows,
        "worst_windows.csv": worst_rows,
        "symbol_metrics.csv": symbol_rows,
        "weight_sensitivity.csv": weight_rows,
    }.items():
        pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
    manifest = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": {"start": str(start.date()), "end": str(end.date())},
        "scenarios": dict(scenario_items),
        "common": {
            "target_gross_cap": 2.0,
            "actual_gross_hard_cap": 2.2,
            "actual_gross_reduce_to": 2.0,
            "path_dependent_funding": True,
            "strategy_defaults_changed": False,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"financing_gate_research={output_dir}")


if __name__ == "__main__":
    main()
