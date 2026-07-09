#!/usr/bin/env python3
"""Walk-forward review for fixed V4.7 external parameter selections."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_v47_external_params import (  # noqa: E402
    CandidateSpec,
    DEFAULT_BASE_STRATEGY,
    V1BenchmarkRunner,
    as_utc,
    load_data,
    make_strategy_class,
    run_symbol_window,
)


WINDOWS = [
    ("2020_2021", "2020-01-01", "2021-12-31"),
    ("2021_2022", "2021-01-01", "2022-12-31"),
    ("2022_2023", "2022-01-01", "2023-12-31"),
    ("2023_2024", "2023-01-01", "2024-12-31"),
    ("2024_2026", "2024-01-01", "2026-05-18"),
    ("full_2020_2026", "2020-01-01", "2026-05-18"),
]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
SELECTIONS = {
    "production_conservative": PROJECT_ROOT
    / "results"
    / "strategy_review"
    / "v47_external_final_selection"
    / "final_selected_conservative.json",
    "validation_best": PROJECT_ROOT
    / "results"
    / "strategy_review"
    / "v47_external_final_selection"
    / "final_selected_validation_best.json",
}


@dataclass(frozen=True)
class StrategyCase:
    name: str
    selection_name: str
    symbol: str
    spec: CandidateSpec | None


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "strategy_review" / f"v47_walkforward_review_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", output_dir)
    runner.config["symbols"] = SYMBOLS
    data = load_data(runner, runner.config["symbols"])

    rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        cases = [StrategyCase("v7_baseline", "baseline", symbol, None)]
        for selection_name, path in SELECTIONS.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rec = payload.get("recommendations", {}).get(symbol, {"decision": "keep_baseline"})
            cases.append(StrategyCase(selection_name, selection_name, symbol, spec_from_recommendation(symbol, selection_name, rec)))

        for case in cases:
            for window_name, start, end in WINDOWS:
                strategy = None
                strategy_name = DEFAULT_BASE_STRATEGY
                if case.spec is not None:
                    cls = make_strategy_class(DEFAULT_BASE_STRATEGY, case.spec)
                    strategy = cls(
                        initial_capital=float(runner.config["capital"]["initial"]),
                        reserve=float(runner.config["capital"]["reserve"]),
                        fee_rate=float(runner.config["cost"]["fee_rate"]),
                    )
                    strategy_name = case.spec.candidate
                result, _ = run_symbol_window(
                    strategy_name=strategy_name,
                    strategy=strategy,
                    data=data,
                    runner=runner,
                    symbol=symbol,
                    start_ts=as_utc(start),
                    end_ts=as_utc(end),
                    window_name=window_name,
                )
                rows.append({
                    "selection": case.selection_name,
                    "strategy_case": case.name,
                    "symbol": symbol,
                    "window": window_name,
                    "start": start,
                    "end": end,
                    "annual_return": result.strategy_annual_return,
                    "total_return": result.strategy_total_return,
                    "max_drawdown": result.strategy_max_drawdown,
                    "trade_count": result.trade_count,
                    "financing_cost": result.execution_transform_financing_cost,
                    "max_gross": result.execution_transform_max_gross_position,
                    "buyhold_annual_return": result.buyhold_annual_return,
                    "buyhold_total_return": result.buyhold_total_return,
                    "buyhold_max_drawdown": result.buyhold_max_drawdown,
                })

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "walkforward_metrics.csv", index=False)
    summary = build_summary(raw)
    summary.to_csv(output_dir / "walkforward_summary.csv", index=False)
    (output_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")
    print(output_dir)
    print(summary.to_string(index=False))


def spec_from_recommendation(symbol: str, selection_name: str, rec: dict[str, Any]) -> CandidateSpec | None:
    if rec.get("decision") != "use_candidate":
        return None
    return CandidateSpec(
        mode="combined",
        symbol=symbol,
        candidate=f"{selection_name}_{symbol.replace('/', '').lower()}",
        target_pct=rec.get("target_pct"),
        entry_profile=rec.get("entry_profile"),
        min_hold_calls=rec.get("min_hold_calls"),
        trend_mult=rec.get("trend_mult"),
        trend_cap=rec.get("trend_cap"),
        low_recovery_mult=rec.get("low_recovery_mult"),
        low_recovery_cap=rec.get("low_recovery_cap"),
        shock_reduce_step=rec.get("shock_reduce_step"),
        shock_add_step=rec.get("shock_add_step"),
        shock_extra_tiers=rec.get("shock_extra_tiers"),
        shock_max_position=rec.get("shock_max_position"),
        shock_max_gross=rec.get("shock_max_gross"),
    )


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    baseline = raw[raw["strategy_case"] == "v7_baseline"].copy()
    merged = raw.merge(
        baseline[[
            "symbol", "window", "annual_return", "max_drawdown", "trade_count",
            "financing_cost", "max_gross",
        ]].rename(columns={
            "annual_return": "baseline_annual_return",
            "max_drawdown": "baseline_max_drawdown",
            "trade_count": "baseline_trade_count",
            "financing_cost": "baseline_financing_cost",
            "max_gross": "baseline_max_gross",
        }),
        on=["symbol", "window"],
        how="left",
    )
    merged["annual_delta"] = merged["annual_return"] - merged["baseline_annual_return"]
    merged["drawdown_abs_delta"] = merged["max_drawdown"].abs() - merged["baseline_max_drawdown"].abs()
    merged["financing_delta"] = merged["financing_cost"] - merged["baseline_financing_cost"]
    merged["max_gross_delta"] = merged["max_gross"] - merged["baseline_max_gross"]
    review = merged[merged["strategy_case"] != "v7_baseline"].copy()
    rows = []
    for (selection, symbol), group in review.groupby(["selection", "symbol"]):
        wf = group[group["window"] != "full_2020_2026"]
        full = group[group["window"] == "full_2020_2026"].iloc[0]
        rows.append({
            "selection": selection,
            "symbol": symbol,
            "mean_window_annual_delta": wf["annual_delta"].mean(),
            "worst_window_annual_delta": wf["annual_delta"].min(),
            "positive_window_count": int((wf["annual_delta"] > 0).sum()),
            "window_count": int(len(wf)),
            "mean_drawdown_abs_delta": wf["drawdown_abs_delta"].mean(),
            "full_annual_delta": full["annual_delta"],
            "full_drawdown_abs_delta": full["drawdown_abs_delta"],
            "full_financing_delta": full["financing_delta"],
            "full_max_gross_delta": full["max_gross_delta"],
        })
    return pd.DataFrame(rows).sort_values(["selection", "symbol"])


def build_readme(summary: pd.DataFrame) -> str:
    lines = [
        "# V4.7 Walk-Forward Review",
        "",
        "Compared fixed selections against original v7 baseline on overlapping full-window slices.",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
