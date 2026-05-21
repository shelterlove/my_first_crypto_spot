"""Structured result persistence for V1 backtests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def _json_safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_structured_results(
    results,
    scores,
    verdict,
    config,
    output_dir: Path,
    candidate_name: str,
    timestamp: str,
    artifacts: dict[str, list[dict]] | None = None,
) -> Path:
    run_dir = output_dir / candidate_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary in results.values():
        for perf in summary.perfs:
            for window in perf.windows:
                rows.append(asdict(window))
    pd.DataFrame(rows).to_csv(run_dir / "raw_backtest_results.csv", index=False)

    summary_rows = []
    bh_summary = results.get("buy_hold")
    for name, summary in results.items():
        summary_rows.append({
            "strategy": name,
            "score": scores.get(name),
            "window_count": summary.total_window_count(),
            "mean_return": summary.mean_return(),
            "median_return": summary.median_return(),
            "mean_excess_return": summary.mean_excess_return(),
            "median_excess_return": summary.median_excess_return(),
            "win_rate_vs_bh": summary.win_rate_vs_bh(),
            "mean_max_drawdown": summary.mean_max_drawdown(),
            "mean_sharpe": summary.mean_sharpe(),
            "mean_sortino": summary.mean_sortino(),
            "mean_calmar": summary.mean_calmar(),
            "mean_trade_count": summary.mean_trade_count(),
            "mean_exposure": summary.mean_exposure(),
            "mean_turnover": summary.mean_turnover(),
            "retention_ratio": summary.retention_ratio(bh_summary),
            "drawdown_reduction": summary.drawdown_reduction(bh_summary),
            "excess_return_consistency": summary.excess_return_consistency(),
            "strategy_type": "benchmark" if name == "buy_hold" else summary.classify_strategy(bh_summary),
        })
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_metrics.csv", index=False)

    artifacts = artifacts or {}
    if artifacts.get("equity_curves"):
        pd.DataFrame(artifacts["equity_curves"]).to_csv(
            run_dir / "equity_curves.csv.gz",
            index=False,
            compression="gzip",
        )
    if artifacts.get("action_logs"):
        pd.DataFrame(artifacts["action_logs"]).to_csv(
            run_dir / "action_logs.csv.gz",
            index=False,
            compression="gzip",
        )

    with (run_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe({
            "candidate": candidate_name,
            "timestamp": timestamp,
            "scores": scores,
            "verdict": verdict,
            "config": config,
        }), f, indent=2)

    return run_dir
