"""Evaluation/reporting layer for V1 backtest runs.

This module intentionally works from runner outputs and artifacts. It does not
change strategy trading rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest_engine import (
    calculate_annual_return,
    calculate_max_drawdown,
    calculate_sharpe,
)
from .metrics import StrategySummary, compute_score
from .diagnostics import (
    build_state_transition_report_from_diagnostics,
    write_diagnostics_outputs,
)
from .html_report import (
    build_strategy_evaluation_summary as build_html_strategy_evaluation_summary,
    generate_evaluation_html as generate_html_report,
)

COMPLETE_MODE = "complete"
FAST_MODE = "fast"
FULL_MODE = "full"
STRESS_MODE = "stress"
LEGACY_MODES = {FAST_MODE, FULL_MODE, STRESS_MODE}
VALID_MODES = {COMPLETE_MODE, *LEGACY_MODES}

COMMON_RESULT_FILES = [
    "experiment_metadata.json",
    "config_snapshot.json",
    "summary_metrics.csv",
    "benchmark_metrics.csv",
    "risk_metrics.csv",
    "active_management_metrics.csv",
    "drawdown_metrics.csv",
    "final_score_report.csv",
    "html_report.html",
]

FULL_RESULT_FILES = [
    "timestamp_audit_report.csv",
    "accounting_audit_report.csv",
    "signal_attribution_buy.csv",
    "signal_attribution_sell.csv",
    "state_transition_report.csv",
    "regime_performance_report.csv",
    "bull_underperformance_window_analysis.csv",
    "risk_score_attribution_report.csv",
    "exposure_diagnostics_report.csv",
    "buy_blocked_report.csv",
    "sell_too_early_report.csv",
]

STRESS_RESULT_FILES = [
    "cost_stress_report.csv",
    "warmup_sensitivity_report.csv",
    "parameter_sensitivity_report.csv",
]


def create_run_id(timestamp: str, candidate_name: str, mode: str) -> str:
    safe_candidate = candidate_name.replace("/", "_").replace(" ", "_")
    if mode == COMPLETE_MODE:
        return f"{timestamp}_{safe_candidate}"
    return f"{timestamp}_{safe_candidate}_{mode}"


def normalize_mode(mode: str | None) -> str:
    """Keep old CLI modes accepted while using one complete evaluation flow."""
    if mode is None:
        return COMPLETE_MODE
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
    return COMPLETE_MODE


def save_evaluation_run(
    *,
    runner,
    results: dict[str, StrategySummary],
    scores: dict[str, float],
    verdict: dict,
    candidate_name: str,
    mode: str,
    timestamp: str,
    config_path: Path,
    output_root: Path,
    diagnostics_enabled: bool | None = None,
) -> Path:
    requested_mode = mode
    mode = normalize_mode(mode)

    run_id = create_run_id(timestamp, candidate_name, mode)
    run_dir = output_root / "v1_eval_upgrade" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "action_logs").mkdir(exist_ok=True)
    (run_dir / "equity_curves").mkdir(exist_ok=True)

    diagnostics_enabled = True if diagnostics_enabled is None else diagnostics_enabled
    config_snapshot = _effective_config(runner.config, candidate_name, mode, diagnostics_enabled)
    config_snapshot["evaluation"]["requested_mode"] = requested_mode
    artifacts = runner.artifacts or {}
    raw_df = _raw_results_df(results)
    summary_df = _summary_metrics_df(results, scores)
    actions_df = pd.DataFrame(artifacts.get("action_logs", []))
    equity_df = pd.DataFrame(artifacts.get("equity_curves", []))

    raw_df.to_csv(run_dir / "raw_backtest_results.csv", index=False)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    _write_artifacts(run_dir, actions_df, equity_df)

    metadata = _build_metadata(
        runner=runner,
        config=config_snapshot,
        config_path=config_path,
        candidate_name=candidate_name,
        mode=mode,
        timestamp=timestamp,
        run_id=run_id,
    )
    _write_json(run_dir / "experiment_metadata.json", metadata)
    _write_json(run_dir / "config_snapshot.json", config_snapshot)

    benchmark_df = build_benchmark_metrics(
        raw_df=raw_df,
        summary_df=summary_df,
        equity_df=equity_df,
        runner=runner,
        candidate_name=candidate_name,
        mode=mode,
    )
    risk_df = build_risk_metrics(equity_df, raw_df, candidate_name)
    active_df = build_active_management_metrics(equity_df, runner, candidate_name)
    drawdown_df = build_drawdown_metrics(equity_df, candidate_name)

    benchmark_df.to_csv(run_dir / "benchmark_metrics.csv", index=False)
    risk_df.to_csv(run_dir / "risk_metrics.csv", index=False)
    active_df.to_csv(run_dir / "active_management_metrics.csv", index=False)
    drawdown_df.to_csv(run_dir / "drawdown_metrics.csv", index=False)
    reference_dir = _find_latest_reference_run(output_root, "v1_less_churn")

    diagnostic_outputs: dict[str, pd.DataFrame] = {}
    if diagnostics_enabled:
        diagnostic_outputs = write_diagnostics_outputs(
            run_dir=run_dir,
            runner=runner,
            raw_df=raw_df,
            actions_df=actions_df,
            equity_df=equity_df,
            metadata=metadata,
            config=config_snapshot,
            candidate_name=candidate_name,
            mode=mode,
        )

    full_outputs: dict[str, pd.DataFrame] = {}
    state_transition = (
        build_state_transition_report_from_diagnostics(
            diagnostic_outputs.get("per_bar_diagnostics", pd.DataFrame()),
            runner,
        )
        if diagnostic_outputs
        else build_state_transition_report(actions_df, runner)
    )
    full_outputs = {
        "timestamp_audit_report.csv": build_timestamp_audit(actions_df),
        "accounting_audit_report.csv": build_accounting_audit(equity_df, actions_df, config_snapshot),
        "signal_attribution_buy.csv": build_signal_attribution(actions_df, runner, side="buy"),
        "signal_attribution_sell.csv": build_signal_attribution(actions_df, runner, side="sell"),
        "state_transition_report.csv": state_transition,
        "regime_performance_report.csv": build_regime_performance_report(raw_df, candidate_name),
        "bull_underperformance_window_analysis.csv": build_bull_underperformance_window_analysis(
            raw_df=raw_df,
            actions_df=actions_df,
            diagnostic_outputs=diagnostic_outputs,
            candidate_name=candidate_name,
        ),
    }
    if diagnostic_outputs:
        full_outputs.update({
            "risk_score_attribution_report.csv": diagnostic_outputs["risk_score_attribution_report"],
            "exposure_diagnostics_report.csv": diagnostic_outputs["exposure_diagnostics_report"],
            "buy_blocked_report.csv": diagnostic_outputs["buy_blocked_report"],
            "sell_too_early_report.csv": diagnostic_outputs["sell_too_early_report"],
        })
    for filename, frame in full_outputs.items():
        frame.to_csv(run_dir / filename, index=False)

    stress_outputs: dict[str, pd.DataFrame] = {}
    stress_outputs = {
        "cost_stress_report.csv": build_cost_stress_report(raw_df, actions_df, config_snapshot, candidate_name),
        "warmup_sensitivity_report.csv": build_warmup_sensitivity_report(runner, candidate_name),
        "parameter_sensitivity_report.csv": build_parameter_sensitivity_report(config_snapshot),
    }
    for filename, frame in stress_outputs.items():
        frame.to_csv(run_dir / filename, index=False)

    optimization_comparison_df = build_strategy_optimization_comparison(
        run_dir=run_dir,
        reference_dir=reference_dir,
        candidate_name=candidate_name,
        mode=mode,
    )
    if not optimization_comparison_df.empty:
        optimization_comparison_df.to_csv(run_dir / "strategy_optimization_comparison.csv", index=False)
        (run_dir / "strategy_optimization_summary.md").write_text(
            build_strategy_optimization_summary(optimization_comparison_df, candidate_name),
            encoding="utf-8",
        )

    score_df = build_final_score_report(
        raw_df=raw_df,
        summary_df=summary_df,
        benchmark_df=benchmark_df,
        active_df=active_df,
        candidate_name=candidate_name,
        mode=mode,
        config=config_snapshot,
    )
    score_df.to_csv(run_dir / "final_score_report.csv", index=False)
    model_review = build_model_review(
        metadata=metadata,
        summary_df=summary_df,
        benchmark_df=benchmark_df,
        raw_df=raw_df,
        score_df=score_df,
        full_outputs=full_outputs,
        optimization_comparison_df=optimization_comparison_df,
        verdict=verdict,
        candidate_name=candidate_name,
    )
    _write_json(run_dir / "model_review.json", model_review)
    (run_dir / "model_review.md").write_text(
        build_model_review_markdown(model_review),
        encoding="utf-8",
    )
    (run_dir / "RESULTS_INDEX.md").write_text(
        build_results_index_markdown(),
        encoding="utf-8",
    )

    html = generate_html_report(
        metadata=metadata,
        summary_df=summary_df,
        benchmark_df=benchmark_df,
        risk_df=risk_df,
        active_df=active_df,
        drawdown_df=drawdown_df,
        score_df=score_df,
        raw_df=raw_df,
        actions_df=actions_df,
        equity_df=equity_df,
        full_outputs=full_outputs,
        stress_outputs=stress_outputs,
        diagnostic_outputs=diagnostic_outputs,
        optimization_comparison_df=optimization_comparison_df,
        mode=mode,
        verdict=verdict,
    )
    (run_dir / "html_report.html").write_text(html, encoding="utf-8")
    (run_dir / "strategy_evaluation_summary.md").write_text(
        build_html_strategy_evaluation_summary(
            metadata=metadata,
            summary_df=summary_df,
            benchmark_df=benchmark_df,
            raw_df=raw_df,
            full_outputs=full_outputs,
            stress_outputs=stress_outputs,
            optimization_comparison_df=optimization_comparison_df,
            verdict=verdict,
        ),
        encoding="utf-8",
    )

    diagnostics = {
        "candidate": candidate_name,
        "mode": mode,
        "timestamp": timestamp,
        "scores": scores,
        "verdict": verdict,
        "diagnostics_enabled": diagnostics_enabled,
        "output_files": sorted(p.name for p in run_dir.iterdir()),
    }
    _write_json(run_dir / "diagnostics.json", diagnostics)
    return run_dir


def build_model_review(
    *,
    metadata: dict[str, Any],
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    score_df: pd.DataFrame,
    full_outputs: dict[str, pd.DataFrame],
    optimization_comparison_df: pd.DataFrame,
    verdict: dict,
    candidate_name: str,
) -> dict[str, Any]:
    candidate_summary = _single_row(summary_df, "strategy", candidate_name)
    buy_hold_summary = _single_row(summary_df, "strategy", "buy_hold")
    candidate_raw = raw_df[raw_df.get("strategy_name") == candidate_name].copy()
    regime_raw = _group_raw_metrics(candidate_raw, "market_regime")
    symbol_raw = _group_raw_metrics(candidate_raw, "symbol")
    benchmark_summary = _group_benchmark_metrics(benchmark_df)
    comparison_row = _single_row(optimization_comparison_df, "candidate", candidate_name)

    score = _float_or_none(candidate_summary.get("score"))
    score_notes = _score_notes(
        candidate_summary=candidate_summary,
        comparison_row=comparison_row,
        benchmark_summary=benchmark_summary,
        regime_raw=regime_raw,
    )
    return {
        "metadata": {
            "run_id": metadata.get("run_id"),
            "candidate": candidate_name,
            "timestamp": metadata.get("timestamp"),
            "mode": metadata.get("mode"),
            "symbols": metadata.get("symbols"),
            "timeframe": metadata.get("timeframe"),
            "data_start": metadata.get("data_start"),
            "data_end": metadata.get("data_end"),
        },
        "decision": {
            "score": score,
            "recommendation": verdict.get("recommendation"),
            "pass_promotion_criteria": _bool_or_none(comparison_row.get("pass_promotion_criteria")),
            "review_result": _review_result(score_notes),
            "main_notes": score_notes,
        },
        "headline_metrics": {
            "candidate": _headline_metrics(candidate_summary),
            "buy_hold": _headline_metrics(buy_hold_summary),
            "drawdown_reduction_vs_buy_hold": _float_or_none(candidate_summary.get("drawdown_reduction")),
            "retention_ratio_vs_buy_hold": _float_or_none(candidate_summary.get("retention_ratio")),
        },
        "benchmarks": benchmark_summary,
        "by_regime": regime_raw,
        "by_symbol": symbol_raw,
        "optimization_comparison": _compact_comparison(comparison_row),
        "score_components": _score_components(score_df),
        "artifact_policy": {
            "start_here": ["model_review.md", "model_review.json", "summary_metrics.csv"],
            "deep_dive": [
                "strategy_optimization_comparison.csv",
                "regime_performance_report.csv",
                "benchmark_metrics.csv",
                "bull_underperformance_window_analysis.csv",
            ],
            "audit_only": [
                "raw_backtest_results.csv",
                "action_logs.csv.gz",
                "equity_curves.csv.gz",
                "diagnostics/",
            ],
        },
    }


def build_model_review_markdown(review: dict[str, Any]) -> str:
    meta = review["metadata"]
    decision = review["decision"]
    candidate = review["headline_metrics"]["candidate"]
    buy_hold = review["headline_metrics"]["buy_hold"]
    lines = [
        f"# Model Review: {meta.get('candidate')}",
        "",
        "## Decision",
        f"- Result: **{decision.get('review_result')}**",
        f"- Score: `{_fmt_number(decision.get('score'))}`",
        f"- Recommendation: `{decision.get('recommendation')}`",
        f"- Promotion criteria: `{decision.get('pass_promotion_criteria')}`",
        "",
        "## Headline",
        "| Metric | Candidate | Buy & Hold |",
        "|---|---:|---:|",
        f"| Mean return | {_fmt_pct(candidate.get('mean_return'))} | {_fmt_pct(buy_hold.get('mean_return'))} |",
        f"| Median return | {_fmt_pct(candidate.get('median_return'))} | {_fmt_pct(buy_hold.get('median_return'))} |",
        f"| Median excess | {_fmt_pct(candidate.get('median_excess_return'))} | {_fmt_pct(buy_hold.get('median_excess_return'))} |",
        f"| Win rate vs BH | {_fmt_pct(candidate.get('win_rate_vs_bh'))} | {_fmt_pct(buy_hold.get('win_rate_vs_bh'))} |",
        f"| Mean max drawdown | {_fmt_pct(candidate.get('mean_max_drawdown'))} | {_fmt_pct(buy_hold.get('mean_max_drawdown'))} |",
        f"| Avg exposure | {_fmt_pct(candidate.get('mean_exposure'))} | {_fmt_pct(buy_hold.get('mean_exposure'))} |",
        f"| Trades/window | {_fmt_number(candidate.get('mean_trade_count'))} | {_fmt_number(buy_hold.get('mean_trade_count'))} |",
        f"| Turnover | {_fmt_number(candidate.get('mean_turnover'))} | {_fmt_number(buy_hold.get('mean_turnover'))} |",
        "",
        "## Notes",
    ]
    for note in decision.get("main_notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Benchmarks", "| Benchmark | Mean excess | Median excess | Win rate | Avg exposure |", "|---|---:|---:|---:|---:|"])
    for name, row in review.get("benchmarks", {}).items():
        lines.append(
            f"| {name} | {_fmt_pct(row.get('mean_excess_return'))} | "
            f"{_fmt_pct(row.get('median_excess_return'))} | {_fmt_pct(row.get('win_rate'))} | "
            f"{_fmt_pct(row.get('avg_exposure'))} |"
        )
    lines.extend(["", "## Regime", "| Regime | Windows | Mean excess | Median excess | Win rate | Drawdown | Exposure |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in review.get("by_regime", {}).items():
        lines.append(
            f"| {name} | {row.get('windows')} | {_fmt_pct(row.get('mean_excess_return'))} | "
            f"{_fmt_pct(row.get('median_excess_return'))} | {_fmt_pct(row.get('win_rate_vs_bh'))} | "
            f"{_fmt_pct(row.get('mean_max_drawdown'))} | {_fmt_pct(row.get('mean_exposure'))} |"
        )
    lines.extend(["", "## Symbol", "| Symbol | Windows | Mean excess | Median excess | Win rate | Drawdown | Exposure |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in review.get("by_symbol", {}).items():
        lines.append(
            f"| {name} | {row.get('windows')} | {_fmt_pct(row.get('mean_excess_return'))} | "
            f"{_fmt_pct(row.get('median_excess_return'))} | {_fmt_pct(row.get('win_rate_vs_bh'))} | "
            f"{_fmt_pct(row.get('mean_max_drawdown'))} | {_fmt_pct(row.get('mean_exposure'))} |"
        )
    lines.extend([
        "",
        "## Read Order",
        "1. `model_review.md` for the decision.",
        "2. `strategy_optimization_comparison.csv` for promotion criteria.",
        "3. `regime_performance_report.csv` and `benchmark_metrics.csv` for attribution.",
        "4. Raw logs and diagnostics only when debugging a specific failure.",
        "",
    ])
    return "\n".join(lines)


def build_results_index_markdown() -> str:
    return "\n".join([
        "# Results Index",
        "",
        "## Primary Review",
        "- `model_review.md`: concise decision, headline metrics, regime/symbol tables.",
        "- `model_review.json`: machine-readable version of the same review.",
        "- `summary_metrics.csv`: top-level metrics for Buy & Hold and candidate.",
        "",
        "## Strategy Comparison",
        "- `strategy_optimization_comparison.csv`: promotion criteria and baseline comparison.",
        "- `strategy_optimization_summary.md`: short generated comparison summary.",
        "- `benchmark_metrics.csv`: Buy & Hold, exposure-matched, simple EMA168, and previous-best benchmarks.",
        "- `final_score_report.csv`: score inputs and hard-constraint checks.",
        "",
        "## Attribution",
        "- `regime_performance_report.csv`: BULL/MIXED/BEAR behavior.",
        "- `bull_underperformance_window_analysis.csv`: windows where BULL performance lags.",
        "- `signal_attribution_buy.csv` / `signal_attribution_sell.csv`: action reasons.",
        "- `risk_score_attribution_report.csv`: risk-score distribution.",
        "",
        "## Audit And Debug",
        "- `raw_backtest_results.csv`: one row per symbol/window.",
        "- `action_logs.csv.gz`: executed actions.",
        "- `equity_curves.csv.gz`: equity curve records.",
        "- `diagnostics/`: per-bar diagnostics and quality report.",
        "- `timestamp_audit_report.csv` / `accounting_audit_report.csv`: execution and accounting checks.",
        "",
        "## Stress",
        "- `cost_stress_report.csv`: fee/slippage stress.",
        "- `warmup_sensitivity_report.csv`: warmup sensitivity.",
        "- `parameter_sensitivity_report.csv`: configured perturbation placeholders.",
        "",
    ])


def _headline_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "score", "window_count", "mean_return", "median_return",
        "mean_excess_return", "median_excess_return", "win_rate_vs_bh",
        "mean_max_drawdown", "mean_sharpe", "mean_sortino", "mean_calmar",
        "mean_trade_count", "mean_exposure", "mean_turnover",
    ]
    return {key: _float_or_none(row.get(key)) for key in keys}


def _group_raw_metrics(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for name, group in df.groupby(group_col, dropna=False):
        rows[str(name)] = {
            "windows": int(len(group)),
            "mean_return": _float_or_none(group["total_return"].mean()),
            "median_return": _float_or_none(group["total_return"].median()),
            "mean_excess_return": _float_or_none(group["excess_return"].mean()),
            "median_excess_return": _float_or_none(group["excess_return"].median()),
            "win_rate_vs_bh": _float_or_none(group["win_vs_bh"].mean()),
            "mean_max_drawdown": _float_or_none(group["max_drawdown"].mean()),
            "mean_trade_count": _float_or_none(group["trade_count"].mean()),
            "mean_exposure": _float_or_none(group["avg_exposure"].mean()),
            "mean_turnover": _float_or_none(group["turnover"].mean()),
        }
    return rows


def _group_benchmark_metrics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or "benchmark" not in df.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for name, group in df.groupby("benchmark", dropna=False):
        rows[str(name)] = {
            "windows": int(len(group)),
            "benchmark_mean_return": _float_or_none(group["benchmark_return"].mean()),
            "benchmark_median_return": _float_or_none(group["benchmark_return"].median()),
            "strategy_mean_return": _float_or_none(group["strategy_return"].mean()),
            "mean_excess_return": _float_or_none(group["excess_return"].mean()),
            "median_excess_return": _float_or_none(group["excess_return"].median()),
            "win_rate": _float_or_none(group["win"].mean()),
            "avg_exposure": _float_or_none(group["avg_exposure"].mean()),
        }
    return rows


def _score_components(score_df: pd.DataFrame) -> dict[str, Any]:
    if score_df.empty:
        return {}
    return {
        str(row.get("component", row.get("metric", i))): {
            key: _float_or_none(value)
            for key, value in row.items()
            if key not in {"component", "metric"}
        }
        for i, row in score_df.iterrows()
    }


def _compact_comparison(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate", "score", "mean_return", "median_excess_vs_bh",
        "mean_excess_vs_bh", "bull_median_excess_vs_bh", "bull_win_vs_bh",
        "bear_median_excess_vs_bh", "max_drawdown", "avg_exposure",
        "turnover", "trade_count", "pass_promotion_criteria",
        "main_improvement", "main_regression", "recommendation",
    ]
    return {key: _json_scalar(row.get(key)) for key in keys if key in row}


def _score_notes(
    *,
    candidate_summary: dict[str, Any],
    comparison_row: dict[str, Any],
    benchmark_summary: dict[str, dict[str, Any]],
    regime_raw: dict[str, dict[str, Any]],
) -> list[str]:
    notes = []
    median_excess = _float_or_none(candidate_summary.get("median_excess_return"))
    max_dd = _float_or_none(candidate_summary.get("mean_max_drawdown"))
    trade_count = _float_or_none(candidate_summary.get("mean_trade_count"))
    if median_excess is not None:
        notes.append(f"Median excess vs Buy & Hold is {_fmt_pct(median_excess)}.")
    if max_dd is not None:
        notes.append(f"Mean max drawdown is {_fmt_pct(max_dd)}.")
    if trade_count is not None:
        notes.append(f"Average trade count is {_fmt_number(trade_count)} per window.")
    simple = benchmark_summary.get("simple_ema168_filter")
    if simple:
        notes.append(
            "Against simple EMA168 filter: "
            f"mean excess {_fmt_pct(simple.get('mean_excess_return'))}, "
            f"win rate {_fmt_pct(simple.get('win_rate'))}."
        )
    bull = regime_raw.get("bull")
    if bull:
        notes.append(
            "BULL regime: "
            f"median excess {_fmt_pct(bull.get('median_excess_return'))}, "
            f"win rate {_fmt_pct(bull.get('win_rate_vs_bh'))}."
        )
    regression = comparison_row.get("main_regression")
    if isinstance(regression, str) and regression:
        notes.append(regression)
    return notes


def _review_result(notes: list[str]) -> str:
    text = " ".join(notes).lower()
    if "promotion threshold not fully met" in text or "below 0.35" in text:
        return "do_not_promote"
    return "review_required"


def _single_row(df: pd.DataFrame, column: str, value: Any) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    match = df[df[column] == value]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _json_scalar(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)


def _fmt_number(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.4f}"


def _fmt_pct(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _effective_config(
    config: dict[str, Any],
    candidate_name: str,
    mode: str,
    diagnostics_enabled: bool,
) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(config))
    snapshot.setdefault("evaluation", {})
    snapshot["evaluation"]["mode"] = mode
    snapshot["evaluation"]["candidate_name"] = candidate_name
    snapshot["evaluation"]["diagnostics_enabled"] = diagnostics_enabled
    snapshot["evaluation"].setdefault("slippage_bps", 0.0)
    snapshot["evaluation"].setdefault("benchmarks", ["buy_hold", "exposure_matched_buy_hold"])
    if "simple_ema168_filter" not in snapshot["evaluation"]["benchmarks"]:
        snapshot["evaluation"]["benchmarks"].append("simple_ema168_filter")
    snapshot["evaluation"].setdefault("simple_ema168_filter", {"low_position_pct": 0.0})
    snapshot["evaluation"].setdefault("hard_constraints", {
        "min_avg_exposure": 0.55,
        "max_turnover": None,
        "max_drawdown_must_not_exceed_bh": True,
        "median_excess_return_vs_exposure_matched_bh_gt": 0.0,
        "stress_min_return": None,
    })
    snapshot["evaluation"].setdefault("parameter_perturbations", {
        "ema_windows": [[20, 60, 150], [24, 72, 168], [30, 90, 200]],
        "atr_percentile_threshold": [0.75, 0.80, 0.85],
        "donchian_upper_filter": [0.85, 0.90, 0.92, 0.95],
        "bull_confirm_bars": [2, 3, 4],
        "bear_confirm_bars": [1, 2, 3],
        "btc_regime_adjustment": [0.02, 0.03, 0.05],
    })
    snapshot["evaluation"].setdefault("diagnostics", {
        "enabled": diagnostics_enabled,
        "sell_too_early_thresholds": {"20d": 0.05, "60d": 0.10},
    })
    snapshot["evaluation"]["diagnostics"]["enabled"] = diagnostics_enabled
    return snapshot


def _raw_results_df(results: dict[str, StrategySummary]) -> pd.DataFrame:
    rows = []
    for summary in results.values():
        for perf in summary.perfs:
            for window in perf.windows:
                rows.append(asdict(window))
    return pd.DataFrame(rows)


def _summary_metrics_df(results: dict[str, StrategySummary], scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    bh_summary = results.get("buy_hold")
    for name, summary in results.items():
        rows.append({
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
    return pd.DataFrame(rows)


def _write_artifacts(run_dir: Path, actions_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    if actions_df.empty:
        actions_df = pd.DataFrame(columns=_action_log_columns())
    if equity_df.empty:
        equity_df = pd.DataFrame()
    actions_df.to_csv(run_dir / "action_logs" / "action_logs.csv.gz", index=False, compression="gzip")
    equity_df.to_csv(run_dir / "equity_curves" / "equity_curves.csv.gz", index=False, compression="gzip")
    # Compatibility copies for older local inspection scripts.
    actions_df.to_csv(run_dir / "action_logs.csv.gz", index=False, compression="gzip")
    equity_df.to_csv(run_dir / "equity_curves.csv.gz", index=False, compression="gzip")


def _build_metadata(
    *,
    runner,
    config: dict[str, Any],
    config_path: Path,
    candidate_name: str,
    mode: str,
    timestamp: str,
    run_id: str,
) -> dict[str, Any]:
    config_text = json.dumps(config, sort_keys=True, default=str)
    data_start, data_end = _data_range(runner)
    fee_rate = config.get("cost", {}).get("fee_rate", 0.0)
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": mode,
        "git_commit_hash": _git_commit_hash(),
        "strategy_name": candidate_name,
        "candidate_name": candidate_name,
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "symbols": config.get("symbols", []),
        "timeframe": config.get("timeframe"),
        "data_start": str(data_start) if data_start is not None else None,
        "data_end": str(data_end) if data_end is not None else None,
        "rolling_window_config": config.get("windows", []),
        "warmup_bars": config.get("warmup_bars"),
        "fee_rate": fee_rate,
        "slippage_bps": config.get("evaluation", {}).get("slippage_bps", 0.0),
        "execution_mode": config.get("execution", {}).get("mode", "next_open"),
        "initial_cash": config.get("capital", {}).get("initial"),
        "reserve": config.get("capital", {}).get("reserve"),
        "benchmark_list": config.get("evaluation", {}).get("benchmarks", []),
        "python_version": platform.python_version(),
        "code_version_notes": None,
    }


def build_benchmark_metrics(
    *,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    runner,
    candidate_name: str,
    mode: str,
) -> pd.DataFrame:
    candidate = raw_df[raw_df["strategy_name"] == candidate_name].copy()
    rows = []
    for _, row in candidate.iterrows():
        exposure_matched = row["avg_exposure"] * row["buy_hold_return"]
        rows.append({
            "strategy": candidate_name,
            "symbol": row["symbol"],
            "window_id": row["window_label"],
            "benchmark": "buy_hold",
            "benchmark_return": row["buy_hold_return"],
            "strategy_return": row["total_return"],
            "excess_return": row["total_return"] - row["buy_hold_return"],
            "win": bool(row["total_return"] >= row["buy_hold_return"]),
            "avg_exposure": row["avg_exposure"],
        })
        rows.append({
            "strategy": candidate_name,
            "symbol": row["symbol"],
            "window_id": row["window_label"],
            "benchmark": "exposure_matched_buy_hold",
            "benchmark_return": exposure_matched,
            "strategy_return": row["total_return"],
            "excess_return": row["total_return"] - exposure_matched,
            "win": bool(row["total_return"] >= exposure_matched),
            "avg_exposure": row["avg_exposure"],
        })

    rows.extend(_simple_ema168_benchmark_rows(runner, candidate_name, candidate))
    rows.extend(_previous_best_rows(runner.config, candidate_name))

    if not rows:
        return pd.DataFrame(columns=[
            "strategy", "symbol", "window_id", "benchmark", "benchmark_return",
            "strategy_return", "excess_return", "win", "avg_exposure",
        ])
    return pd.DataFrame(rows)


def _previous_best_rows(config: dict[str, Any], candidate_name: str) -> list[dict[str, Any]]:
    path = (
        config.get("evaluation", {}).get("previous_best_result_path")
        or config.get("previous_best_result_path")
    )
    if not path:
        return []
    summary_path = Path(path)
    if summary_path.is_dir():
        summary_path = summary_path / "summary_metrics.csv"
    if not summary_path.exists():
        return [{
            "strategy": candidate_name,
            "symbol": "all",
            "window_id": "all",
            "benchmark": "previous_best",
            "benchmark_return": np.nan,
            "strategy_return": np.nan,
            "excess_return": np.nan,
            "win": np.nan,
            "avg_exposure": np.nan,
        }]
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return []
    candidates = summary[summary.get("strategy", pd.Series(dtype=str)) != "buy_hold"]
    if candidates.empty:
        return []
    best = candidates.sort_values("score", ascending=False, na_position="last").iloc[0]
    return [{
        "strategy": candidate_name,
        "symbol": "all",
        "window_id": "all",
        "benchmark": f"previous_best:{best.get('strategy')}",
        "benchmark_return": best.get("mean_return"),
        "strategy_return": np.nan,
        "excess_return": np.nan,
        "win": np.nan,
        "avg_exposure": best.get("mean_exposure"),
    }]


def build_risk_metrics(equity_df: pd.DataFrame, raw_df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    rows = []
    if equity_df.empty:
        return pd.DataFrame(columns=_risk_columns())
    for key, group in _window_groups(equity_df):
        returns = _return_series(group)
        total_return = _total_return(group)
        rows.append({
            **key,
            "strategy": candidate_name,
            "total_return": total_return,
            "CAGR": _annual_return_from_group(total_return, group),
            "annualized_return": _annual_return_from_group(total_return, group),
            "Sharpe": calculate_sharpe(returns, 365),
            "Sortino": _sortino(returns, 365),
            "Calmar": _calmar(total_return, group),
            "max_drawdown": calculate_max_drawdown(returns),
            "VaR_95": returns.quantile(0.05),
            "VaR_99": returns.quantile(0.01),
            "CVaR_95": returns[returns <= returns.quantile(0.05)].mean(),
            "CVaR_99": returns[returns <= returns.quantile(0.01)].mean(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "worst_1d_return": returns.min(),
            "worst_7d_return": _worst_rolling_return(returns, 7),
            "worst_30d_return": _worst_rolling_return(returns, 30),
            "avg_exposure": _avg_exposure(group),
            "median_exposure": _median_exposure(group),
            "position_volatility": _position_pct(group).std(),
        })
    return pd.DataFrame(rows, columns=_risk_columns())


def build_active_management_metrics(equity_df: pd.DataFrame, runner, candidate_name: str) -> pd.DataFrame:
    rows = []
    if equity_df.empty:
        return pd.DataFrame(columns=_active_columns())
    for key, group in _window_groups(equity_df):
        bh_returns = _bh_returns_for_group(group, runner)
        strat_returns = _return_series(group).reset_index(drop=True)
        n = min(len(strat_returns), len(bh_returns))
        strat_returns = strat_returns.iloc[:n]
        bh_returns = bh_returns.iloc[:n]
        active = strat_returns - bh_returns
        tracking_error = active.std() * math.sqrt(365) if len(active) > 1 else np.nan
        beta = _beta(strat_returns, bh_returns)
        corr = strat_returns.corr(bh_returns) if len(strat_returns) > 1 else np.nan
        total_return = _total_return(group)
        bh_total_return = float((1 + bh_returns).prod() - 1) if len(bh_returns) else np.nan
        rows.append({
            **key,
            "strategy": candidate_name,
            "active_return_vs_bh": total_return - bh_total_return,
            "active_return_vs_exposure_matched_bh": total_return - _avg_exposure(group) * bh_total_return,
            "tracking_error_vs_bh": tracking_error,
            "information_ratio_vs_bh": ((active.mean() * 365) / tracking_error) if tracking_error and tracking_error > 0 else np.nan,
            "beta_to_bh": beta,
            "alpha_vs_bh": _annual_return_from_group(total_return, group) - beta * _annualized_from_returns(bh_returns),
            "correlation_to_bh": corr,
            "upside_capture": _capture(strat_returns, bh_returns, positive=True),
            "downside_capture": _capture(strat_returns, bh_returns, positive=False),
            "capture_ratio": _capture_ratio(strat_returns, bh_returns),
            "avg_trade_notional": np.nan,
            "avg_holding_period": np.nan,
            "fee_drag": _fee_drag(group),
            "cash_drag": _cash_drag(group),
            "position_volatility": _position_pct(group).std(),
        })
    return pd.DataFrame(rows, columns=_active_columns())


def build_drawdown_metrics(equity_df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    rows = []
    if equity_df.empty:
        return pd.DataFrame(columns=_drawdown_columns())
    for key, group in _window_groups(equity_df):
        dd = _drawdown_series(group)
        durations = _drawdown_durations(dd)
        rows.append({
            **key,
            "strategy": candidate_name,
            "max_drawdown": dd.min(),
            "max_drawdown_duration": max(durations) if durations else 0,
            "avg_drawdown_duration": float(np.mean(durations)) if durations else 0.0,
            "time_to_recovery": durations[-1] if durations and dd.iloc[-1] < 0 else 0,
            "ulcer_index": math.sqrt(float((dd.clip(upper=0) ** 2).mean())) if len(dd) else np.nan,
            "pain_index": float(abs(dd.clip(upper=0)).mean()) if len(dd) else np.nan,
            "top_5_drawdowns": ";".join(f"{x:.6f}" for x in dd.nsmallest(5).tolist()),
        })
    return pd.DataFrame(rows, columns=_drawdown_columns())


def build_timestamp_audit(actions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if actions_df.empty:
        return pd.DataFrame(columns=_timestamp_columns())
    for _, row in actions_df.iterrows():
        signal_ts = pd.to_datetime(row.get("signal_timestamp"), utc=True, errors="coerce")
        exec_ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        indicator_ts = pd.to_datetime(row.get("indicator_timestamp"), utc=True, errors="coerce")
        if pd.isna(indicator_ts):
            indicator_ts = signal_ts
        btc_ts = pd.to_datetime(row.get("btc_regime_timestamp"), utc=True, errors="coerce")
        errors = []
        if pd.isna(signal_ts) or pd.isna(exec_ts):
            errors.append("missing_signal_or_execution_timestamp")
        elif pd.isna(indicator_ts) or not (indicator_ts <= signal_ts < exec_ts):
            errors.append("indicator_signal_execution_order_failed")
        if pd.isna(btc_ts):
            errors.append("btc_regime_timestamp_missing")
        elif not (btc_ts <= signal_ts < exec_ts):
            errors.append("btc_regime_signal_execution_order_failed")
        rows.append({
            "symbol": row.get("symbol"),
            "window_id": row.get("window_label"),
            "action": row.get("side"),
            "reason": row.get("reason"),
            "buy_setup": _extract_setup(row.get("reason")) if row.get("side") == "buy" else "",
            "sell_reason": _extract_setup(row.get("reason")) if row.get("side") == "sell" else "",
            "signal_timestamp": row.get("signal_timestamp"),
            "execution_timestamp": row.get("timestamp"),
            "indicator_timestamp": str(indicator_ts) if not pd.isna(indicator_ts) else None,
            "btc_regime_timestamp": str(btc_ts) if not pd.isna(btc_ts) else None,
            "timestamp_check_pass": not any(e != "btc_regime_timestamp_missing" for e in errors),
            "timestamp_check_error": ";".join(errors),
        })
    return pd.DataFrame(rows, columns=_timestamp_columns())


def build_accounting_audit(equity_df: pd.DataFrame, actions_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    if equity_df.empty:
        return pd.DataFrame(columns=_accounting_columns())
    fee_rate = config.get("cost", {}).get("fee_rate", 0.0)
    reserve = config.get("capital", {}).get("reserve", 0.0)
    for key, group in _window_groups(equity_df):
        value_cols = [c for c in group.columns if c.endswith("_value") and c != "total_value"]
        qty_cols = [c for c in group.columns if c.endswith("_qty")]
        avg_cost_cols = [c for c in group.columns if c.endswith("_avg_cost")]
        reconstructed = group["cash"] + group[value_cols].fillna(0.0).sum(axis=1)
        _append_check(rows, key, "total_value_reconstruction", (reconstructed - group["total_value"]).abs(), 1e-8)
        _append_check(rows, key, "cash_non_negative", -group["cash"], 1e-8, less_equal=True)
        if qty_cols:
            _append_check(rows, key, "position_qty_non_negative", -group[qty_cols].min(axis=1), 1e-12, less_equal=True)
        position_pct = group[value_cols].sum(axis=1) / group["total_value"].replace(0, np.nan) if value_cols else pd.Series(dtype=float)
        if len(position_pct):
            _append_check(rows, key, "position_pct_below_cap", position_pct - 1.05, 1e-8, less_equal=True)
        _append_check(rows, key, "reserve_monitor", reserve - group["cash"], float("inf"), less_equal=True, notes="informational: cash below reserve is allowed by current engine")
        window_actions = _actions_for_key(actions_df, key)
        if not window_actions.empty:
            fee_error = (window_actions["fee"] - window_actions["notional"] * fee_rate).abs()
            _append_check(rows, key, "fee_matches_notional", fee_error, 1e-8)
        _append_check(rows, key, "final_equity_matches_last_row", pd.Series([0.0]), 1e-8)
        action_delta = abs(int(group["action_count"].sum()) - len(window_actions))
        _append_check(rows, key, "action_log_equity_alignment", pd.Series([float(action_delta)]), 0.0)
        if avg_cost_cols and qty_cols:
            reset_errors = []
            for qty_col, cost_col in zip(qty_cols, avg_cost_cols):
                reset_errors.append(((group[qty_col] <= 1e-12) & (group[cost_col].abs() > 1e-12)).sum())
            _append_check(rows, key, "avg_cost_reset_after_flat", pd.Series(reset_errors, dtype=float), 0.0)
        _append_check(rows, key, "partial_sell_avg_cost_policy", pd.Series([0.0]), 0.0, notes="engine keeps avg_cost unchanged on partial sells")
    return pd.DataFrame(rows, columns=_accounting_columns())


def build_signal_attribution(actions_df: pd.DataFrame, runner, side: str) -> pd.DataFrame:
    columns = _buy_attr_columns() if side == "buy" else _sell_attr_columns()
    if actions_df.empty:
        return pd.DataFrame(columns=columns)
    actions = actions_df[actions_df["side"] == side].copy()
    if actions.empty:
        return pd.DataFrame(columns=columns)
    actions["setup"] = actions["reason"].map(_extract_setup)
    rows = []
    for (symbol, window_id, setup), group in actions.groupby(["symbol", "window_label", "setup"], dropna=False):
        forward = [_forward_returns(row, runner, side=side) for _, row in group.iterrows()]
        fwd = pd.DataFrame(forward)
        if side == "buy":
            rows.append({
                "symbol": symbol,
                "window_id": window_id,
                "buy_setup": setup,
                "count": len(group),
                "avg_next_5d_return": fwd["next_5d_return"].mean(),
                "avg_next_10d_return": fwd["next_10d_return"].mean(),
                "avg_next_20d_return": fwd["next_20d_return"].mean(),
                "avg_next_60d_return": fwd["next_60d_return"].mean(),
                "hit_rate_20d": (fwd["next_20d_return"] > 0).mean(),
                "hit_rate_60d": (fwd["next_60d_return"] > 0).mean(),
                "max_adverse_excursion": fwd["mae"].mean(),
                "max_favorable_excursion": fwd["mfe"].mean(),
                "avg_position_added": group["notional"].mean(),
                "total_fee_cost": group["fee"].sum(),
            })
        else:
            rows.append({
                "symbol": symbol,
                "window_id": window_id,
                "sell_reason": setup,
                "count": len(group),
                "avg_next_5d_return_after_sell": fwd["next_5d_return"].mean(),
                "avg_next_10d_return_after_sell": fwd["next_10d_return"].mean(),
                "avg_next_20d_return_after_sell": fwd["next_20d_return"].mean(),
                "avg_next_60d_return_after_sell": fwd["next_60d_return"].mean(),
                "avoided_drawdown": -fwd["mae"].clip(upper=0).mean(),
                "missed_upside": fwd["mfe"].clip(lower=0).mean(),
                "sell_efficiency": -fwd["next_20d_return"].mean(),
                "total_fee_cost": group["fee"].sum(),
            })
    return pd.DataFrame(rows, columns=columns)


def build_state_transition_report(actions_df: pd.DataFrame, runner) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "raw_state_from", "raw_state_to",
        "confirmed_state_from", "confirmed_state_to", "count",
        "avg_next_20d_return", "avg_next_60d_return",
        "avg_next_20d_drawdown", "avg_next_60d_drawdown", "notes",
    ]
    if actions_df.empty:
        return pd.DataFrame(columns=columns)
    actions = actions_df.copy()
    actions["raw_state"] = actions["reason"].map(lambda x: _extract_reason_field(x, "raw"))
    actions["confirmed_state"] = actions["reason"].map(lambda x: _extract_reason_field(x, "conf"))
    rows = []
    for (symbol, window_id), group in actions.groupby(["symbol", "window_label"], dropna=False):
        group = group.sort_values("timestamp")
        prev_raw = group["raw_state"].shift(1)
        prev_conf = group["confirmed_state"].shift(1)
        tmp = group.assign(raw_from=prev_raw, conf_from=prev_conf).dropna(subset=["raw_from", "conf_from"])
        for keys, trans in tmp.groupby(["raw_from", "raw_state", "conf_from", "confirmed_state"], dropna=False):
            fwd = pd.DataFrame([_forward_returns(row, runner, side="buy") for _, row in trans.iterrows()])
            rows.append({
                "symbol": symbol,
                "window_id": window_id,
                "raw_state_from": keys[0],
                "raw_state_to": keys[1],
                "confirmed_state_from": keys[2],
                "confirmed_state_to": keys[3],
                "count": len(trans),
                "avg_next_20d_return": fwd["next_20d_return"].mean(),
                "avg_next_60d_return": fwd["next_60d_return"].mean(),
                "avg_next_20d_drawdown": fwd["mae_20d"].mean(),
                "avg_next_60d_drawdown": fwd["mae"].mean(),
                "notes": "action-level transition only; full per-bar state history is not yet persisted",
            })
    return pd.DataFrame(rows, columns=columns)


def build_regime_performance_report(raw_df: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    columns = [
        "group_type", "group_value", "strategy", "strategy_return", "buy_hold_return",
        "exposure_matched_bh_return", "excess_return_vs_bh",
        "excess_return_vs_exposure_matched_bh", "max_drawdown", "avg_exposure",
        "trade_count", "win_rate_vs_bh",
    ]
    candidate = raw_df[raw_df["strategy_name"] == candidate_name].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for regime, group in candidate.groupby("market_regime", dropna=False):
        embh = group["avg_exposure"] * group["buy_hold_return"]
        rows.append({
            "group_type": "market_regime",
            "group_value": regime,
            "strategy": candidate_name,
            "strategy_return": group["total_return"].mean(),
            "buy_hold_return": group["buy_hold_return"].mean(),
            "exposure_matched_bh_return": embh.mean(),
            "excess_return_vs_bh": group["excess_return"].mean(),
            "excess_return_vs_exposure_matched_bh": (group["total_return"] - embh).mean(),
            "max_drawdown": group["max_drawdown"].mean(),
            "avg_exposure": group["avg_exposure"].mean(),
            "trade_count": group["trade_count"].mean(),
            "win_rate_vs_bh": group["win_vs_bh"].mean(),
        })
    return pd.DataFrame(rows, columns=columns)


def build_bull_underperformance_window_analysis(
    *,
    raw_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    diagnostic_outputs: dict[str, pd.DataFrame],
    candidate_name: str,
) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "window_start", "window_end", "strategy_return",
        "buy_hold_return", "excess_return", "avg_actual_position_pct",
        "avg_target_position_pct_final", "target_actual_gap_mean",
        "target_reduce_count", "risk_reduce_count", "trend_break_count",
        "bull_guard_count", "strong_bull_suppression_count",
        "recovery_override_count", "target_reduce_sell_too_early_20d",
        "target_reduce_sell_too_early_60d", "risk_reduce_sell_too_early_20d",
        "risk_reduce_sell_too_early_60d", "blocked_by_cooldown_count",
        "blocked_by_trend_risk_count", "blocked_by_btc_bear_count",
        "blocked_by_volatility_count", "blocked_by_donchian_count",
        "avg_reentry_delay_after_sell", "max_reentry_delay_after_sell",
        "recommendation_tag",
    ]
    candidate = raw_df[
        (raw_df["strategy_name"] == candidate_name)
        & (raw_df["market_regime"] == "bull")
    ].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)

    actions = actions_df.copy()
    if not actions.empty:
        actions["setup"] = actions["reason"].map(_extract_setup)
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)

    exposure = diagnostic_outputs.get("exposure_diagnostics_report", pd.DataFrame())
    sell_early = diagnostic_outputs.get("sell_too_early_report", pd.DataFrame())
    summary = diagnostic_outputs.get("diagnostic_summary", pd.DataFrame())
    per_bar = diagnostic_outputs.get("per_bar_diagnostics", pd.DataFrame())

    rows = []
    for _, window in candidate.sort_values("excess_return").iterrows():
        symbol = window["symbol"]
        window_id = window["window_label"]
        window_actions = _window_action_subset(actions, symbol, window_id)
        window_exposure = _first_match(exposure, symbol, window_id)
        window_summary = _first_match(summary, symbol, window_id)
        window_sell_early = sell_early[
            (sell_early.get("symbol") == symbol) & (sell_early.get("window_id") == window_id)
        ] if not sell_early.empty else pd.DataFrame()
        window_per_bar = per_bar[
            (per_bar.get("symbol") == symbol)
            & (per_bar.get("window_id") == window_id)
            & (per_bar.get("is_trading_bar") == True)
        ] if not per_bar.empty else pd.DataFrame()

        target_early = _sell_early_row(window_sell_early, "target-reduce")
        risk_early = _sell_early_row(window_sell_early, "risk-reduce")
        avg_delay, max_delay = _reentry_delay_after_sell(window_per_bar)
        rows.append({
            "symbol": symbol,
            "window_id": window_id,
            "window_start": window["window_start"],
            "window_end": window["window_end"],
            "strategy_return": window["total_return"],
            "buy_hold_return": window["buy_hold_return"],
            "excess_return": window["excess_return"],
            "avg_actual_position_pct": window_exposure.get("avg_actual_position_pct", np.nan),
            "avg_target_position_pct_final": window_exposure.get("avg_target_position_pct_final", np.nan),
            "target_actual_gap_mean": window_exposure.get("target_actual_gap_mean", np.nan),
            "target_reduce_count": int(((window_actions.get("side") == "sell") & (window_actions.get("setup") == "target-reduce")).sum()) if not window_actions.empty else 0,
            "risk_reduce_count": int(((window_actions.get("side") == "sell") & (window_actions.get("setup") == "risk-reduce")).sum()) if not window_actions.empty else 0,
            "trend_break_count": int(window_actions.get("reason", pd.Series(dtype=str)).fillna("").str.contains("trend-break").sum()) if not window_actions.empty else 0,
            "bull_guard_count": int(window_actions.get("reason", pd.Series(dtype=str)).fillna("").str.contains("bull_guard").sum()) if not window_actions.empty else 0,
            "strong_bull_suppression_count": int(window_actions.get("reason", pd.Series(dtype=str)).fillna("").str.contains("strong_bull").sum()) if not window_actions.empty else 0,
            "recovery_override_count": int(window_actions.get("reason", pd.Series(dtype=str)).fillna("").str.contains("recovery_override").sum()) if not window_actions.empty else 0,
            "target_reduce_sell_too_early_20d": target_early.get("sell_too_early_rate_20d", np.nan),
            "target_reduce_sell_too_early_60d": target_early.get("sell_too_early_rate_60d", np.nan),
            "risk_reduce_sell_too_early_20d": risk_early.get("sell_too_early_rate_20d", np.nan),
            "risk_reduce_sell_too_early_60d": risk_early.get("sell_too_early_rate_60d", np.nan),
            "blocked_by_cooldown_count": window_summary.get("blocked_by_cooldown_count", np.nan),
            "blocked_by_trend_risk_count": window_summary.get("blocked_by_trend_risk_count", np.nan),
            "blocked_by_btc_bear_count": window_summary.get("blocked_by_btc_bear_count", np.nan),
            "blocked_by_volatility_count": window_summary.get("blocked_by_volatility_count", np.nan),
            "blocked_by_donchian_count": window_summary.get("blocked_by_donchian_count", np.nan),
            "avg_reentry_delay_after_sell": avg_delay,
            "max_reentry_delay_after_sell": max_delay,
            "recommendation_tag": _bull_underperformance_tag(
                window=window,
                exposure=window_exposure,
                summary=window_summary,
                target_early=target_early,
                risk_early=risk_early,
                avg_reentry_delay=avg_delay,
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def build_cost_stress_report(
    raw_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    config: dict[str, Any],
    candidate_name: str,
) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "scenario", "total_return", "CAGR", "Sharpe",
        "Sortino", "Calmar", "max_drawdown", "excess_return_vs_bh",
        "excess_return_vs_exposure_matched_bh", "win_vs_bh",
        "win_vs_exposure_matched_bh", "return_decay_vs_base", "fee_cost",
        "slippage_cost", "notes",
    ]
    scenarios = [
        ("base", 1.0, config.get("evaluation", {}).get("slippage_bps", 0.0)),
        ("realistic", 1.0, 5.0),
        ("conservative", 2.0, 10.0),
        ("stress", 3.0, 20.0),
    ]
    initial = config.get("capital", {}).get("initial", 100.0)
    base_fee = config.get("cost", {}).get("fee_rate", 0.0)
    candidate = raw_df[raw_df["strategy_name"] == candidate_name]
    rows = []
    for _, row in candidate.iterrows():
        window_actions = actions_df[
            (actions_df.get("symbol") == row["symbol"])
            & (actions_df.get("window_label") == row["window_label"])
        ] if not actions_df.empty else pd.DataFrame()
        notional = window_actions["notional"].sum() if not window_actions.empty else 0.0
        base_fee_cost = window_actions["fee"].sum() if not window_actions.empty else row.get("total_fee_cost", 0.0)
        for scenario, fee_mult, slippage_bps in scenarios:
            extra_fee = max(0.0, base_fee * fee_mult - base_fee) * notional
            slip_cost = notional * slippage_bps / 10_000
            decay = (extra_fee + slip_cost) / initial
            total_return = row["total_return"] - decay
            embh = row["avg_exposure"] * row["buy_hold_return"]
            rows.append({
                "symbol": row["symbol"],
                "window_id": row["window_label"],
                "scenario": scenario,
                "total_return": total_return,
                "CAGR": calculate_annual_return(total_return, 365, 365),
                "Sharpe": row.get("sharpe"),
                "Sortino": row.get("sortino"),
                "Calmar": total_return / abs(row["max_drawdown"]) if row["max_drawdown"] < 0 else np.nan,
                "max_drawdown": row["max_drawdown"],
                "excess_return_vs_bh": total_return - row["buy_hold_return"],
                "excess_return_vs_exposure_matched_bh": total_return - embh,
                "win_vs_bh": total_return >= row["buy_hold_return"],
                "win_vs_exposure_matched_bh": total_return >= embh,
                "return_decay_vs_base": decay,
                "fee_cost": base_fee_cost + extra_fee,
                "slippage_cost": slip_cost,
                "notes": "estimated from base trade notional without rerunning strategy",
            })
    return pd.DataFrame(rows, columns=columns)


def build_warmup_sensitivity_report(runner, candidate_name: str) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "warmup_bars", "total_return", "CAGR", "Sharpe",
        "Calmar", "max_drawdown", "avg_exposure", "turnover", "trade_count",
        "excess_return_vs_bh", "excess_return_vs_exposure_matched_bh", "status",
    ]
    original = runner.config.get("warmup_bars", 200)
    rows = []
    for warmup in [200, 300, 500, 700]:
        try:
            runner.config["warmup_bars"] = warmup
            results = runner.run_all(candidate_name)
            raw = _raw_results_df(results)
            cand = raw[raw["strategy_name"] == candidate_name]
            for _, row in cand.iterrows():
                embh = row["avg_exposure"] * row["buy_hold_return"]
                rows.append({
                    "symbol": row["symbol"],
                    "window_id": row["window_label"],
                    "warmup_bars": warmup,
                    "total_return": row["total_return"],
                    "CAGR": row["cagr"],
                    "Sharpe": row["sharpe"],
                    "Calmar": row["calmar"],
                    "max_drawdown": row["max_drawdown"],
                    "avg_exposure": row["avg_exposure"],
                    "turnover": row["turnover"],
                    "trade_count": row["trade_count"],
                    "excess_return_vs_bh": row["total_return"] - row["buy_hold_return"],
                    "excess_return_vs_exposure_matched_bh": row["total_return"] - embh,
                    "status": "applied",
                })
        except Exception as exc:  # pragma: no cover - defensive report path
            rows.append({
                "symbol": None,
                "window_id": None,
                "warmup_bars": warmup,
                "total_return": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "Calmar": np.nan,
                "max_drawdown": np.nan,
                "avg_exposure": np.nan,
                "turnover": np.nan,
                "trade_count": np.nan,
                "excess_return_vs_bh": np.nan,
                "excess_return_vs_exposure_matched_bh": np.nan,
                "status": f"failed: {exc}",
            })
    runner.config["warmup_bars"] = original
    return pd.DataFrame(rows, columns=columns)


def build_parameter_sensitivity_report(config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "parameter_name", "parameter_value", "symbol", "window_id", "total_return",
        "CAGR", "Sharpe", "Calmar", "max_drawdown", "excess_return_vs_bh",
        "excess_return_vs_exposure_matched_bh", "avg_exposure", "turnover",
        "trade_count", "status", "notes",
    ]
    grid = config.get("evaluation", {}).get("parameter_perturbations", {})
    rows = []
    for name, values in grid.items():
        for value in values:
            rows.append({
                "parameter_name": name,
                "parameter_value": json.dumps(value),
                "symbol": None,
                "window_id": None,
                "total_return": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "Calmar": np.nan,
                "max_drawdown": np.nan,
                "excess_return_vs_bh": np.nan,
                "excess_return_vs_exposure_matched_bh": np.nan,
                "avg_exposure": np.nan,
                "turnover": np.nan,
                "trade_count": np.nan,
                "status": "skipped",
                "notes": "framework only: strategy parameter injection is not wired in this evaluation upgrade",
            })
    return pd.DataFrame(rows, columns=columns)


def build_final_score_report(
    *,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    active_df: pd.DataFrame,
    candidate_name: str,
    mode: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    candidate = raw_df[raw_df["strategy_name"] == candidate_name].copy()
    if candidate.empty:
        return pd.DataFrame(columns=_score_columns())
    embh = candidate["avg_exposure"] * candidate["buy_hold_return"]
    hard = config.get("evaluation", {}).get("hard_constraints", {})
    fail_reasons = []
    if candidate["avg_exposure"].mean() < hard.get("min_avg_exposure", 0.55):
        fail_reasons.append("avg_exposure_below_min")
    max_turnover = hard.get("max_turnover")
    if max_turnover is not None and candidate["turnover"].mean() > max_turnover:
        fail_reasons.append("turnover_above_max")
    if hard.get("max_drawdown_must_not_exceed_bh", True):
        bh = raw_df[raw_df["strategy_name"] == "buy_hold"]
        if not bh.empty and candidate["max_drawdown"].mean() < bh["max_drawdown"].mean():
            fail_reasons.append("drawdown_worse_than_bh")
    if (candidate["total_return"] - embh).median() <= hard.get("median_excess_return_vs_exposure_matched_bh_gt", 0.0):
        fail_reasons.append("median_excess_vs_exposure_matched_bh_not_positive")

    median_excess_embh = (candidate["total_return"] - embh).median()
    median_excess_bh = candidate["excess_return"].median()
    bh = raw_df[raw_df["strategy_name"] == "buy_hold"]
    bh_mdd = bh["max_drawdown"].mean() if not bh.empty else -0.5
    dd_improve = (candidate["max_drawdown"].mean() - bh_mdd) / abs(bh_mdd) if bh_mdd < 0 else 0.0
    downside_capture = active_df["downside_capture"].mean() if not active_df.empty else np.nan
    calmar_sortino = np.nanmean([candidate["calmar"].mean(), candidate["sortino"].mean()])
    consistency = (candidate["excess_return"] > 0).mean()
    turnover_penalty = max(0.0, 1.0 - min(candidate["turnover"].mean(), 10.0) / 10.0)

    score = (
        0.20 * _sigmoid(median_excess_embh)
        + 0.15 * _sigmoid(median_excess_bh)
        + 0.15 * max(0.0, min(1.0, dd_improve))
        + 0.15 * (1.0 - min(max(downside_capture, 0.0), 2.0) / 2.0 if not math.isnan(downside_capture) else 0.5)
        + 0.10 * _bounded_metric(calmar_sortino, 3.0)
        + 0.10 * consistency
        + 0.05 * 0.5
        + 0.05 * 0.5
        + 0.05 * turnover_penalty
    )
    strengths = []
    weaknesses = []
    if median_excess_embh > 0:
        strengths.append("positive median excess vs exposure-matched BH")
    else:
        weaknesses.append("non-positive median excess vs exposure-matched BH")
    if candidate["turnover"].mean() < 6:
        strengths.append("moderate turnover")
    if candidate["max_drawdown"].mean() < -0.35:
        weaknesses.append("material drawdown remains")

    return pd.DataFrame([{
        "strategy": candidate_name,
        "candidate": candidate_name,
        "symbol_group": "all",
        "mode": mode,
        "pass_hard_constraints": len(fail_reasons) == 0,
        "final_score": round(score, 4),
        "rank": 1,
        "fail_reasons": ";".join(fail_reasons),
        "key_strengths": ";".join(strengths),
        "key_weaknesses": ";".join(weaknesses),
    }], columns=_score_columns())


def build_strategy_optimization_comparison(
    *,
    run_dir: Path,
    reference_dir: Path | None,
    candidate_name: str,
    mode: str,
) -> pd.DataFrame:
    if candidate_name == "v1_less_churn":
        return pd.DataFrame()
    rows = []
    for name, path in [("v1_less_churn", reference_dir), (candidate_name, run_dir)]:
        if path is None or not (path / "summary_metrics.csv").exists():
            continue
        rows.append(_optimization_row(name, path))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if {"v1_less_churn", candidate_name}.issubset(set(frame["candidate"])):
        base = frame[frame["candidate"] == "v1_less_churn"].iloc[0]
        cand = frame[frame["candidate"] == candidate_name].iloc[0]
        pass_criteria = (
            cand["bull_median_excess_vs_bh"] >= base["bull_median_excess_vs_bh"] + 0.10
            and cand["bull_win_vs_bh"] >= 0.35
            and cand["bull_underexposure_ratio"] <= base["bull_underexposure_ratio"]
            and cand["bear_median_excess_vs_bh"] >= base["bear_median_excess_vs_bh"] - 0.03
            and cand["max_drawdown"] >= base["max_drawdown"] - 0.03
            and cand["turnover"] <= base["turnover"] * 1.10
            and cand["stress_mean_excess_vs_bh"] > 0
        )
        frame.loc[frame["candidate"] == candidate_name, "pass_promotion_criteria"] = pass_criteria
        frame.loc[frame["candidate"] == candidate_name, "main_improvement"] = (
            f"score {base['score']:.4f}->{cand['score']:.4f}; "
            f"mean_excess_vs_bh {base['mean_excess_vs_bh']:.4f}->{cand['mean_excess_vs_bh']:.4f}; "
            f"bull_median_excess_vs_bh {base['bull_median_excess_vs_bh']:.4f}->{cand['bull_median_excess_vs_bh']:.4f}; "
            f"turnover {base['turnover']:.4f}->{cand['turnover']:.4f}"
        )
        regression_notes = []
        if cand["bull_win_vs_bh"] < 0.35:
            regression_notes.append(
                f"bull_win_vs_bh remains below 0.35 ({base['bull_win_vs_bh']:.4f}->{cand['bull_win_vs_bh']:.4f})"
            )
        else:
            regression_notes.append(
                f"bull_win_vs_bh passed 0.35 ({base['bull_win_vs_bh']:.4f}->{cand['bull_win_vs_bh']:.4f})"
            )
        regression_notes.append(
            f"max_drawdown changed {base['max_drawdown']:.4f}->{cand['max_drawdown']:.4f}"
        )
        if not pass_criteria:
            regression_notes.append("promotion threshold not fully met")
        frame.loc[frame["candidate"] == candidate_name, "main_regression"] = "; ".join(regression_notes)
        frame.loc[frame["candidate"] == candidate_name, "recommendation"] = (
            "promote" if pass_criteria else "do_not_promote_yet"
        )
    return frame


def build_strategy_optimization_summary(comparison: pd.DataFrame, candidate_name: str) -> str:
    if comparison.empty:
        return f"# {candidate_name} 策略优化总结\n\n没有可用对比数据。\n"
    candidate = comparison[comparison["candidate"] == candidate_name]
    base = comparison[comparison["candidate"] == "v1_less_churn"]
    lines = [f"# {candidate_name} 策略优化总结", ""]
    if not candidate.empty and not base.empty:
        c = candidate.iloc[0]
        b = base.iloc[0]
        lines.extend([
            "## 结论",
            f"{candidate_name} 的 score 从 `{b['score']:.4f}` 到 `{c['score']:.4f}`，mean excess vs BH 从 `{b['mean_excess_vs_bh']:.4f}` 到 `{c['mean_excess_vs_bh']:.4f}`。",
            f"bull median excess vs BH 从 `{b['bull_median_excess_vs_bh']:.4f}` 到 `{c['bull_median_excess_vs_bh']:.4f}`，bull win vs BH 从 `{b['bull_win_vs_bh']:.4f}` 到 `{c['bull_win_vs_bh']:.4f}`。",
            f"最大回撤从 `{b['max_drawdown']:.4f}` 到 `{c['max_drawdown']:.4f}`，换手从 `{b['turnover']:.4f}` 到 `{c['turnover']:.4f}`。",
            f"晋级判断：`{c['recommendation']}`。",
            "",
            "## 主要改善",
            str(c.get("main_improvement", "")),
            "",
            "## 剩余问题",
            str(c.get("main_regression", "")),
            "",
            "## 建议",
            "只有当牛市中位数超额、牛市胜率、熊市防守、回撤、换手和成本压力同时满足门槛时，才建议替代 v1_less_churn。",
        ])
    return "\n".join(lines) + "\n"


def _optimization_row(candidate: str, path: Path) -> dict[str, Any]:
    summary = pd.read_csv(path / "summary_metrics.csv")
    raw = pd.read_csv(path / "raw_backtest_results.csv")
    benchmark = pd.read_csv(path / "benchmark_metrics.csv") if (path / "benchmark_metrics.csv").exists() else pd.DataFrame()
    actions = pd.read_csv(path / "action_logs.csv.gz") if (path / "action_logs.csv.gz").exists() else pd.DataFrame()
    cost = pd.read_csv(path / "cost_stress_report.csv") if (path / "cost_stress_report.csv").exists() else pd.DataFrame()
    exposure = pd.read_csv(path / "exposure_diagnostics_report.csv") if (path / "exposure_diagnostics_report.csv").exists() else pd.DataFrame()
    row = summary[summary["strategy"] == candidate].iloc[0]
    cand = raw[raw["strategy_name"] == candidate]
    bull = cand[cand["market_regime"] == "bull"]
    bear = cand[cand["market_regime"] == "bear"]
    embh = benchmark[benchmark["benchmark"] == "exposure_matched_buy_hold"]
    actions["setup"] = actions["reason"].str.extract(r"_(?:buy|sell)_(.*?)_r\d+", expand=False) if not actions.empty else ""
    stress = cost[cost["scenario"] == "stress"] if not cost.empty else pd.DataFrame()
    return {
        "candidate": candidate,
        "score": row.get("score"),
        "mean_return": row.get("mean_return"),
        "median_excess_vs_bh": row.get("median_excess_return"),
        "mean_excess_vs_bh": row.get("mean_excess_return"),
        "mean_excess_vs_exposure_matched_bh": embh["excess_return"].mean() if not embh.empty else np.nan,
        "bull_median_excess_vs_bh": bull["excess_return"].median() if not bull.empty else np.nan,
        "bull_win_vs_bh": (bull["excess_return"] > 0).mean() if not bull.empty else np.nan,
        "bear_median_excess_vs_bh": bear["excess_return"].median() if not bear.empty else np.nan,
        "max_drawdown": row.get("mean_max_drawdown"),
        "avg_exposure": row.get("mean_exposure"),
        "bull_underexposure_ratio": exposure["bull_underexposure_ratio"].mean() if not exposure.empty else np.nan,
        "target_reduce_count": int(((actions.get("side") == "sell") & (actions.get("setup") == "target-reduce")).sum()) if not actions.empty else np.nan,
        "risk_reduce_count": int(((actions.get("side") == "sell") & (actions.get("setup") == "risk-reduce")).sum()) if not actions.empty else np.nan,
        "turnover": row.get("mean_turnover"),
        "trade_count": row.get("mean_trade_count"),
        "stress_mean_excess_vs_bh": stress["excess_return_vs_bh"].mean() if not stress.empty else np.nan,
        "pass_promotion_criteria": False,
        "main_improvement": "",
        "main_regression": "",
        "recommendation": "baseline" if candidate == "v1_less_churn" else "pending",
    }


def _find_latest_reference_run(output_root: Path, candidate_name: str) -> Path | None:
    root = output_root / "v1_eval_upgrade"
    if not root.exists():
        return None
    matches = [
        p for p in root.iterdir()
        if p.is_dir()
        and (p.name.endswith(f"_{candidate_name}") or f"_{candidate_name}_" in p.name)
        and (p / "summary_metrics.csv").exists()
    ]
    return sorted(matches, key=lambda p: p.name)[-1] if matches else None


def _simple_ema168_benchmark_rows(
    runner,
    candidate_name: str,
    candidate_raw: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    config = runner.config
    warmup = config.get("warmup_bars", 200)
    low_position = config.get("evaluation", {}).get("simple_ema168_filter", {}).get("low_position_pct", 0.0)
    fee_rate = config.get("cost", {}).get("fee_rate", 0.0)
    candidate_lookup = {
        (row["symbol"], row["window_label"]): row
        for _, row in candidate_raw.iterrows()
    }
    for symbol in config.get("symbols", []):
        df = runner.load_data(symbol).copy()
        if "ema168" not in df.columns:
            from .strategy_utils import compute_indicators
            df = compute_indicators(df)
        for wc in config.get("windows", []):
            i = 0
            while i + wc["days"] + warmup <= len(df):
                eval_start = i + warmup
                eval_end = i + wc["days"] + warmup
                window = df.iloc[eval_start:eval_end].copy().reset_index(drop=True)
                if window.empty:
                    i += wc["step_days"]
                    continue
                benchmark_return, avg_exposure = _simulate_simple_ema168_window(
                    df=df,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    low_position=low_position,
                    fee_rate=fee_rate,
                )
                window_id = f"{window['timestamp'].iloc[0].date()}~{window['timestamp'].iloc[-1].date()}"
                candidate_row = candidate_lookup.get((symbol, window_id))
                strategy_return = candidate_row["total_return"] if candidate_row is not None else np.nan
                rows.append({
                    "strategy": candidate_name,
                    "symbol": symbol,
                    "window_id": window_id,
                    "benchmark": "simple_ema168_filter",
                    "benchmark_return": benchmark_return,
                    "strategy_return": strategy_return,
                    "excess_return": strategy_return - benchmark_return if not pd.isna(strategy_return) else np.nan,
                    "win": bool(strategy_return >= benchmark_return) if not pd.isna(strategy_return) else np.nan,
                    "avg_exposure": avg_exposure,
                })
                i += wc["step_days"]
    return rows


def _simulate_simple_ema168_window(
    *,
    df: pd.DataFrame,
    eval_start: int,
    eval_end: int,
    low_position: float,
    fee_rate: float,
) -> tuple[float, float]:
    cash = 100.0
    qty = 0.0
    equity_values = []
    exposures = []
    for pos in range(eval_start, eval_end):
        signal_pos = pos - 1
        if signal_pos < 0:
            continue
        signal_row = df.iloc[signal_pos]
        row = df.iloc[pos]
        open_price = float(row["open"])
        close_price = float(row["close"])
        target_pct = 1.0 if signal_row["close"] > signal_row["ema168"] else low_position
        open_equity = cash + qty * open_price
        current_value = qty * open_price
        target_value = open_equity * target_pct
        diff = target_value - current_value
        if diff > 1e-10 and open_price > 0:
            buy_qty = min(cash / (open_price * (1 + fee_rate)), diff / (open_price * (1 + fee_rate)))
            cash -= buy_qty * open_price * (1 + fee_rate)
            qty += buy_qty
        elif diff < -1e-10 and open_price > 0:
            sell_qty = min(qty, -diff / open_price)
            cash += sell_qty * open_price * (1 - fee_rate)
            qty -= sell_qty
        close_equity = cash + qty * close_price
        equity_values.append(close_equity)
        exposures.append((qty * close_price / close_equity) if close_equity > 0 else 0.0)
    if not equity_values:
        return 0.0, 0.0
    return float(equity_values[-1] / 100.0 - 1.0), float(np.mean(exposures))


def _window_groups(df: pd.DataFrame):
    keys = ["symbol", "window_label"]
    for values, group in df.groupby(keys, dropna=False):
        yield {"symbol": values[0], "window_id": values[1]}, group.sort_values("timestamp").reset_index(drop=True)


def _return_series(group: pd.DataFrame) -> pd.Series:
    return group["total_value"].pct_change().fillna(group["total_value"].iloc[0] / 100.0 - 1.0)


def _total_return(group: pd.DataFrame) -> float:
    return float(group["total_value"].iloc[-1] / 100.0 - 1.0)


def _annual_return_from_group(total_return: float, group: pd.DataFrame) -> float:
    return calculate_annual_return(total_return, len(group), 365)


def _annualized_from_returns(returns: pd.Series) -> float:
    return calculate_annual_return(float((1 + returns).prod() - 1), len(returns), 365) if len(returns) else np.nan


def _sortino(returns: pd.Series, periods_per_year: int) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return np.nan
    denom = downside.std() * math.sqrt(periods_per_year)
    return returns.mean() * periods_per_year / denom if denom > 0 else np.nan


def _calmar(total_return: float, group: pd.DataFrame) -> float:
    annual = _annual_return_from_group(total_return, group)
    mdd = calculate_max_drawdown(_return_series(group))
    return annual / abs(mdd) if mdd < 0 else np.nan


def _worst_rolling_return(returns: pd.Series, days: int) -> float:
    if len(returns) < days:
        return np.nan
    return float((1 + returns).rolling(days).apply(np.prod, raw=True).min() - 1)


def _drawdown_series(group: pd.DataFrame) -> pd.Series:
    equity = group["total_value"].astype(float)
    return equity / equity.cummax() - 1


def _drawdown_durations(dd: pd.Series) -> list[int]:
    durations = []
    current = 0
    for value in dd:
        if value < 0:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return durations


def _avg_exposure(group: pd.DataFrame) -> float:
    pct = _position_pct(group)
    return float(pct.mean()) if len(pct) else 0.0


def _median_exposure(group: pd.DataFrame) -> float:
    pct = _position_pct(group)
    return float(pct.median()) if len(pct) else 0.0


def _position_pct(group: pd.DataFrame) -> pd.Series:
    value_cols = [c for c in group.columns if c.endswith("_value") and c != "total_value"]
    if not value_cols:
        return pd.Series(dtype=float)
    return group[value_cols].sum(axis=1) / group["total_value"].replace(0, np.nan)


def _fee_drag(group: pd.DataFrame) -> float:
    return float(group["cumulative_fees"].iloc[-1] / group["total_value"].iloc[-1]) if "cumulative_fees" in group else np.nan


def _cash_drag(group: pd.DataFrame) -> float:
    return float((group["cash"] / group["total_value"].replace(0, np.nan)).mean())


def _bh_returns_for_group(group: pd.DataFrame, runner) -> pd.Series:
    symbol = group["symbol"].iloc[0]
    df = runner.load_data(symbol)
    ts = pd.to_datetime(group["timestamp"], utc=True)
    candles = df.copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    merged = pd.DataFrame({"timestamp": ts}).merge(candles[["timestamp", "close"]], on="timestamp", how="left")
    return merged["close"].pct_change().fillna(0.0)


def _beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(strategy_returns) < 2 or benchmark_returns.var() == 0:
        return np.nan
    return float(strategy_returns.cov(benchmark_returns) / benchmark_returns.var())


def _capture(strategy_returns: pd.Series, benchmark_returns: pd.Series, positive: bool) -> float:
    mask = benchmark_returns > 0 if positive else benchmark_returns < 0
    if not mask.any():
        return np.nan
    bench = benchmark_returns[mask].sum()
    return float(strategy_returns[mask].sum() / bench) if bench != 0 else np.nan


def _capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    up = _capture(strategy_returns, benchmark_returns, True)
    down = _capture(strategy_returns, benchmark_returns, False)
    return up / abs(down) if down and not math.isnan(down) else np.nan


def _actions_for_key(actions_df: pd.DataFrame, key: dict[str, Any]) -> pd.DataFrame:
    if actions_df.empty:
        return actions_df
    return actions_df[(actions_df["symbol"] == key["symbol"]) & (actions_df["window_label"] == key["window_id"])]


def _window_action_subset(actions_df: pd.DataFrame, symbol: str, window_id: str) -> pd.DataFrame:
    if actions_df.empty:
        return actions_df
    return actions_df[
        (actions_df["symbol"] == symbol)
        & (actions_df["window_label"] == window_id)
    ].copy()


def _first_match(frame: pd.DataFrame, symbol: str, window_id: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    rows = frame[
        (frame.get("symbol") == symbol)
        & (frame.get("window_id") == window_id)
    ]
    return {} if rows.empty else rows.iloc[0].to_dict()


def _sell_early_row(frame: pd.DataFrame, reason: str) -> dict[str, Any]:
    if frame.empty or "sell_reason" not in frame:
        return {}
    rows = frame[frame["sell_reason"] == reason]
    return {} if rows.empty else rows.iloc[0].to_dict()


def _reentry_delay_after_sell(per_bar: pd.DataFrame) -> tuple[float, float]:
    if per_bar.empty or "action" not in per_bar:
        return np.nan, np.nan
    actions = per_bar[per_bar["action"].isin(["buy", "sell"])].sort_values("timestamp")
    delays = []
    for idx, row in actions[actions["action"] == "sell"].iterrows():
        later = actions[(actions.index > idx) & (actions["action"] == "buy")]
        if later.empty:
            continue
        days = (pd.to_datetime(later.iloc[0]["timestamp"], utc=True) - pd.to_datetime(row["timestamp"], utc=True)).days
        delays.append(days)
    return (float(np.mean(delays)), float(np.max(delays))) if delays else (np.nan, np.nan)


def _bull_underperformance_tag(
    *,
    window: pd.Series,
    exposure: dict[str, Any],
    summary: dict[str, Any],
    target_early: dict[str, Any],
    risk_early: dict[str, Any],
    avg_reentry_delay: float,
) -> str:
    if risk_early.get("sell_too_early_rate_60d", 0) >= 0.50:
        return "risk_reduce_sold_too_early"
    if target_early.get("sell_too_early_rate_60d", 0) >= 0.35:
        return "target_reduce_sold_too_early"
    if exposure.get("target_actual_gap_mean", 0) >= 0.10:
        return "underexposure"
    if summary.get("blocked_by_trend_risk_count", 0) + summary.get("blocked_by_btc_bear_count", 0) > 100:
        return "blocked_reentry"
    if window.get("symbol") == "BTC/USDT" and window.get("excess_return", 0) < 0:
        return "btc_specific_timing_issue"
    if not pd.isna(avg_reentry_delay) and avg_reentry_delay > 20:
        return "slow_reentry"
    return "not_actionable"


def _append_check(
    rows: list[dict[str, Any]],
    key: dict[str, Any],
    check_name: str,
    errors: pd.Series,
    tolerance: float,
    *,
    less_equal: bool = False,
    notes: str = "",
) -> None:
    errors = pd.to_numeric(errors, errors="coerce").fillna(0.0)
    if less_equal:
        failed = errors > tolerance
        max_abs = float(errors.clip(lower=0).max()) if len(errors) else 0.0
    else:
        failed = errors.abs() > tolerance
        max_abs = float(errors.abs().max()) if len(errors) else 0.0
    rows.append({
        **key,
        "check_name": check_name,
        "pass": bool(not failed.any()),
        "max_abs_error": max_abs,
        "error_count": int(failed.sum()),
        "example_rows": ",".join(map(str, errors[failed].head(5).index.tolist())),
        "notes": notes,
    })


def _forward_returns(row: pd.Series, runner, side: str) -> dict[str, float]:
    symbol = row.get("symbol")
    df = runner.load_data(symbol)
    candles = df.copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    pos = candles["timestamp"].searchsorted(ts, side="right")
    price = float(row.get("price", np.nan))
    out = {}
    closes = candles["close"].astype(float)
    for horizon in [5, 10, 20, 60]:
        idx = pos + horizon - 1
        out[f"next_{horizon}d_return"] = float(closes.iloc[idx] / price - 1) if idx < len(closes) and price > 0 else np.nan
    end_20 = min(pos + 20, len(closes))
    end_60 = min(pos + 60, len(closes))
    future_20 = closes.iloc[pos:end_20]
    future_60 = closes.iloc[pos:end_60]
    out["mae_20d"] = float(future_20.min() / price - 1) if len(future_20) and price > 0 else np.nan
    out["mae"] = float(future_60.min() / price - 1) if len(future_60) and price > 0 else np.nan
    out["mfe"] = float(future_60.max() / price - 1) if len(future_60) and price > 0 else np.nan
    return out


def _extract_setup(reason: str | None) -> str:
    if not reason:
        return ""
    match = re.search(r"_(?:buy|sell)_(.*?)_r\d+", reason)
    return match.group(1) if match else ""


def _extract_reason_field(reason: str | None, prefix: str) -> str:
    if not reason:
        return ""
    match = re.search(rf"_{prefix}([A-Z]+)", reason)
    return match.group(1) if match else ""


def _html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _data_range(runner) -> tuple[Any, Any]:
    starts = []
    ends = []
    for symbol in runner.config.get("symbols", []):
        df = runner.load_data(symbol)
        if not df.empty:
            starts.append(df["timestamp"].min())
            ends.append(df["timestamp"].max())
    return (min(starts) if starts else None, max(ends) if ends else None)


def _sigmoid(value: float) -> float:
    if math.isnan(value):
        return 0.5
    return 1.0 / (1.0 + math.exp(-3 * value))


def _bounded_metric(value: float, cap: float) -> float:
    if math.isnan(value) or cap <= 0:
        return 0.0
    return max(0.0, min(1.0, value / cap))


def _action_log_columns() -> list[str]:
    return [
        "strategy_name", "symbol", "window_name", "window_label", "window_start",
        "window_end", "timestamp", "signal_timestamp", "execution_mode", "side",
        "indicator_timestamp", "btc_regime_timestamp", "quantity", "price",
        "notional", "fee", "reason",
    ]


def _risk_columns() -> list[str]:
    return [
        "symbol", "window_id", "strategy", "total_return", "CAGR",
        "annualized_return", "Sharpe", "Sortino", "Calmar", "max_drawdown",
        "VaR_95", "VaR_99", "CVaR_95", "CVaR_99", "skewness", "kurtosis",
        "worst_1d_return", "worst_7d_return", "worst_30d_return",
        "avg_exposure", "median_exposure", "position_volatility",
    ]


def _active_columns() -> list[str]:
    return [
        "symbol", "window_id", "strategy", "active_return_vs_bh",
        "active_return_vs_exposure_matched_bh", "tracking_error_vs_bh",
        "information_ratio_vs_bh", "beta_to_bh", "alpha_vs_bh",
        "correlation_to_bh", "upside_capture", "downside_capture",
        "capture_ratio", "avg_trade_notional", "avg_holding_period",
        "fee_drag", "cash_drag", "position_volatility",
    ]


def _drawdown_columns() -> list[str]:
    return [
        "symbol", "window_id", "strategy", "max_drawdown",
        "max_drawdown_duration", "avg_drawdown_duration", "time_to_recovery",
        "ulcer_index", "pain_index", "top_5_drawdowns",
    ]


def _timestamp_columns() -> list[str]:
    return [
        "symbol", "window_id", "action", "reason", "buy_setup", "sell_reason",
        "signal_timestamp", "execution_timestamp", "indicator_timestamp",
        "btc_regime_timestamp", "timestamp_check_pass", "timestamp_check_error",
    ]


def _accounting_columns() -> list[str]:
    return [
        "symbol", "window_id", "check_name", "pass", "max_abs_error",
        "error_count", "example_rows", "notes",
    ]


def _buy_attr_columns() -> list[str]:
    return [
        "symbol", "window_id", "buy_setup", "count", "avg_next_5d_return",
        "avg_next_10d_return", "avg_next_20d_return", "avg_next_60d_return",
        "hit_rate_20d", "hit_rate_60d", "max_adverse_excursion",
        "max_favorable_excursion", "avg_position_added", "total_fee_cost",
    ]


def _sell_attr_columns() -> list[str]:
    return [
        "symbol", "window_id", "sell_reason", "count",
        "avg_next_5d_return_after_sell", "avg_next_10d_return_after_sell",
        "avg_next_20d_return_after_sell", "avg_next_60d_return_after_sell",
        "avoided_drawdown", "missed_upside", "sell_efficiency",
        "total_fee_cost",
    ]


def _score_columns() -> list[str]:
    return [
        "strategy", "candidate", "symbol_group", "mode",
        "pass_hard_constraints", "final_score", "rank", "fail_reasons",
        "key_strengths", "key_weaknesses",
    ]


# HTML/Markdown presentation is implemented in html_report.py.
