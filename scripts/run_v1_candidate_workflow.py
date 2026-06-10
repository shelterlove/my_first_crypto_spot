#!/usr/bin/env python3
"""Run a native V1 candidate workflow against a registered baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.backtest_engine import infer_periods_per_year
from crypto_spot_v1.backtest_event_driven import calculate_portfolio_performance, run_rebalance_backtest
from crypto_spot_v1.benchmark import V1BenchmarkRunner, build_strategy
from crypto_spot_v1.evaluation import (
    RESEARCH_MODE,
    _raw_results_df,
    _summary_metrics_df,
    normalize_mode,
    save_evaluation_run,
)


SMOKE_WINDOWS = [
    ("strong_bull", "2019-02-25", "2021-02-24"),
    ("post_covid", "2020-03-21", "2021-03-21"),
    ("path_pollution", "2018-06-30", "2021-06-29"),
    ("bear_rally", "2022-08-01", "2022-12-31"),
    ("bear_defence", "2021-12-11", "2022-12-11"),
]

TARGETED_WINDOWS = [
    ("btc_2023_recovery", "2023-05-01", "2023-08-31"),
    ("eth_2024_recovery", "2024-04-01", "2024-07-31"),
    ("full_dev_tail", "2023-01-01", "2024-12-31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Registered V1 candidate strategy.")
    parser.add_argument("--baseline", default="v2_21E", help="Registered V1 baseline strategy.")
    parser.add_argument(
        "--stage",
        choices=["smoke", "research", "complete", "all"],
        default="smoke",
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "--step-multiplier",
        type=int,
        default=None,
        help="Rolling step multiplier for research/complete. Defaults match run_v1_backtest.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "v1_candidate_workflow"),
        help="Workflow output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_root = (
        Path(args.output_dir)
        / f"{timestamp}_{safe_name(args.candidate)}_vs_{safe_name(args.baseline)}_{args.stage}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        f"# V1 Candidate Workflow: `{args.candidate}` vs `{args.baseline}`",
        "",
        f"- Stage: `{args.stage}`",
        f"- Output: `{run_root}`",
        "",
    ]

    if args.stage in {"smoke", "all"}:
        smoke_dir = run_root / "smoke"
        smoke_result = run_smoke(args.baseline, args.candidate, smoke_dir)
        sections.extend(smoke_report(smoke_result, smoke_dir))

    if args.stage in {"research", "all"}:
        research_dir = run_root / "research"
        research_result = run_evaluation_pair(
            baseline=args.baseline,
            candidate=args.candidate,
            mode="research",
            output_root=research_dir,
            step_multiplier=args.step_multiplier,
        )
        sections.extend(evaluation_report("research", research_result, research_dir))

    if args.stage in {"complete", "all"}:
        complete_dir = run_root / "complete"
        complete_result = run_evaluation_pair(
            baseline=args.baseline,
            candidate=args.candidate,
            mode="complete",
            output_root=complete_dir,
            step_multiplier=args.step_multiplier,
        )
        sections.extend(evaluation_report("complete", complete_result, complete_dir))

    report = "\n".join(sections).rstrip() + "\n"
    (run_root / "workflow_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {run_root}")


def run_smoke(baseline: str, candidate: str, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    config = runner.config
    all_dfs = runner._inject_btc_regime()
    capital = config["capital"]["initial"]
    reserve = config["capital"]["reserve"]
    fee = config["cost"]["fee_rate"]
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    ppy = infer_periods_per_year(config["timeframe"])

    windows = [(f"smoke_{name}", start, end) for name, start, end in SMOKE_WINDOWS]
    windows.extend((f"targeted_{name}", start, end) for name, start, end in TARGETED_WINDOWS)
    strategies = [baseline, candidate]

    summary_rows: list[dict] = []
    action_rows: list[dict] = []
    for window_name, start, end in windows:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        for symbol, df in all_dfs.items():
            eval_matches = df.index[df["timestamp"] >= start_ts].tolist()
            end_matches = df.index[df["timestamp"] <= end_ts].tolist()
            if not eval_matches or not end_matches:
                continue
            eval_start = eval_matches[0]
            eval_end = end_matches[-1] + 1
            backtest_start = max(0, eval_start - 1 if execution_mode != "same_close" else eval_start)
            window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
            backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)
            bh_start = float(window_df["open"].iloc[0] if execution_mode == "next_open" else window_df["close"].iloc[0])
            bh_end = float(window_df["close"].iloc[-1])
            bh_return_pct = (bh_end / bh_start - 1.0) * 100 if bh_start else 0.0

            for strategy_name in strategies:
                strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
                setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
                result_df = run_rebalance_backtest(
                    {symbol: backtest_df},
                    strategy,
                    initial_capital=capital,
                    reserve=reserve,
                    fee_rate=fee,
                    execution_mode=execution_mode,
                )
                full_actions = result_df.attrs.get("action_log")
                result_df = result_df[result_df["timestamp"] >= start_ts].reset_index(drop=True)
                actions = pd.DataFrame() if full_actions is None else full_actions
                if not actions.empty:
                    actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
                    for row in actions.to_dict("records"):
                        action_rows.append({
                            "window": window_name,
                            "pair": symbol,
                            "strategy": strategy_name,
                            **row,
                        })
                perf = calculate_portfolio_performance(
                    result_df,
                    capital,
                    ppy,
                    candle_df=window_df,
                    fee_rate=fee,
                    benchmark_entry_col="open" if execution_mode == "next_open" else "close",
                )
                value_col = f"{symbol}_value"
                summary_rows.append({
                    "window": window_name,
                    "pair": symbol,
                    "strategy": strategy_name,
                    "start": start,
                    "end": end,
                    "final_equity": float(result_df["total_value"].iloc[-1]),
                    "total_return_pct": float(perf["total_return"] * 100),
                    "max_drawdown_pct": float(perf["max_drawdown"] * 100),
                    "bh_total_return_pct": bh_return_pct,
                    "trade_count": int(len(actions)),
                    "buy_count": int((actions["side"] == "buy").sum()) if not actions.empty else 0,
                    "sell_count": int((actions["side"] == "sell").sum()) if not actions.empty else 0,
                    "avg_exposure_pct": float(result_df[value_col].div(result_df["total_value"]).mean() * 100),
                    "total_fee_cost": float(actions["fee"].sum()) if not actions.empty and "fee" in actions else 0.0,
                })

    summary = pd.DataFrame(summary_rows)
    actions = pd.DataFrame(action_rows)
    deltas = compare_summary(summary, baseline, candidate)

    summary.to_csv(output_dir / "summary.csv", index=False)
    actions.to_csv(output_dir / "actions.csv", index=False)
    deltas.to_csv(output_dir / "deltas.csv", index=False)
    return {"summary": summary, "actions": actions, "deltas": deltas}


def run_evaluation_pair(
    *,
    baseline: str,
    candidate: str,
    mode: str,
    output_root: Path,
    step_multiplier: int | None,
) -> dict[str, Path | pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    base_dir = run_registered_evaluation(baseline, mode, output_root, step_multiplier)
    cand_dir = run_registered_evaluation(candidate, mode, output_root, step_multiplier)
    deltas = compare_raw_results(base_dir, cand_dir, baseline, candidate)
    deltas.to_csv(output_root / "candidate_vs_baseline_delta.csv", index=False)
    return {"baseline_dir": base_dir, "candidate_dir": cand_dir, "deltas": deltas}


def run_registered_evaluation(
    candidate_name: str,
    mode: str,
    output_root: Path,
    step_multiplier: int | None,
) -> Path:
    config_path = PROJECT_ROOT / "configs" / "backtest_v1.json"
    runner = V1BenchmarkRunner(config_path, output_dir=output_root)
    effective_mode = normalize_mode(mode)
    effective_step = step_multiplier
    if effective_step is None:
        effective_step = 2 if effective_mode == RESEARCH_MODE else 1
    runner.config.setdefault("evaluation", {})["window_step_multiplier"] = effective_step
    if effective_mode != RESEARCH_MODE:
        return run_minimal_complete_evaluation(
            runner=runner,
            candidate_name=candidate_name,
            config_path=config_path,
            output_root=output_root,
            step_multiplier=effective_step,
        )

    results = runner.run_all(
        candidate_name,
        collect_artifacts=False,
        window_step_multiplier=effective_step,
    )
    scores = runner.score_all(results)
    verdict = runner.check_promotion(results, candidate_name)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return save_evaluation_run(
        runner=runner,
        results=results,
        scores=scores,
        verdict=verdict,
        candidate_name=candidate_name,
        mode=effective_mode,
        timestamp=timestamp,
        config_path=config_path,
        output_root=output_root,
        diagnostics_enabled=(effective_mode != RESEARCH_MODE),
    )


def run_minimal_complete_evaluation(
    *,
    runner: V1BenchmarkRunner,
    candidate_name: str,
    config_path: Path,
    output_root: Path,
    step_multiplier: int,
) -> Path:
    results = runner.run_all(
        candidate_name,
        collect_artifacts=False,
        window_step_multiplier=step_multiplier,
    )
    scores = runner.score_all(results)
    verdict = runner.check_promotion(results, candidate_name)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "v1_eval_upgrade" / f"{timestamp}_{safe_name(candidate_name)}_complete_minimal"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _raw_results_df(results)
    summary_df = _summary_metrics_df(results, scores)
    raw_df.to_csv(run_dir / "raw_backtest_results.csv", index=False)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    (run_dir / "workflow_note.md").write_text(
        "\n".join([
            f"# Minimal Complete Evaluation: `{candidate_name}`",
            "",
            "This run uses the full native V1 rolling grid for complete-stage candidate comparison,",
            "but skips heavy action/equity/per-bar artifacts. Use `scripts/run_v1_backtest.py",
            "--mode complete` for a full audit after the candidate passes this workflow.",
            "",
            f"- Config: `{config_path}`",
            f"- Window step multiplier: `{step_multiplier}`",
            f"- Scores: `{scores}`",
            f"- Verdict: `{verdict}`",
            "",
        ]),
        encoding="utf-8",
    )
    return run_dir


def compare_summary(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    base = summary[summary["strategy"] == baseline].set_index(["window", "pair"])
    cand = summary[summary["strategy"] == candidate].set_index(["window", "pair"])
    rows = []
    for idx, crow in cand.iterrows():
        brow = base.loc[idx]
        rows.append({
            "window": idx[0],
            "pair": idx[1],
            "return_delta_pp": crow["total_return_pct"] - brow["total_return_pct"],
            "mdd_delta_pp": crow["max_drawdown_pct"] - brow["max_drawdown_pct"],
            "trade_delta": int(crow["trade_count"] - brow["trade_count"]),
            "exposure_delta_pp": crow["avg_exposure_pct"] - brow["avg_exposure_pct"],
            "base_return_pct": brow["total_return_pct"],
            "cand_return_pct": crow["total_return_pct"],
            "base_mdd_pct": brow["max_drawdown_pct"],
            "cand_mdd_pct": crow["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def compare_raw_results(base_dir: Path, cand_dir: Path, baseline: str, candidate: str) -> pd.DataFrame:
    base = pd.read_csv(base_dir / "raw_backtest_results.csv")
    cand = pd.read_csv(cand_dir / "raw_backtest_results.csv")
    base = base[base["strategy_name"] == baseline].copy()
    cand = cand[cand["strategy_name"] == candidate].copy()
    keys = ["symbol", "window_label", "window_start", "window_end"]
    merged = cand.merge(base, on=keys, suffixes=("_candidate", "_baseline"))
    for column in ("total_return", "max_drawdown", "excess_return", "trade_count", "avg_exposure", "turnover"):
        merged[f"{column}_delta"] = merged[f"{column}_candidate"] - merged[f"{column}_baseline"]
    return merged


def smoke_report(result: dict[str, pd.DataFrame], output_dir: Path) -> list[str]:
    deltas = result["deltas"]
    actions = result["actions"]
    trigger_count = 0
    if not actions.empty and "reason" in actions.columns:
        trigger_count = int(actions["reason"].fillna("").str.contains("recent_buy_target_reduce_deadband").sum())
    return [
        "## Smoke",
        "",
        f"- Output: `{output_dir}`",
        f"- Windows: `{len(deltas)}` pair windows",
        f"- Negative return deltas: `{int((deltas['return_delta_pp'] < -1e-9).sum())}`",
        f"- Worse drawdown deltas: `{int((deltas['mdd_delta_pp'] < -1e-9).sum())}`",
        f"- Deadband trigger actions: `{trigger_count}`",
        "",
    ]


def evaluation_report(stage: str, result: dict[str, Path | pd.DataFrame], output_dir: Path) -> list[str]:
    deltas = result["deltas"]
    assert isinstance(deltas, pd.DataFrame)
    changed = deltas[
        (deltas["total_return_delta"].abs() > 1e-12)
        | (deltas["max_drawdown_delta"].abs() > 1e-12)
        | (deltas["trade_count_delta"] != 0)
    ]
    return [
        f"## {stage.title()}",
        "",
        f"- Output: `{output_dir}`",
        f"- Baseline run: `{result['baseline_dir']}`",
        f"- Candidate run: `{result['candidate_dir']}`",
        f"- Windows: `{len(deltas)}`",
        f"- Changed windows: `{len(changed)}`",
        f"- Negative return deltas: `{int((deltas['total_return_delta'] < -1e-12).sum())}`",
        f"- Worse drawdown deltas: `{int((deltas['max_drawdown_delta'] < -1e-12).sum())}`",
        f"- Trade delta sum: `{int(deltas['trade_count_delta'].sum())}`",
        f"- Return delta sum: `{deltas['total_return_delta'].sum():.6f}`",
        "",
    ]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
