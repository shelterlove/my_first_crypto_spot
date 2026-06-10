#!/usr/bin/env python3
"""Run one lightweight recent-3-year candidate test from the DB data path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.backtest_engine import infer_periods_per_year  # noqa: E402
from crypto_spot_v1.backtest_event_driven import calculate_portfolio_performance, run_rebalance_backtest  # noqa: E402
from crypto_spot_v1.benchmark import V1BenchmarkRunner, build_strategy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="v2_28C")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--end", default="", help="Optional end timestamp/date. Defaults to latest common DB timestamp.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    data = runner._inject_btc_regime()
    if not data:
        raise SystemExit("No DB data loaded.")

    latest_common = min(df["timestamp"].max() for df in data.values() if not df.empty)
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else latest_common
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    start_ts = end_ts - pd.DateOffset(years=args.years) + pd.Timedelta(days=1)

    run_id = args.run_id or f"recent{args.years}y_{args.candidate}_vs_{args.baseline}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, actions, deltas = run_window(
        data=data,
        baseline=args.baseline,
        candidate=args.candidate,
        start_ts=start_ts,
        end_ts=end_ts,
        runner=runner,
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    actions.to_csv(output_dir / "actions.csv", index=False)
    deltas.to_csv(output_dir / "deltas.csv", index=False)
    report = render_report(args, output_dir, start_ts, end_ts, deltas)
    (output_dir / "recent3y_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_dir}")


def run_window(
    *,
    data: dict[str, pd.DataFrame],
    baseline: str,
    candidate: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    runner: V1BenchmarkRunner,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = runner.config
    capital = config["capital"]["initial"]
    reserve = config["capital"]["reserve"]
    fee = config["cost"]["fee_rate"]
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    ppy = infer_periods_per_year(config["timeframe"])
    strategies = [baseline, candidate]
    summary_rows: list[dict] = []
    action_rows: list[dict] = []

    for symbol, df in data.items():
        starts = df.index[df["timestamp"] >= start_ts].tolist()
        ends = df.index[df["timestamp"] <= end_ts].tolist()
        if not starts or not ends:
            continue
        eval_start = starts[0]
        eval_end = ends[-1] + 1
        backtest_start = max(0, eval_start - 1 if execution_mode != "same_close" else eval_start)
        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)

        for strategy_name in strategies:
            strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
            setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
            result = run_rebalance_backtest(
                {symbol: backtest_df},
                strategy,
                initial_capital=capital,
                reserve=reserve,
                fee_rate=fee,
                execution_mode=execution_mode,
            )
            full_actions = result.attrs.get("action_log")
            result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
            actions = pd.DataFrame() if full_actions is None else full_actions
            if not actions.empty:
                actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
                for row in actions.to_dict("records"):
                    action_rows.append({"symbol": symbol, "strategy": strategy_name, **row})

            perf = calculate_portfolio_performance(
                result,
                capital,
                ppy,
                candle_df=window_df,
                fee_rate=fee,
                benchmark_entry_col="open" if execution_mode == "next_open" else "close",
            )
            summary_rows.append({
                "strategy": strategy_name,
                "symbol": symbol,
                "start": str(start_ts),
                "end": str(end_ts),
                "total_return": float(perf["total_return"]),
                "max_drawdown": float(perf["max_drawdown"]),
                "trade_count": int(len(actions)),
                "final_equity": float(result["total_value"].iloc[-1]),
            })

    summary = pd.DataFrame(summary_rows)
    actions = pd.DataFrame(action_rows)
    deltas = compare(summary, baseline, candidate)
    return summary, actions, deltas


def compare(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    base = summary[summary["strategy"] == baseline]
    cand = summary[summary["strategy"] == candidate]
    merged = cand.merge(base, on="symbol", suffixes=("_candidate", "_baseline"))
    merged["total_return_delta"] = merged["total_return_candidate"] - merged["total_return_baseline"]
    merged["max_drawdown_delta"] = merged["max_drawdown_candidate"] - merged["max_drawdown_baseline"]
    merged["trade_count_delta"] = merged["trade_count_candidate"] - merged["trade_count_baseline"]
    return merged


def render_report(
    args: argparse.Namespace,
    output_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    deltas: pd.DataFrame,
) -> str:
    cols = [
        "symbol",
        "total_return_candidate",
        "total_return_baseline",
        "total_return_delta",
        "max_drawdown_candidate",
        "max_drawdown_baseline",
        "max_drawdown_delta",
        "trade_count_delta",
    ]
    return "\n".join([
        "# Recent 3Y Candidate Test",
        "",
        f"- Candidate: `{args.candidate}`",
        f"- Baseline: `{args.baseline}`",
        f"- Window: `{start_ts}` to `{end_ts}`",
        f"- Output: `{output_dir}`",
        "",
        "## Deltas",
        "",
        deltas[cols].to_markdown(index=False, floatfmt=".6f") if not deltas.empty else "No deltas.",
        "",
        "## Gate",
        "",
        f"- Negative return deltas: `{int((deltas['total_return_delta'] < -1e-9).sum()) if not deltas.empty else 0}`",
        f"- Worse drawdown deltas: `{int((deltas['max_drawdown_delta'] < -1e-9).sum()) if not deltas.empty else 0}`",
        f"- Return delta sum: `{float(deltas['total_return_delta'].sum()) if not deltas.empty else 0.0:.6f}`",
        "",
    ])


if __name__ == "__main__":
    main()
