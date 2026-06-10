#!/usr/bin/env python3
"""Replay path-level counterfactuals on local feather candles.

This diagnostic exists because event-level forward returns were repeatedly
misleading: changing one buy/sell can alter later position state, trade timing,
and drawdown. The script compares registered strategies over full backtest
paths and reports the realized path delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from crypto_spot_v1.backtest_engine import infer_periods_per_year  # noqa: E402
from crypto_spot_v1.backtest_event_driven import (  # noqa: E402
    calculate_portfolio_performance,
    run_rebalance_backtest,
)
from crypto_spot_v1.benchmark import build_strategy  # noqa: E402
from crypto_spot_v1.rolling_windows import run_strategy_rolling  # noqa: E402


PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
SMOKE_WINDOWS = (
    ("strong_bull", "2019-02-25", "2021-02-24"),
    ("post_covid", "2020-03-21", "2021-03-21"),
    ("path_pollution", "2018-06-30", "2021-06-29"),
    ("bear_rally", "2022-08-01", "2022-12-31"),
    ("bear_defence", "2021-12-11", "2022-12-11"),
    ("btc_2023_recovery", "2023-05-01", "2023-08-31"),
    ("eth_2024_recovery", "2024-04-01", "2024-07-31"),
    ("full_dev_tail", "2023-01-01", "2024-12-31"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="v2_28C")
    parser.add_argument("--reference", default="v2_21E")
    parser.add_argument(
        "--candidates",
        default="v2_32A,v2_32B,v2_33A",
        help="Comma-separated registered strategies to replay against baseline.",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "complete", "all"),
        default="all",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "diagnostics"),
    )
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    strategies = [args.reference, args.baseline, *candidates]
    run_id = args.run_id or f"path_counterfactual_replay_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_all_candles(Path(args.data_dir))
    reports: list[str] = [
        "# Path Counterfactual Replay",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Reference: `{args.reference}`",
        f"- Candidates: `{', '.join(candidates)}`",
        f"- Stage: `{args.stage}`",
        f"- Output: `{output_dir}`",
        "",
    ]

    if args.stage in {"smoke", "all"}:
        smoke, actions = run_smoke(candles, strategies)
        smoke.to_csv(output_dir / "smoke_summary.csv", index=False)
        actions.to_csv(output_dir / "smoke_actions.csv", index=False)
        reports.extend(render_stage_report("smoke", smoke, actions, args.baseline, candidates, output_dir))

    if args.stage in {"complete", "all"}:
        complete = run_complete(candles, strategies)
        complete.to_csv(output_dir / "complete_raw.csv", index=False)
        reports.extend(render_stage_report("complete", complete, None, args.baseline, candidates, output_dir))

    report = "\n".join(reports).rstrip() + "\n"
    (output_dir / "path_counterfactual_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_dir}")


def load_all_candles(data_dir: Path) -> dict[str, pd.DataFrame]:
    all_dfs = {pair: load_pair(data_dir, pair) for pair in PAIRS}
    btc = all_dfs["BTC/USDT"].copy()
    btc["btc_regime"] = strategy_utils.compute_btc_regime(btc)
    btc["btc_price_vs_ema72"] = btc["close"] / btc["ema72"] - 1.0
    btc["btc_price_vs_ema168"] = btc["close"] / btc["ema168"] - 1.0
    btc_features = (
        btc.set_index("timestamp")[
            ["btc_regime", "btc_price_vs_ema72", "btc_price_vs_ema168", "ema24_slope", "ema168_slope"]
        ]
        .rename(columns={"ema24_slope": "btc_ema24_slope", "ema168_slope": "btc_ema168_slope"})
        .reset_index()
    )
    for pair, frame in all_dfs.items():
        merged = pd.merge_asof(
            frame.sort_values("timestamp"),
            btc_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged["btc_regime"] = merged["btc_regime"].ffill().fillna("RANGE")
        all_dfs[pair] = merged.reset_index(drop=True)
    return all_dfs


def load_pair(data_dir: Path, pair: str) -> pd.DataFrame:
    path = data_dir / f"{pair.replace('/', '_')}-1d.feather"
    frame = pd.read_feather(path)
    frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.drop(columns=["date"]).sort_values("timestamp").reset_index(drop=True)
    return strategy_utils.compute_indicators(frame)


def run_smoke(candles: dict[str, pd.DataFrame], strategies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    action_rows: list[dict] = []
    capital = 100_000.0
    reserve = 0.2
    fee = 0.001
    ppy = infer_periods_per_year("1d")

    for window_name, start, end in SMOKE_WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        for pair, frame in candles.items():
            start_matches = frame.index[frame["timestamp"] >= start_ts].tolist()
            end_matches = frame.index[frame["timestamp"] <= end_ts].tolist()
            if not start_matches or not end_matches:
                continue
            eval_start = start_matches[0]
            eval_end = end_matches[-1] + 1
            backtest_start = max(0, eval_start - 1)
            window_df = frame.iloc[eval_start:eval_end].reset_index(drop=True)
            backtest_df = frame.iloc[backtest_start:eval_end].reset_index(drop=True)
            for strategy_name in strategies:
                strategy = build_strategy(strategy_name, capital, reserve, fee)
                setattr(strategy, "TARGET_ALLOC", {pair: 1.0})
                result = run_rebalance_backtest(
                    {pair: backtest_df},
                    strategy,
                    initial_capital=capital,
                    reserve=reserve,
                    fee_rate=fee,
                    execution_mode="next_open",
                )
                full_action_log = result.attrs.get("action_log")
                result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
                actions = pd.DataFrame() if full_action_log is None else full_action_log
                if not actions.empty:
                    actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
                    for row in actions.to_dict("records"):
                        action_rows.append({"window_label": window_name, "symbol": pair, "strategy_name": strategy_name, **row})
                perf = calculate_portfolio_performance(
                    result,
                    capital,
                    ppy,
                    candle_df=window_df,
                    fee_rate=fee,
                    benchmark_entry_col="open",
                )
                rows.append({
                    "strategy_name": strategy_name,
                    "symbol": pair,
                    "window_name": "smoke",
                    "window_label": window_name,
                    "window_start": str(start_ts),
                    "window_end": str(end_ts),
                    "total_return": float(perf["total_return"]),
                    "max_drawdown": float(perf["max_drawdown"]),
                    "trade_count": int(len(actions)),
                })
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def run_complete(candles: dict[str, pd.DataFrame], strategies: list[str]) -> pd.DataFrame:
    config = json.loads((PROJECT_ROOT / "configs" / "backtest_v1.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    capital = config["capital"]["initial"]
    reserve = config["capital"]["reserve"]
    fee = config["cost"]["fee_rate"]
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    warmup_bars = config.get("warmup_bars", 200)

    for strategy_name in strategies:
        for pair, frame in candles.items():
            for window in config["windows"]:
                strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
                setattr(strategy, "TARGET_ALLOC", {pair: 1.0})
                metrics = run_strategy_rolling(
                    pair,
                    frame,
                    strategy,
                    strategy_name,
                    window_days=window["days"],
                    step_days=window["step_days"],
                    initial_capital=capital,
                    reserve=reserve,
                    fee_rate=fee,
                    timeframe=config["timeframe"],
                    warmup_bars=warmup_bars,
                    execution_mode=execution_mode,
                    artifact_sink=None,
                    collect_equity_curve=False,
                )
                for metric in metrics:
                    row = asdict(metric)
                    row["window_name"] = window["name"]
                    rows.append(row)
    return pd.DataFrame(rows)


def compare_results(frame: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    base = frame[frame["strategy_name"] == baseline].copy()
    cand = frame[frame["strategy_name"] == candidate].copy()
    keys = ["symbol", "window_name", "window_label", "window_start", "window_end"]
    merged = cand.merge(base, on=keys, suffixes=("_candidate", "_baseline"))
    merged["total_return_delta"] = merged["total_return_candidate"] - merged["total_return_baseline"]
    merged["max_drawdown_delta"] = merged["max_drawdown_candidate"] - merged["max_drawdown_baseline"]
    merged["trade_count_delta"] = merged["trade_count_candidate"] - merged["trade_count_baseline"]
    return merged


def summarize_delta(delta: pd.DataFrame) -> dict[str, float | int]:
    changed = (delta["total_return_delta"].abs() > 1e-12) | (delta["trade_count_delta"].abs() > 1e-12)
    return {
        "windows": int(len(delta)),
        "changed_windows": int(changed.sum()),
        "negative_return_deltas": int((delta["total_return_delta"] < -1e-9).sum()),
        "worse_drawdown_deltas": int((delta["max_drawdown_delta"] < -1e-9).sum()),
        "trade_delta_sum": int(delta["trade_count_delta"].sum()),
        "return_delta_sum": float(delta["total_return_delta"].sum()),
    }


def render_stage_report(
    stage: str,
    frame: pd.DataFrame,
    actions: pd.DataFrame | None,
    baseline: str,
    candidates: list[str],
    output_dir: Path,
) -> list[str]:
    lines = [f"## {stage.title()}", ""]
    for candidate in candidates:
        delta = compare_results(frame, baseline, candidate)
        delta_path = output_dir / f"{stage}_delta_{safe_name(candidate)}_vs_{safe_name(baseline)}.csv"
        delta.to_csv(delta_path, index=False)
        summary = summarize_delta(delta)
        lines.extend([
            f"### `{candidate}` vs `{baseline}`",
            "",
            f"- Windows: `{summary['windows']}`",
            f"- Changed windows: `{summary['changed_windows']}`",
            f"- Negative return deltas: `{summary['negative_return_deltas']}`",
            f"- Worse drawdown deltas: `{summary['worse_drawdown_deltas']}`",
            f"- Trade delta sum: `{summary['trade_delta_sum']}`",
            f"- Return delta sum: `{summary['return_delta_sum']:.6f}`",
            "",
        ])
        by_symbol = (
            delta.groupby("symbol")
            .agg(
                changed=("total_return_delta", lambda s: int((s.abs() > 1e-12).sum())),
                return_delta_sum=("total_return_delta", "sum"),
                negative_return_deltas=("total_return_delta", lambda s: int((s < -1e-9).sum())),
                positive_return_deltas=("total_return_delta", lambda s: int((s > 1e-9).sum())),
                worse_drawdown_deltas=("max_drawdown_delta", lambda s: int((s < -1e-9).sum())),
                trade_delta_sum=("trade_count_delta", "sum"),
            )
            .reset_index()
        )
        by_symbol.to_csv(output_dir / f"{stage}_by_symbol_{safe_name(candidate)}_vs_{safe_name(baseline)}.csv", index=False)
        if actions is not None and not actions.empty:
            diff = action_diff(actions, baseline, candidate)
            diff.to_csv(output_dir / f"{stage}_action_diff_{safe_name(candidate)}_vs_{safe_name(baseline)}.csv", index=False)
            lines.append(f"- Action diff rows: `{len(diff)}`")
            lines.append("")
    return lines


def action_diff(actions: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    keys = ["window_label", "symbol", "timestamp", "side"]
    base = actions[actions["strategy_name"] == baseline][keys + ["reason"]].rename(columns={"reason": "baseline_reason"})
    cand = actions[actions["strategy_name"] == candidate][keys + ["reason"]].rename(columns={"reason": "candidate_reason"})
    missing = base.merge(cand[keys], on=keys, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    missing["diff_type"] = "missing_in_candidate"
    extra = cand.merge(base[keys], on=keys, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns=["_merge"])
    extra["diff_type"] = "extra_in_candidate"
    return pd.concat([missing, extra], ignore_index=True, sort=False)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
