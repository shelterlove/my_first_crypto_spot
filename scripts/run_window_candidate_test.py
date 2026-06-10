#!/usr/bin/env python3
"""Run a lightweight explicit-date candidate window test from the DB data path."""

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
from crypto_spot_v1.strategy_utils import compute_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="v3")
    parser.add_argument("--start", required=True, help="Window start date/timestamp.")
    parser.add_argument("--end", required=True, help="Window end date/timestamp.")
    parser.add_argument("--include-reference", default="", help="Optional extra reference strategy, e.g. v2_36C.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = _parse_utc(args.start)
    end_ts = _parse_utc(args.end)
    if end_ts < start_ts:
        raise SystemExit("--end must be >= --start")

    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    data = runner._inject_btc_regime()
    if not data:
        raise SystemExit("No DB data loaded.")

    run_id = args.run_id or (
        f"window_{args.candidate}_vs_{args.baseline}_"
        f"{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}"
    )
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = [args.baseline, args.candidate]
    if args.include_reference:
        strategies.append(args.include_reference)
    strategies = list(dict.fromkeys(strategies))

    summary, actions, deltas, trade_review = run_window(
        data=data,
        strategies=strategies,
        baseline=args.baseline,
        candidate=args.candidate,
        start_ts=start_ts,
        end_ts=end_ts,
        runner=runner,
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    actions.to_csv(output_dir / "actions.csv", index=False)
    deltas.to_csv(output_dir / "deltas.csv", index=False)
    trade_review.to_csv(output_dir / "trade_review.csv", index=False)
    report = render_report(args, output_dir, start_ts, end_ts, deltas, trade_review)
    (output_dir / "window_report.md").write_text(report, encoding="utf-8")
    (output_dir / "trade_review.md").write_text(render_trade_review_md(args, trade_review), encoding="utf-8")
    print(report)
    print(f"Wrote {output_dir}")


def run_window(
    *,
    data: dict[str, pd.DataFrame],
    strategies: list[str],
    baseline: str,
    candidate: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    runner: V1BenchmarkRunner,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = runner.config
    capital = config["capital"]["initial"]
    reserve = config["capital"]["reserve"]
    fee = config["cost"]["fee_rate"]
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    ppy = infer_periods_per_year(config["timeframe"])
    summary_rows: list[dict] = []
    action_rows: list[dict] = []
    symbol_windows: dict[str, pd.DataFrame] = {}

    for symbol in config["symbols"]:
        df = data.get(symbol)
        if df is None or df.empty:
            continue
        if "ema168" not in df.columns:
            df = compute_indicators(df)
        starts = df.index[df["timestamp"] >= start_ts].tolist()
        ends = df.index[df["timestamp"] <= end_ts].tolist()
        if not starts or not ends:
            continue
        eval_start = starts[0]
        eval_end = ends[-1] + 1
        backtest_start = max(0, eval_start - 1 if execution_mode != "same_close" else eval_start)
        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)
        symbol_windows[symbol] = window_df.copy()

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
            actions = pd.DataFrame() if full_actions is None else full_actions.copy()
            if not actions.empty:
                actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
                actions = enrich_actions(actions, window_df)
                for row in actions.to_dict("records"):
                    action_rows.append({"strategy": strategy_name, **row})

            result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
            result.attrs["action_log"] = actions
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
                "buy_hold_return": float(perf.get("bh_total_return", 0.0)),
                "excess_return": float(perf["total_return"] - perf.get("bh_total_return", 0.0)),
                "max_drawdown": float(perf["max_drawdown"]),
                "trade_count": int(len(actions)),
                "final_equity": float(result["total_value"].iloc[-1]),
                **action_stats(actions),
            })

    summary = pd.DataFrame(summary_rows)
    actions = pd.DataFrame(action_rows)
    deltas = compare(summary, baseline, candidate)
    trade_review = build_trade_review(
        actions=actions,
        candidate=candidate,
        baseline=baseline,
        symbol_windows=symbol_windows,
    )
    return summary, actions, deltas, trade_review


def action_stats(actions: pd.DataFrame) -> dict[str, int | float]:
    if actions.empty:
        return {
            "mixed_trade_pct": 0.0,
            "mixed_mixed_sell_pct": 0.0,
            "target_reduce_count": 0,
            "bull_buy_count": 0,
            "bull_sell_count": 0,
            "bear_mixed_breakout_buy_count": 0,
            "profit_take_count": 0,
            "constructive_mixed_sell_count": 0,
            "mature_giveback_sell_count": 0,
        }
    raw = actions.get("raw_state", pd.Series("", index=actions.index)).fillna("")
    conf = actions.get("confirmed_state", pd.Series("", index=actions.index)).fillna("")
    side = actions.get("side", pd.Series("", index=actions.index)).fillna("")
    setup = actions.get("setup", pd.Series("", index=actions.index)).fillna("")
    guards = actions.get("guards", pd.Series("", index=actions.index)).fillna("")

    mixed = (raw == "MIXED") | (conf == "MIXED")
    mixed_mixed_sell = (raw == "MIXED") & (conf == "MIXED") & (side == "sell")
    bull = (raw == "BULL") | (conf == "BULL")
    bear_mixed_breakout = (
        (side == "buy")
        & setup.isin(["target-gap", "safe-recovery"])
        & ((raw == "BEAR") | (conf == "BEAR") | (raw == "MIXED") | (conf == "MIXED"))
    )
    profit_take = setup.eq("light-profit-take") | guards.str.contains("profit_take", na=False)
    return {
        "mixed_trade_pct": float(mixed.mean()),
        "mixed_mixed_sell_pct": float(mixed_mixed_sell.mean()),
        "target_reduce_count": int(setup.eq("target-reduce").sum()),
        "bull_buy_count": int(((side == "buy") & bull).sum()),
        "bull_sell_count": int(((side == "sell") & bull).sum()),
        "bear_mixed_breakout_buy_count": int(bear_mixed_breakout.sum()),
        "profit_take_count": int(profit_take.sum()),
        "constructive_mixed_sell_count": int(((side == "sell") & actions.get("constructive_mixed_context", pd.Series(False, index=actions.index)).fillna(False)).sum()),
        "mature_giveback_sell_count": int(((side == "sell") & actions.get("mature_bull_giveback_context", pd.Series(False, index=actions.index)).fillna(False)).sum()),
    }


def compare(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    base = summary[summary["strategy"] == baseline]
    cand = summary[summary["strategy"] == candidate]
    merged = cand.merge(base, on="symbol", suffixes=("_candidate", "_baseline"))
    for col in [
        "total_return",
        "excess_return",
        "max_drawdown",
        "trade_count",
        "mixed_trade_pct",
        "mixed_mixed_sell_pct",
        "target_reduce_count",
        "bear_mixed_breakout_buy_count",
        "profit_take_count",
        "constructive_mixed_sell_count",
        "mature_giveback_sell_count",
    ]:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    return merged


def render_report(
    args: argparse.Namespace,
    output_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    deltas: pd.DataFrame,
    trade_review: pd.DataFrame,
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
        "mixed_trade_pct_delta",
        "mixed_mixed_sell_pct_delta",
        "target_reduce_count_delta",
        "profit_take_count_delta",
        "constructive_mixed_sell_count_delta",
        "mature_giveback_sell_count_delta",
    ]
    ret_sum = float(deltas["total_return_delta"].sum()) if not deltas.empty else 0.0
    worse_dd = int((deltas["max_drawdown_delta"] < -0.005).sum()) if not deltas.empty else 0
    worse_returns = int((deltas["total_return_delta"] < -1e-9).sum()) if not deltas.empty else 0
    changed = trade_review[trade_review["change_type"] != "unchanged"] if not trade_review.empty else pd.DataFrame()
    review_lines = summarize_trade_review(changed)
    return "\n".join([
        "# Explicit Window Candidate Test",
        "",
        f"- Candidate: `{args.candidate}`",
        f"- Baseline: `{args.baseline}`",
        f"- Reference: `{args.include_reference or ''}`",
        f"- Window: `{start_ts}` to `{end_ts}`",
        f"- Output: `{output_dir}`",
        "",
        "## Deltas",
        "",
        _table_text(deltas[cols], floatfmt=".6f") if not deltas.empty else "No deltas.",
        "",
        "## Gate",
        "",
        f"- Negative return deltas: `{worse_returns}`",
        f"- Drawdown worse beyond 0.5pp: `{worse_dd}`",
        f"- Return delta sum: `{ret_sum:.6f}`",
        "",
        "## Trade Review",
        "",
        *review_lines,
        "",
    ])


def build_trade_review(
    *,
    actions: pd.DataFrame,
    candidate: str,
    baseline: str,
    symbol_windows: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for symbol, window_df in symbol_windows.items():
        base = actions[(actions["strategy"] == baseline) & (actions["symbol"] == symbol)].copy()
        cand = actions[(actions["strategy"] == candidate) & (actions["symbol"] == symbol)].copy()
        base = enrich_actions(base, window_df)
        cand = enrich_actions(cand, window_df)
        used_base: set[int] = set()

        for cand_idx, cand_row in cand.iterrows():
            match_idx, change_type = match_trade(cand_row, base, used_base)
            matched = base.loc[match_idx] if match_idx is not None else None
            if match_idx is not None:
                used_base.add(int(match_idx))
            rows.append(build_review_row(
                symbol=symbol,
                source="candidate",
                strategy=candidate,
                row=cand_row,
                matched=matched,
                change_type=change_type,
            ))

        for base_idx, base_row in base.iterrows():
            if int(base_idx) in used_base:
                continue
            rows.append(build_review_row(
                symbol=symbol,
                source="baseline_only",
                strategy=baseline,
                row=base_row,
                matched=None,
                change_type="removed",
            ))

    review = pd.DataFrame(rows)
    if review.empty:
        return review
    review = review.sort_values(["symbol", "change_type", "timestamp"]).reset_index(drop=True)
    return review


def enrich_actions(actions: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions
    if "post_30d_return" in actions.columns and "constructive_mixed_context" in actions.columns:
        return actions
    enriched = actions.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
    merge_col = "indicator_timestamp" if "indicator_timestamp" in enriched.columns else "timestamp"
    enriched[merge_col] = pd.to_datetime(enriched[merge_col], utc=True)
    window = window_df.copy()
    window["timestamp"] = pd.to_datetime(window["timestamp"], utc=True)
    feature_cols = [
        "timestamp",
        "close",
        "ema24",
        "ema72",
        "ema168",
        "ema24_slope",
        "ema168_slope",
        "roc_20",
        "atr_pct_rank",
        "donchian_pos",
        "rolling_365d_pos",
        "btc_regime",
    ]
    available = [c for c in feature_cols if c in window.columns]
    enriched = enriched.merge(
        window[available].rename(columns={"timestamp": merge_col}),
        on=merge_col,
        how="left",
        suffixes=("", "_window"),
    )
    enriched["constructive_mixed_context"] = enriched.apply(classify_constructive_mixed_context, axis=1)
    enriched["mature_bull_giveback_context"] = enriched.apply(classify_mature_bull_giveback_context, axis=1)
    for horizon in [7, 14, 30, 60]:
        enriched[f"post_{horizon}d_return"] = enriched.apply(
            lambda row: forward_return(window, row["timestamp"], float(row["price"]), horizon),
            axis=1,
        )
    enriched["reverse_trade_within_30d"] = enriched.apply(
        lambda row: reverse_trade_within_days(enriched, row["timestamp"], str(row["side"]), 30),
        axis=1,
    )
    return enriched


def classify_constructive_mixed_context(row: pd.Series) -> bool:
    try:
        price = float(row.get("price", float("nan")))
        ema72 = _float_or_nan(row.get("ema72"))
        ema168 = _float_or_nan(row.get("ema168"))
        ema168_slope = _float_or_nan(row.get("ema168_slope"))
        rolling = _float_or_nan(row.get("rolling_365d_pos"))
        don = _float_or_nan(row.get("donchian_pos"))
        atr = _float_or_nan(row.get("atr_pct_rank"))
        risk = float(row.get("risk_score", float("nan")))
        trend = float(row.get("trend_risk", float("nan")))
        dd = float(row.get("drawdown_risk", float("nan")))
    except (TypeError, ValueError):
        return False
    if row.get("raw_state") != "MIXED" or row.get("confirmed_state") != "MIXED":
        return False
    if row.get("setup") != "target-reduce":
        return False
    if pd.isna(price) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
        return False
    if pd.isna(rolling) or pd.isna(don) or pd.isna(atr) or pd.isna(risk) or pd.isna(trend) or pd.isna(dd):
        return False
    return bool(
        risk <= 2
        and trend <= 2
        and dd == 0
        and ema72 > ema168
        and ema168_slope > 0
        and price > ema168 * 0.995
        and 0.62 <= rolling <= 0.80
        and 0.30 <= don <= 0.72
        and atr < 0.80
    )


def classify_mature_bull_giveback_context(row: pd.Series) -> bool:
    try:
        price = float(row.get("price", float("nan")))
        ema24 = _float_or_nan(row.get("ema24"))
        ema72 = _float_or_nan(row.get("ema72"))
        ema168 = _float_or_nan(row.get("ema168"))
        ema168_slope = _float_or_nan(row.get("ema168_slope"))
        rolling = _float_or_nan(row.get("rolling_365d_pos"))
        don = _float_or_nan(row.get("donchian_pos"))
        atr = _float_or_nan(row.get("atr_pct_rank"))
        roc20 = _float_or_nan(row.get("roc_20"))
        trend = float(row.get("trend_risk", float("nan")))
        dd = float(row.get("drawdown_risk", float("nan")))
    except (TypeError, ValueError):
        return False
    if row.get("raw_state") != "BULL" or row.get("confirmed_state") != "BULL":
        return False
    if pd.isna(price) or pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
        return False
    if pd.isna(rolling) or pd.isna(don) or pd.isna(atr) or pd.isna(roc20) or pd.isna(trend) or pd.isna(dd):
        return False
    return bool(
        trend <= 1
        and dd == 0
        and price < ema24
        and price > ema72
        and ema72 > ema168
        and ema168_slope > 0
        and 0.78 <= rolling <= 0.90
        and 0.70 <= don <= 0.83
        and atr >= 0.65
        and roc20 <= -0.06
    )


def forward_return(window_df: pd.DataFrame, timestamp: pd.Timestamp, price: float, horizon: int) -> float:
    if price <= 0:
        return float("nan")
    pos = window_df["timestamp"].searchsorted(timestamp, side="right")
    idx = pos + horizon - 1
    if idx >= len(window_df):
        return float("nan")
    future_price = float(window_df.iloc[idx]["close"])
    return future_price / price - 1.0


def reverse_trade_within_days(actions: pd.DataFrame, timestamp: pd.Timestamp, side: str, days: int) -> bool:
    end_ts = timestamp + pd.Timedelta(days=days)
    opposite = "sell" if side == "buy" else "buy"
    later = actions[(actions["timestamp"] > timestamp) & (actions["timestamp"] <= end_ts)]
    return bool((later["side"] == opposite).any())


def match_trade(row: pd.Series, base: pd.DataFrame, used_base: set[int]) -> tuple[int | None, str]:
    exact = base[
        (~base.index.isin(used_base))
        & (base["side"] == row["side"])
        & (base["setup"] == row["setup"])
        & (base["timestamp"] == row["timestamp"])
    ]
    if not exact.empty:
        return int(exact.index[0]), "unchanged"

    nearby = base[
        (~base.index.isin(used_base))
        & (base["side"] == row["side"])
        & (base["setup"] == row["setup"])
    ].copy()
    if nearby.empty:
        return None, "added"
    nearby["distance_days"] = (nearby["timestamp"] - row["timestamp"]).abs().dt.days
    nearby = nearby[nearby["distance_days"] <= 14].sort_values("distance_days")
    if nearby.empty:
        return None, "added"
    idx = int(nearby.index[0])
    change_type = "delayed" if row["timestamp"] > nearby.iloc[0]["timestamp"] else "advanced"
    return idx, change_type


def build_review_row(
    *,
    symbol: str,
    source: str,
    strategy: str,
    row: pd.Series,
    matched: pd.Series | None,
    change_type: str,
) -> dict:
    side = str(row.get("side", ""))
    post_30d = _float_or_nan(row.get("post_30d_return"))
    quality = post_30d if side == "buy" else (-post_30d if pd.notna(post_30d) else float("nan"))
    impact = classify_trade_impact(change_type, side, quality, bool(row.get("reverse_trade_within_30d", False)))
    out = {
        "symbol": symbol,
        "source": source,
        "strategy": strategy,
        "change_type": change_type,
        "impact_label": impact,
        "timestamp": row.get("timestamp"),
        "side": side,
        "setup": row.get("setup", ""),
        "price": _float_or_nan(row.get("price")),
        "guards": row.get("guards", ""),
        "raw_state": row.get("raw_state", ""),
        "confirmed_state": row.get("confirmed_state", ""),
        "risk_score": row.get("risk_score"),
        "trend_risk": row.get("trend_risk"),
        "drawdown_risk": row.get("drawdown_risk"),
        "rolling_365d_pos": _float_or_nan(row.get("rolling_365d_pos")),
        "donchian_pos": _float_or_nan(row.get("donchian_pos")),
        "price_vs_ema168": _ratio(_float_or_nan(row.get("price")), _float_or_nan(row.get("ema168"))),
        "post_7d_return": _float_or_nan(row.get("post_7d_return")),
        "post_14d_return": _float_or_nan(row.get("post_14d_return")),
        "post_30d_return": post_30d,
        "post_60d_return": _float_or_nan(row.get("post_60d_return")),
        "reverse_trade_within_30d": bool(row.get("reverse_trade_within_30d", False)),
        "matched_timestamp": matched.get("timestamp") if matched is not None else pd.NaT,
        "matched_price": _float_or_nan(matched.get("price")) if matched is not None else float("nan"),
        "matched_post_30d_return": _float_or_nan(matched.get("post_30d_return")) if matched is not None else float("nan"),
    }
    if matched is not None and pd.notna(out["matched_timestamp"]):
        out["timing_shift_days"] = int((pd.Timestamp(out["timestamp"]) - pd.Timestamp(out["matched_timestamp"])).days)
        out["quality_delta_30d"] = quality - (
            _float_or_nan(matched.get("post_30d_return"))
            if side == "buy"
            else -_float_or_nan(matched.get("post_30d_return"))
        )
    else:
        out["timing_shift_days"] = pd.NA
        out["quality_delta_30d"] = float("nan")
    return out


def classify_trade_impact(change_type: str, side: str, quality_30d: float, reverse_trade_within_30d: bool) -> str:
    if change_type == "unchanged":
        return "baseline-equivalent"
    if pd.isna(quality_30d):
        return "unknown"
    if change_type == "removed":
        return "removed_trade"
    if change_type in {"advanced", "delayed"}:
        if quality_30d >= 0.08:
            return "timing_helped"
        if quality_30d <= -0.08:
            return "timing_hurt"
        return "timing_neutral"
    if reverse_trade_within_30d and abs(quality_30d) < 0.08:
        return "churn_candidate"
    if side == "buy":
        return "good_added_buy" if quality_30d > 0 else "bad_added_buy"
    return "good_added_sell" if quality_30d > 0 else "bad_added_sell"


def summarize_trade_review(trade_review: pd.DataFrame) -> list[str]:
    if trade_review.empty:
        return ["No changed trades."]
    lines = []
    change_counts = trade_review["change_type"].value_counts().to_dict()
    impact_counts = trade_review["impact_label"].value_counts().head(6).to_dict()
    lines.append(f"- Changed trades: `{len(trade_review)}`")
    lines.append(f"- Change mix: `{change_counts}`")
    lines.append(f"- Impact mix: `{impact_counts}`")
    for symbol in sorted(trade_review["symbol"].unique()):
        part = trade_review[trade_review["symbol"] == symbol]
        best = part.sort_values("quality_delta_30d", ascending=False).head(1)
        worst = part.sort_values("quality_delta_30d", ascending=True).head(1)
        if not best.empty:
            b = best.iloc[0]
            lines.append(
                f"- {symbol} best changed trade: `{b['change_type']}` `{b['side']}/{b['setup']}` on `{pd.Timestamp(b['timestamp']).date()}` impact `{b['impact_label']}`"
            )
        if not worst.empty:
            w = worst.iloc[0]
            lines.append(
                f"- {symbol} worst changed trade: `{w['change_type']}` `{w['side']}/{w['setup']}` on `{pd.Timestamp(w['timestamp']).date()}` impact `{w['impact_label']}`"
            )
    return lines


def render_trade_review_md(args: argparse.Namespace, trade_review: pd.DataFrame) -> str:
    if trade_review.empty:
        return "# Trade Review\n\nNo changed trades."
    cols = [
        "symbol",
        "change_type",
        "impact_label",
        "timestamp",
        "side",
        "setup",
        "price",
        "timing_shift_days",
        "post_30d_return",
        "quality_delta_30d",
        "reverse_trade_within_30d",
        "guards",
    ]
    changed = trade_review[trade_review["change_type"] != "unchanged"].copy()
    top = changed.reindex(
        changed["quality_delta_30d"].abs().sort_values(ascending=False).index
    ).head(30)
    return "\n".join([
        "# Trade Review",
        "",
        f"- Candidate: `{args.candidate}`",
        f"- Baseline: `{args.baseline}`",
        "",
        _table_text(top[cols], floatfmt=".4f") if not top.empty else "No changed trades.",
        "",
    ])


def _float_or_nan(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    return float(value)


def _ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return float("nan")
    return num / den - 1.0


def _table_text(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        return df.to_csv(index=False)


def _parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


if __name__ == "__main__":
    main()
