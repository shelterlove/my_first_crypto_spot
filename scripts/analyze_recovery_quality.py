#!/usr/bin/env python3
"""Classify recovery-buy quality and diagnose post-buy paths.

This is a diagnostic script only. It replays a native strategy path (or consumes
an existing buy_target_path_detail.csv) and labels recovery buy opportunities
without changing strategy trading rules.
"""

from __future__ import annotations

import argparse
import html
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from scripts.diagnose_buy_target_path import load_pair, run_pair_diagnostic  # noqa: E402


PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
WINDOWS = [
    ("strong_bull_recovery", "2019-02-25", "2021-02-24", True),
    ("post_covid_bull", "2020-03-21", "2021-03-21", True),
    ("path_pollution", "2018-06-30", "2021-06-29", False),
    ("bear_rally_counterexample", "2022-08-01", "2022-12-31", False),
    ("bear_defence_counterexample", "2021-12-11", "2022-12-11", False),
]

FEATURES = [
    "current_pct",
    "buy_target",
    "target_gap",
    "risk_score",
    "trend_risk",
    "drawdown_risk",
    "price_vs_ema24",
    "price_vs_ema72",
    "price_vs_ema168",
    "ema24_vs_ema72",
    "ema24_vs_ema168",
    "ema72_vs_ema168",
    "ema24_slope",
    "ema72_slope",
    "ema168_slope",
    "non_bear_days",
    "roc_10",
    "roc_20",
    "atr_pct_rank",
    "donchian_pos",
    "volume_strength",
    "days_since_trend_break",
    "days_since_bear",
    "btc_price_vs_ema24",
    "btc_price_vs_ema72",
    "btc_price_vs_ema168",
    "btc_ema24_slope",
    "btc_ema72_slope",
    "btc_ema168_slope",
    "btc_non_bear_days",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = load_or_build_detail(args)
    detail = enrich_detail(detail, Path(args.datadir))
    events = build_recovery_events(detail, args)
    feature_summary = summarize_features(events)
    rule_candidates = scan_rule_candidates(events, args)
    report = render_report(args, events, feature_summary, rule_candidates)

    events.to_csv(output_dir / "recovery_quality_events.csv", index=False)
    feature_summary.to_csv(output_dir / "recovery_quality_feature_summary.csv", index=False)
    rule_candidates.to_csv(output_dir / "recovery_quality_rule_candidates.csv", index=False)
    (output_dir / "recovery_quality_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovery_quality_report.html").write_text(markdown_to_html(report), encoding="utf-8")

    print("Recovery quality labels")
    summary = summarize_labels(events)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not summary.empty else "No recovery events")
    print("\nTop rule candidates")
    print(rule_candidates.head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}") if not rule_candidates.empty else "No candidate rules")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-name", default="v2_21E")
    parser.add_argument("--input-detail", default="", help="Optional buy_target_path_detail.csv or its parent directory.")
    parser.add_argument("--datadir", default=str(PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="recovery_quality_v2_21E_20260605")
    parser.add_argument("--timerange-start", default="2018-06-30")
    parser.add_argument("--timerange-end", default="2022-12-31")
    parser.add_argument("--lookahead-days", type=int, default=60)
    parser.add_argument("--startup-candles", type=int, default=220)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-notional", type=float, default=0.0)
    parser.add_argument("--reset-at-start", action="store_true")
    parser.add_argument("--candidate-gap", type=float, default=0.05)
    parser.add_argument("--candidate-max-position", type=float, default=0.35)
    parser.add_argument("--max-adverse-30d", type=float, default=-0.12)
    parser.add_argument("--max-adverse-60d", type=float, default=-0.18)
    return parser.parse_args()


def load_or_build_detail(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_detail:
        path = Path(args.input_detail)
        if path.is_dir():
            path = path / "buy_target_path_detail.csv"
        if not path.exists():
            raise SystemExit(f"Missing input detail: {path}")
        detail = pd.read_csv(path)
        detail["date"] = pd.to_datetime(detail["date"], utc=True)
        return detail.sort_values(["pair", "date"]).reset_index(drop=True)

    replay_args = Namespace(**vars(args))
    replay_args.timerange_end = (
        pd.Timestamp(args.timerange_end, tz="UTC") + pd.Timedelta(days=args.lookahead_days)
    ).strftime("%Y-%m-%d")
    rows = []
    btc_regime = load_btc_regime(Path(args.datadir))
    for pair in PAIRS:
        frame = load_pair(Path(args.datadir), pair)
        frame["btc_regime"] = btc_regime.reindex(frame.index).ffill().fillna("RANGE")
        rows.extend(run_pair_diagnostic(pair, frame, replay_args))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail["date"] = pd.to_datetime(detail["date"], utc=True)
    return detail.sort_values(["pair", "date"]).reset_index(drop=True)


def load_btc_regime(datadir: Path) -> pd.Series:
    btc = load_pair(datadir, "BTC/USDT")
    return strategy_utils.compute_btc_regime(btc)


def enrich_detail(detail: pd.DataFrame, datadir: Path) -> pd.DataFrame:
    if detail.empty:
        return detail
    out = detail.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["pair", "date"]).reset_index(drop=True)

    for col in ("price", "ema24", "ema72", "ema168"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["price_vs_ema24"] = safe_ratio(out["price"], out["ema24"]) - 1.0
    out["price_vs_ema72"] = safe_ratio(out["price"], out["ema72"]) - 1.0
    out["price_vs_ema168"] = safe_ratio(out["price"], out["ema168"]) - 1.0
    out["ema24_vs_ema72"] = safe_ratio(out["ema24"], out["ema72"]) - 1.0
    out["ema24_vs_ema168"] = safe_ratio(out["ema24"], out["ema168"]) - 1.0
    out["ema72_vs_ema168"] = safe_ratio(out["ema72"], out["ema168"]) - 1.0
    out["non_bear_days"] = out.groupby("pair", sort=False)["raw_state"].transform(lambda s: consecutive_true(s != "BEAR"))

    trend_break = (out["action"].eq("sell") & out["action_reason"].fillna("").str.contains("trend_break", case=False))
    out["days_since_trend_break"] = days_since_mask(out, trend_break)
    out["days_since_bear"] = days_since_mask(out, out["raw_state"].eq("BEAR"))

    btc = load_pair(datadir, "BTC/USDT").copy()
    btc["raw_state"] = btc.apply(strategy_utils.detect_market_state, axis=1)
    btc["btc_non_bear_days"] = consecutive_true(btc["raw_state"] != "BEAR")
    btc_features = pd.DataFrame({
        "date": pd.Index(btc.index),
        "btc_close": btc["close"],
        "btc_ema24": btc["ema24"],
        "btc_ema72": btc["ema72"],
        "btc_ema168": btc["ema168"],
        "btc_ema24_slope": btc["ema24_slope"],
        "btc_ema72_slope": btc["ema72_slope"],
        "btc_ema168_slope": btc["ema168_slope"],
        "btc_raw_state": btc["raw_state"],
        "btc_non_bear_days": btc["btc_non_bear_days"],
    }).reset_index(drop=True)
    out = out.merge(btc_features, on="date", how="left")
    out["btc_price_vs_ema24"] = safe_ratio(out["btc_close"], out["btc_ema24"]) - 1.0
    out["btc_price_vs_ema72"] = safe_ratio(out["btc_close"], out["btc_ema72"]) - 1.0
    out["btc_price_vs_ema168"] = safe_ratio(out["btc_close"], out["btc_ema168"]) - 1.0

    for horizon in (5, 10, 20, 30, 60):
        out[f"future_ret_{horizon}d"] = out.groupby("pair", sort=False)["price"].shift(-horizon) / out["price"] - 1.0
        out[f"future_down_{horizon}d"] = out.groupby("pair", sort=False)["price"].transform(lambda s: forward_min_return(s, horizon))
    out["window_labels"] = out["date"].map(labels_for_date)
    return out


def build_recovery_events(detail: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for pair, group in detail.groupby("pair", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        for idx, row in group.iterrows():
            source = classify_event_source(row, args)
            if not source:
                continue
            event = row.to_dict()
            event["event_source"] = source
            event.update(post_buy_path(group, idx))
            event["quality_label"] = quality_label(event, args)
            event["failure_modes"] = failure_modes(event)
            rows.append(event)
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows)
    events = events[events["window_labels"].astype(str).ne("")].copy()
    return events.sort_values(["pair", "date", "event_source"]).reset_index(drop=True)


def classify_event_source(row: pd.Series, args: argparse.Namespace) -> str:
    reason = str(row.get("action_reason", ""))
    guard = str(row.get("buy_guard", ""))
    action = str(row.get("action", ""))
    current_pct = float(row.get("current_pct", 0.0) or 0.0)
    target_gap = float(row.get("target_gap", 0.0) or 0.0)
    if action == "buy" and "safe-recovery" in reason:
        return "safe-recovery-buy"
    if action == "buy" and "target-gap" in reason and current_pct < args.candidate_max_position:
        return "target-gap-low-position-buy"
    blocked_candidate = (
        action != "buy"
        and current_pct < args.candidate_max_position
        and target_gap >= args.candidate_gap
        and (
            "tiny_buy" in guard
            or "post-override" in guard
            or "post_override" in guard
            or "cooldown" in guard
            or bool(row.get("cooldown_blocked", False))
        )
    )
    if blocked_candidate:
        return "blocked-gap-candidate"
    return ""


def post_buy_path(group: pd.DataFrame, idx: int) -> dict:
    row = group.iloc[idx]
    out = {}
    future = group.iloc[idx + 1: idx + 61].copy()
    actions = future[future["action"].isin(["buy", "sell"])]
    buys = future[future["action"].eq("buy")]
    sells = future[future["action"].eq("sell")]
    reason = future["action_reason"].fillna("").str.lower()

    for horizon in (5, 10, 20, 30, 60):
        segment = group.iloc[idx + 1: idx + horizon + 1]
        prefix = f"post_{horizon}d"
        reason_h = segment["action_reason"].fillna("").str.lower()
        out[f"{prefix}_target_reduce"] = bool(reason_h.str.contains("target_reduce").any())
        out[f"{prefix}_risk_reduce"] = bool(reason_h.str.contains("risk_reduce").any())
        out[f"{prefix}_trend_break"] = bool(reason_h.str.contains("trend_break").any())
        out[f"{prefix}_buy_count"] = int(segment["action"].eq("buy").sum())
        out[f"{prefix}_sell_count"] = int(segment["action"].eq("sell").sum())
        out[f"{prefix}_max_current_pct"] = float(segment["current_pct"].max()) if not segment.empty else float("nan")
        out[f"{prefix}_mean_target_gap"] = float(segment["target_gap"].mean()) if not segment.empty else float("nan")

    out["post_any_target_reduce_60d"] = bool(reason.str.contains("target_reduce").any())
    out["post_any_risk_reduce_60d"] = bool(reason.str.contains("risk_reduce").any())
    out["post_any_trend_break_60d"] = bool(reason.str.contains("trend_break").any())
    out["days_to_next_buy"] = days_to_first(row["date"], buys)
    out["days_to_next_sell"] = days_to_first(row["date"], sells)
    out["days_to_45pct"] = days_to_threshold(row["date"], future, 0.45)
    out["days_to_60pct"] = days_to_threshold(row["date"], future, 0.60)
    out["days_to_75pct"] = days_to_threshold(row["date"], future, 0.75)
    out["effective_lift_45pct_60d"] = pd.notna(out["days_to_45pct"])
    out["effective_lift_60pct_60d"] = pd.notna(out["days_to_60pct"])
    out["effective_lift_75pct_60d"] = pd.notna(out["days_to_75pct"])
    out["consecutive_small_buy_count_30d"] = consecutive_small_buys(group, idx, horizon=30)
    return out


def quality_label(row: dict, args: argparse.Namespace) -> str:
    ret30 = row.get("future_ret_30d")
    ret60 = row.get("future_ret_60d")
    down30 = row.get("future_down_30d")
    down60 = row.get("future_down_60d")
    trend_break = bool(row.get("post_any_trend_break_60d"))
    lifted = bool(row.get("effective_lift_45pct_60d") or row.get("effective_lift_60pct_60d"))

    low = (
        (pd.notna(ret30) and ret30 < 0)
        or (pd.notna(ret60) and ret60 < 0)
        or (pd.notna(down30) and down30 <= args.max_adverse_30d)
        or (pd.notna(down60) and down60 <= args.max_adverse_60d)
        or trend_break
    )
    high = (
        pd.notna(ret30)
        and pd.notna(ret60)
        and ret30 > 0
        and ret60 > 0
        and (pd.isna(down30) or down30 > args.max_adverse_30d)
        and (pd.isna(down60) or down60 > args.max_adverse_60d)
        and not trend_break
        and lifted
    )
    if high:
        return "HIGH_QUALITY"
    if low:
        return "LOW_QUALITY"
    return "NEUTRAL"


def failure_modes(row: dict) -> str:
    modes = []
    if row.get("post_any_trend_break_60d"):
        modes.append("trend_break_after_buy")
    if row.get("post_any_target_reduce_60d"):
        modes.append("target_reduce_churn")
    if row.get("post_any_risk_reduce_60d"):
        modes.append("risk_reduce_after_buy")
    if row.get("consecutive_small_buy_count_30d", 0) >= 3 and not row.get("effective_lift_45pct_60d"):
        modes.append("repeated_small_probe")
    if pd.isna(row.get("days_to_45pct")) and row.get("future_ret_60d", 0) > 0:
        modes.append("buy_too_small_or_too_slow")
    if row.get("future_ret_30d", 0) < 0 or row.get("future_down_30d", 0) < -0.12:
        modes.append("buy_too_early")
    return "|".join(modes)


def summarize_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for feature in FEATURES:
        if feature not in events.columns:
            continue
        values = pd.to_numeric(events[feature], errors="coerce")
        for label, group in events.assign(_value=values).groupby("quality_label", dropna=False):
            clean = group["_value"].dropna()
            if clean.empty:
                continue
            rows.append({
                "feature": feature,
                "quality_label": label,
                "rows": len(clean),
                "mean": clean.mean(),
                "median": clean.median(),
                "p25": clean.quantile(0.25),
                "p75": clean.quantile(0.75),
            })
    return pd.DataFrame(rows).sort_values(["feature", "quality_label"])


def scan_rule_candidates(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for feature in FEATURES:
        if feature not in events.columns:
            continue
        values = pd.to_numeric(events[feature], errors="coerce").dropna()
        if values.nunique() < 3:
            continue
        thresholds = sorted(set(values.quantile([0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]).dropna()))
        for threshold in thresholds:
            for direction in (">=", "<="):
                mask = apply_rule(events, feature, direction, float(threshold))
                rows.append(score_rule(events, mask, f"{feature} {direction} {threshold:.6g}", 1))
    curated = curated_rule_masks(events)
    for name, mask in curated.items():
        rows.append(score_rule(events, mask, name, name.count(" AND ") + 1))
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["score", "low_reject_pct", "high_recall_pct", "kept_pairs"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )


def curated_rule_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    true = pd.Series(True, index=events.index)
    return {
        "btc_above_ema72 AND alt_ema24_near_ema168 AND non_bear_7d": (
            (events.get("btc_price_vs_ema72", 0) >= 0.05)
            & (events.get("ema24_vs_ema168", 0) >= -0.04)
            & (events.get("non_bear_days", 0) >= 7)
        ),
        "alt_trend_turning AND no_high_vol AND btc_not_bear": (
            (events.get("ema24_slope", 0) > 0)
            & (events.get("ema72_slope", 0) > -0.005)
            & (events.get("atr_pct_rank", 1) < 0.85)
            & (events.get("btc_raw_state", "") != "BEAR")
        ),
        "recovering_structure_conservative": (
            (events.get("price_vs_ema24", -1) > 0)
            & (events.get("ema24_slope", -1) > 0)
            & (events.get("roc_20", 1) < 0.20)
            & (events.get("non_bear_days", 0) >= 10)
            & (events.get("btc_price_vs_ema72", -1) > 0)
        ),
        "all_events": true,
    }


def score_rule(events: pd.DataFrame, mask: pd.Series, rule: str, rule_count: int) -> dict:
    high = events["quality_label"].eq("HIGH_QUALITY")
    low = events["quality_label"].eq("LOW_QUALITY")
    neutral = events["quality_label"].eq("NEUTRAL")
    high_total = int(high.sum())
    low_total = int(low.sum())
    kept_high = int((mask & high).sum())
    kept_low = int((mask & low).sum())
    kept_neutral = int((mask & neutral).sum())
    high_recall = kept_high / high_total if high_total else 0.0
    low_reject = 1.0 - (kept_low / low_total if low_total else 0.0)
    precision = kept_high / (kept_high + kept_low) if kept_high + kept_low else 0.0
    kept_pairs = int(events.loc[mask & high, "pair"].nunique()) if "pair" in events else 0
    counter_low = int((mask & low & events["window_labels"].astype(str).str.contains("bear_", na=False)).sum())
    score = 0.45 * high_recall + 0.35 * low_reject + 0.15 * precision + 0.05 * min(kept_pairs, 3) / 3.0
    return {
        "rule": rule,
        "rule_count": rule_count,
        "score": score,
        "high_total": high_total,
        "low_total": low_total,
        "kept_high": kept_high,
        "kept_low": kept_low,
        "kept_neutral": kept_neutral,
        "high_recall_pct": high_recall * 100.0,
        "low_reject_pct": low_reject * 100.0,
        "precision_pct": precision * 100.0,
        "kept_pairs": kept_pairs,
        "kept_bear_counterexample_low": counter_low,
        "median_kept_ret_60d_pct": events.loc[mask, "future_ret_60d"].median() * 100.0,
        "median_kept_down_60d_pct": events.loc[mask, "future_down_60d"].median() * 100.0,
    }


def summarize_labels(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in events.groupby(["window_labels", "pair", "quality_label"], dropna=False):
        labels, pair, quality = keys
        rows.append({
            "window_labels": labels,
            "pair": pair,
            "quality_label": quality,
            "events": len(group),
            "median_ret_30d_pct": group["future_ret_30d"].median() * 100.0,
            "median_ret_60d_pct": group["future_ret_60d"].median() * 100.0,
            "trend_break_60d_events": int(group["post_any_trend_break_60d"].sum()),
            "lift_45pct_60d_events": int(group["effective_lift_45pct_60d"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["window_labels", "pair", "quality_label"])


def render_report(
    args: argparse.Namespace,
    events: pd.DataFrame,
    feature_summary: pd.DataFrame,
    rule_candidates: pd.DataFrame,
) -> str:
    labels = summarize_labels(events)
    verdict = build_verdict(events, rule_candidates)
    examples = sample_examples(events)
    return "\n".join([
        f"# Recovery Quality Diagnostic: {args.strategy_name}",
        "",
        f"- Timerange: `{args.timerange_start}` to `{args.timerange_end}`",
        f"- Forward/path lookahead: `{args.lookahead_days}` days",
        "- Scope: diagnostic labels only; no strategy rule changes.",
        "- Event sources: safe-recovery buys, target-gap low-position buys, and blocked low-position target-gap candidates.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Label Summary",
        "",
        labels.to_markdown(index=False, floatfmt=".2f") if not labels.empty else "No recovery events.",
        "",
        "## Best Rule Candidates",
        "",
        rule_candidates.head(30).to_markdown(index=False, floatfmt=".2f") if not rule_candidates.empty else "No rule candidates.",
        "",
        "## Feature Summary",
        "",
        feature_summary.to_markdown(index=False, floatfmt=".4f") if not feature_summary.empty else "No feature summary.",
        "",
        "## Example Events",
        "",
        examples.to_markdown(index=False, floatfmt=".2f") if not examples.empty else "No examples.",
        "",
        "## Notes",
        "",
        "- HIGH_QUALITY requires positive 30d/60d returns, bounded 30d/60d adverse path, no 60d trend_break, and an actual lift to at least 45% or 60% exposure.",
        "- LOW_QUALITY catches negative 30d/60d returns, large adverse path, or a 60d trend_break after the event.",
        "- Rule candidates are scored only from contemporaneous features; future returns are used only for labels and scoring.",
        "",
    ])


def build_verdict(events: pd.DataFrame, rules: pd.DataFrame) -> str:
    if events.empty:
        return "No recovery events were detected in the configured diagnostic windows."
    high = events[events["quality_label"] == "HIGH_QUALITY"]
    low = events[events["quality_label"] == "LOW_QUALITY"]
    high_pairs = high["pair"].nunique()
    low_has_bnb_202005 = bool((
        (low["pair"].eq("BNB/USDT"))
        & (low["date"] >= pd.Timestamp("2020-05-01", tz="UTC"))
        & (low["date"] <= pd.Timestamp("2020-05-31", tz="UTC"))
    ).any())
    bear_low = int(low[low["window_labels"].astype(str).str.contains("bear_", na=False)].shape[0])
    best = rules.iloc[0] if not rules.empty else {}
    if high_pairs >= 2 and bear_low > 0 and not rules.empty and best.get("low_reject_pct", 0) >= 50:
        return (
            "The diagnostic set is usable for the next design step: high-quality recoveries span at least two pairs, "
            "bear-market low-quality recoveries are present, and the top contemporaneous rule rejects a meaningful share of low-quality events."
        )
    return (
        "Do not promote a new recovery-position rule from this run alone. Use the event table to inspect whether labels are too sparse, "
        "whether low-quality samples cover the intended counterexamples, and whether the top rules retain multiple pairs."
        f" BNB May 2020 low-quality coverage: `{low_has_bnb_202005}`."
    )


def sample_examples(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    cols = [
        "date",
        "pair",
        "event_source",
        "quality_label",
        "window_labels",
        "current_pct",
        "buy_target",
        "target_gap",
        "future_ret_30d",
        "future_ret_60d",
        "future_down_60d",
        "post_any_trend_break_60d",
        "days_to_45pct",
        "failure_modes",
    ]
    parts = []
    for label in ("HIGH_QUALITY", "LOW_QUALITY", "NEUTRAL"):
        part = events[events["quality_label"].eq(label)].head(8)
        if not part.empty:
            parts.append(part[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def apply_rule(events: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(events[feature], errors="coerce")
    if direction == ">=":
        return values >= threshold
    if direction == "<=":
        return values <= threshold
    raise ValueError(direction)


def labels_for_date(date: pd.Timestamp) -> str:
    labels = []
    for name, start, end, _ in WINDOWS:
        if pd.Timestamp(start, tz="UTC") <= date <= pd.Timestamp(end, tz="UTC"):
            labels.append(name)
    return "|".join(labels)


def safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.astype(float) / right.astype(float).replace(0, float("nan"))


def consecutive_true(mask: pd.Series) -> list[int]:
    streak = 0
    out = []
    for value in mask.fillna(False):
        streak = streak + 1 if bool(value) else 0
        out.append(streak)
    return out


def days_since_mask(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    values = []
    for _, group in frame.groupby("pair", sort=False):
        last_date = None
        for idx, row in group.iterrows():
            if bool(mask.loc[idx]):
                last_date = row["date"]
                values.append(0)
            elif last_date is None:
                values.append(float("nan"))
            else:
                values.append(int((row["date"] - last_date).days))
    return pd.Series(values, index=frame.index)


def forward_min_return(prices: pd.Series, horizon: int) -> pd.Series:
    values = []
    for idx, price in enumerate(prices):
        future = prices.iloc[idx + 1: idx + horizon + 1]
        if future.empty or pd.isna(price) or price == 0:
            values.append(float("nan"))
        else:
            values.append(float(future.min() / price - 1.0))
    return pd.Series(values, index=prices.index)


def days_to_first(start, rows: pd.DataFrame) -> float:
    if rows.empty:
        return float("nan")
    return float((pd.Timestamp(rows.iloc[0]["date"]) - pd.Timestamp(start)).days)


def days_to_threshold(start, rows: pd.DataFrame, threshold: float) -> float:
    hits = rows[rows["current_pct"] >= threshold]
    if hits.empty:
        return float("nan")
    return float((pd.Timestamp(hits.iloc[0]["date"]) - pd.Timestamp(start)).days)


def consecutive_small_buys(group: pd.DataFrame, idx: int, *, horizon: int) -> int:
    start_date = pd.Timestamp(group.iloc[idx]["date"])
    end_date = start_date + pd.Timedelta(days=horizon)
    future = group[(group["date"] > start_date) & (group["date"] <= end_date)]
    count = 0
    max_count = 0
    for _, row in future.iterrows():
        if row["action"] == "buy" and float(row.get("trade_notional", 0.0) or 0.0) == 0.0:
            # Native path detail does not always include notional; fall through to reason/target gap proxy.
            count += 1
        elif row["action"] == "buy" and float(row.get("target_gap", 1.0) or 1.0) <= 0.08:
            count += 1
        elif row["action"] == "buy":
            count += 1
        elif row["action"] == "sell":
            max_count = max(max_count, count)
            count = 0
    return max(max_count, count)


def markdown_to_html(markdown: str) -> str:
    escaped = html.escape(markdown)
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>Recovery Quality Diagnostic</title></head><body><pre>{escaped}</pre></body></html>"


if __name__ == "__main__":
    main()
