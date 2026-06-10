#!/usr/bin/env python3
"""Search contemporaneous RECOVERING score features, including price percentiles.

Diagnostic-only: reads recovery_quality_events.csv, enriches events with recent
path and historical-position features from candles, and searches simple gates.
"""

from __future__ import annotations

import argparse
import html
from itertools import combinations
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovery_quality_v2_21E_20260605"
    / "recovery_quality_events.csv"
)
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"
PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")

BASE_FEATURES = [
    "rolling_365d_pos",
    "donchian_pos",
    "dd_from_120d_high",
    "dd_from_180d_high",
    "price_vs_ema24",
    "price_vs_ema72",
    "price_vs_ema168",
    "ema24_vs_ema168",
    "ema72_vs_ema168",
    "ema24_slope",
    "ema72_slope",
    "roc_10",
    "roc_20",
    "atr_pct_rank",
    "volume_strength",
    "non_bear_days",
    "btc_price_vs_ema72",
    "btc_price_vs_ema168",
    "btc_ema24_slope",
    "btc_ema168_slope",
]

PATH_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ret_14d",
    "ret_30d",
    "pullback_20d_high",
    "pullback_60d_high",
    "bounce_20d_low",
    "bounce_60d_low",
    "price_rank_90d",
    "price_rank_180d",
    "price_rank_365d",
    "realized_vol_20d",
    "up_days_10d",
    "btc_ret_7d",
    "btc_ret_30d",
    "btc_price_rank_180d",
    "btc_bounce_60d_low",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.events))
    enriched = enrich_with_path_features(events, Path(args.datadir))
    single = scan_single_rules(enriched)
    combos = scan_combo_rules(enriched, single)
    curated = evaluate_curated_gates(enriched)
    scorecard = evaluate_structural_scorecard(enriched)
    feature_summary = summarize_features(enriched)
    report = render_report(args, feature_summary, single, combos, curated, scorecard)

    enriched.to_csv(output_dir / "recovery_feature_events.csv", index=False)
    feature_summary.to_csv(output_dir / "recovery_feature_summary.csv", index=False)
    single.to_csv(output_dir / "recovery_feature_single_rules.csv", index=False)
    combos.to_csv(output_dir / "recovery_feature_combo_rules.csv", index=False)
    curated.to_csv(output_dir / "recovery_feature_curated_gates.csv", index=False)
    scorecard.to_csv(output_dir / "recovery_structural_scorecard_summary.csv", index=False)
    (output_dir / "recovery_feature_search_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovery_feature_search_report.html").write_text(markdown_to_html(report), encoding="utf-8")

    print("Top single rules")
    print(single.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}") if not single.empty else "No single rules")
    print("\nTop combo rules")
    print(combos.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}") if not combos.empty else "No combo rules")
    print("\nCurated gates")
    print(curated.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not curated.empty else "No curated gates")
    print("\nStructural scorecard")
    print(scorecard.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not scorecard.empty else "No scorecard")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--datadir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovery_feature_search_v2_21E_20260606")
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing events file: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame[frame["window_labels"].fillna("").ne("")].copy()
    frame["score_window"] = frame["date"].map(score_window_labels)
    frame = frame[frame["score_window"].ne("")].copy()
    frame["is_strong"] = frame["score_window"].str.contains("strong_bull|post_covid", regex=True, na=False)
    frame["is_bear"] = frame["score_window"].str.contains("bear_", na=False)
    frame["is_bnb_may"] = frame["score_window"].str.contains("bnb_2020_05", na=False) & frame["pair"].eq("BNB/USDT")
    frame["good_sample"] = frame["quality_label"].eq("HIGH_QUALITY") & frame["is_strong"]
    frame["bad_sample"] = frame["quality_label"].eq("LOW_QUALITY") & (
        frame["is_bear"] | frame["is_bnb_may"] | frame["score_window"].str.contains("path_pollution", na=False)
    )
    return frame.reset_index(drop=True)


def enrich_with_path_features(events: pd.DataFrame, datadir: Path) -> pd.DataFrame:
    feature_frames = {pair: candle_features(datadir, pair) for pair in PAIRS}
    btc = feature_frames["BTC/USDT"].add_prefix("btc_").rename(columns={"btc_date": "date"})
    parts = []
    for pair in PAIRS:
        pair_features = feature_frames[pair]
        subset = events[events["pair"].eq(pair)].copy()
        if subset.empty:
            continue
        parts.append(subset.merge(pair_features, on="date", how="left").merge(btc, on="date", how="left"))
    return pd.concat(parts, ignore_index=True).sort_values(["pair", "date"]).reset_index(drop=True)


def candle_features(datadir: Path, pair: str) -> pd.DataFrame:
    path = datadir / f"{pair.replace('/', '_')}-1d.feather"
    df = pd.read_feather(path).sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    close = df["close"].astype(float)
    ret = close.pct_change()
    result = pd.DataFrame({"date": df["date"]})
    for days in (1, 3, 7, 14, 30):
        result[f"ret_{days}d"] = close.pct_change(days)
    for days in (20, 60):
        high = close.rolling(days, min_periods=max(5, days // 4)).max()
        low = close.rolling(days, min_periods=max(5, days // 4)).min()
        result[f"pullback_{days}d_high"] = close / high - 1.0
        result[f"bounce_{days}d_low"] = close / low - 1.0
    for days in (90, 180, 365):
        result[f"price_rank_{days}d"] = close.rolling(days, min_periods=max(30, days // 3)).rank(pct=True)
    result["realized_vol_20d"] = ret.rolling(20, min_periods=10).std()
    result["up_days_10d"] = ret.gt(0).rolling(10, min_periods=5).sum()
    return result


def scan_single_rules(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BASE_FEATURES + PATH_FEATURES:
        if feature not in events.columns:
            continue
        values = pd.to_numeric(events[feature], errors="coerce").dropna()
        if values.nunique() < 5:
            continue
        thresholds = sorted(set(values.quantile([0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95]).dropna()))
        for threshold in thresholds:
            for direction in (">=", "<="):
                mask = apply_rule(events, feature, direction, float(threshold))
                row = score_mask(events, mask, f"{feature} {direction} {threshold:.6g}", 1)
                row.update({"feature": feature, "direction": direction, "threshold": float(threshold)})
                if row["good_recall_pct"] >= 8 and row["kept_good_pairs"] >= 2:
                    rows.append(row)
    return sort_rules(pd.DataFrame(rows))


def scan_combo_rules(events: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    seeds = single.head(36).copy()
    rows = []
    for left_idx, right_idx in combinations(seeds.index, 2):
        left = seeds.loc[left_idx]
        right = seeds.loc[right_idx]
        if left["feature"] == right["feature"]:
            continue
        mask = (
            apply_rule(events, left["feature"], left["direction"], left["threshold"])
            & apply_rule(events, right["feature"], right["direction"], right["threshold"])
        )
        row = score_mask(events, mask, f"{left['rule']} AND {right['rule']}", 2)
        if row["good_recall_pct"] >= 8 and row["kept_good_pairs"] >= 2:
            rows.append(row)
    return sort_rules(pd.DataFrame(rows).drop_duplicates("rule") if rows else pd.DataFrame())


def evaluate_curated_gates(events: pd.DataFrame) -> pd.DataFrame:
    e = events
    gates = {
        "macro_repair_mid_history": (
            ((e["btc_price_vs_ema72"] > 0) | (e["btc_ema24_slope"] > 0))
            & e["rolling_365d_pos"].between(0.15, 0.55)
            & e["dd_from_120d_high"].between(0.10, 0.45)
        ),
        "early_recovery_not_extended": (
            (e["btc_price_vs_ema72"] > 0)
            & (e["price_vs_ema168"] < 0.0)
            & (e["roc_20"] < 0.10)
        ),
        "mid_history_pullback": (
            e["rolling_365d_pos"].between(0.20, 0.50)
            & e["donchian_pos"].between(0.35, 0.70)
            & (e["pullback_20d_high"] < -0.03)
            & (e["bounce_60d_low"] < 0.35)
        ),
        "not_chase_recovery": (
            (e["price_rank_180d"] < 0.65)
            & (e["bounce_20d_low"] < 0.25)
            & (e["roc_20"] < 0.15)
            & (e["up_days_10d"] <= 7)
        ),
        "strict_high_cap_candidate": (
            (e["btc_price_vs_ema72"] > 0)
            & e["rolling_365d_pos"].between(0.20, 0.55)
            & (e["price_vs_ema168"] < 0.05)
            & (e["roc_20"] < 0.10)
            & (e["bounce_20d_low"] < 0.25)
            & (e["trend_risk"] == 0)
            & (e["drawdown_risk"] == 0)
        ),
    }
    rows = [score_mask(events, mask, name, name.count("_") + 1) for name, mask in gates.items()]
    return sort_rules(pd.DataFrame(rows))


def evaluate_structural_scorecard(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    base = (
        e["raw_state"].ne("BEAR")
        & (e["current_pct"] < 0.45)
        & (e["target_gap"] >= 0.05)
        & (e["trend_risk"] <= 1)
        & (e["drawdown_risk"] <= 1)
    )
    macro_repair = (e["btc_price_vs_ema72"] > 0) | (e["btc_ema24_slope"] > 0)
    structural_gate = (
        (e["ema72_vs_ema168"] >= -0.03)
        & (e["donchian_pos"] >= 0.54)
        & (e["trend_risk"] == 0)
        & (e["drawdown_risk"] == 0)
        & e["btc_regime"].ne("BEAR")
    )
    score = pd.Series(0, index=e.index, dtype=float)
    score += ((e["ema72_vs_ema168"] >= -0.03) * 2).astype(float)
    score += (e["ema24_vs_ema168"] >= -0.02).astype(float)
    score += macro_repair.astype(float)
    score += ((e["donchian_pos"] >= 0.54) * 2).astype(float)
    score += (e["volume_strength"] <= 0.75).astype(float)
    score += (e["bounce_20d_low"] >= 0.20).astype(float)
    score += (e["pullback_60d_high"] <= -0.25).astype(float)

    cap = pd.Series(35.0, index=e.index)
    cap.loc[base & (score >= 5)] = 50.0
    cap.loc[base & structural_gate & (score >= 7)] = 65.0

    extension_risk = (
        ((e["roc_20"] >= 0.20) & (e["price_vs_ema168"] >= 0.12) & (e["donchian_pos"] >= 0.75))
        | ((e["price_rank_365d"] >= 0.64) & (e["donchian_pos"] >= 0.80) & (e["price_vs_ema168"] >= 0.12))
    )
    chase_risk = (
        ((e["price_rank_365d"] > 0.75) & e["confirmed_state"].ne("BULL"))
        | ((e["bounce_20d_low"] > 0.35) & (e["pullback_20d_high"] > -0.03))
        | extension_risk
    )
    cap.loc[chase_risk] = cap.loc[chase_risk].clip(upper=40.0)
    cap.loc[e["btc_regime"].eq("BEAR")] = cap.loc[e["btc_regime"].eq("BEAR")].clip(upper=35.0)
    cap.loc[e["confirmed_state"].eq("BEAR")] = cap.loc[e["confirmed_state"].eq("BEAR")].clip(upper=40.0)
    cap.loc[e["trend_risk"] >= 2] = cap.loc[e["trend_risk"] >= 2].clip(upper=35.0)

    e["structural_score"] = score
    e["structural_cap_pct"] = cap
    e["structural_gate"] = structural_gate
    e["chase_risk_cap"] = chase_risk
    e["extension_risk_cap"] = extension_risk
    e["structural_tier"] = pd.cut(
        cap,
        bins=[-0.01, 40.0, 55.0, 65.0],
        labels=["LOW_CAP", "MID_CAP", "HIGH_CAP"],
    ).astype(str)

    rows = []
    for keys, group in e.groupby(["score_window", "pair"], dropna=False):
        window, pair = keys
        high = group["quality_label"].eq("HIGH_QUALITY")
        low = group["quality_label"].eq("LOW_QUALITY")
        high_cap = group["structural_cap_pct"] >= 60
        mid_or_high = group["structural_cap_pct"] >= 50
        rows.append({
            "window": window,
            "pair": pair,
            "events": len(group),
            "high_events": int(high.sum()),
            "low_events": int(low.sum()),
            "median_score_high": group.loc[high, "structural_score"].median(),
            "median_score_low": group.loc[low, "structural_score"].median(),
            "high_high_cap_pct": pct((high & high_cap).sum() / high.sum()) if high.any() else float("nan"),
            "low_high_cap_pct": pct((low & high_cap).sum() / low.sum()) if low.any() else float("nan"),
            "high_mid_or_high_cap_pct": pct((high & mid_or_high).sum() / high.sum()) if high.any() else float("nan"),
            "low_mid_or_high_cap_pct": pct((low & mid_or_high).sum() / low.sum()) if low.any() else float("nan"),
            "low_chase_cap_pct": pct((low & group["chase_risk_cap"]).sum() / low.sum()) if low.any() else float("nan"),
            "low_extension_cap_pct": pct((low & group["extension_risk_cap"]).sum() / low.sum()) if low.any() else float("nan"),
            "median_cap_high_pct": group.loc[high, "structural_cap_pct"].median(),
            "median_cap_low_pct": group.loc[low, "structural_cap_pct"].median(),
            "structural_gate_high_events": int((high & group["structural_gate"]).sum()),
            "structural_gate_low_events": int((low & group["structural_gate"]).sum()),
        })
    return pd.DataFrame(rows).sort_values(["window", "pair"])


def summarize_features(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BASE_FEATURES + PATH_FEATURES:
        if feature not in events.columns:
            continue
        for keys, group in events.groupby(["score_window", "quality_label"], dropna=False):
            window, label = keys
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "window": window,
                "quality_label": label,
                "feature": feature,
                "rows": len(values),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
            })
    return pd.DataFrame(rows).sort_values(["window", "feature", "quality_label"])


def apply_rule(events: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(events[feature], errors="coerce")
    return values >= threshold if direction == ">=" else values <= threshold


def score_mask(events: pd.DataFrame, mask: pd.Series, rule: str, rule_count: int) -> dict:
    good = events["good_sample"]
    bad = events["bad_sample"]
    bnb_bad = events["is_bnb_may"] & events["quality_label"].eq("LOW_QUALITY")
    bear_bad = events["is_bear"] & events["quality_label"].eq("LOW_QUALITY")
    good_total = int(good.sum())
    bad_total = int(bad.sum())
    bnb_total = int(bnb_bad.sum())
    bear_total = int(bear_bad.sum())
    kept_good = int((mask & good).sum())
    kept_bad = int((mask & bad).sum())
    kept_bnb = int((mask & bnb_bad).sum())
    kept_bear = int((mask & bear_bad).sum())
    good_recall = kept_good / good_total if good_total else 0.0
    bad_reject = 1.0 - kept_bad / bad_total if bad_total else 0.0
    bnb_reject = 1.0 - kept_bnb / bnb_total if bnb_total else 1.0
    bear_reject = 1.0 - kept_bear / bear_total if bear_total else 1.0
    precision = kept_good / (kept_good + kept_bad) if kept_good + kept_bad else 0.0
    pairs = int(events.loc[mask & good, "pair"].nunique())
    score = (
        0.25 * good_recall
        + 0.25 * bad_reject
        + 0.25 * bnb_reject
        + 0.15 * bear_reject
        + 0.05 * precision
        + 0.05 * min(pairs, 3) / 3.0
    )
    return {
        "rule": rule,
        "rule_count": rule_count,
        "score": score,
        "good_total": good_total,
        "bad_total": bad_total,
        "kept_good": kept_good,
        "kept_bad": kept_bad,
        "kept_bnb_bad": kept_bnb,
        "kept_bear_bad": kept_bear,
        "good_recall_pct": good_recall * 100.0,
        "bad_reject_pct": bad_reject * 100.0,
        "bnb_reject_pct": bnb_reject * 100.0,
        "bear_reject_pct": bear_reject * 100.0,
        "precision_pct": precision * 100.0,
        "kept_good_pairs": pairs,
    }


def sort_rules(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        ["score", "bnb_reject_pct", "bear_reject_pct", "good_recall_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def score_window_labels(date: pd.Timestamp) -> str:
    windows = [
        ("strong_bull_recovery", "2019-02-25", "2021-02-24"),
        ("post_covid_bull", "2020-03-21", "2021-03-21"),
        ("path_pollution", "2018-06-30", "2021-06-29"),
        ("bear_rally_counterexample", "2022-08-01", "2022-12-31"),
        ("bear_defence_counterexample", "2021-12-11", "2022-12-11"),
        ("bnb_2020_05", "2020-05-01", "2020-05-31"),
    ]
    labels = []
    for name, start, end in windows:
        if pd.Timestamp(start, tz="UTC") <= date <= pd.Timestamp(end, tz="UTC"):
            labels.append(name)
    return "|".join(labels)


def pct(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value * 100.0)


def render_report(
    args: argparse.Namespace,
    feature_summary: pd.DataFrame,
    single: pd.DataFrame,
    combos: pd.DataFrame,
    curated: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> str:
    return "\n".join([
        "# Recovery Feature Search",
        "",
        f"- Events: `{args.events}`",
        "- Scope: diagnostic feature search only; no strategy code changes.",
        "- Added path features: historical price rank, recent returns, bounce from rolling lows, pullback from rolling highs, realized vol, and up-day count.",
        "",
        "## Curated Gates",
        "",
        curated.to_markdown(index=False, floatfmt=".2f") if not curated.empty else "No curated gates.",
        "",
        "## Structural Scorecard",
        "",
        scorecard.to_markdown(index=False, floatfmt=".2f") if not scorecard.empty else "No structural scorecard.",
        "",
        "## Top Single Rules",
        "",
        single.head(40).to_markdown(index=False, floatfmt=".2f") if not single.empty else "No single rules.",
        "",
        "## Top Combo Rules",
        "",
        combos.head(40).to_markdown(index=False, floatfmt=".2f") if not combos.empty else "No combo rules.",
        "",
        "## Feature Summary",
        "",
        feature_summary.to_markdown(index=False, floatfmt=".4f") if not feature_summary.empty else "No feature summary.",
        "",
    ])


def markdown_to_html(markdown: str) -> str:
    escaped = html.escape(markdown)
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>Recovery Feature Search</title></head><body><pre>{escaped}</pre></body></html>"


if __name__ == "__main__":
    main()
