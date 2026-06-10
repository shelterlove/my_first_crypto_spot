#!/usr/bin/env python3
"""Evaluate a RECOVERING quality scorecard on recovery-quality events.

This script is diagnostic-only. It reads recovery_quality_events.csv, applies a
contemporaneous feature score, and reports whether higher score/cap tiers align
with HIGH_QUALITY events while suppressing LOW_QUALITY counterexamples.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovery_quality_v2_21E_20260605"
    / "recovery_quality_events.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"

WINDOWS = [
    ("strong_bull_recovery", "2019-02-25", "2021-02-24", True),
    ("post_covid_bull", "2020-03-21", "2021-03-21", True),
    ("path_pollution", "2018-06-30", "2021-06-29", False),
    ("bear_rally_counterexample", "2022-08-01", "2022-12-31", False),
    ("bear_defence_counterexample", "2021-12-11", "2022-12-11", False),
    ("bnb_2020_05", "2020-05-01", "2020-05-31", False),
]

SCORE_COLUMNS = [
    "score_btc_above_ema72",
    "score_btc_ema24_rising",
    "score_btc_not_bear",
    "score_alt_above_ema24",
    "score_alt_ema24_rising",
    "score_alt_near_ema168",
    "score_non_bear_7d",
    "score_non_bear_14d",
    "score_atr_not_high",
    "score_momentum_not_overheated",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.input))
    scored = apply_scorecard(events)
    scored = apply_variants(scored)
    scored = scored[scored["score_window"].ne("")].copy()
    indicator_summary = summarize_indicators(scored)
    variant_summary = summarize_variants(scored)
    window_summary = summarize_windows(scored)
    tier_summary = summarize_tiers(scored)
    pair_summary = summarize_pairs(scored)
    failure_summary = summarize_failures(scored)
    examples = sample_events(scored)
    report = render_report(
        args,
        indicator_summary,
        variant_summary,
        window_summary,
        tier_summary,
        pair_summary,
        failure_summary,
        examples,
    )

    scored.to_csv(output_dir / "recovery_scorecard_events.csv", index=False)
    indicator_summary.to_csv(output_dir / "recovery_scorecard_indicator_summary.csv", index=False)
    variant_summary.to_csv(output_dir / "recovery_scorecard_variant_summary.csv", index=False)
    window_summary.to_csv(output_dir / "recovery_scorecard_window_summary.csv", index=False)
    tier_summary.to_csv(output_dir / "recovery_scorecard_tier_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "recovery_scorecard_pair_summary.csv", index=False)
    failure_summary.to_csv(output_dir / "recovery_scorecard_failure_summary.csv", index=False)
    (output_dir / "recovery_scorecard_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovery_scorecard_report.html").write_text(markdown_to_html(report), encoding="utf-8")

    print("Window summary")
    print(window_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not window_summary.empty else "No events")
    print("\nTier summary")
    print(tier_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not tier_summary.empty else "No tiers")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovery_scorecard_v2_21E_20260606")
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input events: {path}. Run scripts/analyze_recovery_quality.py first.")
    events = pd.read_csv(path)
    events["date"] = pd.to_datetime(events["date"], utc=True)
    return events.sort_values(["pair", "date"]).reset_index(drop=True)


def apply_scorecard(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["base_eligible"] = (
        out["raw_state"].ne("BEAR")
        & (out["current_pct"] < 0.45)
        & (out["target_gap"] >= 0.05)
        & (out["trend_risk"] <= 2)
        & (out["drawdown_risk"] <= 1)
    )
    out["hard_no_add"] = (
        ((out["confirmed_state"].eq("BEAR")) & (out["btc_regime"].eq("BEAR")))
        | (out["days_since_trend_break"].notna() & (out["days_since_trend_break"] <= 20))
    )

    out["score_btc_above_ema72"] = out["btc_price_vs_ema72"] > 0
    out["score_btc_ema24_rising"] = out["btc_ema24_slope"] > 0
    out["score_btc_not_bear"] = out["btc_regime"].ne("BEAR")
    out["score_alt_above_ema24"] = out["price_vs_ema24"] > 0
    out["score_alt_ema24_rising"] = out["ema24_slope"] > 0
    out["score_alt_near_ema168"] = (out["ema24_vs_ema168"] > -0.04) | (out["ema72_vs_ema168"] > -0.05)
    out["score_non_bear_7d"] = out["non_bear_days"] >= 7
    out["score_non_bear_14d"] = out["non_bear_days"] >= 14
    out["score_atr_not_high"] = out["atr_pct_rank"].fillna(1.0) < 0.85
    out["score_momentum_not_overheated"] = (out["roc_20"].fillna(1.0) < 0.20) & (out["donchian_pos"].fillna(1.0) < 0.90)

    out["raw_score"] = out[SCORE_COLUMNS].astype(bool).sum(axis=1)
    out["penalty"] = 0
    out.loc[out["btc_regime"].eq("BEAR"), "penalty"] += 2
    out.loc[out["trend_risk"] >= 2, "penalty"] += 2
    out.loc[out["atr_pct_rank"].fillna(0.0) >= 0.90, "penalty"] += 1
    out.loc[out["roc_20"].fillna(0.0) >= 0.25, "penalty"] += 1
    out.loc[(out["price_vs_ema72"] < 0) & (out["ema72_slope"] < 0), "penalty"] += 1
    out.loc[out["days_since_bear"].notna() & (out["days_since_bear"] <= 30), "penalty"] += 1
    out.loc[out["unrealized_pnl_pct"].fillna(0.0) < -0.10, "penalty"] += 1
    out["recovery_score"] = (out["raw_score"] - out["penalty"]).clip(lower=0, upper=10)

    out["score_cap_pct"] = out["recovery_score"].map(score_to_cap)
    out.loc[~out["base_eligible"], "score_cap_pct"] = 0.0
    out.loc[out["hard_no_add"], "score_cap_pct"] = out.loc[out["hard_no_add"], "current_pct"] * 100.0
    out.loc[out["btc_regime"].eq("BEAR"), "score_cap_pct"] = out.loc[out["btc_regime"].eq("BEAR"), "score_cap_pct"].clip(upper=35.0)
    out.loc[out["confirmed_state"].eq("BEAR"), "score_cap_pct"] = out.loc[out["confirmed_state"].eq("BEAR"), "score_cap_pct"].clip(upper=40.0)
    out.loc[out["trend_risk"] >= 2, "score_cap_pct"] = out.loc[out["trend_risk"] >= 2, "score_cap_pct"].clip(upper=35.0)
    out.loc[out["atr_pct_rank"].fillna(0.0) >= 0.90, "score_cap_pct"] = out.loc[out["atr_pct_rank"].fillna(0.0) >= 0.90, "score_cap_pct"].clip(upper=40.0)
    out.loc[(out["roc_20"].fillna(0.0) >= 0.30) & out["confirmed_state"].ne("BULL"), "score_cap_pct"] = (
        out.loc[(out["roc_20"].fillna(0.0) >= 0.30) & out["confirmed_state"].ne("BULL"), "score_cap_pct"].clip(upper=40.0)
    )

    out["score_tier"] = pd.cut(
        out["score_cap_pct"],
        bins=[-0.01, 0.01, 40.0, 55.0, 65.0],
        labels=["NO_ADD", "LOW_CAP", "MID_CAP", "HIGH_CAP"],
    ).astype(str)
    out["score_window"] = out["date"].map(labels_for_date)
    out["would_lift_over_current"] = out["score_cap_pct"] > (out["current_pct"] * 100.0 + 5.0)
    out["is_bear_counterexample"] = out["score_window"].str.contains("bear_", na=False)
    return out


def score_to_cap(score: int) -> float:
    if score < 5:
        return 35.0
    if score <= 6:
        return 50.0
    return 65.0


def apply_variants(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["variant_original_cap_pct"] = out["score_cap_pct"]

    out["variant_raw_cap_pct"] = out["raw_score"].map(score_to_cap)
    out.loc[~out["base_eligible"], "variant_raw_cap_pct"] = 0.0
    out["variant_raw_cap_pct"] = apply_common_hard_caps(out, out["variant_raw_cap_pct"])

    persistence_mid = out["raw_score"].ge(5) & out["score_non_bear_7d"]
    persistence_high = (
        out["raw_score"].ge(7)
        & out["score_non_bear_14d"]
        & out["score_btc_not_bear"]
        & out["score_alt_above_ema24"]
    )
    out["variant_persistence_cap_pct"] = 35.0
    out.loc[persistence_mid, "variant_persistence_cap_pct"] = 50.0
    out.loc[persistence_high, "variant_persistence_cap_pct"] = 65.0
    out.loc[~out["base_eligible"], "variant_persistence_cap_pct"] = 0.0
    out["variant_persistence_cap_pct"] = apply_common_hard_caps(out, out["variant_persistence_cap_pct"])

    risk_mid = out["raw_score"].ge(5) & (out["trend_risk"] <= 1) & (out["drawdown_risk"] <= 1)
    risk_high = (
        out["raw_score"].ge(7)
        & (out["trend_risk"] == 0)
        & (out["drawdown_risk"] == 0)
        & out["score_atr_not_high"]
        & out["score_momentum_not_overheated"]
    )
    out["variant_risk_first_cap_pct"] = 35.0
    out.loc[risk_mid, "variant_risk_first_cap_pct"] = 50.0
    out.loc[risk_high, "variant_risk_first_cap_pct"] = 65.0
    out.loc[~out["base_eligible"], "variant_risk_first_cap_pct"] = 0.0
    out["variant_risk_first_cap_pct"] = apply_common_hard_caps(out, out["variant_risk_first_cap_pct"])

    strict_mid = (
        out["raw_score"].ge(6)
        & out["score_btc_not_bear"]
        & out["score_alt_above_ema24"]
        & (out["trend_risk"] <= 1)
    )
    strict_high = (
        out["raw_score"].ge(8)
        & out["score_btc_not_bear"]
        & out["score_btc_above_ema72"]
        & out["score_alt_above_ema24"]
        & out["score_alt_ema24_rising"]
        & out["score_non_bear_7d"]
        & (out["trend_risk"] == 0)
    )
    out["variant_strict_cap_pct"] = 35.0
    out.loc[strict_mid, "variant_strict_cap_pct"] = 50.0
    out.loc[strict_high, "variant_strict_cap_pct"] = 65.0
    out.loc[~out["base_eligible"], "variant_strict_cap_pct"] = 0.0
    out["variant_strict_cap_pct"] = apply_common_hard_caps(out, out["variant_strict_cap_pct"])
    return out


def apply_common_hard_caps(scored: pd.DataFrame, cap: pd.Series) -> pd.Series:
    cap = cap.astype(float).copy()
    cap.loc[scored["hard_no_add"]] = scored.loc[scored["hard_no_add"], "current_pct"] * 100.0
    cap.loc[scored["btc_regime"].eq("BEAR")] = cap.loc[scored["btc_regime"].eq("BEAR")].clip(upper=35.0)
    cap.loc[scored["confirmed_state"].eq("BEAR")] = cap.loc[scored["confirmed_state"].eq("BEAR")].clip(upper=40.0)
    cap.loc[scored["trend_risk"] >= 2] = cap.loc[scored["trend_risk"] >= 2].clip(upper=35.0)
    cap.loc[scored["atr_pct_rank"].fillna(0.0) >= 0.90] = cap.loc[scored["atr_pct_rank"].fillna(0.0) >= 0.90].clip(upper=40.0)
    overheated = (scored["roc_20"].fillna(0.0) >= 0.30) & scored["confirmed_state"].ne("BULL")
    cap.loc[overheated] = cap.loc[overheated].clip(upper=40.0)
    return cap


def summarize_windows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, group in scored.groupby("score_window", dropna=False):
        high = group["quality_label"].eq("HIGH_QUALITY")
        low = group["quality_label"].eq("LOW_QUALITY")
        rows.append({
            "window": window,
            "events": len(group),
            "pairs": group["pair"].nunique(),
            "high_quality_events": int(high.sum()),
            "low_quality_events": int(low.sum()),
            "median_score_high": group.loc[high, "recovery_score"].median(),
            "median_score_low": group.loc[low, "recovery_score"].median(),
            "high_high_cap_pct": pct((high & group["score_tier"].eq("HIGH_CAP")).mean()),
            "low_high_cap_pct": pct((low & group["score_tier"].eq("HIGH_CAP")).mean()),
            "low_no_or_low_cap_pct": pct((low & group["score_tier"].isin(["NO_ADD", "LOW_CAP"])).mean()),
            "median_cap_high_pct": group.loc[high, "score_cap_pct"].median(),
            "median_cap_low_pct": group.loc[low, "score_cap_pct"].median(),
            "median_ret_60d_high_pct": group.loc[high, "future_ret_60d"].median() * 100.0,
            "median_ret_60d_low_pct": group.loc[low, "future_ret_60d"].median() * 100.0,
        })
    return pd.DataFrame(rows).sort_values("window")


def summarize_indicators(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(["score_window", "pair"], dropna=False):
        window, pair = keys
        high = group["quality_label"].eq("HIGH_QUALITY")
        low = group["quality_label"].eq("LOW_QUALITY")
        for col in SCORE_COLUMNS:
            high_rate = group.loc[high, col].mean() if high.any() else float("nan")
            low_rate = group.loc[low, col].mean() if low.any() else float("nan")
            rows.append({
                "window": window,
                "pair": pair,
                "indicator": col.replace("score_", ""),
                "events": len(group),
                "high_events": int(high.sum()),
                "low_events": int(low.sum()),
                "high_hit_pct": pct(high_rate),
                "low_hit_pct": pct(low_rate),
                "high_minus_low_pct": pct(high_rate - low_rate) if pd.notna(high_rate) and pd.notna(low_rate) else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["window", "pair", "high_minus_low_pct"], ascending=[True, True, False])


def summarize_variants(scored: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "original_penalty_score": "variant_original_cap_pct",
        "raw_score_hard_caps": "variant_raw_cap_pct",
        "persistence_gate": "variant_persistence_cap_pct",
        "risk_first_gate": "variant_risk_first_cap_pct",
        "strict_structure_gate": "variant_strict_cap_pct",
    }
    rows = []
    for keys, group in scored.groupby(["score_window", "pair"], dropna=False):
        window, pair = keys
        high = group["quality_label"].eq("HIGH_QUALITY")
        low = group["quality_label"].eq("LOW_QUALITY")
        for name, col in variants.items():
            high_cap = group[col] >= 60.0
            mid_or_high = group[col] >= 50.0
            rows.append({
                "window": window,
                "pair": pair,
                "variant": name,
                "events": len(group),
                "high_events": int(high.sum()),
                "low_events": int(low.sum()),
                "high_high_cap_pct": pct((high & high_cap).sum() / high.sum()) if high.any() else float("nan"),
                "low_high_cap_pct": pct((low & high_cap).sum() / low.sum()) if low.any() else float("nan"),
                "high_mid_or_high_cap_pct": pct((high & mid_or_high).sum() / high.sum()) if high.any() else float("nan"),
                "low_mid_or_high_cap_pct": pct((low & mid_or_high).sum() / low.sum()) if low.any() else float("nan"),
                "median_cap_high_pct": group.loc[high, col].median(),
                "median_cap_low_pct": group.loc[low, col].median(),
            })
    return pd.DataFrame(rows).sort_values(["window", "pair", "variant"])


def summarize_tiers(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(["score_window", "score_tier"], dropna=False):
        window, tier = keys
        rows.append({
            "window": window,
            "score_tier": tier,
            "events": len(group),
            "pairs": group["pair"].nunique(),
            "high_quality_pct": pct(group["quality_label"].eq("HIGH_QUALITY").mean()),
            "low_quality_pct": pct(group["quality_label"].eq("LOW_QUALITY").mean()),
            "median_score": group["recovery_score"].median(),
            "median_cap_pct": group["score_cap_pct"].median(),
            "median_ret_30d_pct": group["future_ret_30d"].median() * 100.0,
            "median_ret_60d_pct": group["future_ret_60d"].median() * 100.0,
            "median_down_60d_pct": group["future_down_60d"].median() * 100.0,
        })
    return pd.DataFrame(rows).sort_values(["window", "score_tier"])


def summarize_pairs(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(["score_window", "pair"], dropna=False):
        window, pair = keys
        high = group["quality_label"].eq("HIGH_QUALITY")
        low = group["quality_label"].eq("LOW_QUALITY")
        rows.append({
            "window": window,
            "pair": pair,
            "events": len(group),
            "high_quality_events": int(high.sum()),
            "low_quality_events": int(low.sum()),
            "median_score": group["recovery_score"].median(),
            "high_cap_events": int(group["score_tier"].eq("HIGH_CAP").sum()),
            "low_quality_high_cap_events": int((low & group["score_tier"].eq("HIGH_CAP")).sum()),
            "median_cap_pct": group["score_cap_pct"].median(),
        })
    return pd.DataFrame(rows).sort_values(["window", "pair"])


def summarize_failures(scored: pd.DataFrame) -> pd.DataFrame:
    focus = scored[
        scored["quality_label"].eq("LOW_QUALITY")
        & scored["score_tier"].isin(["MID_CAP", "HIGH_CAP"])
    ].copy()
    if focus.empty:
        return pd.DataFrame(columns=["window", "failure_mode", "events", "median_score", "median_cap_pct"])
    rows = []
    exploded = focus.assign(failure_mode=focus["failure_modes"].fillna("").str.split("|")).explode("failure_mode")
    exploded["failure_mode"] = exploded["failure_mode"].replace("", "unclassified")
    for keys, group in exploded.groupby(["score_window", "failure_mode"], dropna=False):
        window, mode = keys
        rows.append({
            "window": window,
            "failure_mode": mode,
            "events": len(group),
            "median_score": group["recovery_score"].median(),
            "median_cap_pct": group["score_cap_pct"].median(),
        })
    return pd.DataFrame(rows).sort_values(["window", "events"], ascending=[True, False])


def sample_events(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "pair",
        "score_window",
        "quality_label",
        "event_source",
        "recovery_score",
        "raw_score",
        "penalty",
        "score_tier",
        "score_cap_pct",
        "current_pct",
        "future_ret_60d",
        "future_down_60d",
        "failure_modes",
    ]
    parts = []
    masks = [
        scored["quality_label"].eq("HIGH_QUALITY") & scored["score_tier"].eq("HIGH_CAP"),
        scored["quality_label"].eq("LOW_QUALITY") & scored["score_tier"].eq("HIGH_CAP"),
        scored["score_window"].eq("bnb_2020_05"),
        scored["score_window"].str.contains("bear_", na=False) & scored["quality_label"].eq("LOW_QUALITY"),
    ]
    for mask in masks:
        part = scored[mask].head(10)
        if not part.empty:
            parts.append(part[cols])
    return pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()


def render_report(
    args: argparse.Namespace,
    indicator_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> str:
    verdict = build_verdict(window_summary, tier_summary, pair_summary)
    return "\n".join([
        "# Recovery Scorecard Diagnostic",
        "",
        f"- Input: `{args.input}`",
        "- Scope: scorecard diagnostics only; no strategy code changes.",
        "- Score max: 10 contemporaneous points, minus risk penalties.",
        "- Cap mapping: score <5 => 35%, score 5-6 => 50%, score >=7 => 65%, then hard caps.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Window Summary",
        "",
        window_summary.to_markdown(index=False, floatfmt=".2f") if not window_summary.empty else "No window summary.",
        "",
        "## Tier Summary",
        "",
        tier_summary.to_markdown(index=False, floatfmt=".2f") if not tier_summary.empty else "No tier summary.",
        "",
        "## Variant Summary",
        "",
        variant_summary.to_markdown(index=False, floatfmt=".2f") if not variant_summary.empty else "No variant summary.",
        "",
        "## Indicator Separation",
        "",
        indicator_summary.to_markdown(index=False, floatfmt=".2f") if not indicator_summary.empty else "No indicator summary.",
        "",
        "## Pair Summary",
        "",
        pair_summary.to_markdown(index=False, floatfmt=".2f") if not pair_summary.empty else "No pair summary.",
        "",
        "## Low-Quality Events That Still Receive Mid/High Cap",
        "",
        failure_summary.head(50).to_markdown(index=False, floatfmt=".2f") if not failure_summary.empty else "No low-quality events reached mid/high cap.",
        "",
        "## Examples",
        "",
        examples.to_markdown(index=False, floatfmt=".2f") if not examples.empty else "No examples.",
        "",
    ])


def build_verdict(window_summary: pd.DataFrame, tier_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> str:
    if window_summary.empty:
        return "No scoreable events."
    bear = window_summary[window_summary["window"].str.contains("bear_", na=False)]
    strong = window_summary[window_summary["window"].str.contains("strong_bull|post_covid", regex=True, na=False)]
    bnb_may = window_summary[window_summary["window"].eq("bnb_2020_05")]
    bear_low_high = bear["low_high_cap_pct"].max() if not bear.empty else float("nan")
    strong_high_high = strong["high_high_cap_pct"].median() if not strong.empty else float("nan")
    bnb_low_high = bnb_may["low_high_cap_pct"].max() if not bnb_may.empty else float("nan")
    if pd.notna(bear_low_high) and bear_low_high > 10:
        return (
            f"Current scorecard is too permissive for 2022 counterexamples: worst low-quality HIGH_CAP rate is {bear_low_high:.2f}%. "
            "Tighten BTC/confirmed-BEAR caps or require stronger non-BEAR persistence before allowing 65%."
        )
    if pd.notna(strong_high_high) and strong_high_high < 30:
        return (
            f"Current scorecard is too conservative in strong recovery windows: median high-quality HIGH_CAP rate is {strong_high_high:.2f}%. "
            "The score may need a lower high-cap threshold, but only if 2022 remains suppressed."
        )
    return (
        "Current scorecard is worth a second diagnostic pass: it preserves meaningful high-cap access in strong recovery windows while keeping "
        f"2022 low-quality HIGH_CAP near {bear_low_high:.2f}% and BNB 2020-05 near {bnb_low_high:.2f}%."
    )


def labels_for_date(date: pd.Timestamp) -> str:
    labels = []
    for name, start, end, _ in WINDOWS:
        if pd.Timestamp(start, tz="UTC") <= date <= pd.Timestamp(end, tz="UTC"):
            labels.append(name)
    return "|".join(labels)


def pct(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value * 100.0)


def markdown_to_html(markdown: str) -> str:
    escaped = html.escape(markdown)
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>Recovery Scorecard</title></head><body><pre>{escaped}</pre></body></html>"


if __name__ == "__main__":
    main()
