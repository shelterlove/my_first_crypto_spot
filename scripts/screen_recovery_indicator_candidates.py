#!/usr/bin/env python3
"""Screen recovery-quality indicators without changing strategy code.

The output is a research aid for the external market-score layer.  It searches
simple low-quality recovery gates using only existing diagnostic events:

- pre-event structural indicators available on the candidate day;
- short post-event path indicators available after 5 or 10 daily bars.

Rules are scored as conservative risk controls: blocking bad recovery samples is
useful, blocking good recovery samples is expensive.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovery_feature_search_structural_v2_21E_20260606"
    / "recovery_feature_events.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"

PRE_FEATURES = [
    "current_pct",
    "target_gap",
    "risk_score",
    "trend_risk",
    "drawdown_risk",
    "price_vs_ema24",
    "price_vs_ema72",
    "price_vs_ema168",
    "ema24_vs_ema72",
    "ema72_vs_ema168",
    "ema24_slope",
    "ema72_slope",
    "ema168_slope",
    "rolling_365d_pos",
    "price_rank_90d",
    "price_rank_180d",
    "price_rank_365d",
    "dd_from_120d_high",
    "dd_from_180d_high",
    "donchian_pos",
    "roc_10",
    "roc_20",
    "volume_strength",
    "atr_pct_rank",
    "up_days_10d",
    "btc_price_vs_ema72",
    "btc_price_vs_ema168",
    "btc_ema24_slope",
    "btc_ema168_slope",
    "btc_price_rank_180d",
    "btc_ret_30d",
]

POST_BASE_FEATURES = [
    "price_vs_ema24",
    "price_vs_ema72",
    "price_vs_ema168",
    "ema24_vs_ema72",
    "ema72_vs_ema168",
    "ema24_slope",
    "ema72_slope",
    "ema168_slope",
    "rolling_365d_pos",
    "price_rank_180d",
    "price_rank_365d",
    "donchian_pos",
    "roc_10",
    "roc_20",
    "volume_strength",
    "atr_pct_rank",
    "up_days_10d",
]

PATH_BOOLEAN_FEATURES = [
    "post_5d_target_reduce",
    "post_5d_risk_reduce",
    "post_5d_trend_break",
    "post_10d_target_reduce",
    "post_10d_risk_reduce",
    "post_10d_trend_break",
]

PATH_NUMERIC_FEATURES = [
    "post_5d_buy_count",
    "post_5d_sell_count",
    "post_5d_max_current_pct",
    "post_5d_mean_target_gap",
    "post_10d_buy_count",
    "post_10d_sell_count",
    "post_10d_max_current_pct",
    "post_10d_mean_target_gap",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.events))
    events = add_post_indicator_features(events)
    sample_summary = summarize_samples(events)
    predicates = build_predicates(events)
    single = score_predicates(events, predicates, max_terms=1)
    combos2 = score_combinations(events, single, predicates, terms=2)
    combos3 = score_combinations(events, single, predicates, terms=3)
    curated = score_curated(events)
    all_rules = pd.concat(
        [single.assign(source="single"), combos2.assign(source="combo2"), combos3.assign(source="combo3"), curated.assign(source="curated")],
        ignore_index=True,
    )
    all_rules = sort_rules(all_rules)

    events.to_csv(output_dir / "indicator_screen_events.csv", index=False)
    sample_summary.to_csv(output_dir / "indicator_screen_sample_summary.csv", index=False)
    single.to_csv(output_dir / "indicator_screen_single_rules.csv", index=False)
    combos2.to_csv(output_dir / "indicator_screen_combo2_rules.csv", index=False)
    combos3.to_csv(output_dir / "indicator_screen_combo3_rules.csv", index=False)
    curated.to_csv(output_dir / "indicator_screen_curated_rules.csv", index=False)
    all_rules.to_csv(output_dir / "indicator_screen_all_rules.csv", index=False)
    report = render_report(args, sample_summary, all_rules, curated)
    (output_dir / "indicator_screen_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovery_indicator_screen_20260608")
    parser.add_argument("--min-bad-hits", type=int, default=4)
    parser.add_argument("--max-good-hit-rate", type=float, default=0.20)
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing events file: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame[frame["window_labels"].fillna("").ne("")].copy()
    frame = frame[
        frame["raw_state"].ne("BEAR")
        & frame["target_gap"].ge(0.04)
        & frame["current_pct"].lt(0.55)
        & frame["buy_setup"].isin(["safe-recovery", "target-gap"])
    ].copy()
    frame["sample_label"] = "OTHER"
    frame.loc[frame["good_sample"].astype(bool), "sample_label"] = "GOOD"
    frame.loc[frame["bad_sample"].astype(bool), "sample_label"] = "BAD"
    return frame.sort_values(["pair", "date"]).reset_index(drop=True)


def add_post_indicator_features(events: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in events.groupby("pair", sort=False):
        group = group.sort_values("date").reset_index(drop=True).copy()
        for horizon in (5, 10):
            shifted = group[POST_BASE_FEATURES].shift(-horizon)
            for feature in POST_BASE_FEATURES:
                group[f"post_{horizon}d_{feature}"] = shifted[feature]
                group[f"post_{horizon}d_{feature}_delta"] = shifted[feature] - group[feature]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def summarize_samples(events: pd.DataFrame) -> pd.DataFrame:
    labelled = events[events["sample_label"].isin(["GOOD", "BAD"])]
    rows = [{
        "scope": "all",
        "events": len(events),
        "labelled": len(labelled),
        "good": int((events["sample_label"] == "GOOD").sum()),
        "bad": int((events["sample_label"] == "BAD").sum()),
        "pairs": ",".join(sorted(labelled["pair"].dropna().unique())),
        "windows": ",".join(sorted(labelled["score_window"].dropna().unique())),
    }]
    for pair, group in labelled.groupby("pair", sort=True):
        rows.append({
            "scope": pair,
            "events": len(events[events["pair"].eq(pair)]),
            "labelled": len(group),
            "good": int((group["sample_label"] == "GOOD").sum()),
            "bad": int((group["sample_label"] == "BAD").sum()),
            "pairs": pair,
            "windows": ",".join(sorted(group["score_window"].dropna().unique())),
        })
    return pd.DataFrame(rows)


def build_predicates(events: pd.DataFrame) -> list[dict]:
    predicates: list[dict] = []
    feature_sets = [
        ("pre", PRE_FEATURES),
        ("path", PATH_NUMERIC_FEATURES),
        ("post", [f"post_{h}d_{feature}" for h in (5, 10) for feature in POST_BASE_FEATURES]),
        ("post_delta", [f"post_{h}d_{feature}_delta" for h in (5, 10) for feature in POST_BASE_FEATURES]),
    ]
    for phase, features in feature_sets:
        for feature in features:
            if feature not in events.columns:
                continue
            values = pd.to_numeric(events[feature], errors="coerce").dropna()
            if values.nunique() < 8:
                continue
            quantiles = values.quantile([0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90]).dropna()
            for threshold in sorted(set(float(v) for v in quantiles)):
                for direction in ("<=", ">="):
                    predicates.append({
                        "phase": phase,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "rule": f"{feature} {direction} {threshold:.6g}",
                    })
    for feature in PATH_BOOLEAN_FEATURES:
        if feature in events.columns:
            predicates.append({
                "phase": "path_bool",
                "feature": feature,
                "direction": "is",
                "threshold": 1.0,
                "rule": f"{feature} is True",
            })
    return predicates


def score_predicates(events: pd.DataFrame, predicates: list[dict], max_terms: int) -> pd.DataFrame:
    rows = []
    for pred in predicates:
        mask = apply_predicate(events, pred)
        row = score_mask(events, mask, pred["rule"], max_terms, [pred])
        if keep_rule(row):
            rows.append(row)
    return sort_rules(pd.DataFrame(rows))


def score_combinations(events: pd.DataFrame, single: pd.DataFrame, predicates: list[dict], terms: int) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    predicate_by_rule = {pred["rule"]: pred for pred in predicates}
    seeds = single.head(50 if terms == 2 else 18).copy()
    rows = []
    for combo in combinations(seeds["rule"], terms):
        preds = [predicate_by_rule[rule] for rule in combo]
        features = {pred["feature"] for pred in preds}
        if len(features) != len(preds):
            continue
        if terms >= 3 and not any(pred["phase"].startswith("post") or pred["phase"].startswith("path") for pred in preds):
            continue
        mask = pd.Series(True, index=events.index)
        for pred in preds:
            mask &= apply_predicate(events, pred)
        row = score_mask(events, mask, " AND ".join(combo), terms, preds)
        if keep_rule(row):
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return sort_rules(pd.DataFrame(rows).drop_duplicates("rule"))


def score_curated(events: pd.DataFrame) -> pd.DataFrame:
    e = events
    rules = {
        "post5_target_reduce_and_momentum_fails": (
            e["post_5d_target_reduce"].astype(bool)
            & (e["post_5d_roc_20_delta"] <= -0.08)
            & (e["post_5d_price_vs_ema72_delta"] <= -0.03)
        ),
        "post10_target_reduce_no_lift": (
            e["post_10d_target_reduce"].astype(bool)
            & (e["post_10d_max_current_pct"] < 0.45)
            & (e["post_10d_mean_target_gap"] >= 0.04)
        ),
        "post5_breadth_failure": (
            (e["post_5d_up_days_10d"] <= 4)
            & (e["post_5d_volume_strength"] <= 0.90)
            & (e["post_5d_donchian_pos_delta"] <= -0.15)
        ),
        "post10_ema_reject": (
            (e["post_10d_price_vs_ema168"] < 0)
            & (e["post_10d_ema72_vs_ema168"] < -0.03)
            & (e["post_10d_roc_20"] <= 0)
        ),
        "pre_high_percentile_weak_reclaim": (
            (e["price_rank_180d"] >= 0.70)
            & (e["donchian_pos"] <= 0.60)
            & (e["roc_20"] <= 0.05)
            & (e["btc_price_vs_ema72"] <= 0.05)
        ),
        "pre_low_percentile_unresolved_breakdown": (
            (e["price_rank_365d"] <= 0.35)
            & (e["price_vs_ema168"] < 0)
            & (e["ema72_vs_ema168"] < -0.03)
            & (e["volume_strength"] <= 1.05)
        ),
        "hybrid_pre_weak_post_no_lift": (
            (e["price_vs_ema168"] < 0)
            & (e["ema72_vs_ema168"] < -0.02)
            & (e["post_10d_max_current_pct"] < 0.45)
            & (e["post_10d_buy_count"] <= 1)
        ),
    }
    rows = [score_mask(events, mask, name, name.count("_") + 1, []) for name, mask in rules.items()]
    return sort_rules(pd.DataFrame([row for row in rows if keep_rule(row, min_bad_hits=2, max_good_hit_rate=0.35)]))


def apply_predicate(events: pd.DataFrame, pred: dict) -> pd.Series:
    if pred["direction"] == "is":
        return events[pred["feature"]].astype(bool)
    values = pd.to_numeric(events[pred["feature"]], errors="coerce")
    if pred["direction"] == "<=":
        return values <= pred["threshold"]
    return values >= pred["threshold"]


def score_mask(events: pd.DataFrame, mask: pd.Series, rule: str, terms: int, preds: list[dict]) -> dict:
    labelled = events["sample_label"].isin(["GOOD", "BAD"])
    good = events["sample_label"].eq("GOOD")
    bad = events["sample_label"].eq("BAD")
    hit = mask.fillna(False)
    hit_labelled = hit & labelled
    hit_good = hit & good
    hit_bad = hit & bad
    bad_total = int(bad.sum())
    good_total = int(good.sum())
    precision = int(hit_bad.sum()) / int(hit_labelled.sum()) if int(hit_labelled.sum()) else 0.0
    bad_recall = int(hit_bad.sum()) / bad_total if bad_total else 0.0
    good_hit_rate = int(hit_good.sum()) / good_total if good_total else 0.0
    bad_windows = events.loc[hit_bad, "score_window"].nunique()
    bad_pairs = events.loc[hit_bad, "pair"].nunique()
    good_pairs = events.loc[hit_good, "pair"].nunique()
    utility = int(hit_bad.sum()) - 3.0 * int(hit_good.sum()) + 0.25 * bad_windows + 0.25 * bad_pairs
    phases = ",".join(sorted({pred.get("phase", "") for pred in preds if pred}))
    return {
        "rule": rule,
        "terms": terms,
        "phases": phases,
        "hits": int(hit.sum()),
        "labelled_hits": int(hit_labelled.sum()),
        "bad_hits": int(hit_bad.sum()),
        "good_hits": int(hit_good.sum()),
        "bad_total": bad_total,
        "good_total": good_total,
        "bad_precision": precision,
        "bad_recall": bad_recall,
        "good_hit_rate": good_hit_rate,
        "bad_pairs": int(bad_pairs),
        "good_pairs": int(good_pairs),
        "bad_windows": int(bad_windows),
        "first_bad_date": str(events.loc[hit_bad, "date"].min().date()) if int(hit_bad.sum()) else "",
        "last_bad_date": str(events.loc[hit_bad, "date"].max().date()) if int(hit_bad.sum()) else "",
        "bad_pair_list": ",".join(sorted(events.loc[hit_bad, "pair"].unique())),
        "bad_window_list": ",".join(sorted(events.loc[hit_bad, "score_window"].unique())),
        "utility": float(utility),
    }


def keep_rule(row: dict, min_bad_hits: int = 4, max_good_hit_rate: float = 0.20) -> bool:
    if row["bad_hits"] < min_bad_hits:
        return False
    if row["good_hit_rate"] > max_good_hit_rate:
        return False
    if row["bad_precision"] < 0.55:
        return False
    if row["bad_pairs"] < 1 or row["bad_windows"] < 1:
        return False
    return True


def sort_rules(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        ["utility", "bad_precision", "bad_hits", "good_hit_rate", "terms"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def render_report(args: argparse.Namespace, samples: pd.DataFrame, all_rules: pd.DataFrame, curated: pd.DataFrame) -> str:
    cols = [
        "source",
        "rule",
        "terms",
        "phases",
        "bad_hits",
        "good_hits",
        "bad_precision",
        "bad_recall",
        "good_hit_rate",
        "bad_pairs",
        "bad_windows",
        "utility",
    ]
    lines = [
        "# Recovery Indicator Screen",
        "",
        f"- Events: `{args.events}`",
        f"- Min bad hits: `{args.min_bad_hits}`",
        f"- Max good hit rate: `{args.max_good_hit_rate:.0%}`",
        "",
        "## Samples",
        "",
        samples.to_markdown(index=False),
        "",
        "## Best Rules",
        "",
        all_rules[cols].head(30).to_markdown(index=False, floatfmt=".4f") if not all_rules.empty else "No rule passed screening.",
        "",
        "## Curated Rules",
        "",
        curated[cols[1:]].head(20).to_markdown(index=False, floatfmt=".4f") if not curated.empty else "No curated rule passed screening.",
        "",
        "## Interpretation",
        "",
        "- Prefer post-path rules when they rank near the top; they are closer to the intended external risk layer.",
        "- Treat pre-event rules as context only unless they later pass path-level counterfactual replay.",
        "- A screened rule is not a strategy candidate until it passes path replay with no negative return or drawdown deltas.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
