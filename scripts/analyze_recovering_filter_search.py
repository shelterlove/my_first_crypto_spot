#!/usr/bin/env python3
"""Search diagnostic filters that separate true RECOVERING from 2022 false rallies."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovering_transition_v2_21E_partial_exit_tagfix_20260605"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"

STRONG_WINDOWS = ("strong_bull_main", "strong_bull_recovery", "post_covid_bull")
BEAR_WINDOWS = ("bear_rally_counterexample", "bear_defence_counterexample")
GOOD_FUTURE_RET_60D = 0.0
GOOD_FUTURE_DOWN_60D = -0.20
BAD_FUTURE_RET_60D = 0.0
BAD_FUTURE_DOWN_60D = -0.15

FEATURES = [
    "ema24_slope",
    "ema72_slope",
    "ema168_slope",
    "price_vs_ema168",
    "ema24_vs_ema168",
    "ema72_vs_ema168",
    "dd_from_120d_high",
    "dd_from_180d_high",
    "rolling_365d_pos",
    "volume_strength",
    "roc_10",
    "roc_20",
    "atr_pct_rank",
    "donchian_pos",
    "raw_state_consecutive_non_bear",
    "ema24_above_ema72_days",
    "btc_price_vs_ema24",
    "btc_price_vs_ema72",
    "btc_ema24_vs_ema72",
    "btc_ema24_vs_ema72_slope_proxy",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.input_dir))
    triggers = build_trigger_samples(rows)
    single = scan_single_filters(triggers, min_good_recall=args.min_good_recall)
    combos = scan_combo_filters(triggers, single, min_good_recall=args.min_good_recall)
    triples = scan_triple_filters(triggers, single, min_good_recall=args.min_triple_good_recall)
    curated = evaluate_curated_rules(triggers)
    zero_bad = summarize_zero_bad_rules(single, combos, triples)
    feature_summary = summarize_features(triggers)
    report = render_report(args, triggers, feature_summary, single, combos, zero_bad)

    triggers.to_csv(output_dir / "recovering_filter_trigger_samples.csv", index=False)
    feature_summary.to_csv(output_dir / "recovering_filter_feature_summary.csv", index=False)
    single.to_csv(output_dir / "recovering_filter_single_rules.csv", index=False)
    combos.to_csv(output_dir / "recovering_filter_combo_rules.csv", index=False)
    triples.to_csv(output_dir / "recovering_filter_triple_rules.csv", index=False)
    curated.to_csv(output_dir / "recovering_filter_curated_rules.csv", index=False)
    zero_bad.to_csv(output_dir / "recovering_filter_zero_2022_bad_rules.csv", index=False)
    (output_dir / "recovering_filter_search_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovering_filter_search_report.html").write_text(markdown_to_simple_html(report), encoding="utf-8")

    print("Top single filters")
    print(single.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}") if not single.empty else "No single filter")
    print("\nTop combo filters")
    print(combos.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}") if not combos.empty else "No combo filter")
    print("\nTop triple filters")
    print(triples.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}") if not triples.empty else "No triple filter")
    print("\nCurated filter checks")
    print(curated.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not curated.empty else "No curated filter")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovering_filter_search_v2_21E_20260605")
    parser.add_argument("--min-good-recall", type=float, default=0.55)
    parser.add_argument("--min-triple-good-recall", type=float, default=0.40)
    return parser.parse_args()


def load_rows(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "recovering_transition_detail.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/analyze_recovering_transition.py first.")
    rows = pd.read_csv(path)
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    rows["btc_price_vs_ema24"] = rows["btc_price"] / rows["btc_ema24"] - 1.0
    rows["btc_price_vs_ema72"] = rows["btc_price"] / rows["btc_ema72"] - 1.0
    rows["btc_ema24_vs_ema72"] = rows["btc_ema24"] / rows["btc_ema72"] - 1.0
    rows["btc_ema24_vs_ema72_slope_proxy"] = rows.groupby("pair", sort=False)["btc_ema24_vs_ema72"].diff(5)
    rows["sample_group"] = rows["window_labels"].map(label_sample_group)
    return rows


def label_sample_group(labels: str) -> str:
    text = str(labels)
    if any(name in text for name in BEAR_WINDOWS):
        return "bear_counterexample"
    if any(name in text for name in STRONG_WINDOWS):
        return "strong_bull"
    return "other"


def build_trigger_samples(rows: pd.DataFrame) -> pd.DataFrame:
    triggers = rows[rows["recovering_candidate"].fillna(False)].copy()
    triggers = triggers[triggers["sample_group"].isin(["strong_bull", "bear_counterexample"])].copy()
    triggers["good_recovery"] = (
        (triggers["sample_group"] == "strong_bull")
        & (triggers["future_ret_60d"] > GOOD_FUTURE_RET_60D)
        & (triggers["future_down_60d"] > GOOD_FUTURE_DOWN_60D)
    )
    triggers["bad_recovery"] = (
        (triggers["sample_group"] == "bear_counterexample")
        & ((triggers["future_ret_60d"] < BAD_FUTURE_RET_60D) | (triggers["future_down_60d"] < BAD_FUTURE_DOWN_60D))
    )
    triggers["adverse_recovery"] = (
        (triggers["future_ret_60d"] <= GOOD_FUTURE_RET_60D)
        | (triggers["future_down_60d"] <= GOOD_FUTURE_DOWN_60D)
    )
    triggers["quality_label"] = "neutral"
    triggers.loc[triggers["good_recovery"], "quality_label"] = "good"
    triggers.loc[triggers["bad_recovery"], "quality_label"] = "bad"
    return triggers.reset_index(drop=True)


def scan_single_filters(triggers: pd.DataFrame, *, min_good_recall: float) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        if feature not in triggers.columns:
            continue
        values = pd.to_numeric(triggers[feature], errors="coerce").dropna()
        if values.nunique() < 3:
            continue
        thresholds = sorted(set(values.quantile([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]).dropna()))
        for threshold in thresholds:
            for direction in (">=", "<="):
                mask = apply_rule(triggers, feature, direction, float(threshold))
                row = score_mask(
                    triggers,
                    mask,
                    rule=f"{feature} {direction} {threshold:.6g}",
                    rule_count=1,
                )
                row.update({"feature": feature, "direction": direction, "threshold": float(threshold)})
                if row["good_recall_pct"] >= min_good_recall * 100:
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["score", "bad_reject_pct", "precision_pct", "good_recall_pct"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )


def scan_combo_filters(triggers: pd.DataFrame, single: pd.DataFrame, *, min_good_recall: float) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    seeds = single.head(24).copy()
    rows = []
    for left_idx, right_idx in combinations(seeds.index, 2):
        left = seeds.loc[left_idx]
        right = seeds.loc[right_idx]
        if left["feature"] == right["feature"]:
            continue
        left_mask = apply_rule(triggers, left["feature"], left["direction"], left["threshold"])
        right_mask = apply_rule(triggers, right["feature"], right["direction"], right["threshold"])
        mask = left_mask & right_mask
        row = score_mask(
            triggers,
            mask,
            rule=f"{left['rule']} AND {right['rule']}",
            rule_count=2,
        )
        if row["good_recall_pct"] >= min_good_recall * 100:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates("rule")
        .sort_values(["score", "bad_reject_pct", "precision_pct", "good_recall_pct"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )


def scan_triple_filters(triggers: pd.DataFrame, single: pd.DataFrame, *, min_good_recall: float) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame()
    focus_features = {
        "btc_price_vs_ema72",
        "ema24_vs_ema168",
        "ema72_vs_ema168",
        "ema168_slope",
        "raw_state_consecutive_non_bear",
        "ema24_above_ema72_days",
        "roc_20",
        "volume_strength",
        "atr_pct_rank",
    }
    seeds = single[single["feature"].isin(focus_features)].head(40).copy()
    rows = []
    for first_idx, second_idx, third_idx in combinations(seeds.index, 3):
        rules = [seeds.loc[first_idx], seeds.loc[second_idx], seeds.loc[third_idx]]
        features = [rule["feature"] for rule in rules]
        if len(set(features)) != len(features):
            continue
        if "btc_price_vs_ema72" not in features:
            continue
        mask = pd.Series(True, index=triggers.index)
        parts = []
        for rule in rules:
            mask &= apply_rule(triggers, rule["feature"], rule["direction"], rule["threshold"])
            parts.append(rule["rule"])
        row = score_mask(
            triggers,
            mask,
            rule=" AND ".join(parts),
            rule_count=3,
        )
        if row["good_recall_pct"] >= min_good_recall * 100 and row["kept_strong_pairs"] >= 2:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates("rule")
        .sort_values(
            ["kept_bad", "adverse_reject_pct", "good_recall_pct", "kept_strong_pairs", "score"],
            ascending=[True, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def evaluate_curated_rules(triggers: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "btc95_alt24_near_ema168": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
        ),
        "btc95_alt24_near_ema168_nonbear7": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
            & (triggers["raw_state_consecutive_non_bear"] >= 7)
        ),
        "btc95_alt24_near_ema168_nonbear10": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
            & (triggers["raw_state_consecutive_non_bear"] >= 10)
        ),
        "btc95_alt24_near_ema168_ema168_slope": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
            & (triggers["ema168_slope"] >= -0.00729914)
        ),
        "btc95_alt24_near_ema168_roc20_cap": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
            & (triggers["roc_20"] <= 0.170355)
        ),
        "btc95_alt24_near_ema168_nonbear7_roc20_cap": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema24_vs_ema168"] >= -0.0322949)
            & (triggers["raw_state_consecutive_non_bear"] >= 7)
            & (triggers["roc_20"] <= 0.170355)
        ),
        "btc95_alt72_near_ema168_nonbear7": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema72_vs_ema168"] >= -0.045001)
            & (triggers["raw_state_consecutive_non_bear"] >= 7)
        ),
        "btc95_alt72_near_ema168_roc20_cap": (
            (triggers["btc_price_vs_ema72"] >= 0.0946377)
            & (triggers["ema72_vs_ema168"] >= -0.045001)
            & (triggers["roc_20"] <= 0.170355)
        ),
    }
    rows = []
    for name, mask in candidates.items():
        row = score_mask(triggers, mask, rule=name, rule_count=name.count("_") + 1)
        row["description"] = describe_curated_rule(name)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["kept_bad", "adverse_reject_pct", "good_recall_pct"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def describe_curated_rule(name: str) -> str:
    descriptions = {
        "btc95_alt24_near_ema168": "BTC strongly above EMA72; alt EMA24 is near EMA168.",
        "btc95_alt24_near_ema168_nonbear7": "Base structural rule plus at least 7 non-BEAR days.",
        "btc95_alt24_near_ema168_nonbear10": "Base structural rule plus at least 10 non-BEAR days.",
        "btc95_alt24_near_ema168_ema168_slope": "Base structural rule plus long EMA slope no longer materially falling.",
        "btc95_alt24_near_ema168_roc20_cap": "Base structural rule plus 20d momentum not overheated.",
        "btc95_alt24_near_ema168_nonbear7_roc20_cap": "Base structural rule plus persistence and no 20d momentum spike.",
        "btc95_alt72_near_ema168_nonbear7": "BTC strong; alt EMA72 near EMA168; at least 7 non-BEAR days.",
        "btc95_alt72_near_ema168_roc20_cap": "BTC strong; alt EMA72 near EMA168; 20d momentum not overheated.",
    }
    return descriptions.get(name, "")


def summarize_zero_bad_rules(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[(rows["kept_bad"] == 0) & (rows["kept_good"] > 0)].copy()
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.sort_values(
            ["adverse_reject_pct", "good_recall_pct", "kept_strong_pairs"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def apply_rule(rows: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(rows[feature], errors="coerce")
    if direction == ">=":
        return values >= threshold
    if direction == "<=":
        return values <= threshold
    raise ValueError(direction)


def score_mask(triggers: pd.DataFrame, mask: pd.Series, *, rule: str, rule_count: int) -> dict:
    good = triggers["good_recovery"]
    bad = triggers["bad_recovery"]
    adverse = triggers["adverse_recovery"]
    strong = triggers["sample_group"] == "strong_bull"
    bear = triggers["sample_group"] == "bear_counterexample"
    kept_good = int((mask & good).sum())
    kept_bad = int((mask & bad).sum())
    kept_adverse = int((mask & adverse).sum())
    good_total = int(good.sum())
    bad_total = int(bad.sum())
    adverse_total = int(adverse.sum())
    kept_strong_pairs = int(triggers.loc[mask & strong, "pair"].nunique())
    kept_bear_pairs = int(triggers.loc[mask & bear, "pair"].nunique())
    good_recall = kept_good / good_total if good_total else 0.0
    bad_reject = 1.0 - (kept_bad / bad_total if bad_total else 0.0)
    adverse_reject = 1.0 - (kept_adverse / adverse_total if adverse_total else 0.0)
    precision = kept_good / (kept_good + kept_bad) if kept_good + kept_bad else 0.0
    breadth_bonus = min(kept_strong_pairs, 3) / 3.0
    score = (0.45 * good_recall) + (0.40 * bad_reject) + (0.10 * precision) + (0.05 * breadth_bonus)
    return {
        "rule": rule,
        "rule_count": rule_count,
        "score": score,
        "good_total": good_total,
        "bad_total": bad_total,
        "kept_good": kept_good,
        "kept_bad": kept_bad,
        "kept_adverse": kept_adverse,
        "good_recall_pct": good_recall * 100.0,
        "bad_reject_pct": bad_reject * 100.0,
        "adverse_reject_pct": adverse_reject * 100.0,
        "precision_pct": precision * 100.0,
        "kept_strong_pairs": kept_strong_pairs,
        "kept_bear_pairs": kept_bear_pairs,
        "kept_strong_days": int((mask & strong).sum()),
        "kept_bear_days": int((mask & bear).sum()),
        "median_kept_future_ret_60d_pct": triggers.loc[mask, "future_ret_60d"].median() * 100.0,
        "median_kept_future_down_60d_pct": triggers.loc[mask, "future_down_60d"].median() * 100.0,
    }


def summarize_features(triggers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        if feature not in triggers.columns:
            continue
        for label, group in triggers.groupby("quality_label", dropna=False):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "feature": feature,
                "quality_label": label,
                "rows": len(values),
                "median": values.median(),
                "mean": values.mean(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
            })
    return pd.DataFrame(rows).sort_values(["feature", "quality_label"])


def render_report(
    args: argparse.Namespace,
    triggers: pd.DataFrame,
    feature_summary: pd.DataFrame,
    single: pd.DataFrame,
    combos: pd.DataFrame,
    zero_bad: pd.DataFrame,
) -> str:
    good = int(triggers["good_recovery"].sum())
    bad = int(triggers["bad_recovery"].sum())
    adverse = int(triggers["adverse_recovery"].sum())
    neutral = int((triggers["quality_label"] == "neutral").sum())
    best = combos.iloc[0]["rule"] if not combos.empty else single.iloc[0]["rule"] if not single.empty else "无"
    return "\n".join([
        "# RECOVERING 判断指标筛选诊断",
        "",
        f"- Input: `{args.input_dir}`",
        f"- Good 样本: {good} 天；2022 Bad 样本: {bad} 天；全局 adverse 样本: {adverse} 天；Neutral 样本: {neutral} 天。",
        "- Good 定义：强牛窗口 RECOVERING 触发后 60d 收益为正，且 60d 内最大下行不低于 -20%。",
        "- Bad 定义：2022 反例窗口 RECOVERING 触发后 60d 收益为负，或 60d 内最大下行低于 -15%。",
        "- Adverse 定义：所有窗口中 60d 收益不为正，或 60d 内最大下行低于 -20%。用于复核规则是否只是在过滤 2022。",
        "- 本脚本只找候选判断指标，不改策略，不把未来收益指标放入策略规则。",
        "",
        "## 初步结论",
        "",
        f"当前最强的诊断组合是：`{best}`。",
        "如果组合规则仍保留大量 2022 bad 样本，RECOVERING 方向应继续暂停；如果它能保留多币种强牛样本并大幅剔除 2022，再进入下一轮人工复核。",
        "",
        "## 单指标 Top 规则",
        "",
        single.head(30).to_markdown(index=False, floatfmt=".2f") if not single.empty else "无可用单指标规则。",
        "",
        "## 双指标 Top 规则",
        "",
        combos.head(30).to_markdown(index=False, floatfmt=".2f") if not combos.empty else "无可用双指标规则。",
        "",
        "## 剔除全部 2022 Bad 的规则",
        "",
        zero_bad.head(30).to_markdown(index=False, floatfmt=".2f") if not zero_bad.empty else "没有规则能在当前召回门槛下剔除全部 2022 Bad。",
        "",
        "## 指标分布摘要",
        "",
        feature_summary.to_markdown(index=False, floatfmt=".4f") if not feature_summary.empty else "无指标摘要。",
        "",
    ])


def markdown_to_simple_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>RECOVERING Filter Search</title></head><body><pre>{escaped}</pre></body></html>"


if __name__ == "__main__":
    main()
