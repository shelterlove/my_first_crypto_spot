#!/usr/bin/env python3
"""Analyze external risk-cap layers on historical MIXED bars.

This is a diagnostic script only. It evaluates cap labels on already-built
MIXED profile samples, using older rows for research and the latest rows for
out-of-sample validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixed-dir",
        default=str(PROJECT_ROOT / "results" / "diagnostics" / "mixed_historical_profiles_oos2y_db_20260608"),
        help="Directory produced by analyze_mixed_historical_profiles.py.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="external_risk_cap_design_mixed_oos2y_20260609")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixed_dir = Path(args.mixed_dir)
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(mixed_dir / "mixed_historical_train_bars.csv", parse_dates=["timestamp"])
    test = pd.read_csv(mixed_dir / "mixed_historical_test_bars.csv", parse_dates=["timestamp"])
    all_rows = pd.read_csv(mixed_dir / "mixed_historical_bars.csv", parse_dates=["timestamp"])

    frames = []
    summaries = []
    for split_name, frame in (("train", train), ("test", test), ("all", all_rows)):
        labelled = add_cap_labels(frame)
        labelled["split"] = split_name
        frames.append(labelled)
        summaries.append(summarize_caps(labelled, split_name))

    labelled_all = pd.concat(frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    pair_summary = summarize_caps_by_pair(labelled_all)
    composition = summarize_composition(labelled_all)
    report = render_report(summary, pair_summary, composition, train, test)

    labelled_all.to_csv(output_dir / "external_risk_cap_labeled_mixed_bars.csv", index=False)
    summary.to_csv(output_dir / "external_risk_cap_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "external_risk_cap_pair_summary.csv", index=False)
    composition.to_csv(output_dir / "external_risk_cap_composition.csv", index=False)
    (output_dir / "external_risk_cap_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {output_dir}")


def add_cap_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cap = pd.Series(0.60, index=out.index, dtype=float)
    label = pd.Series("DEFAULT_MIXED_CAP60", index=out.index, dtype=str)

    reclaim = out["profile"].eq("RECLAIM_EMA72_LONG_DOWN")
    supportive = out["btc_signal"].eq("BTC_SUPPORTIVE")
    range_improving = out["btc_signal"].eq("BTC_RANGE_IMPROVING")
    range_weak = out["btc_signal"].eq("BTC_RANGE_WEAK")
    btc_bear = out["btc_signal"].eq("BTC_BEAR")
    pullback = out["profile"].eq("PULLBACK_ABOVE_EMA168")
    below_down = out["profile"].eq("BELOW_EMA168_LONG_DOWN")
    above_up = out["profile"].eq("ABOVE_EMA72_LONG_UP")
    lost_up = out["profile"].eq("LOST_EMA168_LONG_UP")

    vol_high = out["volume_strength"].ge(1.15)
    sharp_neg = out["roc_20"].lt(-0.08)
    mom_neg = out["roc_20"].lt(0)
    rank_high = out["rolling_365d_pos"].ge(0.75)
    rank_midlow = out["rolling_365d_pos"].le(0.55)
    donchian_low = out["donchian_pos"].lt(0.35)
    below_ema72 = out["price_vs_ema72"].lt(0)
    failed_structure = below_ema72 & mom_neg & out["ema24_slope"].lt(0)

    cap75 = (reclaim & range_improving & ~sharp_neg) | (reclaim & vol_high & ~btc_bear & ~range_weak)
    cap.loc[cap75] = 0.75
    label.loc[cap75] = "RECLAIM_PARTIAL_RELEASE_CAP75"

    cap100 = reclaim & supportive
    cap.loc[cap100] = 1.00
    label.loc[cap100] = "SUPPORTIVE_RECLAIM_CAP100"

    caution = (
        range_weak
        | pullback
        | (above_up & (rank_high | donchian_low | btc_bear))
        | (lost_up & supportive & (below_ema72 | vol_high | out["atr_pct_rank"].ge(0.85)))
    )
    cap.loc[caution] = np.minimum(cap.loc[caution], 0.40)
    label.loc[caution] = "CAUTION_CAP40"

    critical = (
        (pullback & (vol_high | sharp_neg | range_weak))
        | (below_down & (range_weak | btc_bear))
        | (range_weak & failed_structure)
        | (btc_bear & rank_midlow & out["price_vs_ema72"].ge(0))
    )
    cap.loc[critical] = np.minimum(cap.loc[critical], 0.25)
    label.loc[critical] = "CRITICAL_CAP25"

    # Strict supportive reclaim is the only positive override.
    cap.loc[cap100] = 1.00
    label.loc[cap100] = "SUPPORTIVE_RECLAIM_CAP100"

    out["external_risk_cap"] = cap
    out["external_risk_cap_label"] = label
    return out


def summarize_caps(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    result = frame.groupby("external_risk_cap", dropna=False).agg(
        count=("pair", "size"),
        ret30_med=("fwd_ret_30d", "median"),
        ret60_med=("fwd_ret_60d", "median"),
        pos60_rate=("fwd_ret_60d", lambda s: float((s > 0).mean())),
        down60_med=("fwd_down_60d", "median"),
        pairs=("pair", lambda s: ",".join(sorted(s.unique()))),
    ).reset_index()
    result.insert(0, "split", split_name)
    return result.sort_values(["split", "external_risk_cap"]).reset_index(drop=True)


def summarize_caps_by_pair(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "external_risk_cap", "pair"], dropna=False)
        .agg(
            count=("pair", "size"),
            ret60_med=("fwd_ret_60d", "median"),
            pos60_rate=("fwd_ret_60d", lambda s: float((s > 0).mean())),
            down60_med=("fwd_down_60d", "median"),
        )
        .reset_index()
        .sort_values(["split", "external_risk_cap", "pair"])
    )


def summarize_composition(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "external_risk_cap", "external_risk_cap_label", "profile", "btc_signal"], dropna=False)
        .agg(count=("pair", "size"))
        .reset_index()
        .sort_values(["split", "external_risk_cap", "count"], ascending=[True, True, False])
    )


def render_report(
    summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    composition: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    keep_pairs = pair_summary[pair_summary["count"] >= 8].copy()
    comp_keep = composition[composition["count"] >= 8].copy()
    return "\n".join(
        [
            "# External Risk Cap Design",
            "",
            "- Scope: MIXED bars only; diagnostic labels, no strategy changes.",
            f"- Train rows: `{len(train)}` from `{train['timestamp'].min()}` to `{train['timestamp'].max()}`.",
            f"- Test rows: `{len(test)}` from `{test['timestamp'].min()}` to `{test['timestamp'].max()}`.",
            "",
            "## Cap Summary",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Pair Summary",
            "",
            keep_pairs.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Main Composition",
            "",
            comp_keep.to_markdown(index=False),
            "",
            "## Proposed Interpretation",
            "",
            "- `100%`: only strict supportive reclaim. This is the only full-release condition.",
            "- `75%`: partial reclaim release. It may be absent in a test split, so do not overfit it.",
            "- `60%`: default MIXED repair zone.",
            "- `40%`: caution zone. Slow or cap new exposure.",
            "- `25%`: critical zone. Allow only probe-sized exposure unless the core strategy is already defensive.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
