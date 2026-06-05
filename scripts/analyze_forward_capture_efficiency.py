#!/usr/bin/env python3
"""Measure whether low exposure missed future upside or reflected useful defence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analyze_mixed_regime_labels import label_mixed_quality, label_mixed_row  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"
DEFAULT_GLOB = "buy_target_path_v2_21E_*_partial_exit_tagfix_20260605"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = load_details(args)
    if detail.empty:
        raise SystemExit("No buy_target_path_detail.csv rows found.")
    enriched = enrich(detail)
    summary = build_summary(enriched)
    top = top_missed_segments(summary)

    enriched.to_csv(output_dir / "forward_capture_detail.csv", index=False)
    summary.to_csv(output_dir / "forward_capture_summary.csv", index=False)
    top.to_csv(output_dir / "forward_capture_top_segments.csv", index=False)
    (output_dir / "forward_capture_report.md").write_text(
        render_report(args, summary, top),
        encoding="utf-8",
    )

    print("Top missed beta segments")
    print(top.head(args.top_n).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory containing buy_target_path_detail.csv. Can be repeated.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_GLOB,
        help="Glob under results/diagnostics used when --input-dir is omitted.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default="forward_capture_v2_21E_partial_exit_tagfix_20260605")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_details(args: argparse.Namespace) -> pd.DataFrame:
    paths = []
    if args.input_dir:
        paths = [Path(item) / "buy_target_path_detail.csv" for item in args.input_dir]
    else:
        paths = [
            path / "buy_target_path_detail.csv"
            for path in sorted(DEFAULT_OUTPUT_DIR.glob(args.input_glob))
        ]

    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_run"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["source_run", "pair", "date"]).reset_index(drop=True)
    for horizon in (20, 60, 90):
        col = f"future_ret_{horizon}d"
        out[col] = (
            out.groupby(["source_run", "pair"], sort=False)["price"].shift(-horizon) / out["price"] - 1.0
        )
        out[f"future_down_{horizon}d"] = (
            out.groupby(["source_run", "pair"], sort=False)["price"]
            .transform(lambda s: forward_min_return(s, horizon))
        )
        out[f"missed_beta_{horizon}d"] = out[col].clip(lower=0.0) * (1.0 - out["current_pct"].clip(0.0, 1.0))
        out[f"target_missed_beta_{horizon}d"] = out[col].clip(lower=0.0) * (1.0 - out["buy_target"].clip(0.0, 1.0))
        out[f"execution_gap_beta_{horizon}d"] = (
            out[col].clip(lower=0.0) * (out["buy_target"] - out["current_pct"]).clip(lower=0.0)
        )
        out[f"defence_value_{horizon}d"] = (-out[f"future_down_{horizon}d"].clip(upper=0.0)) * (
            1.0 - out["current_pct"].clip(0.0, 1.0)
        )

    out["market_state"] = out["raw_state"]
    out["mixed_label"] = out.apply(label_mixed_row, axis=1)
    out["mixed_quality"] = out.apply(label_mixed_quality, axis=1)
    out["strong_uptrend"] = (
        (out["price"] > out["ema24"])
        & (out["ema24"] > out["ema72"])
        & (out["ema72"] > out["ema168"])
        & (out["ema168_slope"] > 0)
        & (out["btc_regime"] != "BEAR")
    )
    out["high_atr"] = out["atr_pct_rank"] >= 0.80
    out["low_position"] = out["current_pct"] < 0.60
    out["target_gap_bucket"] = pd.cut(
        out["target_gap"],
        bins=[-10.0, 0.02, 0.10, 0.25, 10.0],
        labels=["no_gap", "small_gap", "medium_gap", "large_gap"],
    )
    out["position_bucket"] = pd.cut(
        out["current_pct"],
        bins=[-0.01, 0.35, 0.60, 0.80, 1.01],
        labels=["lt35", "35_60", "60_80", "gt80"],
    )
    out["vol_bucket"] = pd.cut(
        out["vol_multiplier"],
        bins=[-0.01, 0.90, 0.97, 0.999, 1.01],
        labels=["lt90", "90_97", "97_100", "one"],
    )
    out["action_family"] = out["action_reason"].fillna("").map(classify_action)
    return out


def forward_min_return(prices: pd.Series, horizon: int) -> pd.Series:
    values = []
    for idx, price in enumerate(prices):
        future = prices.iloc[idx + 1: idx + horizon + 1]
        if future.empty or pd.isna(price) or price == 0:
            values.append(float("nan"))
        else:
            values.append(float(future.min() / price - 1.0))
    return pd.Series(values, index=prices.index)


def classify_action(reason: str) -> str:
    if not reason:
        return "hold"
    for key in ("target-reduce", "risk-reduce", "trend-break", "safe-recovery", "pullback", "trend-cont", "target-gap"):
        if key in reason:
            return key
    return "other"


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    group_sets = [
        ["source_run"],
        ["pair"],
        ["confirmed_state"],
        ["raw_state", "confirmed_state"],
        ["btc_regime"],
        ["mixed_quality"],
        ["strong_uptrend"],
        ["high_atr"],
        ["strong_uptrend", "high_atr"],
        ["position_bucket"],
        ["target_gap_bucket"],
        ["vol_bucket"],
        ["action_family"],
        ["confirmed_state", "high_atr"],
        ["confirmed_state", "position_bucket"],
        ["mixed_quality", "high_atr"],
        ["pair", "confirmed_state"],
        ["pair", "strong_uptrend", "high_atr"],
    ]
    rows = [summary_row(["all"], ("all",), detail)]
    for cols in group_sets:
        for keys, group in detail.groupby(cols, dropna=False, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append(summary_row(cols, keys, group))
    return pd.DataFrame(rows).sort_values(["missed_beta_60d_mean", "rows"], ascending=[False, False])


def summary_row(group_cols: list[str], keys: tuple, group: pd.DataFrame) -> dict:
    row = {
        "segment": segment_name(group_cols, keys),
        "rows": len(group),
        "mean_current_pct": group["current_pct"].mean() * 100,
        "mean_buy_target_pct": group["buy_target"].mean() * 100,
        "mean_target_gap_pct": group["target_gap"].mean() * 100,
        "mean_vol_multiplier": group["vol_multiplier"].mean(),
        "future_ret_60d_mean_pct": group["future_ret_60d"].mean() * 100,
        "future_ret_60d_median_pct": group["future_ret_60d"].median() * 100,
        "future_ret_60d_positive_rate_pct": (group["future_ret_60d"] > 0).mean() * 100,
    }
    for horizon in (20, 60, 90):
        row[f"missed_beta_{horizon}d_mean"] = group[f"missed_beta_{horizon}d"].mean() * 100
        row[f"target_missed_beta_{horizon}d_mean"] = group[f"target_missed_beta_{horizon}d"].mean() * 100
        row[f"execution_gap_beta_{horizon}d_mean"] = group[f"execution_gap_beta_{horizon}d"].mean() * 100
        row[f"defence_value_{horizon}d_mean"] = group[f"defence_value_{horizon}d"].mean() * 100
    row["target_share_of_missed_60d_pct"] = safe_ratio(
        row["target_missed_beta_60d_mean"],
        row["missed_beta_60d_mean"],
    ) * 100
    row["execution_share_of_missed_60d_pct"] = safe_ratio(
        row["execution_gap_beta_60d_mean"],
        row["missed_beta_60d_mean"],
    ) * 100
    row["defence_to_missed_60d"] = safe_ratio(
        row["defence_value_60d_mean"],
        row["missed_beta_60d_mean"],
    )
    return row


def segment_name(group_cols: list[str], keys: tuple) -> str:
    if group_cols == ["all"]:
        return "all"
    return "|".join(f"{col}={key}" for col, key in zip(group_cols, keys))


def safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def top_missed_segments(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[summary["rows"] >= 20].copy()
    return candidates.sort_values(
        ["missed_beta_60d_mean", "execution_gap_beta_60d_mean"],
        ascending=[False, False],
    )


def render_report(args: argparse.Namespace, summary: pd.DataFrame, top: pd.DataFrame) -> str:
    cols = [
        "segment", "rows", "mean_current_pct", "mean_buy_target_pct",
        "future_ret_60d_mean_pct", "missed_beta_60d_mean",
        "target_missed_beta_60d_mean", "execution_gap_beta_60d_mean",
        "defence_value_60d_mean", "target_share_of_missed_60d_pct",
        "execution_share_of_missed_60d_pct",
    ]
    return "\n".join([
        "# Forward Capture Efficiency",
        "",
        f"- Run id: `{args.run_id}`",
        "",
        "## Interpretation",
        "",
        "- `missed_beta`: future upside multiplied by uninvested position.",
        "- `target_missed_beta`: missed upside caused by target itself staying below 100%.",
        "- `execution_gap_beta`: missed upside where target was above actual position.",
        "- `defence_value`: future drawdown avoided by uninvested position.",
        "",
        "## Top Missed Beta Segments",
        "",
        top[cols].head(args.top_n).to_markdown(index=False, floatfmt=".2f"),
        "",
        "## All Summary",
        "",
        summary[cols].head(args.top_n * 2).to_markdown(index=False, floatfmt=".2f"),
        "",
    ])


if __name__ == "__main__":
    main()
