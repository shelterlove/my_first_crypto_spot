#!/usr/bin/env python3
"""Diagnose MIXED low-position buy execution blockers and narrow unblock candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "diagnostics" / "buy_target_path_v2_21E_full_dev_partial_exit_tagfix_20260605"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = load_detail(args)
    rows = enrich(detail)
    summary = summarize(rows)
    candidate = summarize_candidate(rows)

    rows.to_csv(output_dir / "mixed_execution_blocker_detail.csv", index=False)
    summary.to_csv(output_dir / "mixed_execution_blocker_summary.csv", index=False)
    candidate.to_csv(output_dir / "mixed_execution_candidate_summary.csv", index=False)
    (output_dir / "mixed_execution_blocker_report.md").write_text(
        render_report(args, summary, candidate),
        encoding="utf-8",
    )

    print("Candidate summary")
    print(candidate.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTop blocker summary")
    print(summary.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="mixed_execution_blockers_v2_21E_partial_exit_tagfix_20260605")
    return parser.parse_args()


def load_detail(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.input_dir) / "buy_target_path_detail.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values(["pair", "date"]).reset_index(drop=True)


def enrich(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    for horizon in (20, 60, 90):
        out[f"future_ret_{horizon}d"] = (
            out.groupby("pair", sort=False)["price"].shift(-horizon) / out["price"] - 1.0
        )
        out[f"future_down_{horizon}d"] = out.groupby("pair", sort=False)["price"].transform(
            lambda s: forward_min_return(s, horizon)
        )

    guard = out["buy_guard"].fillna("")
    out["has_btc_bear_tgap"] = guard.str.contains("btc-bear-tgap", na=False)
    out["has_tiny_skip"] = guard.str.contains("tiny_buy_skipped", na=False)
    out["has_cost_guard"] = guard.str.contains("v2_4_cost", na=False)
    out["has_vol_guard"] = guard.str.contains("v2_4_vol", na=False)
    out["hold_with_gap"] = (
        (out["action"].fillna("hold") == "hold")
        & (out["target_gap"] > out["min_adj_threshold"])
        & (out["buy_target"] >= 0.60)
        & (out["current_pct"] < 0.35)
        & (out["raw_state"] == "MIXED")
        & (out["confirmed_state"] == "MIXED")
    )
    out["structural_recovery"] = (
        (out["price"] > out["ema24"])
        & (out["ema24_slope"] > 0)
        & (out["ema72_slope"] >= 0)
        & (out["roc_10"] > 0)
        & (out["trend_risk"] == 0)
        & (out["drawdown_risk"] == 0)
        & (out["risk_score"] == 0)
        & (out["atr_pct_rank"] < 0.75)
    )
    out["strict_unblock_candidate"] = (
        out["hold_with_gap"]
        & out["structural_recovery"]
        & out["has_btc_bear_tgap"]
        & out["has_tiny_skip"]
        & (out["btc_regime"] == "BEAR")
    )
    out["refined_unblock_candidate"] = (
        out["strict_unblock_candidate"]
        & (out["volume_strength"] >= 1.15)
        & (out["ema72_slope"] >= 0.035)
    )
    out["candidate_buy_pct"] = 0.0
    out.loc[out["strict_unblock_candidate"], "candidate_buy_pct"] = (
        out.loc[out["strict_unblock_candidate"], "target_gap"].clip(upper=0.08)
    )
    out["refined_candidate_buy_pct"] = 0.0
    out.loc[out["refined_unblock_candidate"], "refined_candidate_buy_pct"] = (
        out.loc[out["refined_unblock_candidate"], "target_gap"].clip(upper=0.08)
    )
    out["sample_period"] = out["date"].map(label_period)
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


def label_period(date: pd.Timestamp) -> str:
    if pd.Timestamp("2019-01-01", tz="UTC") <= date <= pd.Timestamp("2019-04-30", tz="UTC"):
        return "2019_rebound"
    if pd.Timestamp("2022-01-01", tz="UTC") <= date <= pd.Timestamp("2022-12-31", tz="UTC"):
        return "2022_bear"
    if pd.Timestamp("2020-03-21", tz="UTC") <= date <= pd.Timestamp("2021-03-21", tz="UTC"):
        return "2020_2021_bull"
    return "other"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    focus = rows[
        (rows["raw_state"] == "MIXED")
        & (rows["confirmed_state"] == "MIXED")
        & (rows["current_pct"] < 0.35)
        & (rows["buy_target"] >= 0.60)
    ].copy()
    group_sets = [
        ["pair"],
        ["btc_regime"],
        ["sample_period"],
        ["has_btc_bear_tgap", "has_tiny_skip"],
        ["has_cost_guard"],
        ["structural_recovery"],
        ["sample_period", "structural_recovery"],
        ["pair", "sample_period"],
    ]
    rows_out = [summary_row("all_focus", focus)]
    for cols in group_sets:
        for keys, group in focus.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            name = "|".join(f"{col}={key}" for col, key in zip(cols, keys))
            rows_out.append(summary_row(name, group))
    return pd.DataFrame(rows_out).sort_values(["rows", "future_ret_60d_mean_pct"], ascending=[False, False])


def summarize_candidate(rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, flag_col, buy_col in (
        ("strict", "strict_unblock_candidate", "candidate_buy_pct"),
        ("refined", "refined_unblock_candidate", "refined_candidate_buy_pct"),
    ):
        candidate = rows[rows[flag_col]].copy()
        groups = [(f"{label}:all_candidate", candidate)]
        groups.extend((f"{label}:period={period}", group) for period, group in candidate.groupby("sample_period"))
        groups.extend((f"{label}:pair={pair}", group) for pair, group in candidate.groupby("pair"))
        frames.extend(candidate_row(name, group, buy_col) for name, group in groups if not group.empty)
    return pd.DataFrame(frames)


def summary_row(name: str, group: pd.DataFrame) -> dict:
    return {
        "segment": name,
        "rows": len(group),
        "hold_rate_pct": (group["action"].fillna("hold").eq("hold").mean() * 100) if not group.empty else float("nan"),
        "btc_bear_tgap_pct": group["has_btc_bear_tgap"].mean() * 100 if not group.empty else float("nan"),
        "tiny_skip_pct": group["has_tiny_skip"].mean() * 100 if not group.empty else float("nan"),
        "cost_guard_pct": group["has_cost_guard"].mean() * 100 if not group.empty else float("nan"),
        "structural_recovery_pct": group["structural_recovery"].mean() * 100 if not group.empty else float("nan"),
        "mean_current_pct": group["current_pct"].mean() * 100,
        "mean_buy_target_pct": group["buy_target"].mean() * 100,
        "mean_target_gap_pct": group["target_gap"].mean() * 100,
        "future_ret_20d_mean_pct": group["future_ret_20d"].mean() * 100,
        "future_ret_60d_mean_pct": group["future_ret_60d"].mean() * 100,
        "future_ret_90d_mean_pct": group["future_ret_90d"].mean() * 100,
        "future_down_60d_mean_pct": group["future_down_60d"].mean() * 100,
    }


def candidate_row(name: str, group: pd.DataFrame, buy_col: str) -> dict:
    return {
        "segment": name,
        "rows": len(group),
        "mean_candidate_buy_pct": group[buy_col].mean() * 100,
        "total_candidate_buy_pct": group[buy_col].sum() * 100,
        "future_ret_20d_mean_pct": group["future_ret_20d"].mean() * 100,
        "future_ret_60d_mean_pct": group["future_ret_60d"].mean() * 100,
        "future_ret_90d_mean_pct": group["future_ret_90d"].mean() * 100,
        "future_ret_60d_positive_rate_pct": (group["future_ret_60d"] > 0).mean() * 100,
        "future_down_60d_mean_pct": group["future_down_60d"].mean() * 100,
        "worst_future_down_60d_pct": group["future_down_60d"].min() * 100,
        "first_date": group["date"].min().strftime("%Y-%m-%d"),
        "last_date": group["date"].max().strftime("%Y-%m-%d"),
    }


def render_report(args: argparse.Namespace, summary: pd.DataFrame, candidate: pd.DataFrame) -> str:
    return "\n".join([
        "# MIXED Low-Position Execution Blockers",
        "",
        f"- Input: `{args.input_dir}`",
        "",
        "## Candidate Summary",
        "",
        candidate.to_markdown(index=False, floatfmt=".2f") if not candidate.empty else "No strict unblock candidate rows.",
        "",
        "## Blocker Summary",
        "",
        summary.head(40).to_markdown(index=False, floatfmt=".2f"),
        "",
        "Strict candidate rule:",
        "",
        "- raw_state == confirmed_state == MIXED",
        "- current_pct < 35% and buy_target >= 60%",
        "- BTC regime is BEAR, target-gap is shrunk and tiny buy is skipped",
        "- trend_risk == drawdown_risk == risk_score == 0",
        "- price > ema24, ema24_slope > 0, ema72_slope >= 0, roc_10 > 0, atr_pct_rank < 0.75",
        "- candidate buy size is capped at 8% of portfolio value",
        "- refined candidate also requires volume_strength >= 1.15 and ema72_slope >= 0.035",
        "",
    ])


if __name__ == "__main__":
    main()
