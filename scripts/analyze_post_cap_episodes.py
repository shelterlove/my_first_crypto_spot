#!/usr/bin/env python3
"""Analyze post-capitulation recovery episodes from buy-target diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "buy_target_path_v2_21E_long_structure_2018_2024_20260604"
    / "buy_target_path_detail.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = load_detail(Path(args.detail_csv))
    episodes, daily = build_episodes(detail, args)
    summary = summarize(episodes)

    episodes.to_csv(output_dir / "post_cap_episodes.csv", index=False)
    daily.to_csv(output_dir / "post_cap_episode_days.csv", index=False)
    summary.to_csv(output_dir / "post_cap_episode_summary.csv", index=False)
    (output_dir / "post_cap_episode_report.md").write_text(
        render_report(args, episodes, summary),
        encoding="utf-8",
    )

    print("Post-cap episode summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not summary.empty else "No episodes")
    print("\nEpisodes")
    print(episodes.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not episodes.empty else "No episodes")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", default=str(DEFAULT_DETAIL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="post_cap_episodes_v2_21E_20260604")
    parser.add_argument("--dd180-lt", type=float, default=0.35)
    parser.add_argument("--rolling365-pos-lt", type=float, default=0.25)
    parser.add_argument("--ema168-slope-gt", type=float, default=-0.03)
    parser.add_argument("--cluster-gap-days", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=120)
    return parser.parse_args()


def load_detail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    return df.sort_values(["pair", "date"]).reset_index(drop=True)


def build_episodes(detail: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate = (
        detail["lagging_recovery_lift_trigger"].fillna(False)
        & detail["dd_from_180d_high"].lt(args.dd180_lt)
        & detail["rolling_365d_pos"].lt(args.rolling365_pos_lt)
        & detail["ema168_slope"].gt(args.ema168_slope_gt)
    )
    gate_days = detail[gate].copy()
    episode_rows = []
    daily_rows = []
    episode_id = 0
    for pair, group in gate_days.groupby("pair", sort=False):
        prev_date = None
        cluster_id = 0
        for _, row in group.iterrows():
            if prev_date is None or (row["date"] - prev_date).days > args.cluster_gap_days:
                cluster_id += 1
                episode_id += 1
                episode, daily = enrich_episode(
                    episode_id=episode_id,
                    pair=pair,
                    cluster_id=cluster_id,
                    start=row,
                    detail=detail[detail["pair"].eq(pair)].copy(),
                    args=args,
                )
                episode_rows.append(episode)
                daily_rows.extend(daily)
            prev_date = row["date"]
    return pd.DataFrame(episode_rows), pd.DataFrame(daily_rows)


def enrich_episode(
    *,
    episode_id: int,
    pair: str,
    cluster_id: int,
    start: pd.Series,
    detail: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    start_date = start["date"]
    end_date = start_date + pd.Timedelta(days=args.horizon_days)
    window = detail[(detail["date"] >= start_date) & (detail["date"] <= end_date)].copy()
    if window.empty:
        return {}, []

    first_buy = first_row(window, window["action"].eq("buy"))
    first_target_reduce = first_row(
        window,
        window["action"].eq("sell") & window["action_reason"].str.contains("target-reduce", na=False),
    )
    first_trend_break = first_row(
        window,
        window["action"].eq("sell") & window["action_reason"].str.contains("trend-break", na=False),
    )
    first_safe_recovery = first_row(
        window,
        window["action"].eq("buy") & window["action_reason"].str.contains("safe-recovery", na=False),
    )

    episode = {
        "episode_id": episode_id,
        "pair": pair,
        "cluster_id": cluster_id,
        "start_date": fmt_date(start_date),
        "start_price": start["price"],
        "start_current_pct": start["current_pct"],
        "start_buy_target": start["buy_target"],
        "start_target_gap": start["target_gap"],
        "start_raw_state": start["raw_state"],
        "start_confirmed_state": start["confirmed_state"],
        "start_trend_risk": start["trend_risk"],
        "start_btc_regime": start["btc_regime"],
        "start_dd180": start["dd_from_180d_high"],
        "start_rolling365_pos": start["rolling_365d_pos"],
        "start_ema168_slope": start["ema168_slope"],
        "start_buy_guard": start["buy_guard"],
        "gate_days_in_cluster": count_initial_gate_days(window, args),
        "tiny_blocked_gate_days": count_initial_tiny_days(window, args),
        "first_buy_days": days_between(start_date, first_buy),
        "first_buy_date": row_date(first_buy),
        "first_buy_reason": row_value(first_buy, "action_reason"),
        "first_safe_recovery_days": days_between(start_date, first_safe_recovery),
        "first_target_reduce_days": days_between(start_date, first_target_reduce),
        "first_target_reduce_date": row_date(first_target_reduce),
        "first_target_reduce_reason": row_value(first_target_reduce, "action_reason"),
        "first_trend_break_days": days_between(start_date, first_trend_break),
        "buy_count_120d": int(window["action"].eq("buy").sum()),
        "sell_count_120d": int(window["action"].eq("sell").sum()),
        "target_reduce_count_120d": int(window["action_reason"].str.contains("target-reduce", na=False).sum()),
        "days_to_50_pct": days_to_exposure(window, start_date, 0.50),
        "days_to_70_pct": days_to_exposure(window, start_date, 0.70),
        "days_to_90_pct": days_to_exposure(window, start_date, 0.90),
        "exposure_30d_pct": exposure_at(window, start_date, 30),
        "exposure_60d_pct": exposure_at(window, start_date, 60),
        "exposure_90d_pct": exposure_at(window, start_date, 90),
        "price_ret_30d_pct": price_return_at(window, start["price"], start_date, 30),
        "price_ret_60d_pct": price_return_at(window, start["price"], start_date, 60),
        "price_ret_90d_pct": price_return_at(window, start["price"], start_date, 90),
        "price_ret_120d_pct": price_return_at(window, start["price"], start_date, 120),
    }
    if first_target_reduce is not None:
        episode["ret_20d_after_target_reduce_pct"] = price_return_at(
            window, first_target_reduce["price"], first_target_reduce["date"], 20
        )
        episode["ret_60d_after_target_reduce_pct"] = price_return_at(
            window, first_target_reduce["price"], first_target_reduce["date"], 60
        )
    episode["primary_bottleneck"] = classify_bottleneck(episode)

    daily_rows = []
    for _, row in window.iterrows():
        daily_rows.append({
            "episode_id": episode_id,
            "pair": pair,
            "date": fmt_date(row["date"]),
            "days_after_start": int((row["date"] - start_date).days),
            "price": row["price"],
            "current_pct": row["current_pct"],
            "buy_target": row["buy_target"],
            "target_gap": row["target_gap"],
            "action": row["action"],
            "action_reason": row["action_reason"],
            "raw_state": row["raw_state"],
            "confirmed_state": row["confirmed_state"],
            "trend_risk": row["trend_risk"],
            "buy_guard": row["buy_guard"],
        })
    return episode, daily_rows


def first_row(frame: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    rows = frame[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def count_initial_gate_days(window: pd.DataFrame, args: argparse.Namespace) -> int:
    count = 0
    for _, row in window.iterrows():
        if (
            bool(row.get("lagging_recovery_lift_trigger"))
            and row.get("dd_from_180d_high") < args.dd180_lt
            and row.get("rolling_365d_pos") < args.rolling365_pos_lt
            and row.get("ema168_slope") > args.ema168_slope_gt
        ):
            count += 1
            continue
        break
    return count


def count_initial_tiny_days(window: pd.DataFrame, args: argparse.Namespace) -> int:
    return int(
        window.head(count_initial_gate_days(window, args))["buy_guard"]
        .fillna("")
        .str.contains("tiny_buy_skipped")
        .sum()
    )


def days_to_exposure(window: pd.DataFrame, start_date: pd.Timestamp, threshold: float) -> int | None:
    hits = window[window["current_pct"] >= threshold]
    if hits.empty:
        return None
    return int((hits.iloc[0]["date"] - start_date).days)


def exposure_at(window: pd.DataFrame, start_date: pd.Timestamp, days: int) -> float | None:
    row = row_at_or_after(window, start_date + pd.Timedelta(days=days))
    if row is None:
        return None
    return float(row["current_pct"] * 100)


def price_return_at(window: pd.DataFrame, start_price: float, start_date: pd.Timestamp, days: int) -> float | None:
    row = row_at_or_after(window, start_date + pd.Timedelta(days=days))
    if row is None or not start_price:
        return None
    return (float(row["price"]) / float(start_price) - 1.0) * 100


def row_at_or_after(window: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    rows = window[window["date"] >= date]
    if rows.empty:
        return None
    return rows.iloc[0]


def days_between(start: pd.Timestamp, row: pd.Series | None) -> int | None:
    if row is None:
        return None
    return int((row["date"] - start).days)


def row_date(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return fmt_date(row["date"])


def row_value(row: pd.Series | None, key: str) -> str:
    if row is None:
        return ""
    return str(row.get(key, ""))


def fmt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def classify_bottleneck(row: dict) -> str:
    if row.get("first_trend_break_days") is not None and row["first_trend_break_days"] <= 10:
        return "failed_recovery"
    if row.get("first_buy_days") is None:
        return "no_rebuy"
    if row["first_buy_days"] >= 7:
        return "buy_delay"
    if (
        row.get("first_target_reduce_days") is not None
        and row["first_target_reduce_days"] <= 45
        and row.get("ret_60d_after_target_reduce_pct", 0) > 10
    ):
        return "target_reduce_churn"
    return "normal_rebuild"


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = []
    for key, group in episodes.groupby("primary_bottleneck", dropna=False):
        rows.append({
            "primary_bottleneck": key,
            "episodes": len(group),
            "median_first_buy_days": group["first_buy_days"].median(),
            "median_days_to_70_pct": group["days_to_70_pct"].median(),
            "median_price_ret_90d_pct": group["price_ret_90d_pct"].median(),
            "median_target_reduce_days": group["first_target_reduce_days"].median(),
        })
    total = {
        "primary_bottleneck": "all",
        "episodes": len(episodes),
        "median_first_buy_days": episodes["first_buy_days"].median(),
        "median_days_to_70_pct": episodes["days_to_70_pct"].median(),
        "median_price_ret_90d_pct": episodes["price_ret_90d_pct"].median(),
        "median_target_reduce_days": episodes["first_target_reduce_days"].median(),
    }
    return pd.DataFrame([total, *rows])


def render_report(args: argparse.Namespace, episodes: pd.DataFrame, summary: pd.DataFrame) -> str:
    return "\n".join([
        "# Post-Capitulation Recovery Episodes",
        "",
        f"Gate: `dd_from_180d_high < {args.dd180_lt}`, "
        f"`rolling_365d_pos < {args.rolling365_pos_lt}`, "
        f"`ema168_slope > {args.ema168_slope_gt}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No episodes.",
        "",
        "## Episodes",
        "",
        episodes.to_markdown(index=False) if not episodes.empty else "No episodes.",
        "",
    ])


if __name__ == "__main__":
    main()
