#!/usr/bin/env python3
"""Diagnose whether a RECOVERING transition state explains delayed rebuilds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "buy_target_path_v2_21E_full_dev_partial_exit_tagfix_20260605"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"

WINDOWS = [
    ("strong_bull_main", "2018-12-27", "2021-12-26", True),
    ("strong_bull_recovery", "2019-02-25", "2021-02-24", True),
    ("post_covid_bull", "2020-03-21", "2021-03-21", True),
    ("path_pollution_guardrail", "2018-06-30", "2021-06-29", False),
    ("bear_rally_counterexample", "2022-08-01", "2022-12-31", False),
    ("bear_defence_counterexample", "2021-12-11", "2022-12-11", False),
]
LOW_THRESHOLDS = (0.25, 0.35)
HIGH_THRESHOLD = 0.60
RECOVERING_TARGET_FLOOR = 0.60
RECOVERING_MAX_BUY_CAP = 0.18


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = enrich(load_detail(Path(args.input_dir)))
    episodes = build_recovery_episodes(detail, max_episode_days=args.max_episode_days)
    delay_summary = summarize_delays(episodes)
    blocker_summary = summarize_blockers(episodes)
    trigger_summary = summarize_triggers(detail)
    report = render_report(args, episodes, delay_summary, blocker_summary, trigger_summary)

    detail.to_csv(output_dir / "recovering_transition_detail.csv", index=False)
    delay_summary.to_csv(output_dir / "recovering_delay_summary.csv", index=False)
    blocker_summary.to_csv(output_dir / "recovering_blocker_summary.csv", index=False)
    trigger_summary.to_csv(output_dir / "recovering_trigger_summary.csv", index=False)
    (output_dir / "recovering_transition_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovering_transition_report.html").write_text(markdown_to_simple_html(report), encoding="utf-8")

    print("Recovering delay summary")
    print(delay_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not delay_summary.empty else "No recovery episodes")
    print("\nRecovering trigger summary")
    print(trigger_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not trigger_summary.empty else "No recovering triggers")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovering_transition_v2_21E_partial_exit_tagfix_20260605")
    parser.add_argument("--max-episode-days", type=int, default=180)
    return parser.parse_args()


def load_detail(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "buy_target_path_detail.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/diagnose_buy_target_path.py first.")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values(["pair", "date"]).reset_index(drop=True)


def enrich(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    btc = (
        out[out["pair"] == "BTC/USDT"]
        .set_index("date")[["price", "ema24", "ema72", "raw_state", "btc_regime"]]
        .rename(columns={
            "price": "btc_price",
            "ema24": "btc_ema24",
            "ema72": "btc_ema72",
            "raw_state": "btc_raw_state",
            "btc_regime": "btc_detail_regime",
        })
    )
    out = out.join(btc, on="date")
    out["btc_clear_bear"] = (out["btc_regime"] == "BEAR") | (out["btc_raw_state"] == "BEAR")
    out["btc_reclaimed_ema24_ema72"] = (out["btc_price"] > out["btc_ema24"]) & (out["btc_price"] > out["btc_ema72"])
    out["recovering_btc_filter"] = (~out["btc_clear_bear"]) | out["btc_reclaimed_ema24_ema72"]
    out["recovering_candidate"] = (
        (out["current_pct"] < 0.35)
        & (out["raw_state"] != "BEAR")
        & (out["trend_risk"] == 0)
        & (out["drawdown_risk"] == 0)
        & (out["price"] > out["ema24"])
        & (out["ema24_slope"] > 0)
        & (out["roc_10"] > 0)
        & out["recovering_btc_filter"]
    )
    out["recovering_exit_condition"] = (
        (out["raw_state"] == "BEAR")
        | (out["trend_risk"] > 0)
        | (out["price"] <= out["ema24"])
        | (out["btc_clear_bear"] & ~out["btc_reclaimed_ema24_ema72"])
    )
    out["recovering_target_gap_pct"] = (
        (RECOVERING_TARGET_FLOOR - out["current_pct"]).clip(lower=0.0) * 100.0
    )
    candidate_buy = out["recovering_target_gap_pct"] / 100.0
    out["blocked_recovery_buy_pct"] = (
        (candidate_buy.clip(upper=RECOVERING_MAX_BUY_CAP) - out["executable_buy_pct"].fillna(0.0)).clip(lower=0.0)
        * 100.0
    )
    for horizon in (20, 60, 90):
        out[f"future_ret_{horizon}d"] = out.groupby("pair", sort=False)["price"].shift(-horizon) / out["price"] - 1.0
    out["future_down_60d"] = out.groupby("pair", sort=False)["price"].transform(lambda s: forward_min_return(s, 60))
    out["bear_false_recovery_flag"] = (
        out["recovering_candidate"]
        & (out["date"] >= pd.Timestamp("2022-08-01", tz="UTC"))
        & (out["date"] <= pd.Timestamp("2022-12-31", tz="UTC"))
        & ((out["future_ret_60d"] < 0) | (out["future_down_60d"] < -0.15))
    )
    out["window_labels"] = out["date"].map(labels_for_date)
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


def labels_for_date(date: pd.Timestamp) -> str:
    labels = []
    for name, start, end, _ in WINDOWS:
        if pd.Timestamp(start, tz="UTC") <= date <= pd.Timestamp(end, tz="UTC"):
            labels.append(name)
    return "|".join(labels)


def build_recovery_episodes(detail: pd.DataFrame, *, max_episode_days: int) -> pd.DataFrame:
    rows = []
    for window_name, start_raw, end_raw, is_strong_bull in WINDOWS:
        start = pd.Timestamp(start_raw, tz="UTC")
        end = pd.Timestamp(end_raw, tz="UTC")
        for pair, pair_frame in detail.groupby("pair", sort=False):
            window = pair_frame[(pair_frame["date"] >= start) & (pair_frame["date"] <= end)].copy()
            if window.empty:
                continue
            for low_threshold in LOW_THRESHOLDS:
                rows.extend(detect_window_episodes(
                    window=window,
                    window_name=window_name,
                    is_strong_bull=is_strong_bull,
                    pair=pair,
                    low_threshold=low_threshold,
                    max_episode_days=max_episode_days,
                ))
    return pd.DataFrame(rows)


def detect_window_episodes(
    *,
    window: pd.DataFrame,
    window_name: str,
    is_strong_bull: bool,
    pair: str,
    low_threshold: float,
    max_episode_days: int,
) -> list[dict]:
    rows = []
    frame = window.reset_index(drop=True)
    cursor = 0
    event_index = 1
    while cursor < len(frame):
        low_hits = frame.index[(frame.index >= cursor) & (frame["current_pct"] < low_threshold)]
        if len(low_hits) == 0:
            break
        start_idx = int(low_hits[0])
        end_limit = min(len(frame) - 1, start_idx + max_episode_days)
        high_hits = frame.index[
            (frame.index >= start_idx)
            & (frame.index <= end_limit)
            & (frame["current_pct"] >= HIGH_THRESHOLD)
        ]
        actual_idx = int(high_hits[0]) if len(high_hits) else None
        stop_idx = actual_idx if actual_idx is not None else end_limit
        segment = frame.iloc[start_idx:stop_idx + 1].copy()
        rows.append(episode_row(
            segment=segment,
            window_name=window_name,
            is_strong_bull=is_strong_bull,
            pair=pair,
            low_threshold=low_threshold,
            event_index=event_index,
            recovered=actual_idx is not None,
        ))
        cursor = stop_idx + 1
        event_index += 1
    return rows


def episode_row(
    *,
    segment: pd.DataFrame,
    window_name: str,
    is_strong_bull: bool,
    pair: str,
    low_threshold: float,
    event_index: int,
    recovered: bool,
) -> dict:
    start = segment.iloc[0]
    end = segment.iloc[-1]
    recovering = segment[segment["recovering_candidate"]]
    bull = segment[segment["confirmed_state"] == "BULL"]
    actionable = segment[segment["target_gap"] > segment["min_adj_threshold"]]
    delay_segment = segment
    if not recovering.empty:
        delay_segment = segment[segment["date"] >= recovering.iloc[0]["date"]]

    delay_days = days_between(start["date"], end["date"]) if recovered else None
    recovering_lead_days = days_between(recovering.iloc[0]["date"], end["date"]) if recovered and not recovering.empty else None
    bull_lead_days = days_between(bull.iloc[0]["date"], end["date"]) if recovered and not bull.empty else None
    price_ret_during_delay = pct_return(recovering.iloc[0]["price"], end["price"]) if recovered and not recovering.empty else None

    first_recovering = recovering.iloc[0] if not recovering.empty else None
    return {
        "window": window_name,
        "is_strong_bull_window": is_strong_bull,
        "pair": pair,
        "low_threshold_pct": low_threshold * 100,
        "event_index": event_index,
        "low_date": format_date(start["date"]),
        "low_current_pct": start["current_pct"] * 100,
        "actual_recovery_date": format_date(end["date"]) if recovered else "",
        "actual_recovery_current_pct": end["current_pct"] * 100 if recovered else None,
        "recovered_to_60": recovered,
        "days_low_to_60": delay_days,
        "first_recovering_date": format_date(first_recovering["date"]) if first_recovering is not None else "",
        "first_confirmed_bull_date": format_date(bull.iloc[0]["date"]) if not bull.empty else "",
        "recovery_delay_days": recovering_lead_days,
        "confirmed_bull_lead_days": bull_lead_days,
        "recovering_vs_confirmed_bull_early_days": (
            days_between(first_recovering["date"], bull.iloc[0]["date"])
            if first_recovering is not None and not bull.empty
            else None
        ),
        "price_return_during_delay_pct": price_ret_during_delay,
        "missed_recovery_beta_60d": missed_beta(first_recovering) if first_recovering is not None else None,
        "recovering_target_gap_pct": first_recovering["recovering_target_gap_pct"] if first_recovering is not None else None,
        "blocked_recovery_buy_pct": first_recovering["blocked_recovery_buy_pct"] if first_recovering is not None else None,
        "bear_false_recovery_flag": bool(recovering["bear_false_recovery_flag"].any()) if not recovering.empty else False,
        "recovering_trigger_days": len(recovering),
        "confirmed_state_lag_days": int((delay_segment["confirmed_state"] != "BULL").sum()),
        "buy_target_insufficient_days": int((delay_segment["buy_target"] < HIGH_THRESHOLD).sum()),
        "cooldown_blocked_days": int(delay_segment["cooldown_blocked"].fillna(False).sum()),
        "tiny_buy_skip_days": int(delay_segment["buy_guard"].fillna("").str.contains("tiny_buy_skipped").sum()),
        "btc_bear_target_gap_shrink_days": int(
            ((delay_segment["btc_tgap_mult"].fillna(1.0) < 1.0)
             | delay_segment["buy_guard"].fillna("").str.contains("btc-bear-tgap")).sum()
        ),
        "max_buy_limit_days": int(
            ((delay_segment["target_gap"] > delay_segment["adjusted_max_buy"].fillna(0.0))
             & (delay_segment["adjusted_max_buy"].fillna(0.0) > 0)).sum()
        ),
        "atr_donchian_filter_days": int(
            (delay_segment["buy_guard"].fillna("").str.contains("vol")
             | (delay_segment["atr_pct_rank"].fillna(0.0) >= 0.85)
             | (delay_segment["donchian_pos"].fillna(0.5) <= 0.30)).sum()
        ),
        "actionable_days": len(actionable),
        "mean_buy_target_pct": segment["buy_target"].mean() * 100,
        "mean_current_pct": segment["current_pct"].mean() * 100,
        "mean_target_gap_pct": segment["target_gap"].mean() * 100,
    }


def missed_beta(row: pd.Series) -> float:
    gap = max(RECOVERING_TARGET_FLOOR - float(row["current_pct"]), 0.0)
    future = row.get("future_ret_60d")
    if pd.isna(future):
        return float("nan")
    return gap * float(future) * 100.0


def summarize_delays(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    group_cols = ["window", "is_strong_bull_window", "low_threshold_pct"]
    rows = []
    for keys, group in episodes.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        recovered = group[group["recovered_to_60"]]
        triggered = group[group["first_recovering_date"].astype(str) != ""]
        row.update({
            "episodes": len(group),
            "pairs_with_recovering": triggered["pair"].nunique(),
            "recovering_trigger_rate_pct": len(triggered) / len(group) * 100,
            "recovered_to_60_rate_pct": group["recovered_to_60"].mean() * 100,
            "median_days_low_to_60": recovered["days_low_to_60"].median(),
            "median_recovery_delay_days": recovered["recovery_delay_days"].median(),
            "median_confirmed_bull_lead_days": recovered["confirmed_bull_lead_days"].median(),
            "median_recovering_vs_bull_early_days": triggered["recovering_vs_confirmed_bull_early_days"].median(),
            "median_price_return_during_delay_pct": triggered["price_return_during_delay_pct"].median(),
            "median_missed_recovery_beta_60d": triggered["missed_recovery_beta_60d"].median(),
            "median_recovering_target_gap_pct": triggered["recovering_target_gap_pct"].median(),
            "median_blocked_recovery_buy_pct": triggered["blocked_recovery_buy_pct"].median(),
            "bear_false_recovery_episodes": int(group["bear_false_recovery_flag"].sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["window", "low_threshold_pct"])


def summarize_blockers(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    blocker_cols = [
        "confirmed_state_lag_days",
        "buy_target_insufficient_days",
        "cooldown_blocked_days",
        "tiny_buy_skip_days",
        "btc_bear_target_gap_shrink_days",
        "max_buy_limit_days",
        "atr_donchian_filter_days",
    ]
    rows = []
    for keys, group in episodes.groupby(["window", "low_threshold_pct"], dropna=False):
        window, low_threshold = keys
        total_days = group["recovery_delay_days"].fillna(group["days_low_to_60"]).fillna(0).sum()
        for col in blocker_cols:
            rows.append({
                "window": window,
                "low_threshold_pct": low_threshold,
                "blocker": col.replace("_days", ""),
                "episodes": len(group),
                "total_blocker_days": int(group[col].sum()),
                "mean_blocker_days_per_episode": group[col].mean(),
                "share_of_delay_days_pct": (group[col].sum() / total_days * 100) if total_days else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["window", "low_threshold_pct", "total_blocker_days"], ascending=[True, True, False])


def summarize_triggers(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, start_raw, end_raw, is_strong_bull in WINDOWS:
        start = pd.Timestamp(start_raw, tz="UTC")
        end = pd.Timestamp(end_raw, tz="UTC")
        frame = detail[(detail["date"] >= start) & (detail["date"] <= end)]
        for pair, group in frame.groupby("pair", sort=False):
            triggers = group[group["recovering_candidate"]]
            rows.append({
                "window": name,
                "is_strong_bull_window": is_strong_bull,
                "pair": pair,
                "days": len(group),
                "recovering_trigger_days": len(triggers),
                "first_recovering_date": format_date(triggers.iloc[0]["date"]) if not triggers.empty else "",
                "median_future_ret_60d_pct": triggers["future_ret_60d"].median() * 100 if not triggers.empty else None,
                "median_future_down_60d_pct": triggers["future_down_60d"].median() * 100 if not triggers.empty else None,
                "bear_false_recovery_days": int(triggers["bear_false_recovery_flag"].sum()) if not triggers.empty else 0,
                "mean_blocked_recovery_buy_pct": triggers["blocked_recovery_buy_pct"].mean() if not triggers.empty else None,
            })
    return pd.DataFrame(rows).sort_values(["window", "pair"])


def render_report(
    args: argparse.Namespace,
    episodes: pd.DataFrame,
    delay_summary: pd.DataFrame,
    blocker_summary: pd.DataFrame,
    trigger_summary: pd.DataFrame,
) -> str:
    verdict = build_verdict(episodes, trigger_summary)
    return "\n".join([
        "# RECOVERING 过渡态诊断报告",
        "",
        f"- Input: `{args.input_dir}`",
        "- Baseline: `v2_21E / CryptoSpotV221E`",
        "- 规则只做诊断标签，不改变交易逻辑。",
        "",
        "## 结论",
        "",
        verdict,
        "",
        "## 延迟摘要",
        "",
        delay_summary.to_markdown(index=False, floatfmt=".2f") if not delay_summary.empty else "No recovery episodes.",
        "",
        "## 阻塞归因",
        "",
        "注：阻塞项不是互斥分类，同一天可以同时命中 confirmed-state 滞后、tiny buy skip 和 BTC target-gap shrink，因此 share 可能超过 100%。",
        "",
        blocker_summary.head(60).to_markdown(index=False, floatfmt=".2f") if not blocker_summary.empty else "No blockers.",
        "",
        "## RECOVERING 触发检查",
        "",
        trigger_summary.to_markdown(index=False, floatfmt=".2f") if not trigger_summary.empty else "No recovering triggers.",
        "",
        "## 诊断规则",
        "",
        "- `current_pct < 35%`",
        "- `raw_state != BEAR`，`trend_risk == 0`，`drawdown_risk == 0`",
        "- price > EMA24，EMA24 slope > 0，ROC10 > 0",
        "- BTC 不处于明确 BEAR，或 BTC 已重新站上 EMA24/EMA72",
        "- 观察目标 floor 为 60%，单次恢复买入观察 cap 为 18%，不修改 sell path。",
        "",
        "## 必须回答",
        "",
        answer_required_questions(episodes, trigger_summary),
        "",
    ])


def build_verdict(episodes: pd.DataFrame, trigger_summary: pd.DataFrame) -> str:
    if episodes.empty:
        return "没有检测到可评估的低仓位恢复 episode，暂不进入候选策略设计。"
    strong = episodes[(episodes["is_strong_bull_window"]) & (episodes["first_recovering_date"].astype(str) != "")]
    bear_triggers = trigger_summary[trigger_summary["window"].isin(["bear_rally_counterexample", "bear_defence_counterexample"])]
    median_delay = strong["recovery_delay_days"].median()
    median_ret = strong["price_return_during_delay_pct"].median()
    pairs = strong["pair"].nunique()
    false_days = int(bear_triggers["bear_false_recovery_days"].sum()) if not bear_triggers.empty else 0
    bear_trigger_days = int(bear_triggers["recovering_trigger_days"].sum()) if not bear_triggers.empty else 0

    pass_delay = not pd.isna(median_delay) and median_delay >= 10
    pass_cost = not pd.isna(median_ret) and median_ret > 0
    pass_breadth = pairs >= 2
    pass_bear = false_days == 0 or false_days <= max(2, bear_trigger_days * 0.10)
    if pass_delay and pass_cost and pass_breadth and pass_bear:
        return (
            "诊断证据支持进入最小候选设计：强牛 RECOVERING 相对实际 60% 仓位恢复有至少 10 天的中位提前量，"
            "延迟期间价格中位收益为正，样本不集中在单一币种，且 2022 反例误触发可控。"
        )
    return (
        "诊断证据不足以进入候选策略设计。需要同时满足：强牛中位提前量 >= 10 天、延迟期间价格中位收益为正、"
        "至少覆盖两个币种、且 2022 反例误触发可控。"
    )


def answer_required_questions(episodes: pd.DataFrame, trigger_summary: pd.DataFrame) -> str:
    if episodes.empty:
        return "未检测到 episode，四个问题均无法成立。"
    strong = episodes[episodes["is_strong_bull_window"]]
    triggered = strong[strong["first_recovering_date"].astype(str) != ""]
    bear = trigger_summary[trigger_summary["window"].isin(["bear_rally_counterexample", "bear_defence_counterexample"])]
    delay = triggered["recovery_delay_days"].median()
    bull_early = triggered["recovering_vs_confirmed_bull_early_days"].median()
    price_ret = triggered["price_return_during_delay_pct"].median()
    false_days = int(bear["bear_false_recovery_days"].sum()) if not bear.empty else 0
    trigger_days = int(bear["recovering_trigger_days"].sum()) if not bear.empty else 0
    enter_candidate = build_verdict(episodes, trigger_summary).startswith("诊断证据支持")
    return "\n".join([
        f"- 强牛跑输是否主要来自 `<35% -> >60%` 恢复延迟：中位 RECOVERING 到实际 60% 延迟为 {fmt_num(delay)} 天，延迟期间价格中位收益 {fmt_num(price_ret)}%。",
        f"- `RECOVERING` 是否比 confirmed BULL 更早且更有效：RECOVERING 相对 confirmed BULL 中位提前 {fmt_num(bull_early)} 天。",
        f"- 2022 熊市反弹是否会误触发：2022 两个反例窗口共触发 {trigger_days} 天，其中 false-recovery 标记 {false_days} 天。",
        f"- 是否值得进入候选策略设计：{'是' if enter_candidate else '否'}。",
    ])


def markdown_to_simple_html(markdown: str) -> str:
    escaped = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>RECOVERING Transition Report</title></head><body><pre>{escaped}</pre></body></html>"


def days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)).days)


def pct_return(start: float, end: float) -> float | None:
    if pd.isna(start) or float(start) == 0:
        return None
    return (float(end) / float(start) - 1.0) * 100.0


def format_date(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def fmt_num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
