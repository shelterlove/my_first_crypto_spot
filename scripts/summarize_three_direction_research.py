#!/usr/bin/env python3
"""Summarize the three post-v2_21E research directions without tuning on validation."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREQTRADE_DIR = PROJECT_ROOT / "results" / "freqtrade_eval"
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    target_reduce = read_csv(Path(args.target_reduce_dir) / "target_reduce_regret_summary.csv")
    forward_capture = read_csv(Path(args.forward_capture_dir) / "forward_capture_summary.csv")
    mixed_blockers = read_csv(Path(args.mixed_blocker_dir) / "mixed_execution_candidate_summary.csv")
    v24c_quick_delta = compare_rolling(Path(args.v21e_quick_dir), Path(args.v24c_quick_dir))
    v24c_standard_delta = compare_rolling(Path(args.v21e_standard_dir), Path(args.v24c_standard_dir))
    v24c_smoke = build_smoke_summary(args)
    v24d_trial = compare_smoke_to_standard_baseline(args.v24d_smoke_dir, Path(args.v21e_standard_dir))

    direction_summary = build_direction_summary(
        target_reduce=target_reduce,
        forward_capture=forward_capture,
        mixed_blockers=mixed_blockers,
        v24c_quick_delta=v24c_quick_delta,
        v24c_standard_delta=v24c_standard_delta,
        v24c_smoke=v24c_smoke,
    )
    rolling_delta = pd.concat(
        [
            v24c_quick_delta.assign(scope="quick"),
            v24c_standard_delta.assign(scope="standard"),
        ],
        ignore_index=True,
    )

    direction_summary.to_csv(output_dir / "three_direction_decision_summary.csv", index=False)
    rolling_delta.to_csv(output_dir / "v2_24C_vs_v2_21E_rolling_delta.csv", index=False)
    v24c_smoke.to_csv(output_dir / "v2_24C_smoke_guardrails.csv", index=False)
    v24d_trial.to_csv(output_dir / "v2_24D_rejected_smoke_trial.csv", index=False)

    report = render_markdown(
        args=args,
        direction_summary=direction_summary,
        target_reduce=target_reduce,
        forward_capture=forward_capture,
        mixed_blockers=mixed_blockers,
        rolling_delta=rolling_delta,
        v24c_smoke=v24c_smoke,
        v24d_trial=v24d_trial,
    )
    (output_dir / "three_direction_research_report.md").write_text(report, encoding="utf-8")
    (output_dir / "three_direction_research_report.html").write_text(render_html(report), encoding="utf-8")

    print(direction_summary.to_string(index=False))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DIAG_DIR))
    parser.add_argument("--run-id", default="three_direction_research_v2_21E_20260605")
    parser.add_argument(
        "--target-reduce-dir",
        default=str(DIAG_DIR / "target_reduce_regret_v2_21E_dev_standard_partial_exit_tagfix_20260605"),
    )
    parser.add_argument(
        "--forward-capture-dir",
        default=str(DIAG_DIR / "forward_capture_v2_21E_full_dev_partial_exit_tagfix_20260605"),
    )
    parser.add_argument(
        "--mixed-blocker-dir",
        default=str(DIAG_DIR / "mixed_execution_blockers_v2_21E_full_dev_partial_exit_tagfix_20260605"),
    )
    parser.add_argument(
        "--v21e-quick-dir",
        default=str(FREQTRADE_DIR / "rolling_v2_21E_dev_partial_exit_fix_quick_20260604"),
    )
    parser.add_argument(
        "--v21e-standard-dir",
        default=str(FREQTRADE_DIR / "rolling_v2_21E_dev_partial_exit_fix_standard_20260604"),
    )
    parser.add_argument(
        "--v24c-quick-dir",
        default=str(FREQTRADE_DIR / "rolling_v2_24C_nearflat_dev_quick_20260605"),
    )
    parser.add_argument(
        "--v24c-standard-dir",
        default=str(FREQTRADE_DIR / "rolling_v2_24C_nearflat_dev_standard_20260605"),
    )
    parser.add_argument(
        "--v24c-smoke-dir",
        action="append",
        default=[
            str(FREQTRADE_DIR / "smoke_v2_24C_nearflat_20190201_20190430_binance_20260605"),
            str(FREQTRADE_DIR / "smoke_v2_24C_nearflat_1095w001_20180630_20210629_binance_20260605"),
            str(FREQTRADE_DIR / "smoke_v2_24C_nearflat_20220801_20221231_binance_20260605"),
        ],
    )
    parser.add_argument(
        "--v24d-smoke-dir",
        action="append",
        default=[
            str(FREQTRADE_DIR / "smoke_v2_24D_bull_20181227_20191227_binance_20260605"),
            str(FREQTRADE_DIR / "smoke_v2_24D_bull_20190225_20210224_binance_20260605"),
        ],
    )
    return parser.parse_args()


def read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def compare_rolling(reference_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    ref = read_csv(reference_dir / "rolling_summary.csv")
    cand = read_csv(candidate_dir / "rolling_summary.csv")
    keys = ["window_days", "step_days", "window_start", "window_end"]
    merged = cand.merge(ref, on=keys, suffixes=("_candidate", "_reference"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "window_days": row["window_days"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "candidate_return_pct": row["portfolio_return_pct_candidate"],
                "reference_return_pct": row["portfolio_return_pct_reference"],
                "return_delta_pct": row["portfolio_return_pct_candidate"] - row["portfolio_return_pct_reference"],
                "candidate_excess_pct": row["excess_return_pct_candidate"],
                "reference_excess_pct": row["excess_return_pct_reference"],
                "excess_delta_pct": row["excess_return_pct_candidate"] - row["excess_return_pct_reference"],
                "drawdown_delta_pct": row["max_drawdown_pct_candidate"] - row["max_drawdown_pct_reference"],
                "exposure_delta_pct": row["avg_exposure_pct_candidate"] - row["avg_exposure_pct_reference"],
            }
        )
    return pd.DataFrame(rows)


def build_smoke_summary(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for item in args.v24c_smoke_dir:
        path = Path(item) / "summary.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        portfolio = summary[summary["mode"] == "single_fixed_aggregate"]
        if portfolio.empty:
            continue
        row = portfolio.iloc[0]
        rows.append(
            {
                "run_dir": str(Path(item).relative_to(PROJECT_ROOT)) if Path(item).is_absolute() else item,
                "timerange": row["timerange"],
                "return_pct": row["total_return_pct"],
                "buyhold_pct": row["buyhold_total_return_pct"],
                "excess_pct": row["total_excess_pct"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "avg_exposure_pct": row["avg_exposure_pct"],
            }
        )
    return pd.DataFrame(rows)


def compare_smoke_to_standard_baseline(smoke_dirs: list[str], baseline_dir: Path) -> pd.DataFrame:
    baseline = read_csv(baseline_dir / "rolling_summary.csv")
    rows = []
    for item in smoke_dirs:
        path = Path(item) / "summary.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        portfolio = summary[summary["mode"] == "single_fixed_aggregate"]
        if portfolio.empty:
            continue
        row = portfolio.iloc[0]
        timerange = str(row["timerange"])
        start, end = timerange.split("-", 1)
        base_rows = baseline[
            (baseline["window_start"].astype(str) == start)
            & (baseline["window_end"].astype(str) == end)
        ]
        base = base_rows.iloc[0] if not base_rows.empty else pd.Series(dtype=object)
        rows.append(
            {
                "run_dir": str(Path(item).relative_to(PROJECT_ROOT)) if Path(item).is_absolute() else item,
                "timerange": timerange,
                "candidate_return_pct": row["total_return_pct"],
                "reference_return_pct": value_or_nan(base, "portfolio_return_pct"),
                "return_delta_pct": row["total_return_pct"] - value_or_nan(base, "portfolio_return_pct"),
                "candidate_excess_pct": row["total_excess_pct"],
                "reference_excess_pct": value_or_nan(base, "excess_return_pct"),
                "excess_delta_pct": row["total_excess_pct"] - value_or_nan(base, "excess_return_pct"),
                "candidate_drawdown_pct": row["max_drawdown_pct"],
                "reference_drawdown_pct": value_or_nan(base, "max_drawdown_pct"),
                "drawdown_delta_pct": row["max_drawdown_pct"] - value_or_nan(base, "max_drawdown_pct"),
            }
        )
    return pd.DataFrame(rows)


def build_direction_summary(
    *,
    target_reduce: pd.DataFrame,
    forward_capture: pd.DataFrame,
    mixed_blockers: pd.DataFrame,
    v24c_quick_delta: pd.DataFrame,
    v24c_standard_delta: pd.DataFrame,
    v24c_smoke: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            summarize_target_reduce_direction(target_reduce),
            summarize_probe_direction(mixed_blockers, v24c_quick_delta, v24c_standard_delta, v24c_smoke),
            summarize_bull_hold_direction(forward_capture, target_reduce),
        ]
    )


def summarize_target_reduce_direction(summary: pd.DataFrame) -> dict:
    high_atr = row_for(summary, "high_atr_target_reduce")
    structural = row_for(summary, "low_risk_structural_intact")
    all_row = row_for(summary, "all")
    evidence = "weak"
    recommendation = "暂不做全局 target-reduce 放宽"
    next_action = "只保留 high-ATR target-reduce 作为候选观察，不直接改策略"
    if not high_atr.empty and high_atr.get("events", 0) >= 20:
        if high_atr.get("regret_90d_rate_pct", 0) >= 70 and high_atr.get("median_ret_90d_pct", 0) > 20:
            evidence = "medium"
            recommendation = "可研究高 ATR 强势段 target-reduce 软化，但必须加 BULL/结构过滤"
            next_action = "如推进，只允许限制单次 target-reduce，不拦截 risk/trend/bear sell"
    if not structural.empty and structural.get("median_ret_60d_pct", 0) < 0:
        recommendation += "；低风险结构完整段没有稳定后悔，不能作为泛化依据"
    return {
        "direction": "target_reduce_regret",
        "evidence_strength": evidence,
        "candidate_allowed": evidence == "medium",
        "primary_signal": metric_text(high_atr, "high_atr_target_reduce", ["events", "regret_90d_rate_pct", "median_ret_90d_pct"]),
        "risk_signal": metric_text(all_row, "all_target_reduce", ["events", "regret_60d_rate_pct", "median_ret_60d_pct", "sell_helped_20d_rate_pct"]),
        "recommendation": recommendation,
        "next_action": next_action,
    }


def summarize_probe_direction(
    mixed_blockers: pd.DataFrame,
    quick_delta: pd.DataFrame,
    standard_delta: pd.DataFrame,
    smoke: pd.DataFrame,
) -> dict:
    refined = row_for(mixed_blockers, "refined:all_candidate")
    quick_mean = quick_delta["excess_delta_pct"].mean()
    standard_mean = standard_delta["excess_delta_pct"].mean()
    standard_median = standard_delta["excess_delta_pct"].median()
    worst_std = standard_delta["excess_delta_pct"].min()
    smoke_1095 = row_contains(smoke, "timerange", "20180630-20210629")
    evidence = "low"
    candidate_allowed = False
    recommendation = "不继续把 probe 放大成正式候选"
    if not refined.empty and refined.get("rows", 0) >= 10 and refined.get("future_ret_60d_positive_rate_pct", 0) >= 90:
        evidence = "medium"
    if evidence == "medium" and quick_mean > 0 and standard_mean > 0 and standard_median >= -0.1:
        candidate_allowed = False
        recommendation = "只作为受限研究方向保留；24C 的实测改善不足以替代 21E"
    return {
        "direction": "probe_vs_core_buy",
        "evidence_strength": evidence,
        "candidate_allowed": candidate_allowed,
        "primary_signal": metric_text(refined, "refined_probe", ["rows", "future_ret_60d_mean_pct", "future_ret_60d_positive_rate_pct", "worst_future_down_60d_pct"]),
        "risk_signal": (
            f"24C quick mean delta {quick_mean:.2f}pp, standard mean delta {standard_mean:.2f}pp, "
            f"standard median delta {standard_median:.2f}pp, worst standard delta {worst_std:.2f}pp; "
            f"1095 guardrail return {value_or_nan(smoke_1095, 'return_pct'):.2f}%"
        ),
        "recommendation": recommendation,
        "next_action": "若未来重开，只能从 current_pct<5%、position cap<=8%、cooldown 的 near-flat probe 开始",
    }


def summarize_bull_hold_direction(forward_capture: pd.DataFrame, target_reduce: pd.DataFrame) -> dict:
    strong = row_for(forward_capture, "strong_uptrend=True|high_atr=False")
    mixed_low = row_for(forward_capture, "confirmed_state=MIXED|position_bucket=lt35")
    high_atr_tr = row_for(target_reduce, "high_atr_target_reduce")
    evidence = "medium" if not high_atr_tr.empty and high_atr_tr.get("regret_90d_rate_pct", 0) >= 70 else "low"
    candidate_allowed = evidence == "medium"
    recommendation = (
        "优先研究 BULL/强趋势持仓维持，而不是 MIXED 全局抬仓"
        if candidate_allowed
        else "先不写候选，继续定位 strong BULL 内的减仓事件"
    )
    return {
        "direction": "bull_hold_maintenance",
        "evidence_strength": evidence,
        "candidate_allowed": candidate_allowed,
        "primary_signal": metric_text(high_atr_tr, "high_atr_target_reduce", ["events", "regret_90d_rate_pct", "median_ret_90d_pct"]),
        "risk_signal": metric_text(mixed_low, "mixed_low_position", ["rows", "missed_beta_60d_mean", "defence_value_60d_mean", "defence_to_missed_60d"]),
        "recommendation": recommendation,
        "next_action": "候选规则必须限定 confirmed BULL、trend/drawdown/risk 全为 0、BTC 非 BEAR、均线多头排列",
    }


def row_for(frame: pd.DataFrame, segment: str) -> pd.Series:
    if frame.empty or "segment" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame[frame["segment"].astype(str) == segment]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def row_contains(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame[frame[column].astype(str).str.contains(value, na=False)]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def value_or_nan(row: pd.Series, column: str) -> float:
    if row.empty or column not in row:
        return float("nan")
    return float(row[column])


def metric_text(row: pd.Series, label: str, columns: list[str]) -> str:
    if row.empty:
        return f"{label}: missing"
    parts = []
    for col in columns:
        if col in row and pd.notna(row[col]):
            value = row[col]
            if isinstance(value, float):
                parts.append(f"{col}={value:.2f}")
            else:
                parts.append(f"{col}={value}")
    return f"{label}: " + ", ".join(parts)


def render_markdown(
    *,
    args: argparse.Namespace,
    direction_summary: pd.DataFrame,
    target_reduce: pd.DataFrame,
    forward_capture: pd.DataFrame,
    mixed_blockers: pd.DataFrame,
    rolling_delta: pd.DataFrame,
    v24c_smoke: pd.DataFrame,
    v24d_trial: pd.DataFrame,
) -> str:
    quick = rolling_delta[rolling_delta["scope"] == "quick"]
    standard = rolling_delta[rolling_delta["scope"] == "standard"]
    return "\n".join(
        [
            "# 三方向研究汇总",
            "",
            "## 结论",
            "",
            direction_summary.to_markdown(index=False),
            "",
            "## 24C 反证检查",
            "",
            f"- quick mean excess delta: {quick['excess_delta_pct'].mean():.2f}pp; median: {quick['excess_delta_pct'].median():.2f}pp; worst: {quick['excess_delta_pct'].min():.2f}pp",
            f"- standard mean excess delta: {standard['excess_delta_pct'].mean():.2f}pp; median: {standard['excess_delta_pct'].median():.2f}pp; worst: {standard['excess_delta_pct'].min():.2f}pp",
            "- 解释：near-flat probe 修复了 24B 的 1095d 路径污染，但整体改善偏小，不能替代 21E。",
            "",
            "### Smoke Guardrails",
            "",
            table_or_empty(v24c_smoke),
            "",
            "## 24D 候选试验结果",
            "",
            "24D 是按本轮诊断设计的最小 BULL/high-ATR target-reduce 软化候选；两个强牛 smoke 都没有相对 21E 改善，因此已按规则淘汰，不进入 quick。",
            "",
            table_or_empty(v24d_trial),
            "",
            "## Target-Reduce 证据",
            "",
            table_or_empty(target_reduce.head(12)),
            "",
            "## Probe / 小单污染证据",
            "",
            table_or_empty(mixed_blockers),
            "",
            "## Forward Capture / BULL 持仓证据",
            "",
            table_or_empty(forward_capture.head(20)),
            "",
            "## 下一步执行建议",
            "",
            "1. 不上线 24B/24C，也不把 probe 继续放大。",
            "2. 优先从 BULL 强趋势持仓维持设计一个最小候选，规则必须只影响 target-reduce。",
            "3. 候选先跑 smoke：2018-12-27~2019-12-27、2019-02-25~2021-02-24、2018-06-30~2021-06-29、2022-08-01~2022-12-31。",
            "4. smoke 通过后才跑 quick；quick 通过后才跑 standard；validation 只观察，不调参。",
            "",
            "## Inputs",
            "",
            f"- target_reduce: `{args.target_reduce_dir}`",
            f"- forward_capture: `{args.forward_capture_dir}`",
            f"- mixed_blocker: `{args.mixed_blocker_dir}`",
            f"- v21E quick/standard: `{args.v21e_quick_dir}`, `{args.v21e_standard_dir}`",
            f"- v24C quick/standard: `{args.v24c_quick_dir}`, `{args.v24c_standard_dir}`",
            "",
        ]
    )


def table_or_empty(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data._"
    return frame.to_markdown(index=False, floatfmt=".2f")


def render_html(markdown: str) -> str:
    body = html.escape(markdown)
    body = body.replace("\n", "<br>\n")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Three Direction Research</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.5;"
        "max-width:1200px;margin:32px auto;padding:0 20px;color:#20242a}"
        "code{background:#f3f4f6;padding:2px 4px;border-radius:4px}"
        "</style></head><body>"
        f"{body}</body></html>"
    )


if __name__ == "__main__":
    main()
