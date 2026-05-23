"""Human-facing HTML and Markdown reports for V1 evaluations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PERCENT_HINTS = (
    "return", "cagr", "drawdown", "excess", "win_rate", "win_vs",
    "exposure", "position_pct", "capture", "underexposure", "overexposure",
    "fee_drag", "cash_drag", "var_", "cvar_", "worst_", "volatility",
    "target_actual_gap", "avg_target", "avg_actual",
)

NON_PERCENT_FIELDS = {
    "score", "final_score", "rank", "trade_count", "mean_trade_count",
    "turnover", "mean_turnover", "annualized_turnover", "count",
    "window_count", "windows", "rows", "error_count", "warning_count",
}

LABELS = {
    "strategy": "策略",
    "candidate": "候选策略",
    "symbol": "币种",
    "window_id": "窗口",
    "window_label": "窗口",
    "window_start": "窗口开始",
    "window_end": "窗口结束",
    "market_regime": "市场状态",
    "benchmark": "基准",
    "score": "评分",
    "final_score": "最终评分",
    "total_return": "策略收益",
    "strategy_return": "策略收益",
    "benchmark_return": "基准收益",
    "buy_hold_return": "Buy & Hold 收益",
    "excess_return": "超额收益",
    "mean_return": "平均收益",
    "median_return": "中位数收益",
    "mean_excess_return": "平均超额",
    "median_excess_return": "中位数超额",
    "mean_excess_vs_bh": "平均超额 vs BH",
    "median_excess_vs_bh": "中位数超额 vs BH",
    "mean_excess_vs_exposure_matched_bh": "平均超额 vs 等仓位 BH",
    "median_excess_vs_exposure_matched_bh": "中位数超额 vs 等仓位 BH",
    "stress_mean_excess_vs_bh": "压力成本平均超额 vs BH",
    "CAGR": "年化收益",
    "annualized_return": "年化收益",
    "Sharpe": "Sharpe",
    "Sortino": "Sortino",
    "Calmar": "Calmar",
    "max_drawdown": "最大回撤",
    "mean_max_drawdown": "平均最大回撤",
    "avg_exposure": "平均仓位",
    "mean_exposure": "平均仓位",
    "median_exposure": "中位数仓位",
    "turnover": "换手",
    "mean_turnover": "平均换手",
    "trade_count": "交易次数",
    "mean_trade_count": "平均交易次数",
    "fee_cost": "费用",
    "total_fee_cost": "总费用",
    "fee_drag": "费用拖累",
    "win": "胜出",
    "win_rate": "胜率",
    "win_rate_vs_bh": "胜率 vs BH",
    "win_vs_bh": "胜出 vs BH",
    "bull_median_excess_vs_bh": "牛市中位数超额 vs BH",
    "bear_median_excess_vs_bh": "熊市中位数超额 vs BH",
    "bull_underexposure_ratio": "牛市低仓位比例",
    "target_reduce_count": "target-reduce 次数",
    "risk_reduce_count": "risk-reduce 次数",
    "scenario": "压力场景",
    "return_decay_vs_base": "收益衰减 vs base",
    "pass": "通过",
    "check_name": "检查项",
    "error_count": "错误数",
    "warning_count": "警告数",
    "recommendation": "建议",
    "main_improvement": "主要改善",
    "main_regression": "主要退化",
}


def generate_evaluation_html(
    *,
    metadata: dict[str, Any],
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    active_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    score_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    full_outputs: dict[str, pd.DataFrame],
    stress_outputs: dict[str, pd.DataFrame],
    diagnostic_outputs: dict[str, pd.DataFrame],
    optimization_comparison_df: pd.DataFrame,
    mode: str,
    verdict: dict,
) -> str:
    candidate = str(metadata.get("candidate_name") or verdict.get("candidate") or "")
    candidate_summary = _candidate_summary(summary_df, candidate)
    comparison_row = _comparison_row(optimization_comparison_df, candidate)
    audit = _audit_summary(full_outputs, diagnostic_outputs)

    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>V1 标准化回测报告</title>",
        _style(),
        "</head><body><div class='container'>",
        "<h1>V1 标准化回测报告</h1>",
        _subtitle(metadata, candidate),
        _config_grid(metadata),
        _verdict_box(candidate, candidate_summary, comparison_row, audit),
        _dashboard(candidate_summary, benchmark_df, comparison_row),
        _table_section("总体排名", _overall_ranking(summary_df, optimization_comparison_df), highlight=candidate),
        _table_section("BH 对比仪表盘", _benchmark_summary(benchmark_df)),
        _table_section("5 维度评分与晋级约束", score_df),
        _table_section("按币种表现", _symbol_performance(raw_df, candidate)),
        _table_section("按窗口大小", _window_label_performance(raw_df, candidate)),
        _table_section("市场状态拆分", _regime_performance(raw_df, candidate)),
        _table_section("版本对比", _version_comparison(optimization_comparison_df)),
        _table_section("牛市弱势窗口归因", _safe_head(full_outputs.get("bull_underperformance_window_analysis.csv"), 40)),
        _table_section("候选策略滚动窗口详情", _candidate_windows(raw_df, candidate), max_rows=90),
        _table_section("最佳窗口 Top 10", _top_bottom_windows(raw_df, candidate, top=True)),
        _table_section("最差窗口 Bottom 10", _top_bottom_windows(raw_df, candidate, top=False)),
        _table_section("仓位与换手诊断", _position_diagnostics(summary_df, diagnostic_outputs)),
        _table_section("交易行为归因", _action_reason_summary(actions_df)),
        _table_section("Risk Score 归因", _safe_head(diagnostic_outputs.get("risk_score_attribution_report"), 40)),
        _table_section("Buy Blocked / Missed Opportunity", _safe_head(diagnostic_outputs.get("buy_blocked_report"), 40)),
        _table_section("Sell Too Early", _safe_head(diagnostic_outputs.get("sell_too_early_report"), 40)),
        _table_section("成本压力测试", _cost_stress_summary(stress_outputs.get("cost_stress_report.csv", pd.DataFrame()))),
        _table_section("审计摘要", audit),
        _diagnostics_block(diagnostic_outputs, full_outputs),
        "</div></body></html>",
    ]
    return "\n".join(sections)


def build_strategy_evaluation_summary(
    *,
    metadata: dict[str, Any],
    summary_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    full_outputs: dict[str, pd.DataFrame],
    stress_outputs: dict[str, pd.DataFrame],
    optimization_comparison_df: pd.DataFrame,
    verdict: dict,
) -> str:
    candidate = str(metadata.get("candidate_name") or verdict.get("candidate") or "")
    row = _candidate_summary(summary_df, candidate)
    comparison = _comparison_row(optimization_comparison_df, candidate)
    audit = _audit_summary(full_outputs, {})
    return "\n".join([
        "# 策略回测评估总结",
        "",
        "## 实验信息",
        f"- run_id: `{metadata.get('run_id')}`",
        f"- candidate: `{candidate}`",
        f"- symbols: `{', '.join(metadata.get('symbols', []))}`",
        f"- execution_mode: `{metadata.get('execution_mode')}`",
        "",
        "## 核心结果",
        f"- score: `{_fmt(row.get('score'), 'score')}`",
        f"- mean_return: `{_fmt(row.get('mean_return'), 'mean_return')}`",
        f"- median_excess_return: `{_fmt(row.get('median_excess_return'), 'median_excess_return')}`",
        f"- win_rate_vs_bh: `{_fmt(row.get('win_rate_vs_bh'), 'win_rate_vs_bh')}`",
        f"- mean_max_drawdown: `{_fmt(row.get('mean_max_drawdown'), 'mean_max_drawdown')}`",
        f"- mean_turnover: `{_fmt(row.get('mean_turnover'), 'mean_turnover')}`",
        "",
        "## 晋级判断",
        f"- recommendation: `{comparison.get('recommendation', 'review')}`",
        f"- main_improvement: `{comparison.get('main_improvement', '')}`",
        f"- main_regression: `{comparison.get('main_regression', '')}`",
        "",
        "## 审计摘要",
        *[f"- {r['check_name']}: pass={r['pass']}, error_count={r['error_count']}" for _, r in audit.iterrows()],
        "",
        "## 详细结果",
        "完整窗口、归因、诊断、压力测试和审计明细见同目录 CSV；HTML 用于人工评审和快速定位问题。",
    ])


def _subtitle(metadata: dict[str, Any], candidate: str) -> str:
    ts = metadata.get("timestamp", "")
    return f"<p class='subtitle'>生成时间: {_esc(ts)} | 候选策略: <strong>{_esc(candidate)}</strong></p>"


def _config_grid(metadata: dict[str, Any]) -> str:
    windows = metadata.get("rolling_window_config") or []
    window_names = ", ".join(str(w.get("name", w)) for w in windows) if isinstance(windows, list) else str(windows)
    items = [
        ("币种", ", ".join(metadata.get("symbols", []))),
        ("窗口", window_names),
        ("资金", f"{metadata.get('initial_cash')} + reserve {metadata.get('reserve')}"),
        ("费率", _fmt(metadata.get("fee_rate"), "fee_rate")),
        ("执行", metadata.get("execution_mode")),
        ("周期", metadata.get("timeframe")),
        ("warmup", metadata.get("warmup_bars")),
        ("run_id", metadata.get("run_id")),
    ]
    cards = "".join(
        f"<div class='config-item'><div class='label'>{_esc(k)}</div><div class='value'>{_esc(v)}</div></div>"
        for k, v in items
    )
    return f"<div class='config-grid'>{cards}</div>"


def _verdict_box(candidate: str, summary: dict[str, Any], comparison: dict[str, Any], audit: pd.DataFrame) -> str:
    recommendation = str(comparison.get("recommendation") or "review")
    passed = audit["error_count"].sum() == 0 if not audit.empty else True
    promote = recommendation in {"promote", "promote_candidate", "baseline"} and passed
    css = "pass" if promote or passed else "fail"
    title = "建议晋级" if promote else "暂不建议晋级" if recommendation == "do_not_promote_yet" else "需要复核"
    stats = [
        ("候选评分", _fmt(summary.get("score"), "score")),
        ("平均收益", _fmt(summary.get("mean_return"), "mean_return")),
        ("中位数超额", _fmt(summary.get("median_excess_return"), "median_excess_return")),
        ("胜率 vs BH", _fmt(summary.get("win_rate_vs_bh"), "win_rate_vs_bh")),
        ("平均回撤", _fmt(summary.get("mean_max_drawdown"), "mean_max_drawdown")),
        ("交易次数", _fmt(summary.get("mean_trade_count"), "mean_trade_count")),
    ]
    stat_html = "".join(f"<span class='verdict-stat'><strong>{_esc(v)}</strong> {_esc(k)}</span>" for k, v in stats)
    notes = " | ".join(filter(None, [comparison.get("main_improvement", ""), comparison.get("main_regression", "")]))
    return (
        f"<div class='verdict {css}'><h3>{_esc(title)}：<strong>{_esc(candidate)}</strong></h3>"
        f"{stat_html}<div class='verdict-note'>{_esc(notes or recommendation)}</div></div>"
    )


def _dashboard(summary: dict[str, Any], benchmark_df: pd.DataFrame, comparison: dict[str, Any]) -> str:
    bh = _benchmark_lookup(benchmark_df, "buy_hold")
    exposure_bh = _benchmark_lookup(benchmark_df, "exposure_matched_buy_hold")
    cards = [
        ("收益保留/平均收益", _fmt(summary.get("mean_return"), "mean_return"), "所有滚动窗口均值", "neutral"),
        ("平均超额 vs BH", _fmt(summary.get("mean_excess_return"), "mean_excess_return"), "策略收益 - BH 收益", _tone(summary.get("mean_excess_return"))),
        ("超额 vs 等仓位 BH", _fmt(exposure_bh.get("mean_excess_return"), "mean_excess_return"), "控制平均仓位后的超额", _tone(exposure_bh.get("mean_excess_return"))),
        ("窗口胜率 vs BH", _fmt(summary.get("win_rate_vs_bh"), "win_rate_vs_bh"), "跑赢 BH 的窗口占比", _tone((summary.get("win_rate_vs_bh") or 0) - 0.5)),
        ("平均最大回撤", _fmt(summary.get("mean_max_drawdown"), "mean_max_drawdown"), "越接近 0 越好", _tone(-(abs(summary.get("mean_max_drawdown") or 0) - 0.4))),
        ("压力成本超额", _fmt(comparison.get("stress_mean_excess_vs_bh"), "stress_mean_excess_vs_bh"), "stress 场景均值", _tone(comparison.get("stress_mean_excess_vs_bh"))),
    ]
    return "<div class='dashboard'>" + "".join(_card(*card) for card in cards) + "</div>"


def _card(label: str, value: str, sub: str, tone: str) -> str:
    return (
        f"<div class='card'><div class='card-label'>{_esc(label)}</div>"
        f"<div class='card-value {tone}'>{_esc(value)}</div><div class='card-sub'>{_esc(sub)}</div></div>"
    )


def _overall_ranking(summary_df: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy", "score", "mean_return", "median_return", "mean_excess_return",
        "median_excess_return", "win_rate_vs_bh", "mean_max_drawdown",
        "mean_sharpe", "mean_sortino", "mean_calmar", "mean_trade_count",
    ]
    out = summary_df[[c for c in cols if c in summary_df.columns]].copy()
    if not comparison.empty and "candidate" in comparison.columns:
        extra_cols = ["candidate", "mean_excess_vs_exposure_matched_bh", "bull_median_excess_vs_bh", "bear_median_excess_vs_bh", "recommendation"]
        extra = comparison[[c for c in extra_cols if c in comparison.columns]].rename(columns={"candidate": "strategy"})
        out = out.merge(extra, on="strategy", how="left")
    return out.sort_values("score", ascending=False, na_position="last") if "score" in out else out


def _benchmark_summary(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()
    return benchmark_df.groupby("benchmark", dropna=False).agg(
        windows=("window_id", "count"),
        mean_strategy_return=("strategy_return", "mean"),
        mean_benchmark_return=("benchmark_return", "mean"),
        mean_excess_return=("excess_return", "mean"),
        median_excess_return=("excess_return", "median"),
        win_rate=("win", "mean"),
    ).reset_index()


def _symbol_performance(raw_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    df = _candidate_raw(raw_df, candidate)
    if df.empty:
        return pd.DataFrame()
    return df.groupby("symbol", dropna=False).agg(
        window_count=("window_label", "count"),
        mean_return=("total_return", "mean"),
        median_return=("total_return", "median"),
        buy_hold_return=("buy_hold_return", "mean"),
        mean_excess_return=("excess_return", "mean"),
        win_rate_vs_bh=("win_vs_bh", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        avg_exposure=("avg_exposure", "mean"),
        turnover=("turnover", "mean"),
        trade_count=("trade_count", "mean"),
    ).reset_index()


def _window_label_performance(raw_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    df = _candidate_raw(raw_df, candidate)
    if df.empty:
        return pd.DataFrame()
    return df.groupby("window_label", dropna=False).agg(
        window_count=("symbol", "count"),
        mean_return=("total_return", "mean"),
        median_return=("total_return", "median"),
        mean_excess_return=("excess_return", "mean"),
        win_rate_vs_bh=("win_vs_bh", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        avg_exposure=("avg_exposure", "mean"),
        turnover=("turnover", "mean"),
        trade_count=("trade_count", "mean"),
        cagr=("cagr", "mean"),
    ).reset_index()


def _regime_performance(raw_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    df = _candidate_raw(raw_df, candidate)
    if df.empty or "market_regime" not in df:
        return pd.DataFrame()
    return df.groupby("market_regime", dropna=False).agg(
        window_count=("window_label", "count"),
        mean_return=("total_return", "mean"),
        median_return=("total_return", "median"),
        mean_excess_return=("excess_return", "mean"),
        median_excess_return=("excess_return", "median"),
        win_rate_vs_bh=("win_vs_bh", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        avg_exposure=("avg_exposure", "mean"),
        trade_count=("trade_count", "mean"),
    ).reset_index()


def _version_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate", "score", "mean_return", "mean_excess_vs_bh",
        "median_excess_vs_bh", "mean_excess_vs_exposure_matched_bh",
        "bull_median_excess_vs_bh", "bear_median_excess_vs_bh",
        "max_drawdown", "turnover", "trade_count", "stress_mean_excess_vs_bh",
        "recommendation",
    ]
    return comparison[[c for c in cols if c in comparison.columns]].copy() if not comparison.empty else pd.DataFrame()


def _candidate_windows(raw_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    df = _candidate_raw(raw_df, candidate)
    cols = [
        "symbol", "window_label", "window_start", "window_end", "total_return",
        "buy_hold_return", "excess_return", "max_drawdown", "sharpe",
        "sortino", "cagr", "trade_count", "avg_exposure", "market_regime",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def _top_bottom_windows(raw_df: pd.DataFrame, candidate: str, *, top: bool) -> pd.DataFrame:
    df = _candidate_windows(raw_df, candidate)
    if df.empty or "excess_return" not in df:
        return pd.DataFrame()
    return df.sort_values("excess_return", ascending=not top).head(10)


def _position_diagnostics(summary_df: pd.DataFrame, diagnostic_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_cols = [c for c in ["strategy", "mean_exposure", "mean_turnover", "mean_trade_count"] if c in summary_df.columns]
    base = summary_df[base_cols].copy() if base_cols else pd.DataFrame()
    exposure = diagnostic_outputs.get("exposure_diagnostics_report", pd.DataFrame())
    if exposure.empty:
        return base
    diag = exposure.groupby("symbol", dropna=False).agg(
        avg_actual_position_pct=("avg_actual_position_pct", "mean"),
        avg_target_position_pct_final=("avg_target_position_pct_final", "mean"),
        target_actual_gap_mean=("target_actual_gap_mean", "mean"),
        underexposed_bar_ratio=("underexposed_bar_ratio", "mean"),
        bull_underexposure_ratio=("bull_underexposure_ratio", "mean"),
        bear_overexposure_ratio=("bear_overexposure_ratio", "mean"),
    ).reset_index()
    return pd.concat([base, diag], ignore_index=True, sort=False)


def _action_reason_summary(actions_df: pd.DataFrame) -> pd.DataFrame:
    if actions_df.empty or "reason" not in actions_df:
        return pd.DataFrame()
    df = actions_df.copy()
    df["action_reason"] = df["reason"].fillna("")
    return df.groupby(["side", "action_reason"], dropna=False).agg(
        count=("action_reason", "count"),
        avg_trade_notional=("notional", "mean"),
        total_fee_cost=("fee", "sum"),
    ).reset_index().sort_values("count", ascending=False).head(40)


def _cost_stress_summary(stress: pd.DataFrame) -> pd.DataFrame:
    if stress.empty:
        return pd.DataFrame()
    return stress.groupby("scenario", dropna=False).agg(
        window_count=("window_id", "count"),
        mean_return=("total_return", "mean"),
        mean_excess_vs_bh=("excess_return_vs_bh", "mean"),
        mean_excess_vs_exposure_matched_bh=("excess_return_vs_exposure_matched_bh", "mean"),
        win_vs_bh=("win_vs_bh", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        fee_cost=("fee_cost", "mean"),
        return_decay_vs_base=("return_decay_vs_base", "mean"),
    ).reset_index()


def _audit_summary(full_outputs: dict[str, pd.DataFrame], diagnostic_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    timestamp = full_outputs.get("timestamp_audit_report.csv", pd.DataFrame())
    accounting = full_outputs.get("accounting_audit_report.csv", pd.DataFrame())
    quality = diagnostic_outputs.get("diagnostic_quality_report", pd.DataFrame())
    rows = [
        _audit_row("timestamp_audit", timestamp.get("timestamp_check_pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
        _audit_row("accounting_audit", accounting.get("pass", pd.Series(dtype=bool)).fillna(False).eq(False).sum()),
    ]
    if not quality.empty:
        errors = pd.to_numeric(quality.get("error_count", 0), errors="coerce").fillna(0).sum()
        warnings = pd.to_numeric(quality.get("warning_count", 0), errors="coerce").fillna(0).sum()
        rows.append({"check_name": "diagnostic_quality", "pass": errors == 0, "error_count": int(errors), "warning_count": int(warnings)})
    return pd.DataFrame(rows)


def _audit_row(name: str, error_count: int | float) -> dict[str, Any]:
    count = int(error_count)
    return {"check_name": name, "pass": count == 0, "error_count": count, "warning_count": 0}


def _diagnostics_block(diagnostic_outputs: dict[str, pd.DataFrame], full_outputs: dict[str, pd.DataFrame]) -> str:
    if not diagnostic_outputs:
        return "<h2>策略诊断</h2><p class='muted'>本次未生成 diagnostics。</p>"
    quality = diagnostic_outputs.get("diagnostic_quality_report", pd.DataFrame())
    summary = diagnostic_outputs.get("diagnostic_summary", pd.DataFrame())
    exposure = diagnostic_outputs.get("exposure_diagnostics_report", pd.DataFrame())
    state = full_outputs.get("state_transition_report.csv", pd.DataFrame())
    warnings = quality[(quality["pass"] == False) | (quality["warning_count"] > 0)] if not quality.empty else pd.DataFrame()
    return "\n".join([
        "<h2>策略诊断摘要</h2>",
        _table_html(quality, max_rows=20),
        "<h2>Per-Bar 状态覆盖</h2>",
        _table_html(summary, max_rows=30),
        "<h2>目标仓位 vs 实际仓位</h2>",
        _table_html(exposure, max_rows=30),
        "<h2>状态转移预测力</h2>",
        _table_html(state, max_rows=30),
        "<h2>关键诊断警告</h2>",
        _table_html(warnings, max_rows=20),
    ])


def _table_section(title: str, frame: pd.DataFrame, *, max_rows: int = 80, highlight: str | None = None) -> str:
    return f"<h2>{_esc(title)}</h2>\n{_table_html(frame, max_rows=max_rows, highlight=highlight)}"


def _table_html(frame: pd.DataFrame | None, *, max_rows: int = 80, highlight: str | None = None) -> str:
    if frame is None or frame.empty:
        return "<p class='muted'>无数据。</p>"
    limited = frame.head(max_rows)
    headers = "".join(f"<th>{_esc(_label(c))}</th>" for c in limited.columns)
    rows = []
    for _, row in limited.iterrows():
        first_value = str(row.iloc[0]) if len(row) else ""
        row_class = "candidate" if highlight and first_value == highlight else ""
        cells = "".join(f"<td>{_format_with_tone(row[col], col)}</td>" for col in limited.columns)
        rows.append(f"<tr class='{row_class}'>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _format_with_tone(value: Any, column: str) -> str:
    text = _esc(_fmt(value, column))
    if not _is_numeric(value):
        return text
    c = column.lower()
    if any(k in c for k in ["excess", "return", "sharpe", "sortino", "calmar", "score", "win_rate"]):
        number = float(value)
        cls = "up" if number > 0 else "down" if number < 0 else ""
        return f"<span class='{cls}'>{text}</span>" if cls else text
    if "drawdown" in c:
        return f"<span class='down'>{text}</span>"
    return text


def _candidate_summary(summary_df: pd.DataFrame, candidate: str) -> dict[str, Any]:
    if summary_df.empty or "strategy" not in summary_df:
        return {}
    rows = summary_df[summary_df["strategy"] == candidate]
    return {} if rows.empty else rows.iloc[0].to_dict()


def _comparison_row(comparison: pd.DataFrame, candidate: str) -> dict[str, Any]:
    if comparison.empty or "candidate" not in comparison:
        return {}
    rows = comparison[comparison["candidate"] == candidate]
    if rows.empty:
        rows = comparison[comparison.get("recommendation", pd.Series(dtype=str)) != "baseline"]
    return {} if rows.empty else rows.iloc[-1].to_dict()


def _benchmark_lookup(benchmark_df: pd.DataFrame, name: str) -> dict[str, Any]:
    summary = _benchmark_summary(benchmark_df)
    rows = summary[summary["benchmark"] == name] if not summary.empty else pd.DataFrame()
    return {} if rows.empty else rows.iloc[0].to_dict()


def _candidate_raw(raw_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if raw_df.empty or "strategy_name" not in raw_df:
        return pd.DataFrame()
    return raw_df[raw_df["strategy_name"] == candidate].copy()


def _safe_head(frame: pd.DataFrame | None, rows: int) -> pd.DataFrame:
    return pd.DataFrame() if frame is None else frame.head(rows)


def _label(column: str) -> str:
    label = LABELS.get(column, column)
    return f"{label} ({column})" if label != column else column


def _fmt(value: Any, column: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if not _is_numeric(value):
        return str(value)
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return ""
    if column and _is_percent_column(column):
        return f"{number * 100:.1f}%"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:.4f}"


def _is_percent_column(column: str) -> bool:
    c = column.lower()
    if c in NON_PERCENT_FIELDS:
        return False
    if c == "fee_rate":
        return True
    return any(token in c for token in PERCENT_HINTS)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool)


def _tone(value: Any) -> str:
    if not _is_numeric(value) or math.isnan(float(value)):
        return "neutral"
    return "up" if float(value) > 0 else "down" if float(value) < 0 else "neutral"


def _esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _style() -> str:
    return """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "Segoe UI", sans-serif;
       background: #f1f5f9; color: #1e293b; padding: 2rem; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.65rem; margin-bottom: .25rem; }
h2 { font-size: 1.18rem; margin: 2rem 0 .75rem; padding-bottom: .4rem; border-bottom: 2px solid #e2e8f0; }
.subtitle { color: #64748b; margin-bottom: 1.5rem; font-size: .9rem; }
.table-wrap { overflow-x: auto; margin: .75rem 0 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: .5rem .75rem; text-align: right; font-size: .84rem; white-space: nowrap; }
th { background: #f8fafc; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }
td { border-bottom: 1px solid #f1f5f9; }
tr:last-child td { border-bottom: none; }
th:first-child, td:first-child { text-align: left; }
tr:hover td { background: #f8fafc; }
tr.candidate td { background: #eff6ff; }
.muted { color: #94a3b8; font-style: italic; }
.up { color: #16a34a; font-weight: 600; }
.down { color: #dc2626; font-weight: 600; }
.neutral { color: #334155; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .75rem; margin: .75rem 0 1.5rem; }
.config-item { background: #fff; padding: .75rem 1rem; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,.06); font-size: .85rem; }
.config-item .label { color: #64748b; margin-bottom: .2rem; }
.config-item .value { font-weight: 600; overflow-wrap: anywhere; }
.verdict { padding: 1.25rem 1.5rem; border-radius: 10px; margin: 1rem 0 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,.08); }
.verdict.pass { background: #f0fdf4; border: 1px solid #86efac; }
.verdict.fail { background: #fef2f2; border: 1px solid #fca5a5; }
.verdict h3 { font-size: 1.1rem; margin-bottom: .65rem; }
.verdict-stat { display: inline-block; margin-right: 1.35rem; margin-bottom: .35rem; font-size: .85rem; }
.verdict-stat strong { font-size: 1.05rem; }
.verdict-note { margin-top: .45rem; color: #475569; font-size: .85rem; }
.dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .75rem; margin: .75rem 0 1.5rem; }
.card { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.06); text-align: center; }
.card-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }
.card-value { font-size: 1.35rem; font-weight: 700; margin-top: .25rem; }
.card-sub { font-size: .75rem; color: #94a3b8; margin-top: .15rem; }
</style>
"""
