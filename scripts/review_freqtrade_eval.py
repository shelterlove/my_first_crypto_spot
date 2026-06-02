#!/usr/bin/env python3
"""Score and render a compact HTML review for Freqtrade evaluations."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "results" / "freqtrade_eval"
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"

LABELS = {
    "score": "综合评分",
    "decision": "评审结论",
    "grade": "等级",
    "component": "评分项",
    "weight": "权重",
    "score_0_100": "单项得分",
    "weighted_points": "加权得分",
    "detail": "计算细节",
    "check": "检查项",
    "pass": "是否通过",
    "purpose": "用途",
    "mode": "模式",
    "pair": "币种",
    "metric": "指标",
    "candidate": "候选策略",
    "reference": "参考策略",
    "delta": "变化",
    "windows": "窗口数",
    "window_days": "窗口天数",
    "window_start": "开始日期",
    "window_end": "结束日期",
    "total_return_pct": "策略收益",
    "buyhold_total_return_pct": "买入持有收益",
    "total_excess_pct": "超额收益",
    "portfolio_return_pct": "组合收益",
    "buyhold_return_pct": "买入持有收益",
    "excess_return_pct": "超额收益",
    "mean_return_pct": "平均策略收益",
    "mean_buyhold_pct": "平均买入持有收益",
    "mean_excess_pct": "平均超额",
    "median_excess_pct": "中位超额",
    "win_rate_pct": "胜率",
    "worst_excess_pct": "最差超额",
    "best_excess_pct": "最佳超额",
    "max_drawdown_pct": "最大回撤",
    "worst_drawdown_pct": "最差回撤",
    "mean_max_drawdown_pct": "平均最大回撤",
    "avg_exposure_pct": "平均仓位",
    "mean_exposure_pct": "平均仓位",
    "trade_count": "交易次数",
    "mean_trade_count": "平均交易次数",
    "underwater_days": "水下天数",
    "cagr_pct": "年化收益",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
}

VALUE_LABELS = {
    "return_score": "收益能力",
    "robustness_score": "跨窗口稳定性",
    "risk_score": "风险控制",
    "risk_adjusted_score": "风险调整收益",
    "behavior_score": "交易行为",
    "aggregate_excess_positive": "组合全周期超额为正",
    "all_pairs_full_excess_positive": "所有单币全周期超额为正",
    "max_drawdown_not_extreme": "最大回撤未超过硬上限",
    "rolling_available": "滚动窗口数据存在",
    "rule_generality_review_passed": "规则通用性人工复核通过",
    "defense_integrity_review_passed": "防守完整性人工复核通过",
    "full_excess_not_worse_than_reference": "全周期超额不弱于参考版本",
    "max_drawdown_not_materially_worse": "最大回撤未显著恶化",
    "rolling_median_excess_not_worse": "滚动中位超额未恶化",
    "rolling_win_rate_not_worse": "滚动胜率未恶化",
    "worst_rolling_excess_not_worse": "最差滚动超额未恶化",
    "single_pair_median_not_materially_worse": "单币滚动中位表现未显著恶化",
    "trade_count_same_regime": "交易频率仍在同一量级",
    "promote_reference": "晋级为参考版本",
    "paper_trade_candidate": "可进入模拟盘候选",
    "research_only": "仅保留研究",
    "reject": "拒绝",
    "True": "通过",
    "False": "未通过",
    "single": "单币",
    "single_fixed_aggregate": "固定分仓组合",
    "PORTFOLIO": "组合",
}


def main() -> None:
    args = parse_args()
    candidate = load_eval(
        baseline_dir=Path(args.baseline_dir),
        quick_dir=optional_path(args.quick_dir),
        standard_dir=optional_path(args.standard_dir),
    )
    reference = None
    if args.reference_baseline_dir:
        reference = load_eval(
            baseline_dir=Path(args.reference_baseline_dir),
            quick_dir=optional_path(args.reference_quick_dir),
            standard_dir=optional_path(args.reference_standard_dir),
        )

    review = build_review(
        strategy=args.strategy,
        candidate=candidate,
        reference=reference,
        rule_generality_pass=args.rule_generality_pass,
        defense_integrity_pass=args.defense_integrity_pass,
    )
    output_dir = Path(args.output_dir) if args.output_dir else candidate.baseline_dir / "review"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(review["components"]).to_csv(output_dir / "score_components.csv", index=False)
    pd.DataFrame(review["checks"]).to_csv(output_dir / "promotion_checks.csv", index=False)
    (output_dir / "score.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(review, candidate, reference), encoding="utf-8")
    print(pd.DataFrame(review["components"]).to_string(index=False))
    print(f"\nscore={review['score']:.2f} grade={review['grade']} decision={review['decision']}")
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="v2_20D")
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASE / "baseline_v2_20D_20260602"))
    parser.add_argument("--quick-dir", default=str(DEFAULT_BASE / "rolling_v2_20D_quick_20260602"))
    parser.add_argument("--standard-dir", default=str(DEFAULT_BASE / "rolling_v2_20D_standard_20260602"))
    parser.add_argument("--reference-baseline-dir", default="")
    parser.add_argument("--reference-quick-dir", default="")
    parser.add_argument("--reference-standard-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rule-generality-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--defense-integrity-pass", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def optional_path(value: str) -> Path | None:
    return Path(value) if value else None


class Evaluation:
    def __init__(self, *, baseline_dir: Path, quick_dir: Path | None, standard_dir: Path | None):
        self.baseline_dir = baseline_dir
        self.quick_dir = quick_dir
        self.standard_dir = standard_dir
        self.summary = read_csv(baseline_dir / "summary.csv")
        self.trades = read_csv(baseline_dir / "trades.csv", required=False)
        self.equity = read_csv(baseline_dir / "single_fixed_portfolio_equity.csv", required=False)
        self.quick = read_csv(quick_dir / "rolling_summary.csv", required=False) if quick_dir else pd.DataFrame()
        self.quick_pairs = read_csv(quick_dir / "rolling_pair_summary.csv", required=False) if quick_dir else pd.DataFrame()
        self.standard = read_csv(standard_dir / "rolling_summary.csv", required=False) if standard_dir else pd.DataFrame()
        self.standard_pairs = read_csv(standard_dir / "rolling_pair_summary.csv", required=False) if standard_dir else pd.DataFrame()
        self.stats = equity_stats(self.equity)
        self.buyhold_stats = equity_stats(build_buyhold_equity(self))

    @property
    def aggregate(self) -> pd.Series:
        rows = self.summary[self.summary["mode"] == "single_fixed_aggregate"]
        if rows.empty:
            raise ValueError(f"No single_fixed_aggregate row in {self.baseline_dir / 'summary.csv'}")
        return rows.iloc[0]

    @property
    def pairs(self) -> pd.DataFrame:
        return self.summary[self.summary["mode"] == "single"].copy()


def load_eval(*, baseline_dir: Path, quick_dir: Path | None, standard_dir: Path | None) -> Evaluation:
    return Evaluation(
        baseline_dir=baseline_dir.resolve(),
        quick_dir=quick_dir.resolve() if quick_dir else None,
        standard_dir=standard_dir.resolve() if standard_dir else None,
    )


def read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def build_review(
    *,
    strategy: str,
    candidate: Evaluation,
    reference: Evaluation | None,
    rule_generality_pass: bool,
    defense_integrity_pass: bool,
) -> dict[str, Any]:
    components = score_components(candidate)
    score = sum(item["weighted_points"] for item in components)
    checks = promotion_checks(
        candidate,
        reference,
        rule_generality_pass=rule_generality_pass,
        defense_integrity_pass=defense_integrity_pass,
    )
    decision = decide(score, checks, reference)
    return {
        "strategy": strategy,
        "score": round(score, 4),
        "grade": grade(score),
        "decision": decision,
        "components": components,
        "checks": checks,
        "summary": summary_payload(candidate),
        "reference_summary": summary_payload(reference) if reference else None,
    }


def score_components(evaluation: Evaluation) -> list[dict[str, Any]]:
    returns, returns_detail = score_returns(evaluation)
    robustness, robustness_detail = score_robustness(evaluation)
    risk, risk_detail = score_risk(evaluation)
    adjusted, adjusted_detail = score_risk_adjusted(evaluation)
    behavior, behavior_detail = score_behavior(evaluation)
    return [
        component("return_score", 25, returns, returns_detail),
        component("robustness_score", 25, robustness, robustness_detail),
        component("risk_score", 20, risk, risk_detail),
        component("risk_adjusted_score", 20, adjusted, adjusted_detail),
        component("behavior_score", 10, behavior, behavior_detail),
    ]


def component(name: str, weight: float, value: float, detail: str = "") -> dict[str, Any]:
    value = clamp_score(value)
    return {
        "component": name,
        "weight": weight,
        "score_0_100": round(value, 2),
        "weighted_points": round(value * weight / 100, 4),
        "detail": detail,
    }


def score_returns(evaluation: Evaluation) -> tuple[float, str]:
    agg = evaluation.aggregate
    pairs = evaluation.pairs
    strat_ret = pct_to_ratio(agg["total_return_pct"])
    bh_ret = pct_to_ratio(agg["buyhold_total_return_pct"])
    excess_pp = num(agg["total_excess_pct"])
    relative = relative_return_score(strat_ret, bh_ret, excess_pp)
    excess = score_between(excess_pp, -50, 200)
    pair_positive = num((pairs["total_excess_pct"] > 0).mean() * 100) if not pairs.empty else 0.0
    min_pair_excess = num(pairs["total_excess_pct"].min()) if not pairs.empty else 0.0
    min_pair = score_between(min_pair_excess, -50, 100)
    cagr = relative_metric_score(evaluation.stats["cagr"], evaluation.buyhold_stats["cagr"], bad_ratio=0.80, good_ratio=1.20)
    score = 0.30 * excess + 0.25 * relative + 0.20 * pair_positive + 0.15 * min_pair + 0.10 * cagr
    detail = (
        f"aggregate_excess={excess_pp:.2f}pp, strategy/bh_return={safe_ratio(strat_ret, bh_ret):.2f}x, "
        f"pair_positive={pair_positive:.1f}%, min_pair_excess={min_pair_excess:.2f}pp, "
        f"strategy/bh_cagr={safe_ratio(evaluation.stats['cagr'], evaluation.buyhold_stats['cagr']):.2f}x"
    )
    return score, detail


def score_robustness(evaluation: Evaluation) -> tuple[float, str]:
    rolling = preferred_rolling(evaluation)
    pair_summary = preferred_pair_rolling(evaluation)
    if rolling.empty:
        return 0.0, "no rolling data"
    median_excess = num(rolling["excess_return_pct"].median())
    win_rate = num((rolling["excess_return_pct"] > 0).mean() * 100)
    worst_excess = num(rolling["excess_return_pct"].min())
    median_score = score_between(median_excess, -40, 10)
    win_score = score_between(win_rate, 35, 55)
    worst_score = score_between(worst_excess, -300, -100)
    pair_score = 0.0
    pair_medians = ""
    if not pair_summary.empty:
        pair_scores = [score_between(v, -40, 10) for v in pair_summary["median_excess_pct"]]
        pair_score = num(pd.Series(pair_scores).mean())
        pair_medians = ", ".join(f"{row.pair}:{row.median_excess_pct:.2f}pp" for row in pair_summary.itertuples())
    score = 0.35 * median_score + 0.25 * win_score + 0.25 * worst_score + 0.15 * pair_score
    detail = (
        f"windows={len(rolling)}, median_excess={median_excess:.2f}pp, win_rate={win_rate:.1f}%, "
        f"worst_excess={worst_excess:.2f}pp, pair_medians={pair_medians}"
    )
    return score, detail


def score_risk(evaluation: Evaluation) -> tuple[float, str]:
    agg = evaluation.aggregate
    pairs = evaluation.pairs
    rolling = preferred_rolling(evaluation)
    agg_dd = abs(num(agg["max_drawdown_pct"]))
    worst_pair_dd = abs(num(pairs["max_drawdown_pct"].min())) if not pairs.empty else agg_dd
    worst_rolling_dd = abs(num(rolling["max_drawdown_pct"].min())) if not rolling.empty else agg_dd
    underwater_days = num(agg.get("underwater_days", 0))
    bh_dd = abs(num(evaluation.buyhold_stats["max_drawdown_pct"]))
    bh_underwater = num(evaluation.buyhold_stats["underwater_days"])
    relative_dd = score_lower_better(safe_ratio(agg_dd, bh_dd), bad=1.00, good=0.60)
    relative_underwater = score_lower_better(safe_ratio(underwater_days, bh_underwater), bad=1.20, good=0.70)
    relative = 0.60 * relative_dd + 0.40 * relative_underwater
    score = (
        0.30 * score_lower_better(agg_dd, bad=70, good=40)
        + 0.20 * score_lower_better(worst_pair_dd, bad=75, good=45)
        + 0.20 * score_lower_better(worst_rolling_dd, bad=75, good=45)
        + 0.15 * score_lower_better(underwater_days, bad=1200, good=500)
        + 0.15 * relative
    )
    detail = (
        f"aggregate_dd={-agg_dd:.2f}%, worst_pair_dd={-worst_pair_dd:.2f}%, "
        f"worst_rolling_dd={-worst_rolling_dd:.2f}%, underwater_days={underwater_days:.0f}, "
        f"dd_vs_bh={safe_ratio(agg_dd, bh_dd):.2f}x, underwater_vs_bh={safe_ratio(underwater_days, bh_underwater):.2f}x"
    )
    return score, detail


def score_risk_adjusted(evaluation: Evaluation) -> tuple[float, str]:
    stats = evaluation.stats
    bh = evaluation.buyhold_stats
    absolute = (
        0.40 * score_between(stats["calmar"], 0.0, 1.20)
        + 0.35 * score_between(stats["sortino"], 0.0, 1.80)
        + 0.25 * score_between(stats["sharpe"], 0.0, 1.20)
    )
    relative = (
        0.40 * relative_metric_score(stats["calmar"], bh["calmar"], bad_ratio=0.80, good_ratio=1.20)
        + 0.35 * relative_metric_score(stats["sortino"], bh["sortino"], bad_ratio=0.80, good_ratio=1.20)
        + 0.25 * relative_metric_score(stats["sharpe"], bh["sharpe"], bad_ratio=0.80, good_ratio=1.20)
    )
    score = 0.60 * absolute + 0.40 * relative
    detail = (
        f"cagr={stats['cagr'] * 100:.2f}%, sharpe={stats['sharpe']:.2f}, "
        f"sortino={stats['sortino']:.2f}, calmar={stats['calmar']:.2f}; "
        f"bh_cagr={bh['cagr'] * 100:.2f}%, bh_sharpe={bh['sharpe']:.2f}, "
        f"bh_sortino={bh['sortino']:.2f}, bh_calmar={bh['calmar']:.2f}"
    )
    return score, detail


def score_behavior(evaluation: Evaluation) -> tuple[float, str]:
    agg = evaluation.aggregate
    pairs = evaluation.pairs
    years = timerange_years(str(agg["timerange"]))
    trade_counts = pairs["trade_count"].fillna(0) if not pairs.empty else pd.Series([0.0])
    trades_per_pair_year = num(trade_counts.mean() / years)
    exposure = num(agg["avg_exposure_pct"])
    trade_score = trade_frequency_score(trades_per_pair_year)
    exposure_score = band_score(exposure, low_bad=35, low_good=50, high_good=70, high_bad=85)
    min_trades = max(num(trade_counts.min()), 1.0)
    max_trades = num(trade_counts.max())
    balance_ratio = max_trades / min_trades
    balance_score = score_lower_better(balance_ratio, bad=4.0, good=2.0)
    score = 0.45 * trade_score + 0.35 * exposure_score + 0.20 * balance_score
    detail = (
        f"trades_per_pair_year={trades_per_pair_year:.2f}, mean_exposure={exposure:.2f}%, "
        f"pair_trade_balance={balance_ratio:.2f}x"
    )
    return score, detail


def preferred_rolling(evaluation: Evaluation) -> pd.DataFrame:
    return evaluation.standard if not evaluation.standard.empty else evaluation.quick


def preferred_pair_rolling(evaluation: Evaluation) -> pd.DataFrame:
    return evaluation.standard_pairs if not evaluation.standard_pairs.empty else evaluation.quick_pairs


def promotion_checks(
    candidate: Evaluation,
    reference: Evaluation | None,
    *,
    rule_generality_pass: bool,
    defense_integrity_pass: bool,
) -> list[dict[str, Any]]:
    agg = candidate.aggregate
    pairs = candidate.pairs
    rolling = preferred_rolling(candidate)
    checks = [
        check("aggregate_excess_positive", num(agg["total_excess_pct"]) > 0, f"aggregate_excess={num(agg['total_excess_pct']):.2f}pp"),
        check(
            "all_pairs_full_excess_positive",
            not pairs.empty and bool((pairs["total_excess_pct"] > 0).all()),
            "per_pair_excess=" + ", ".join(f"{row.pair}:{row.total_excess_pct:.2f}pp" for row in pairs.itertuples()),
        ),
        check("max_drawdown_not_extreme", num(agg["max_drawdown_pct"]) >= -70, f"aggregate_dd={num(agg['max_drawdown_pct']):.2f}%"),
        check("rolling_available", not rolling.empty, f"windows={len(rolling)}"),
        check(
            "rule_generality_review_passed",
            rule_generality_pass,
            "人工复核：规则不得包含 BTC/ETH/BNB 专属阈值或只为单一历史片段服务",
        ),
        check(
            "defense_integrity_review_passed",
            defense_integrity_pass,
            "人工复核：BEAR 退出、trend-break、risk-reduce 防守不得被削弱",
        ),
    ]
    if reference is None:
        return checks

    ref_agg = reference.aggregate
    checks.extend([
        check(
            "full_excess_not_worse_than_reference",
            num(agg["total_excess_pct"]) >= num(ref_agg["total_excess_pct"]) - 1,
            delta_detail(num(agg["total_excess_pct"]), num(ref_agg["total_excess_pct"]), "pp"),
        ),
        check(
            "max_drawdown_not_materially_worse",
            num(agg["max_drawdown_pct"]) >= num(ref_agg["max_drawdown_pct"]) - 3,
            delta_detail(num(agg["max_drawdown_pct"]), num(ref_agg["max_drawdown_pct"]), "pp"),
        ),
    ])

    cand_roll = preferred_rolling(candidate)
    ref_roll = preferred_rolling(reference)
    if not cand_roll.empty and not ref_roll.empty:
        checks.extend([
            check(
                "rolling_median_excess_not_worse",
                num(cand_roll["excess_return_pct"].median()) >= num(ref_roll["excess_return_pct"].median()) - 1,
                delta_detail(num(cand_roll["excess_return_pct"].median()), num(ref_roll["excess_return_pct"].median()), "pp"),
            ),
            check(
                "rolling_win_rate_not_worse",
                num((cand_roll["excess_return_pct"] > 0).mean() * 100) >= num((ref_roll["excess_return_pct"] > 0).mean() * 100) - 2,
                delta_detail(num((cand_roll["excess_return_pct"] > 0).mean() * 100), num((ref_roll["excess_return_pct"] > 0).mean() * 100), "pp"),
            ),
            check(
                "worst_rolling_excess_not_worse",
                num(cand_roll["excess_return_pct"].min()) >= num(ref_roll["excess_return_pct"].min()) - 5,
                delta_detail(num(cand_roll["excess_return_pct"].min()), num(ref_roll["excess_return_pct"].min()), "pp"),
            ),
        ])

    cand_pairs = preferred_pair_rolling(candidate)
    ref_pairs = preferred_pair_rolling(reference)
    if not cand_pairs.empty and not ref_pairs.empty:
        merged = cand_pairs[["pair", "median_excess_pct", "mean_trade_count"]].merge(
            ref_pairs[["pair", "median_excess_pct", "mean_trade_count"]],
            on="pair",
            suffixes=("_candidate", "_reference"),
        )
        worst_pair_delta = num((merged["median_excess_pct_candidate"] - merged["median_excess_pct_reference"]).min())
        cand_trades = num(merged["mean_trade_count_candidate"].mean())
        ref_trades = max(num(merged["mean_trade_count_reference"].mean()), 1.0)
        checks.extend([
            check(
                "single_pair_median_not_materially_worse",
                worst_pair_delta >= -10,
                f"worst_pair_median_delta={worst_pair_delta:.2f}pp",
            ),
            check(
                "trade_count_same_regime",
                cand_trades <= ref_trades * 1.5,
                f"candidate_mean_trades={cand_trades:.2f}, reference_mean_trades={ref_trades:.2f}",
            ),
        ])
    return checks


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "pass": bool(passed), "detail": detail}


def decide(score: float, checks: list[dict[str, Any]], reference: Evaluation | None) -> str:
    failed = [item for item in checks if not item["pass"]]
    if failed:
        return "research_only" if score >= 65 else "reject"
    if reference is not None and score >= 75:
        return "promote_reference"
    if score >= 80:
        return "paper_trade_candidate"
    return "research_only"


def grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 75:
        return "B+"
    if score >= 65:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def summary_payload(evaluation: Evaluation | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    agg = evaluation.aggregate
    rolling = preferred_rolling(evaluation)
    stats = evaluation.stats
    bh = evaluation.buyhold_stats
    return {
        "baseline_dir": str(evaluation.baseline_dir),
        "total_return_pct": round(num(agg["total_return_pct"]), 4),
        "buyhold_total_return_pct": round(num(agg["buyhold_total_return_pct"]), 4),
        "total_excess_pct": round(num(agg["total_excess_pct"]), 4),
        "max_drawdown_pct": round(num(agg["max_drawdown_pct"]), 4),
        "underwater_days": round(num(agg.get("underwater_days", 0)), 4),
        "avg_exposure_pct": round(num(agg["avg_exposure_pct"]), 4),
        "cagr_pct": round(stats["cagr"] * 100, 4),
        "sharpe": round(stats["sharpe"], 4),
        "sortino": round(stats["sortino"], 4),
        "calmar": round(stats["calmar"], 4),
        "buyhold_cagr_pct": round(bh["cagr"] * 100, 4),
        "buyhold_sharpe": round(bh["sharpe"], 4),
        "buyhold_sortino": round(bh["sortino"], 4),
        "buyhold_calmar": round(bh["calmar"], 4),
        "rolling_windows": int(len(rolling)) if not rolling.empty else 0,
        "rolling_median_excess_pct": round(num(rolling["excess_return_pct"].median()), 4) if not rolling.empty else None,
        "rolling_win_rate_pct": round(num((rolling["excess_return_pct"] > 0).mean() * 100), 4) if not rolling.empty else None,
        "rolling_worst_excess_pct": round(num(rolling["excess_return_pct"].min()), 4) if not rolling.empty else None,
    }


def build_buyhold_equity(evaluation: Evaluation) -> pd.DataFrame:
    if evaluation.equity.empty or evaluation.pairs.empty:
        return pd.DataFrame()
    dates = pd.DataFrame({"date": normalize_dates(evaluation.equity["date"])})
    sleeves: list[pd.Series] = []
    for row in evaluation.pairs.itertuples():
        data_path = DATA_DIR / f"{str(row.pair).replace('/', '_')}-1d.feather"
        if not data_path.exists():
            return pd.DataFrame()
        try:
            prices = pd.read_feather(data_path, columns=["date", "close"])
        except Exception:
            return pd.DataFrame()
        prices = prices.sort_values("date").copy()
        prices["date"] = normalize_dates(prices["date"])
        aligned = pd.merge_asof(dates, prices, on="date", direction="backward")
        aligned["close"] = aligned["close"].ffill().bfill()
        if aligned["close"].isna().all():
            return pd.DataFrame()
        start_close = num(aligned["close"].iloc[0])
        sleeves.append(num(row.start_wallet) * aligned["close"] / start_close)
    equity = sum(sleeves)
    return pd.DataFrame({"date": dates["date"], "equity": equity})


def equity_stats(df: pd.DataFrame) -> dict[str, float]:
    empty = {
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown_pct": 0.0,
        "underwater_days": 0.0,
    }
    if df.empty or "equity" not in df:
        return empty
    frame = df[["date", "equity"]].dropna().copy()
    if len(frame) < 3:
        return empty
    frame["date"] = normalize_dates(frame["date"])
    frame = frame.sort_values("date")
    equity = frame["equity"].astype(float)
    returns = equity.pct_change().dropna()
    years = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days / 365.25, 0.1)
    total = equity.iloc[-1] / equity.iloc[0]
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    drawdown = equity / equity.cummax() - 1
    max_dd = abs(num(drawdown.min()))
    underwater_days = num((drawdown < -1e-9).sum())
    sharpe = annualized_ratio(returns)
    sortino = annualized_ratio(returns, downside_only=True)
    calmar = cagr / max_dd if max_dd > 0 else 0.0
    return {
        "cagr": finite(cagr),
        "sharpe": finite(sharpe),
        "sortino": finite(sortino),
        "calmar": finite(calmar),
        "max_drawdown_pct": finite(-max_dd * 100),
        "underwater_days": underwater_days,
    }


def annualized_ratio(returns: pd.Series, *, downside_only: bool = False) -> float:
    if returns.empty:
        return 0.0
    denominator_source = returns[returns < 0] if downside_only else returns
    std = num(denominator_source.std())
    if std <= 0:
        return 0.0
    return finite(num(returns.mean()) / std * math.sqrt(365))


def normalize_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(None).astype("datetime64[ns]")


def relative_return_score(strategy_return: float, benchmark_return: float, excess_pp: float) -> float:
    if benchmark_return > 0:
        return score_between(safe_ratio(strategy_return, benchmark_return), 0.80, 1.20)
    return score_between(excess_pp, 0, 100)


def relative_metric_score(strategy: float, benchmark: float, *, bad_ratio: float, good_ratio: float) -> float:
    if benchmark > 0:
        return score_between(safe_ratio(strategy, benchmark), bad_ratio, good_ratio)
    return score_between((strategy - benchmark) * 100, 0, 30)


def trade_frequency_score(trades_per_pair_year: float) -> float:
    if 1 <= trades_per_pair_year <= 12:
        return 100.0
    if trades_per_pair_year < 1:
        return score_between(trades_per_pair_year, 0, 1)
    return score_lower_better(trades_per_pair_year, bad=20, good=12)


def band_score(value: float, *, low_bad: float, low_good: float, high_good: float, high_bad: float) -> float:
    if low_good <= value <= high_good:
        return 100.0
    if value < low_good:
        return score_between(value, low_bad, low_good)
    return score_lower_better(value, bad=high_bad, good=high_good)


def score_between(value: float, bad: float, good: float) -> float:
    if good == bad:
        return 100.0 if value >= good else 0.0
    return clamp_score((value - bad) / (good - bad) * 100)


def score_lower_better(value: float, *, bad: float, good: float) -> float:
    if bad == good:
        return 100.0 if value <= good else 0.0
    return clamp_score((bad - value) / (bad - good) * 100)


def pct_to_ratio(value: Any) -> float:
    return num(value) / 100


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return finite(numerator / denominator)


def num(value: Any) -> float:
    try:
        return finite(float(value))
    except (TypeError, ValueError):
        return 0.0


def finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, finite(value)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, finite(value)))


def render_html(review: dict[str, Any], candidate: Evaluation, reference: Evaluation | None) -> str:
    strategy = esc(review["strategy"])
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{strategy} Freqtrade 策略评审</title>",
        style(),
        "</head><body><main>",
        f"<header><h1>{strategy} Freqtrade 策略评审</h1><p>固定分仓单币评估 + 等权组合评估。超额收益使用百分点（pp）表示。</p></header>",
        score_cards(review),
        table_section("评分拆解", pd.DataFrame(review["components"])),
        table_section("晋级检查", pd.DataFrame(review["checks"])),
        scoring_policy_section(),
        risk_adjusted_section(review),
        table_section("全周期表现：组合与单币", format_baseline(candidate.summary)),
        rolling_section("快速滚动回测", candidate.quick, candidate.quick_pairs),
        rolling_section("标准滚动回测", candidate.standard, candidate.standard_pairs),
        worst_windows_section(preferred_rolling(candidate)),
    ]
    if reference is not None:
        sections.append(reference_delta_section(candidate, reference))
    sections.extend([
        "<section><h2>指标说明</h2><p><strong>超额收益</strong> = 策略收益 - 买入持有收益，单位是百分点（pp）。例如最差超额 -246.76 pp 表示策略比买入持有少赚 246.76 个百分点，并不是策略亏损超过 100%。</p></section>",
        "</main></body></html>",
    ])
    return "\n".join(sections)


def score_cards(review: dict[str, Any]) -> str:
    summary = review["summary"]
    cards = [
        ("综合评分", f"{review['score']:.2f}", review["grade"]),
        ("评审结论", translate_value(review["decision"]), ""),
        ("全周期超额", fmt_pp(summary["total_excess_pct"]), "百分点"),
        ("最大回撤", fmt_pct(summary["max_drawdown_pct"]), ""),
        ("Calmar", f"{summary['calmar']:.2f}", ""),
        ("滚动中位超额", fmt_pp(summary["rolling_median_excess_pct"]), "百分点"),
    ]
    body = "".join(
        f"<div class='card'><div class='label'>{esc(label)}</div><div class='value'>{esc(value)}</div><div class='hint'>{esc(hint)}</div></div>"
        for label, value, hint in cards
    )
    return f"<section class='cards'>{body}</section>"


def format_baseline(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mode", "pair", "total_return_pct", "buyhold_total_return_pct", "total_excess_pct",
        "max_drawdown_pct", "underwater_days", "avg_exposure_pct", "trade_count", "win_rate_pct",
    ]
    return df[cols].copy()


def rolling_section(title: str, rolling: pd.DataFrame, pairs: pd.DataFrame) -> str:
    if rolling.empty:
        return f"<section><h2>{esc(title)}</h2><p>No rolling data.</p></section>"
    agg = pd.DataFrame([{
        "windows": len(rolling),
        "mean_return_pct": rolling["portfolio_return_pct"].mean(),
        "mean_buyhold_pct": rolling["buyhold_return_pct"].mean(),
        "mean_excess_pct": rolling["excess_return_pct"].mean(),
        "median_excess_pct": rolling["excess_return_pct"].median(),
        "win_rate_pct": (rolling["excess_return_pct"] > 0).mean() * 100,
        "worst_excess_pct": rolling["excess_return_pct"].min(),
        "worst_drawdown_pct": rolling["max_drawdown_pct"].min(),
    }])
    return "\n".join([
        f"<section><h2>{esc(title)}</h2>",
        df_to_html(agg),
        "<h3>单币汇总</h3>",
        df_to_html(pairs) if not pairs.empty else "<p>No pair summary.</p>",
        "</section>",
    ])


def worst_windows_section(rolling: pd.DataFrame) -> str:
    if rolling.empty:
        return ""
    cols = [
        "window_days", "window_start", "window_end", "portfolio_return_pct",
        "buyhold_return_pct", "excess_return_pct", "max_drawdown_pct", "avg_exposure_pct",
    ]
    return table_section("最差滚动窗口", rolling.sort_values("excess_return_pct").head(12)[cols])


def reference_delta_section(candidate: Evaluation, reference: Evaluation) -> str:
    cand = candidate.aggregate
    ref = reference.aggregate
    rows = [{
        "metric": "total_excess_pct",
        "candidate": fmt_pp(cand["total_excess_pct"]),
        "reference": fmt_pp(ref["total_excess_pct"]),
        "delta": fmt_pp(num(cand["total_excess_pct"]) - num(ref["total_excess_pct"])),
    }, {
        "metric": "max_drawdown_pct",
        "candidate": fmt_pct(cand["max_drawdown_pct"]),
        "reference": fmt_pct(ref["max_drawdown_pct"]),
        "delta": fmt_pp(num(cand["max_drawdown_pct"]) - num(ref["max_drawdown_pct"])),
    }, {
        "metric": "calmar",
        "candidate": f"{candidate.stats['calmar']:.2f}",
        "reference": f"{reference.stats['calmar']:.2f}",
        "delta": f"{candidate.stats['calmar'] - reference.stats['calmar']:.2f}",
    }]
    if not candidate.standard.empty and not reference.standard.empty:
        rows.extend([{
            "metric": "median_excess_pct",
            "candidate": fmt_pp(candidate.standard["excess_return_pct"].median()),
            "reference": fmt_pp(reference.standard["excess_return_pct"].median()),
            "delta": fmt_pp(candidate.standard["excess_return_pct"].median() - reference.standard["excess_return_pct"].median()),
        }, {
            "metric": "worst_excess_pct",
            "candidate": fmt_pp(candidate.standard["excess_return_pct"].min()),
            "reference": fmt_pp(reference.standard["excess_return_pct"].min()),
            "delta": fmt_pp(candidate.standard["excess_return_pct"].min() - reference.standard["excess_return_pct"].min()),
        }])
    return table_section("相对参考版本变化", pd.DataFrame(rows))


def risk_adjusted_section(review: dict[str, Any]) -> str:
    summary = review["summary"]
    rows = pd.DataFrame([{
        "metric": "cagr_pct",
        "candidate": fmt_pct(summary["cagr_pct"]),
        "reference": fmt_pct(summary["buyhold_cagr_pct"]),
    }, {
        "metric": "sharpe",
        "candidate": f"{summary['sharpe']:.2f}",
        "reference": f"{summary['buyhold_sharpe']:.2f}",
    }, {
        "metric": "sortino",
        "candidate": f"{summary['sortino']:.2f}",
        "reference": f"{summary['buyhold_sortino']:.2f}",
    }, {
        "metric": "calmar",
        "candidate": f"{summary['calmar']:.2f}",
        "reference": f"{summary['buyhold_calmar']:.2f}",
    }])
    return table_section("风险调整指标：策略 vs 买入持有", rows)


def table_section(title: str, df: pd.DataFrame) -> str:
    return f"<section><h2>{esc(title)}</h2>{df_to_html(df)}</section>"


def df_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    headers = "".join(f"<th>{esc(label(str(col)))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{esc(format_value(value, str(col)))}</td>" for col, value in row.items()) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def format_value(value: Any, col: str) -> str:
    if pd.isna(value):
        return ""
    if col in {"window_start", "window_end", "window_days", "step_days", "underwater_days", "windows"}:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        if "excess" in col:
            return f"{value:.2f} pp"
        if any(token in col for token in ("pct", "rate", "drawdown", "return", "exposure")):
            return f"{value:.2f}%"
        return f"{value:.2f}"
    return translate_value(str(value))


def scoring_policy_section() -> str:
    rows = pd.DataFrame([
        {"component": "return_score", "weight": 25, "purpose": "全周期超额、相对买入持有收益、单币覆盖、最弱单币、CAGR 相对表现"},
        {"component": "robustness_score", "weight": 25, "purpose": "滚动中位超额、滚动胜率、最差滚动超额、单币滚动中位表现"},
        {"component": "risk_score", "weight": 20, "purpose": "绝对回撤、单币回撤、滚动回撤、水下天数、相对买入持有风险"},
        {"component": "risk_adjusted_score", "weight": 20, "purpose": "Sharpe、Sortino、Calmar 的绝对表现与相对买入持有表现"},
        {"component": "behavior_score", "weight": 10, "purpose": "交易频率、平均仓位、三币交易次数是否均衡"},
    ])
    return table_section("评分规则", rows)


def timerange_years(timerange: str) -> float:
    if "-" not in timerange:
        return 1.0
    start, end = timerange.split("-", 1)
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    return max((end_ts - start_ts).days / 365.25, 0.1)


def parse_date(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y%m%d")


def delta_detail(candidate: float, reference: float, unit: str) -> str:
    return f"candidate={candidate:.2f}{unit}, reference={reference:.2f}{unit}, delta={candidate - reference:.2f}{unit}"


def fmt_pct(value: Any) -> str:
    return "" if value is None else f"{num(value):.2f}%"


def fmt_pp(value: Any) -> str:
    return "" if value is None else f"{num(value):.2f} pp"


def label(value: str) -> str:
    return LABELS.get(value, value)


def translate_value(value: str) -> str:
    if value in VALUE_LABELS:
        return VALUE_LABELS[value]
    return value


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def style() -> str:
    return """
<style>
:root { color-scheme: light; --bg: #f6f7f9; --panel: #fff; --text: #17202a; --muted: #687385; --line: #dce1e8; --accent: #0f766e; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
header { margin-bottom: 18px; }
h1 { margin: 0 0 6px; font-size: 28px; }
h2 { margin: 0 0 12px; font-size: 18px; }
h3 { margin: 18px 0 8px; font-size: 14px; color: var(--muted); }
p { color: var(--muted); }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin: 14px 0; }
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; background: transparent; border: 0; padding: 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 24px; font-weight: 650; margin-top: 4px; }
.hint { color: var(--muted); font-size: 12px; min-height: 18px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 720px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; background: #fafbfc; }
td { font-variant-numeric: tabular-nums; }
@media (max-width: 760px) { main { padding: 14px; } .cards { grid-template-columns: 1fr; } }
</style>
"""


if __name__ == "__main__":
    main()
