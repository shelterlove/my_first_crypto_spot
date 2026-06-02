#!/usr/bin/env python3
"""Score and render a compact HTML review for Freqtrade evaluations."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "results" / "freqtrade_eval"


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
        logic_score=args.logic_score,
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
    parser.add_argument("--logic-score", type=float, default=85.0)
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
        self.quick = read_csv(quick_dir / "rolling_summary.csv", required=False) if quick_dir else pd.DataFrame()
        self.quick_pairs = read_csv(quick_dir / "rolling_pair_summary.csv", required=False) if quick_dir else pd.DataFrame()
        self.standard = read_csv(standard_dir / "rolling_summary.csv", required=False) if standard_dir else pd.DataFrame()
        self.standard_pairs = read_csv(standard_dir / "rolling_pair_summary.csv", required=False) if standard_dir else pd.DataFrame()

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
    logic_score: float,
) -> dict[str, Any]:
    components = score_components(candidate, logic_score)
    score = sum(item["weighted_points"] for item in components)
    checks = promotion_checks(candidate, reference)
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


def score_components(evaluation: Evaluation, logic_score: float) -> list[dict[str, Any]]:
    long_term, long_term_detail = score_long_term(evaluation)
    rolling, rolling_detail = score_rolling(evaluation)
    risk, risk_detail = score_risk(evaluation)
    trade, trade_detail = score_trade_quality(evaluation)
    components = [
        component("long_term_excess", 25, long_term, long_term_detail),
        component("rolling_stability", 30, rolling, rolling_detail),
        component("risk_control", 25, risk, risk_detail),
        component("trade_quality", 10, trade, trade_detail),
        component("logic_consistency", 10, clamp(logic_score, 0, 100), "manual review score"),
    ]
    return components


def component(name: str, weight: float, value: float, detail: str = "") -> dict[str, Any]:
    value = clamp(value, 0, 100)
    return {
        "component": name,
        "weight": weight,
        "score_0_100": round(value, 2),
        "weighted_points": round(value * weight / 100, 4),
        "detail": detail,
    }


def score_long_term(evaluation: Evaluation) -> tuple[float, str]:
    agg = evaluation.aggregate
    pairs = evaluation.pairs
    excess_ratio = float(agg["total_excess_pct"]) / max(abs(float(agg["buyhold_total_return_pct"])), 100.0)
    aggregate_score = clamp(excess_ratio / 0.50 * 100, 0, 100)
    pair_positive = (pairs["total_excess_pct"] > 0).mean() * 100 if not pairs.empty else 0.0
    min_pair_excess = float(pairs["total_excess_pct"].min()) if not pairs.empty else 0.0
    min_pair_score = clamp(min_pair_excess / 100 * 100, 0, 100)
    score = 0.55 * aggregate_score + 0.25 * pair_positive + 0.20 * min_pair_score
    detail = (
        f"aggregate_excess={float(agg['total_excess_pct']):.2f}pp, "
        f"excess_vs_bh={excess_ratio:.2f}x, pair_positive={pair_positive:.1f}%, "
        f"min_pair_excess={min_pair_excess:.2f}pp"
    )
    return score, detail


def score_rolling(evaluation: Evaluation) -> tuple[float, str]:
    rolling = evaluation.standard if not evaluation.standard.empty else evaluation.quick
    pair_summary = evaluation.standard_pairs if not evaluation.standard_pairs.empty else evaluation.quick_pairs
    if rolling.empty:
        return 0.0, "no rolling data"
    median_excess = float(rolling["excess_return_pct"].median())
    win_rate = float((rolling["excess_return_pct"] > 0).mean() * 100)
    worst_excess = float(rolling["excess_return_pct"].min())
    median_score = clamp((median_excess + 50) / 50 * 100, 0, 100)
    win_score = clamp(win_rate / 55 * 100, 0, 100)
    worst_score = clamp((worst_excess + 300) / 300 * 100, 0, 100)
    if pair_summary.empty:
        pair_score = 50.0
        pair_detail = "no pair rolling summary"
    else:
        pair_score = float(pair_summary["median_excess_pct"].apply(lambda x: clamp((x + 40) / 60 * 100, 0, 100)).mean())
        pair_detail = "pair_medians=" + ",".join(
            f"{row['pair']}:{float(row['median_excess_pct']):.2f}pp"
            for _, row in pair_summary.iterrows()
        )
    score = 0.35 * median_score + 0.25 * win_score + 0.25 * worst_score + 0.15 * pair_score
    detail = (
        f"windows={len(rolling)}, median_excess={median_excess:.2f}pp, "
        f"win_rate={win_rate:.1f}%, worst_excess={worst_excess:.2f}pp, {pair_detail}"
    )
    return score, detail


def score_risk(evaluation: Evaluation) -> tuple[float, str]:
    agg = evaluation.aggregate
    pairs = evaluation.pairs
    rolling = evaluation.standard if not evaluation.standard.empty else evaluation.quick
    agg_dd = abs(float(agg["max_drawdown_pct"]))
    pair_worst_dd = abs(float(pairs["max_drawdown_pct"].min())) if not pairs.empty else agg_dd
    rolling_worst_dd = abs(float(rolling["max_drawdown_pct"].min())) if not rolling.empty else agg_dd
    dd_score = clamp((75 - agg_dd) / 35 * 100, 0, 100)
    pair_score = clamp((75 - pair_worst_dd) / 35 * 100, 0, 100)
    rolling_score = clamp((75 - rolling_worst_dd) / 35 * 100, 0, 100)
    score = 0.45 * dd_score + 0.30 * pair_score + 0.25 * rolling_score
    detail = (
        f"aggregate_dd=-{agg_dd:.2f}%, worst_pair_dd=-{pair_worst_dd:.2f}%, "
        f"worst_rolling_dd=-{rolling_worst_dd:.2f}%"
    )
    return score, detail


def score_trade_quality(evaluation: Evaluation) -> tuple[float, str]:
    pairs = evaluation.pairs
    if pairs.empty:
        return 50.0, "no pair rows"
    timerange = str(pairs.iloc[0].get("timerange", ""))
    years = max(timerange_years(timerange), 1.0)
    trades_per_pair_year = float(pairs["trade_count"].fillna(0).mean()) / years
    trade_score = 100 if 1 <= trades_per_pair_year <= 12 else clamp(100 - abs(trades_per_pair_year - 8) * 6, 0, 100)
    exposure = float(pairs["avg_exposure_pct"].mean())
    exposure_score = clamp(100 - abs(exposure - 60) * 2, 0, 100)
    score = 0.65 * trade_score + 0.35 * exposure_score
    detail = f"trades_per_pair_year={trades_per_pair_year:.2f}, mean_exposure={exposure:.2f}%"
    return score, detail


def promotion_checks(candidate: Evaluation, reference: Evaluation | None) -> list[dict[str, Any]]:
    checks = intrinsic_checks(candidate)
    if reference is None:
        return checks
    checks.extend(reference_checks(candidate, reference))
    return checks


def intrinsic_checks(evaluation: Evaluation) -> list[dict[str, Any]]:
    agg = evaluation.aggregate
    rolling = evaluation.standard if not evaluation.standard.empty else evaluation.quick
    pairs = evaluation.pairs
    return [
        check("aggregate_excess_positive", float(agg["total_excess_pct"]) > 0, f"{float(agg['total_excess_pct']):.2f} percentage points"),
        check("all_pairs_full_excess_positive", bool((pairs["total_excess_pct"] > 0).all()), ""),
        check("max_drawdown_not_extreme", float(agg["max_drawdown_pct"]) >= -65, f"{float(agg['max_drawdown_pct']):.2f}%"),
        check("rolling_available", not rolling.empty, f"{len(rolling)} windows"),
    ]


def reference_checks(candidate: Evaluation, reference: Evaluation) -> list[dict[str, Any]]:
    cand_agg = candidate.aggregate
    ref_agg = reference.aggregate
    cand_roll = candidate.standard if not candidate.standard.empty else candidate.quick
    ref_roll = reference.standard if not reference.standard.empty else reference.quick
    checks = [
        check(
            "full_excess_not_worse_than_reference",
            float(cand_agg["total_excess_pct"]) >= float(ref_agg["total_excess_pct"]) - 1,
            delta_detail(float(cand_agg["total_excess_pct"]), float(ref_agg["total_excess_pct"]), "pp"),
        ),
        check(
            "max_drawdown_not_materially_worse",
            float(cand_agg["max_drawdown_pct"]) >= float(ref_agg["max_drawdown_pct"]) - 3,
            delta_detail(float(cand_agg["max_drawdown_pct"]), float(ref_agg["max_drawdown_pct"]), "pp"),
        ),
    ]
    if not cand_roll.empty and not ref_roll.empty:
        checks.extend([
            check(
                "rolling_median_excess_not_worse",
                float(cand_roll["excess_return_pct"].median()) >= float(ref_roll["excess_return_pct"].median()) - 1,
                delta_detail(float(cand_roll["excess_return_pct"].median()), float(ref_roll["excess_return_pct"].median()), "pp"),
            ),
            check(
                "rolling_win_rate_not_worse",
                (cand_roll["excess_return_pct"] > 0).mean() * 100 >= (ref_roll["excess_return_pct"] > 0).mean() * 100 - 2,
                delta_detail((cand_roll["excess_return_pct"] > 0).mean() * 100, (ref_roll["excess_return_pct"] > 0).mean() * 100, "pp"),
            ),
            check(
                "worst_rolling_excess_not_worse",
                float(cand_roll["excess_return_pct"].min()) >= float(ref_roll["excess_return_pct"].min()) - 1,
                delta_detail(float(cand_roll["excess_return_pct"].min()), float(ref_roll["excess_return_pct"].min()), "pp"),
            ),
        ])
    return checks


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "pass": bool(passed), "detail": detail}


def decide(score: float, checks: list[dict[str, Any]], reference: Evaluation | None) -> str:
    failed = [item for item in checks if not item["pass"]]
    if failed:
        return "research_only" if score >= 65 else "reject"
    if reference is not None and score >= 70:
        return "promote_reference"
    if score >= 75:
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
    rolling = evaluation.standard if not evaluation.standard.empty else evaluation.quick
    return {
        "baseline_dir": str(evaluation.baseline_dir),
        "total_return_pct": round(float(agg["total_return_pct"]), 4),
        "buyhold_total_return_pct": round(float(agg["buyhold_total_return_pct"]), 4),
        "total_excess_pct": round(float(agg["total_excess_pct"]), 4),
        "max_drawdown_pct": round(float(agg["max_drawdown_pct"]), 4),
        "avg_exposure_pct": round(float(agg["avg_exposure_pct"]), 4),
        "rolling_windows": int(len(rolling)) if not rolling.empty else 0,
        "rolling_median_excess_pct": round(float(rolling["excess_return_pct"].median()), 4) if not rolling.empty else None,
        "rolling_win_rate_pct": round(float((rolling["excess_return_pct"] > 0).mean() * 100), 4) if not rolling.empty else None,
        "rolling_worst_excess_pct": round(float(rolling["excess_return_pct"].min()), 4) if not rolling.empty else None,
    }


def render_html(review: dict[str, Any], candidate: Evaluation, reference: Evaluation | None) -> str:
    strategy = esc(review["strategy"])
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{strategy} Freqtrade Review</title>",
        style(),
        "</head><body><main>",
        f"<header><h1>{strategy} Freqtrade Review</h1><p>Fixed-allocation per-pair and aggregate review.</p></header>",
        score_cards(review),
        table_section("Score Components", pd.DataFrame(review["components"])),
        table_section("Promotion Checks", pd.DataFrame(review["checks"])),
        scoring_policy_section(),
        table_section("Baseline: Aggregate And Pairs", format_baseline(candidate.summary)),
        rolling_section("Quick Rolling", candidate.quick, candidate.quick_pairs),
        rolling_section("Standard Rolling", candidate.standard, candidate.standard_pairs),
        worst_windows_section(candidate.standard if not candidate.standard.empty else candidate.quick),
    ]
    if reference is not None:
        sections.append(reference_delta_section(candidate, reference))
    sections.extend([
        "<section><h2>Metric Notes</h2><p><strong>Excess return</strong> means strategy return minus Buy & Hold return. A value below -100% means the strategy lagged Buy & Hold by more than 100 percentage points; it does not mean the strategy lost more than 100%.</p></section>",
        "</main></body></html>",
    ])
    return "\n".join(sections)


def score_cards(review: dict[str, Any]) -> str:
    summary = review["summary"]
    cards = [
        ("Score", f"{review['score']:.2f}", review["grade"]),
        ("Decision", review["decision"], ""),
        ("Full Excess", fmt_pp(summary["total_excess_pct"]), "percentage points"),
        ("Max Drawdown", fmt_pct(summary["max_drawdown_pct"]), ""),
        ("Rolling Median Excess", fmt_pp(summary["rolling_median_excess_pct"]), "percentage points"),
        ("Worst Rolling Excess", fmt_pp(summary["rolling_worst_excess_pct"]), "percentage points"),
    ]
    body = "".join(
        f"<div class='card'><div class='label'>{esc(label)}</div><div class='value'>{esc(value)}</div><div class='hint'>{esc(hint)}</div></div>"
        for label, value, hint in cards
    )
    return f"<section class='cards'>{body}</section>"


def format_baseline(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mode", "pair", "total_return_pct", "buyhold_total_return_pct", "total_excess_pct",
        "max_drawdown_pct", "avg_exposure_pct", "trade_count", "win_rate_pct",
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
        "<h3>By Pair</h3>",
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
    return table_section("Worst Rolling Windows", rolling.sort_values("excess_return_pct").head(12)[cols])


def reference_delta_section(candidate: Evaluation, reference: Evaluation) -> str:
    cand = candidate.aggregate
    ref = reference.aggregate
    rows = [{
        "metric": "full_excess_pct",
        "candidate": fmt_pp(cand["total_excess_pct"]),
        "reference": fmt_pp(ref["total_excess_pct"]),
        "delta": fmt_pp(cand["total_excess_pct"] - ref["total_excess_pct"]),
    }, {
        "metric": "max_drawdown_pct",
        "candidate": fmt_pct(cand["max_drawdown_pct"]),
        "reference": fmt_pct(ref["max_drawdown_pct"]),
        "delta": fmt_pp(cand["max_drawdown_pct"] - ref["max_drawdown_pct"]),
    }]
    if not candidate.standard.empty and not reference.standard.empty:
        rows.extend([{
            "metric": "standard_median_excess_pct",
            "candidate": fmt_pp(candidate.standard["excess_return_pct"].median()),
            "reference": fmt_pp(reference.standard["excess_return_pct"].median()),
            "delta": fmt_pp(candidate.standard["excess_return_pct"].median() - reference.standard["excess_return_pct"].median()),
        }, {
            "metric": "standard_worst_excess_pct",
            "candidate": fmt_pp(candidate.standard["excess_return_pct"].min()),
            "reference": fmt_pp(reference.standard["excess_return_pct"].min()),
            "delta": fmt_pp(candidate.standard["excess_return_pct"].min() - reference.standard["excess_return_pct"].min()),
        }])
    return table_section("Reference Delta", pd.DataFrame(rows))


def table_section(title: str, df: pd.DataFrame) -> str:
    return f"<section><h2>{esc(title)}</h2>{df_to_html(df)}</section>"


def df_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    headers = "".join(f"<th>{esc(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{esc(format_value(value, str(col)))}</td>" for col, value in row.items()) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def format_value(value: Any, col: str) -> str:
    if pd.isna(value):
        return ""
    if col in {"window_start", "window_end", "window_days", "step_days"}:
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
    return str(value)


def scoring_policy_section() -> str:
    rows = pd.DataFrame([
        {"component": "long_term_excess", "weight": 25, "purpose": "Full-period excess, pair coverage, weakest pair excess"},
        {"component": "rolling_stability", "weight": 30, "purpose": "Rolling median excess, win rate, worst excess, pair medians"},
        {"component": "risk_control", "weight": 25, "purpose": "Aggregate, pair, and rolling max drawdown discipline"},
        {"component": "trade_quality", "weight": 10, "purpose": "Trade frequency and average exposure sanity"},
        {"component": "logic_consistency", "weight": 10, "purpose": "Manual score for whether the rule fits the long-term strategy philosophy"},
    ])
    return table_section("Scoring Policy", rows)


def timerange_years(timerange: str) -> float:
    if "-" not in timerange:
        return 1.0
    start, end = timerange.split("-", 1)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return max((end_ts - start_ts).days / 365.25, 0.1)


def delta_detail(candidate: float, reference: float, unit: str) -> str:
    return f"candidate={candidate:.2f}{unit}, reference={reference:.2f}{unit}, delta={candidate - reference:.2f}{unit}"


def fmt_pct(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f}%"


def fmt_pp(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f} pp"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


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
