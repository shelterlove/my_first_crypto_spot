#!/usr/bin/env python3
"""Counterfactual diagnostics for the external recovery overlay.

The goal is to evaluate the overlay as a future permission layer.  Current
v2_21E often marks weak target-gap recovery days as tiny-buy skipped, so a
simple post-buy cap may not actually block a future tiny-buy relaxation.  This
script compares cap-style and deny-style overlays on those candidate events.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovery_feature_search_structural_v2_21E_20260606"
    / "recovery_feature_events.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"
SIZE_MULT_RE = re.compile(
    r"(?:mixed_[a-z]+|v2_4_vol_[a-z_]+|v2_4_cost_[a-z]+|post-override-tgap|btc-bear-tgap)_x([0-9.]+)"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.events))
    candidates = build_candidates(events, cap=args.cap)
    summary = summarize(candidates)
    window_summary = summarize_by_window(candidates)
    rule_summary = summarize_rules(candidates)
    report = render_report(args, summary, window_summary, rule_summary, candidates)

    candidates.to_csv(output_dir / "external_overlay_counterfactual_events.csv", index=False)
    summary.to_csv(output_dir / "external_overlay_counterfactual_summary.csv", index=False)
    window_summary.to_csv(output_dir / "external_overlay_counterfactual_window_summary.csv", index=False)
    rule_summary.to_csv(output_dir / "external_overlay_counterfactual_rule_summary.csv", index=False)
    (output_dir / "external_overlay_counterfactual_report.md").write_text(report, encoding="utf-8")

    print("Summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWindow summary")
    print(window_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRule summary")
    print(rule_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="external_overlay_counterfactual_20260606")
    parser.add_argument("--cap", type=float, default=0.30)
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing events file: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["window_labels"].fillna("").ne("")].copy()
    frame["is_bnb_may"] = frame.get("is_bnb_may", False).astype(bool)
    frame["good_sample"] = frame.get("good_sample", False).astype(bool)
    frame["bad_sample"] = frame.get("bad_sample", False).astype(bool)
    frame["bad_counterfactual"] = (
        frame["bad_sample"]
        | (frame["is_bnb_may"] & frame["quality_label"].eq("LOW_QUALITY"))
    )
    frame["good_counterfactual"] = frame["good_sample"]
    return frame


def build_candidates(events: pd.DataFrame, cap: float) -> pd.DataFrame:
    candidates = events[
        events["raw_state"].eq("MIXED")
        & events["current_pct"].lt(0.35)
        & events["target_gap"].ge(0.05)
        & events["buy_setup"].eq("target-gap")
        & (events["bad_counterfactual"] | events["good_counterfactual"])
    ].copy()

    candidates["relaxed_buy_pct"] = candidates.apply(estimate_relaxed_buy_pct, axis=1)
    candidates = candidates[candidates["relaxed_buy_pct"].gt(0)].copy()

    candidates["gate_e"] = gate_e(candidates)
    candidates["gate_f"] = gate_f(candidates)
    candidates["cap_allowed_pct"] = (cap - candidates["current_pct"]).clip(lower=0.0)

    for label in ("e", "f"):
        gate_col = f"gate_{label}"
        candidates[f"cap_{label}_buy_pct"] = candidates[["relaxed_buy_pct", "cap_allowed_pct"]].min(axis=1)
        candidates.loc[~candidates[gate_col], f"cap_{label}_buy_pct"] = candidates.loc[
            ~candidates[gate_col], "relaxed_buy_pct"
        ]
        candidates[f"deny_{label}_buy_pct"] = candidates["relaxed_buy_pct"]
        candidates.loc[candidates[gate_col], f"deny_{label}_buy_pct"] = 0.0
        candidates[f"cap_{label}_blocked_pct"] = (
            candidates["relaxed_buy_pct"] - candidates[f"cap_{label}_buy_pct"]
        ).clip(lower=0.0)
        candidates[f"deny_{label}_blocked_pct"] = (
            candidates["relaxed_buy_pct"] - candidates[f"deny_{label}_buy_pct"]
        ).clip(lower=0.0)

    for horizon in (30, 60):
        ret_col = f"future_ret_{horizon}d"
        down_col = f"future_down_{horizon}d"
        for mode in ("cap_e", "cap_f", "deny_e", "deny_f"):
            blocked = candidates[f"{mode}_blocked_pct"]
            candidates[f"{mode}_blocked_ret_{horizon}d"] = blocked * candidates[ret_col]
            candidates[f"{mode}_blocked_down_{horizon}d"] = blocked * candidates[down_col]
            candidates[f"{mode}_allowed_ret_{horizon}d"] = candidates[f"{mode}_buy_pct"] * candidates[ret_col]
    return candidates


def estimate_relaxed_buy_pct(row: pd.Series) -> float:
    base = numeric(row.get("base_max_buy"))
    if base <= 0:
        return 0.0
    guard = "-".join(
        str(row.get(column, ""))
        for column in ("buy_guard", "cooldown_guard", "lifted_buy_guard")
        if pd.notna(row.get(column, ""))
    )
    mult = 1.0
    for match in SIZE_MULT_RE.finditer(guard):
        mult *= float(match.group(1))
    return max(0.0, min(numeric(row.get("target_gap")), base * mult))


def gate_e(frame: pd.DataFrame) -> pd.Series:
    r1 = frame["rolling_365d_pos"].ge(0.3198) & frame["donchian_pos"].lt(0.54)
    r2 = frame["ema72_vs_ema168"].le(-0.05) & frame["volume_strength"].le(0.8709)
    return r1 | r2


def gate_f(frame: pd.DataFrame) -> pd.Series:
    return frame["btc_price_vs_ema72"].ge(0.096) & frame["ema72_vs_ema168"].le(-0.05)


def summarize(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ("cap_e", "cap_f", "deny_e", "deny_f"):
        blocked = candidates[f"{mode}_blocked_pct"].gt(1e-12)
        hit = candidates[blocked]
        rows.append({
            "mode": mode,
            "events": len(candidates),
            "blocked_events": int(blocked.sum()),
            "blocked_bad": int(hit["bad_counterfactual"].sum()),
            "blocked_good": int(hit["good_counterfactual"].sum()),
            "blocked_bnb_may_bad": int((hit["is_bnb_may"] & hit["bad_counterfactual"]).sum()),
            "blocked_buy_pct_sum": float(hit[f"{mode}_blocked_pct"].sum()),
            "blocked_ret_30d_sum": float(hit[f"{mode}_blocked_ret_30d"].sum()),
            "blocked_ret_60d_sum": float(hit[f"{mode}_blocked_ret_60d"].sum()),
            "blocked_down_30d_sum": float(hit[f"{mode}_blocked_down_30d"].sum()),
            "blocked_down_60d_sum": float(hit[f"{mode}_blocked_down_60d"].sum()),
        })
    return pd.DataFrame(rows)


def summarize_by_window(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ("cap_e", "cap_f", "deny_e", "deny_f"):
        blocked_col = f"{mode}_blocked_pct"
        hit = candidates[candidates[blocked_col].gt(1e-12)]
        for window, group in hit.groupby("score_window", dropna=False):
            rows.append({
                "mode": mode,
                "score_window": window,
                "blocked_events": len(group),
                "blocked_bad": int(group["bad_counterfactual"].sum()),
                "blocked_good": int(group["good_counterfactual"].sum()),
                "blocked_bnb_may_bad": int((group["is_bnb_may"] & group["bad_counterfactual"]).sum()),
                "blocked_buy_pct_sum": float(group[blocked_col].sum()),
                "blocked_ret_60d_sum": float(group[f"{mode}_blocked_ret_60d"].sum()),
                "first_date": str(group["date"].min().date()),
                "last_date": str(group["date"].max().date()),
                "pairs": ",".join(sorted(group["pair"].unique())),
            })
    return pd.DataFrame(rows).sort_values(["mode", "blocked_bnb_may_bad", "blocked_bad"], ascending=[True, False, False])


def summarize_rules(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = {
        "E_R1_mid_history_weak_range": candidates["rolling_365d_pos"].ge(0.3198) & candidates["donchian_pos"].lt(0.54),
        "E_R2_deep_ema_weak_low_volume": candidates["ema72_vs_ema168"].le(-0.05) & candidates["volume_strength"].le(0.8709),
        "F_btc_hot_local_ema_weak": gate_f(candidates),
    }
    for name, mask in rules.items():
        group = candidates[mask]
        rows.append({
            "rule": name,
            "events": len(group),
            "bad": int(group["bad_counterfactual"].sum()),
            "good": int(group["good_counterfactual"].sum()),
            "bnb_may_bad": int((group["is_bnb_may"] & group["bad_counterfactual"]).sum()),
            "relaxed_buy_pct_sum": float(group["relaxed_buy_pct"].sum()),
            "relaxed_ret_60d_sum": float((group["relaxed_buy_pct"] * group["future_ret_60d"]).sum()),
            "pairs": ",".join(sorted(group["pair"].unique())),
        })
    return pd.DataFrame(rows)


def render_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> str:
    bnb_may = candidates[candidates["is_bnb_may"] & candidates["bad_counterfactual"]].copy()
    return "\n".join([
        "# External Overlay Counterfactual",
        "",
        f"- events: `{args.events}`",
        f"- cap: `{args.cap:.0%}`",
        f"- candidate rows: `{len(candidates)}`",
        "",
        "## Mode Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Rule Summary",
        "",
        rule_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Window Summary",
        "",
        window_summary.to_markdown(index=False, floatfmt=".4f") if not window_summary.empty else "No blocked rows.",
        "",
        "## BNB 2020-05 Bad Candidate Sample",
        "",
        bnb_may[[
            "date", "pair", "current_pct", "target_gap", "relaxed_buy_pct",
            "gate_e", "gate_f", "future_ret_30d", "future_ret_60d",
            "ema72_vs_ema168", "rolling_365d_pos", "donchian_pos",
            "btc_price_vs_ema72",
        ]].head(30).to_markdown(index=False, floatfmt=".4f") if not bnb_may.empty else "No BNB May rows.",
        "",
    ])


def numeric(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
