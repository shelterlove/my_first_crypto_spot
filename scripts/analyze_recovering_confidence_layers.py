#!/usr/bin/env python3
"""Simulate layered RECOVERING confidence targets without changing strategy logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "recovering_transition_v2_21E_partial_exit_tagfix_20260605"
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

LAYER_ORDER = ["LOW", "MID", "HIGH_STABLE", "HIGH_CONSERVATIVE", "HIGH_STRICT"]
LAYER_TARGET_FLOORS = {
    "LOW": 0.42,
    "MID": 0.52,
    "HIGH_STABLE": 0.58,
    "HIGH_CONSERVATIVE": 0.62,
    "HIGH_STRICT": 0.65,
}
LAYER_MAX_BUY_CAPS = {
    "LOW": 0.05,
    "MID": 0.08,
    "HIGH_STABLE": 0.12,
    "HIGH_CONSERVATIVE": 0.12,
    "HIGH_STRICT": 0.15,
}

BTC_PRICE_VS_EMA72_MIN = 0.095
ALT_EMA24_VS_EMA168_MIN = -0.032
ALT_EMA72_VS_EMA168_MIN = -0.045
NON_BEAR_DAYS_MIN = 7
ROC20_CAP = 0.17
EMA168_SLOPE_MIN = -0.0073


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = add_confidence_layers(load_detail(Path(args.input_dir)))
    exposure = build_exposure_simulation(detail)
    layer_summary = summarize_layers(exposure)
    window_summary = summarize_windows(exposure)
    gate = evaluate_gate(exposure, layer_summary)
    report = render_report(args, layer_summary, window_summary, gate)

    detail.to_csv(output_dir / "recovering_confidence_detail.csv", index=False)
    exposure.to_csv(output_dir / "recovering_confidence_exposure_simulation.csv", index=False)
    layer_summary.to_csv(output_dir / "recovering_confidence_layer_summary.csv", index=False)
    window_summary.to_csv(output_dir / "recovering_confidence_window_summary.csv", index=False)
    (output_dir / "recovering_confidence_report.md").write_text(report, encoding="utf-8")
    (output_dir / "recovering_confidence_report.html").write_text(markdown_to_simple_html(report), encoding="utf-8")

    print("Confidence layer summary")
    print(layer_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not layer_summary.empty else "No layer rows")
    print("\nGate")
    print(gate.to_string(index=False, float_format=lambda x: f"{x:.2f}") if not gate.empty else "No gate rows")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="recovering_confidence_layers_v2_21E_20260605")
    return parser.parse_args()


def load_detail(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "recovering_transition_detail.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/analyze_recovering_transition.py first.")
    rows = pd.read_csv(path)
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    return rows.sort_values(["pair", "date"]).reset_index(drop=True)


def add_confidence_layers(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    ensure_btc_features(out)

    base = out["recovering_candidate"].fillna(False) & ~out["recovering_exit_condition"].fillna(False)
    mid = (
        base
        & (out["btc_price_vs_ema72"] >= BTC_PRICE_VS_EMA72_MIN)
        & (out["ema24_vs_ema168"] >= ALT_EMA24_VS_EMA168_MIN)
    )
    high_stable = mid & (out["raw_state_consecutive_non_bear"] >= NON_BEAR_DAYS_MIN)
    high_conservative = mid & (out["roc_20"] <= ROC20_CAP)
    high_strict = high_stable & high_conservative

    out["recovering_low"] = base
    out["recovering_mid"] = mid
    out["recovering_high_stable"] = high_stable
    out["recovering_high_conservative"] = high_conservative
    out["recovering_high_strict"] = high_strict
    out["recovering_confidence_layer"] = "NONE"
    out.loc[base, "recovering_confidence_layer"] = "LOW"
    out.loc[mid, "recovering_confidence_layer"] = "MID"
    out.loc[high_stable, "recovering_confidence_layer"] = "HIGH_STABLE"
    out.loc[high_conservative, "recovering_confidence_layer"] = "HIGH_CONSERVATIVE"
    out.loc[high_strict, "recovering_confidence_layer"] = "HIGH_STRICT"

    out["layer_target_floor"] = out["recovering_confidence_layer"].map(LAYER_TARGET_FLOORS).fillna(0.0)
    out["layer_max_buy_cap"] = out["recovering_confidence_layer"].map(LAYER_MAX_BUY_CAPS).fillna(0.0)
    out["layer_target_gap"] = (out["layer_target_floor"] - out["current_pct"]).clip(lower=0.0)
    out["layer_theoretical_buy_pct"] = out[["layer_target_gap", "layer_max_buy_cap"]].min(axis=1)
    out["layer_target_delta_vs_actual"] = (out["layer_target_floor"] - out["buy_target"]).clip(lower=0.0)
    out["layer_theoretical_extra_buy_pct"] = (
        out["layer_theoretical_buy_pct"] - out["executable_buy_pct"].fillna(0.0)
    ).clip(lower=0.0)
    out["strong_bull_window"] = out["window_labels"].fillna("").map(
        lambda text: any(name in str(text) for name, _, _, strong in WINDOWS if strong)
    )
    out["bear_counterexample_window"] = out["window_labels"].fillna("").map(
        lambda text: any(name in str(text) for name in ("bear_rally_counterexample", "bear_defence_counterexample"))
    )
    out["adverse_60d"] = (out["future_ret_60d"] <= 0.0) | (out["future_down_60d"] <= -0.20)
    out["bad_2022_60d"] = (
        out["bear_counterexample_window"]
        & ((out["future_ret_60d"] < 0.0) | (out["future_down_60d"] < -0.15))
    )
    return out


def ensure_btc_features(rows: pd.DataFrame) -> None:
    if "btc_price_vs_ema72" not in rows:
        rows["btc_price_vs_ema72"] = rows["btc_price"] / rows["btc_ema72"] - 1.0
    if "btc_price_vs_ema24" not in rows:
        rows["btc_price_vs_ema24"] = rows["btc_price"] / rows["btc_ema24"] - 1.0
    if "btc_ema24_vs_ema72" not in rows:
        rows["btc_ema24_vs_ema72"] = rows["btc_ema24"] / rows["btc_ema72"] - 1.0


def build_exposure_simulation(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_name, start_raw, end_raw, is_strong in WINDOWS:
        start = pd.Timestamp(start_raw, tz="UTC")
        end = pd.Timestamp(end_raw, tz="UTC")
        frame = detail[(detail["date"] >= start) & (detail["date"] <= end)].copy()
        for layer in LAYER_ORDER:
            flag = layer_flag(layer)
            layer_rows = frame[frame[flag]].copy()
            if layer_rows.empty:
                rows.append(empty_exposure_row(window_name, is_strong, layer))
                continue
            for pair, group in layer_rows.groupby("pair", sort=False):
                rows.append(exposure_row(window_name, is_strong, layer, pair, group))
            rows.append(exposure_row(window_name, is_strong, layer, "ALL", layer_rows))
    return pd.DataFrame(rows)


def empty_exposure_row(window: str, is_strong: bool, layer: str) -> dict:
    return {
        "window": window,
        "is_strong_bull_window": is_strong,
        "layer": layer,
        "pair": "ALL",
        "trigger_days": 0,
        "first_trigger_date": "",
        "pairs": 0,
        "target_floor_pct": LAYER_TARGET_FLOORS[layer] * 100.0,
        "max_buy_cap_pct": LAYER_MAX_BUY_CAPS[layer] * 100.0,
        "mean_current_pct": None,
        "mean_theoretical_buy_pct": None,
        "mean_extra_buy_pct": None,
        "total_extra_buy_pct_days": 0.0,
        "median_future_ret_60d_pct": None,
        "median_future_down_60d_pct": None,
        "positive_60d_rate_pct": None,
        "adverse_60d_days": 0,
        "bad_2022_days": 0,
    }


def exposure_row(window: str, is_strong: bool, layer: str, pair: str, group: pd.DataFrame) -> dict:
    theoretical_buy = compute_layer_theoretical_buy(group, layer)
    extra_buy = (theoretical_buy - group["executable_buy_pct"].fillna(0.0)).clip(lower=0.0)
    return {
        "window": window,
        "is_strong_bull_window": is_strong,
        "layer": layer,
        "pair": pair,
        "trigger_days": len(group),
        "first_trigger_date": format_date(group["date"].min()),
        "pairs": group["pair"].nunique(),
        "target_floor_pct": LAYER_TARGET_FLOORS[layer] * 100.0,
        "max_buy_cap_pct": LAYER_MAX_BUY_CAPS[layer] * 100.0,
        "mean_current_pct": group["current_pct"].mean() * 100.0,
        "mean_theoretical_buy_pct": theoretical_buy.mean() * 100.0,
        "mean_extra_buy_pct": extra_buy.mean() * 100.0,
        "total_extra_buy_pct_days": extra_buy.sum() * 100.0,
        "median_future_ret_60d_pct": group["future_ret_60d"].median() * 100.0,
        "median_future_down_60d_pct": group["future_down_60d"].median() * 100.0,
        "positive_60d_rate_pct": (group["future_ret_60d"] > 0).mean() * 100.0,
        "adverse_60d_days": int(group["adverse_60d"].sum()),
        "bad_2022_days": int(group["bad_2022_60d"].sum()),
    }


def compute_layer_theoretical_buy(group: pd.DataFrame, layer: str) -> pd.Series:
    floor = LAYER_TARGET_FLOORS[layer]
    cap = LAYER_MAX_BUY_CAPS[layer]
    gap = (floor - group["current_pct"]).clip(lower=0.0)
    return gap.clip(upper=cap)


def summarize_layers(exposure: pd.DataFrame) -> pd.DataFrame:
    all_rows = exposure[exposure["pair"] == "ALL"].copy()
    if all_rows.empty:
        return pd.DataFrame()
    rows = []
    for layer, group in all_rows.groupby("layer", sort=False):
        strong = group[group["is_strong_bull_window"]]
        bear = group[group["window"].isin(["bear_rally_counterexample", "bear_defence_counterexample"])]
        rows.append({
            "layer": layer,
            "target_floor_pct": LAYER_TARGET_FLOORS[layer] * 100.0,
            "max_buy_cap_pct": LAYER_MAX_BUY_CAPS[layer] * 100.0,
            "strong_trigger_days": int(strong["trigger_days"].sum()),
            "strong_pairs_max": int(strong["pairs"].max()) if not strong.empty else 0,
            "strong_median_future_ret_60d_pct": weighted_median(strong, "median_future_ret_60d_pct", "trigger_days"),
            "strong_positive_60d_rate_pct": weighted_mean(strong, "positive_60d_rate_pct", "trigger_days"),
            "bear_trigger_days": int(bear["trigger_days"].sum()),
            "bear_bad_2022_days": int(bear["bad_2022_days"].sum()),
            "bear_median_future_ret_60d_pct": weighted_median(bear, "median_future_ret_60d_pct", "trigger_days"),
            "bear_adverse_days": int(bear["adverse_60d_days"].sum()),
            "mean_extra_buy_pct": weighted_mean(group, "mean_extra_buy_pct", "trigger_days"),
            "total_extra_buy_pct_days": group["total_extra_buy_pct_days"].sum(),
        })
    return pd.DataFrame(rows)


def summarize_windows(exposure: pd.DataFrame) -> pd.DataFrame:
    all_rows = exposure[exposure["pair"] == "ALL"].copy()
    if all_rows.empty:
        return pd.DataFrame()
    cols = [
        "window",
        "layer",
        "trigger_days",
        "pairs",
        "first_trigger_date",
        "target_floor_pct",
        "mean_extra_buy_pct",
        "median_future_ret_60d_pct",
        "median_future_down_60d_pct",
        "positive_60d_rate_pct",
        "adverse_60d_days",
        "bad_2022_days",
    ]
    return all_rows[cols].sort_values(["window", "layer"])


def evaluate_gate(exposure: pd.DataFrame, layer_summary: pd.DataFrame) -> pd.DataFrame:
    if layer_summary.empty:
        return pd.DataFrame()
    rows = []
    by_layer = layer_summary.set_index("layer")

    mid = by_layer.loc["MID"] if "MID" in by_layer.index else None
    low = by_layer.loc["LOW"] if "LOW" in by_layer.index else None
    high_stable = by_layer.loc["HIGH_STABLE"] if "HIGH_STABLE" in by_layer.index else None
    high_conservative = by_layer.loc["HIGH_CONSERVATIVE"] if "HIGH_CONSERVATIVE" in by_layer.index else None

    rows.append(gate_row(
        "MID covers at least two strong-bull pairs",
        bool(mid is not None and mid["strong_pairs_max"] >= 2),
        mid["strong_pairs_max"] if mid is not None else None,
    ))
    rows.append(gate_row(
        "MID keeps 2022 bad days at zero or near zero",
        bool(mid is not None and mid["bear_bad_2022_days"] <= 1),
        mid["bear_bad_2022_days"] if mid is not None else None,
    ))
    rows.append(gate_row(
        "LOW target floor remains <= 45%",
        bool(low is not None and low["target_floor_pct"] <= 45.0),
        low["target_floor_pct"] if low is not None else None,
    ))
    high_pass = False
    high_value = None
    for row in (high_stable, high_conservative):
        if row is None:
            continue
        bear_trigger_days = float(row["bear_trigger_days"])
        adverse_reject = 100.0 if bear_trigger_days == 0 else (1.0 - row["bear_adverse_days"] / bear_trigger_days) * 100.0
        if high_value is None or adverse_reject > high_value:
            high_value = adverse_reject
        high_pass = high_pass or adverse_reject >= 90.0
    rows.append(gate_row(
        "HIGH_STABLE or HIGH_CONSERVATIVE rejects >= 90% bear adverse days",
        high_pass,
        high_value,
    ))
    rows.append(gate_row(
        "MID strong-bull median 60d return is positive",
        bool(mid is not None and mid["strong_median_future_ret_60d_pct"] > 0),
        mid["strong_median_future_ret_60d_pct"] if mid is not None else None,
    ))
    return pd.DataFrame(rows)


def gate_row(check: str, passed: bool, value: float | None) -> dict:
    return {"check": check, "passed": passed, "observed_value": value}


def layer_flag(layer: str) -> str:
    return {
        "LOW": "recovering_low",
        "MID": "recovering_mid",
        "HIGH_STABLE": "recovering_high_stable",
        "HIGH_CONSERVATIVE": "recovering_high_conservative",
        "HIGH_STRICT": "recovering_high_strict",
    }[layer]


def weighted_mean(rows: pd.DataFrame, value_col: str, weight_col: str) -> float:
    rows = rows[pd.notna(rows[value_col]) & (rows[weight_col] > 0)]
    if rows.empty:
        return float("nan")
    return float((rows[value_col] * rows[weight_col]).sum() / rows[weight_col].sum())


def weighted_median(rows: pd.DataFrame, value_col: str, weight_col: str) -> float:
    rows = rows[pd.notna(rows[value_col]) & (rows[weight_col] > 0)]
    if rows.empty:
        return float("nan")
    values = rows.loc[rows.index.repeat(rows[weight_col].astype(int)), value_col]
    return float(values.median()) if not values.empty else float("nan")


def render_report(
    args: argparse.Namespace,
    layer_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    can_candidate = bool(not gate.empty and gate["passed"].all())
    verdict = (
        "分层 RECOVERING 诊断通过，可以进入最小候选设计。"
        if can_candidate
        else "分层 RECOVERING 诊断尚未完全通过，建议继续停留在诊断或只做更保守候选。"
    )
    return "\n".join([
        "# RECOVERING 分层置信度诊断",
        "",
        f"- Input: `{args.input_dir}`",
        "- Baseline: `v2_21E / CryptoSpotV221E`",
        "- 本报告只做分层暴露模拟，不改策略交易逻辑。",
        "",
        "## 结论",
        "",
        verdict,
        "",
        "## 通过门槛",
        "",
        gate.to_markdown(index=False, floatfmt=".2f") if not gate.empty else "No gate checks.",
        "",
        "## 分层摘要",
        "",
        layer_summary.to_markdown(index=False, floatfmt=".2f") if not layer_summary.empty else "No layer summary.",
        "",
        "## 窗口摘要",
        "",
        window_summary.to_markdown(index=False, floatfmt=".2f") if not window_summary.empty else "No window summary.",
        "",
        "## 默认分层规则",
        "",
        "- LOW: 原始 RECOVERING，target floor 42%，max buy 5%。",
        "- MID: LOW + BTC price vs EMA72 >= 9.5% + alt EMA24 vs EMA168 >= -3.2%，target floor 52%，max buy 8%。",
        "- HIGH_STABLE: MID + raw_state_consecutive_non_bear >= 7，target floor 58%，max buy 12%。",
        "- HIGH_CONSERVATIVE: MID + ROC20 <= 17%，target floor 62%，max buy 12%。",
        "- HIGH_STRICT: MID + non-BEAR >= 7 + ROC20 <= 17%，target floor 65%，max buy 15%。",
        "",
    ])


def markdown_to_simple_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>RECOVERING Confidence Layers</title></head><body><pre>{escaped}</pre></body></html>"


def format_date(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
