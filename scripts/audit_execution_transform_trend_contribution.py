#!/usr/bin/env python3
"""Audit execution-transform trend leverage contribution."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    daily = _build_daily_frame(args.clean_dir, args.baseline_dir)
    reason_summary = _reason_summary(daily)
    trend_phase_summary = _trend_phase_summary(daily)
    trend_windows = _trend_windows(daily)
    overall = _overall_summary(daily, args.clean_dir, args.baseline_dir)

    daily.to_csv(args.output_dir / "trend_leverage_daily.csv", index=False)
    reason_summary.to_csv(args.output_dir / "trend_leverage_reason_year_symbol.csv", index=False)
    trend_phase_summary.to_csv(args.output_dir / "trend_leverage_phase_summary.csv", index=False)
    trend_windows.to_csv(args.output_dir / "trend_leverage_windows.csv", index=False)
    overall.to_csv(args.output_dir / "trend_leverage_overall.csv", index=False)

    print(f"Wrote {args.output_dir}")
    print(overall.to_string(index=False))


def _build_daily_frame(clean_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    audit = _read_csv(clean_dir / "execution_transform_audit.csv")
    clean_eq = _equity_returns(clean_dir / "equity_curves.csv", "clean")
    base_eq = _equity_returns(baseline_dir / "equity_curves.csv", "baseline")
    prices = _read_csv(clean_dir / "prices.csv")
    lifecycle = _read_optional_context(clean_dir)

    for frame in (audit, clean_eq, base_eq, prices, lifecycle):
        if not frame.empty and "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    daily = audit.merge(clean_eq, on=["symbol", "timestamp"], how="left")
    daily = daily.merge(base_eq, on=["symbol", "timestamp"], how="left")
    if not prices.empty:
        price_cols = [c for c in ["symbol", "timestamp", "close", "price_norm"] if c in prices.columns]
        daily = daily.merge(prices[price_cols], on=["symbol", "timestamp"], how="left")
    if not lifecycle.empty:
        daily = daily.merge(lifecycle, on=["symbol", "timestamp"], how="left")

    for col in ["clean_return", "baseline_return"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        daily[f"{col}_log"] = np.log1p(daily[col].clip(lower=-0.999999999))
    daily["excess_return_log"] = daily["clean_return_log"] - daily["baseline_return_log"]
    daily["year"] = daily["timestamp"].dt.year
    daily["is_trend_confirmed"] = daily["transform_reason"].astype(str).eq("trend_confirmed")
    return daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _equity_returns(path: Path, prefix: str) -> pd.DataFrame:
    eq = _read_csv(path)
    if eq.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", f"{prefix}_return", f"{prefix}_total_value", f"{prefix}_position_pct"])
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
    eq = eq.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    eq[f"{prefix}_return"] = eq.groupby("symbol")["total_value"].pct_change().fillna(0.0)
    cols = ["symbol", "timestamp", f"{prefix}_return", "total_value", "position_pct"]
    out = eq[cols].rename(columns={
        "total_value": f"{prefix}_total_value",
        "position_pct": f"{prefix}_position_pct",
    })
    return out


def _read_optional_context(clean_dir: Path) -> pd.DataFrame:
    lifecycle = _read_csv(clean_dir / "lifecycle_state_shadow.csv")
    if lifecycle.empty:
        return pd.DataFrame()
    cols = [
        "symbol", "timestamp", "lifecycle_state", "regime", "raw_state",
        "confirmed_state", "risk_score", "trend_risk", "drawdown_risk",
        "structural_bear", "low_location_shadow", "recovery_active_shadow",
        "trend_confirmed_shadow", "distribution_shadow", "selected_setup",
        "sizing_setup", "action_setup",
    ]
    return lifecycle[[c for c in cols if c in lifecycle.columns]].copy()


def _reason_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return _summarize_groups(daily, ["symbol", "year", "transform_reason"])


def _trend_phase_summary(daily: pd.DataFrame) -> pd.DataFrame:
    trend = daily[daily["is_trend_confirmed"]].copy()
    if trend.empty:
        return pd.DataFrame()
    group_cols = ["symbol", "year"]
    for col in ["confirmed_state", "lifecycle_state", "regime"]:
        if col in trend.columns:
            group_cols.append(col)
    return _summarize_groups(trend, group_cols)


def _trend_windows(daily: pd.DataFrame) -> pd.DataFrame:
    trend = daily[daily["is_trend_confirmed"]].copy()
    if trend.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for symbol, group in trend.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        gaps = group["timestamp"].diff().dt.days.fillna(1)
        group["window_id"] = (gaps > 2).cumsum()
        for _, window in group.groupby("window_id", sort=False):
            row = _summarize_frame(window)
            row.update({
                "symbol": symbol,
                "start": window["timestamp"].iloc[0],
                "end": window["timestamp"].iloc[-1],
                "start_close": float(window["close"].iloc[0]) if "close" in window.columns and not pd.isna(window["close"].iloc[0]) else np.nan,
                "end_close": float(window["close"].iloc[-1]) if "close" in window.columns and not pd.isna(window["close"].iloc[-1]) else np.nan,
            })
            if not math.isnan(row["start_close"]) and row["start_close"] > 0.0:
                row["price_return"] = row["end_close"] / row["start_close"] - 1.0
            else:
                row["price_return"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cols = ["symbol", "start", "end", "days", "price_return", "clean_return", "baseline_return", "excess_return"]
    other = [c for c in out.columns if c not in cols]
    return out[cols + other].sort_values(["symbol", "start"]).reset_index(drop=True)


def _overall_summary(daily: pd.DataFrame, clean_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    rows = []
    for symbol, group in daily.groupby("symbol", sort=False):
        trend = group[group["is_trend_confirmed"]]
        nontrend = group[~group["is_trend_confirmed"]]
        row = {
            "symbol": symbol,
            "days": int(len(group)),
            "trend_days": int(len(trend)),
            "trend_day_pct": float(len(trend) / len(group)) if len(group) else 0.0,
            "trend_clean_return": _compound_log(trend["clean_return_log"]),
            "trend_baseline_return": _compound_log(trend["baseline_return_log"]),
            "trend_excess_return": _compound_log(trend["excess_return_log"]),
            "nontrend_excess_return": _compound_log(nontrend["excess_return_log"]),
            "financing_cost": float(group.get("financing_cost_today", pd.Series(dtype=float)).sum()),
            "trend_financing_cost": float(trend.get("financing_cost_today", pd.Series(dtype=float)).sum()),
            "gross_warning_count": int(group.get("gross_warning", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "max_gross_position": float(group.get("gross_position", pd.Series([0.0])).astype(float).max()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    metrics = []
    for label, path in [("clean", clean_dir), ("baseline", baseline_dir)]:
        m = _read_csv(path / "metrics.csv")
        if not m.empty:
            row = m.iloc[0]
            metrics.append({
                "symbol": f"__{label}_portfolio__",
                "days": 0,
                "trend_days": 0,
                "trend_day_pct": 0.0,
                "portfolio_total_return": row.get("strategy_total_return", np.nan),
                "portfolio_annual_return": row.get("strategy_annual_return", np.nan),
                "portfolio_max_drawdown": row.get("strategy_max_drawdown", np.nan),
                "portfolio_avg_position": row.get("avg_position_pct", np.nan),
                "portfolio_trade_count": row.get("trade_count", np.nan),
            })
    return pd.concat([out, pd.DataFrame(metrics)], ignore_index=True, sort=False)


def _summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(_summarize_frame(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_frame(frame: pd.DataFrame) -> dict:
    return {
        "days": int(len(frame)),
        "clean_return": _compound_log(frame["clean_return_log"]),
        "baseline_return": _compound_log(frame["baseline_return_log"]),
        "excess_return": _compound_log(frame["excess_return_log"]),
        "clean_log_return": float(frame["clean_return_log"].sum()),
        "baseline_log_return": float(frame["baseline_return_log"].sum()),
        "excess_log_return": float(frame["excess_return_log"].sum()),
        "avg_raw_position_pct": _mean(frame, "raw_position_pct"),
        "avg_actual_position_pct": _mean(frame, "actual_position_pct"),
        "avg_transformed_target_pct": _mean(frame, "transformed_target_pct"),
        "max_actual_position_pct": _max(frame, "actual_position_pct"),
        "max_gross_position": _max(frame, "gross_position"),
        "financing_cost": float(frame.get("financing_cost_today", pd.Series(dtype=float)).sum()),
        "gross_warning_count": int(frame.get("gross_warning", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
    }


def _compound_log(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(math.exp(float(values.sum())) - 1.0)


def _mean(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].astype(float).mean()) if col in frame.columns and not frame.empty else np.nan


def _max(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].astype(float).max()) if col in frame.columns and not frame.empty else np.nan


if __name__ == "__main__":
    main()
