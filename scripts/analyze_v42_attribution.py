#!/usr/bin/env python3
"""Build V4.2 attribution tables from strategy review result directories."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


WINDOWS = {
    "full": (None, None),
    "2021-2024": ("2021-01-01", "2024-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025-2026": ("2025-01-01", "2026-12-31"),
}
ACTION_HASH_COLUMNS = [
    "timestamp",
    "signal_timestamp",
    "indicator_timestamp",
    "btc_regime_timestamp",
    "execution_mode",
    "symbol",
    "side",
    "quantity",
    "price",
    "notional",
    "fee",
    "reason",
    "setup",
    "risk_score",
    "trend_risk",
    "drawdown_risk",
    "raw_state",
    "confirmed_state",
    "target_pct",
    "guards",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "strategy_review" / "v42_attribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path) for path in args.result_dirs]
    runs = [run for run in runs if run is not None]
    if not runs:
        raise SystemExit("No readable result directories supplied.")

    write_csv(strategy_attribution_summary(runs), args.output_dir / "strategy_attribution_summary.csv")
    write_csv(monthly_delta(runs), args.output_dir / "monthly_delta.csv")
    write_csv(symbol_window_metrics(runs), args.output_dir / "symbol_window_metrics.csv")
    write_csv(trend_cont_forward_90d(runs), args.output_dir / "trend_cont_forward_90d.csv")
    write_csv(bear_base_path_summary(runs), args.output_dir / "bear_base_path_summary.csv")
    write_csv(sleeve_pnl_summary(runs), args.output_dir / "sleeve_pnl_summary.csv")
    print(f"Wrote attribution tables to {args.output_dir}")


def load_run(path: Path) -> dict | None:
    if not path.exists():
        print(f"Skipping missing directory: {path}", file=sys.stderr)
        return None
    run = {
        "path": path,
        "strategy": infer_strategy_name(path),
        "metrics": read_csv(path / "metrics.csv"),
        "equity": read_csv(path / "equity_curves.csv"),
        "prices": read_csv(path / "prices.csv"),
        "actions": read_csv(path / "actions.csv"),
        "diagnostics": read_csv(path / "diagnostics.csv"),
        "sleeve_events": read_csv(path / "sleeve_events.csv"),
        "sleeve_daily": read_csv(path / "sleeve_daily.csv"),
        "base_deferred": read_csv(path / "base_deferred_candidates.csv"),
    }
    for key in ("equity", "prices", "actions", "sleeve_events", "sleeve_daily", "base_deferred"):
        frame = run[key]
        if not frame.empty and "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return run


def infer_strategy_name(path: Path) -> str:
    metrics = read_csv(path / "metrics.csv")
    if not metrics.empty and "strategy" in metrics.columns:
        return str(metrics["strategy"].iloc[0])
    name = path.name
    if "_full_" in name:
        return name.split("_full_")[0]
    return name


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def strategy_attribution_summary(runs: list[dict]) -> pd.DataFrame:
    rows = []
    combo_hash = first_action_hash(runs, "v4_2_exp_btc_tc_off_base_exit")
    for run in runs:
        composite = composite_from_equity(run["equity"])
        actions = run["actions"]
        diagnostics = run["diagnostics"]
        sleeve_summary = summarize_sleeve_final(run["sleeve_daily"])
        for window, (start, end) in WINDOWS.items():
            comp = filter_window(composite, start, end)
            if comp.empty:
                continue
            row = {
                "strategy": run["strategy"],
                "result_dir": str(run["path"]),
                "window": window,
                **return_metrics(comp["timestamp"], comp["strategy_equity"]),
                "avg_position_pct": float(comp.get("avg_position_pct", pd.Series(dtype=float)).mean()),
                "trade_count": int(len(filter_window(actions, start, end))),
                "action_hash": action_hash(actions),
                "actions_match_combo": bool(combo_hash and action_hash(actions) == combo_hash),
            }
            row.update(prefix_dict(diagnostic_totals(diagnostics), "diag_"))
            row.update(prefix_dict(sleeve_summary, "sleeve_"))
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_delta(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        comp = composite_from_equity(run["equity"])
        if comp.empty:
            continue
        comp = comp.dropna(subset=["timestamp"]).copy()
        comp["month"] = comp["timestamp"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        month_end = comp.sort_values("timestamp").groupby("month").tail(1)
        month_start = comp.sort_values("timestamp").groupby("month").head(1)
        merged = month_end[["month", "strategy_equity"]].merge(
            month_start[["month", "strategy_equity"]],
            on="month",
            suffixes=("_end", "_start"),
        )
        merged["monthly_return"] = merged["strategy_equity_end"] / merged["strategy_equity_start"] - 1.0
        merged["strategy"] = run["strategy"]
        rows.append(merged[["month", "strategy", "monthly_return", "strategy_equity_end"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    base_strategy = out["strategy"].iloc[0]
    base = out[out["strategy"] == base_strategy][["month", "monthly_return"]].rename(columns={"monthly_return": "base_monthly_return"})
    out = out.merge(base, on="month", how="left")
    out["delta_vs_first"] = out["monthly_return"] - out["base_monthly_return"]
    out["baseline_strategy"] = base_strategy
    return out


def symbol_window_metrics(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        equity = run["equity"]
        actions = run["actions"]
        if equity.empty:
            continue
        for symbol, sym_eq in equity.groupby("symbol"):
            sym_eq = sym_eq.sort_values("timestamp")
            value_col = f"{symbol}_value"
            if value_col not in sym_eq.columns:
                continue
            series = sym_eq["total_value"].astype(float)
            for window, (start, end) in WINDOWS.items():
                win = filter_window(sym_eq, start, end)
                if win.empty:
                    continue
                row = {
                    "strategy": run["strategy"],
                    "symbol": symbol,
                    "window": window,
                    **return_metrics(win["timestamp"], win["total_value"].astype(float) / float(series.iloc[0])),
                    "trade_count": int(len(filter_window(actions[actions.get("symbol") == symbol], start, end))) if not actions.empty else 0,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def trend_cont_forward_90d(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        actions = run["actions"]
        prices = run["prices"]
        if actions.empty or prices.empty or "setup" not in actions.columns:
            continue
        buys = actions[(actions["side"] == "buy") & (actions["setup"] == "trend-cont")]
        for _, action in buys.iterrows():
            px = prices[prices["symbol"] == action["symbol"]].sort_values("timestamp")
            future = px[px["timestamp"] >= action["timestamp"]]
            if future.empty:
                continue
            anchor = float(action["price"])
            fwd = future[future["timestamp"] >= action["timestamp"] + pd.Timedelta(days=90)]
            end_row = fwd.iloc[0] if not fwd.empty else future.iloc[-1]
            rows.append({
                "strategy": run["strategy"],
                "symbol": action["symbol"],
                "timestamp": action["timestamp"],
                "price": anchor,
                "forward_timestamp": end_row["timestamp"],
                "forward_close": float(end_row["close"]),
                "forward_90d_return": float(end_row["close"]) / anchor - 1.0 if anchor > 0 else np.nan,
                "reason": action.get("reason", ""),
            })
    return pd.DataFrame(rows)


def bear_base_path_summary(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        events = run["sleeve_events"]
        deferred = run["base_deferred"]
        if not events.empty:
            base_events = events[events.get("sleeve", "") == "base"].copy()
            for symbol, group in base_events.groupby("symbol"):
                rows.append({
                    "strategy": run["strategy"],
                    "symbol": symbol,
                    "base_buy_count": int((group["side"] == "buy").sum()),
                    "base_sell_count": int((group["side"] == "sell").sum()),
                    "base_buy_notional": float(group.loc[group["side"] == "buy", "notional"].sum()),
                    "base_sell_notional": float(group.loc[group["side"] == "sell", "notional"].sum()),
                    "base_realized_pnl": float(group["realized_pnl_after"].iloc[-1]) if "realized_pnl_after" in group.columns and not group.empty else 0.0,
                    "deferred_candidate_count": int(len(deferred[deferred.get("symbol") == symbol])) if not deferred.empty else 0,
                })
        elif not deferred.empty:
            for symbol, group in deferred.groupby("symbol"):
                rows.append({
                    "strategy": run["strategy"],
                    "symbol": symbol,
                    "base_buy_count": 0,
                    "base_sell_count": 0,
                    "base_buy_notional": 0.0,
                    "base_sell_notional": 0.0,
                    "base_realized_pnl": 0.0,
                    "deferred_candidate_count": int(len(group)),
                })
    return pd.DataFrame(rows)


def sleeve_pnl_summary(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        daily = run["sleeve_daily"]
        if daily.empty:
            continue
        daily = daily.sort_values("timestamp")
        for symbol, group in daily.groupby("symbol"):
            last = group.iloc[-1]
            main_pnl = float(last.get("main_realized_pnl", 0.0)) + float(last.get("main_unrealized_pnl", 0.0))
            base_pnl = float(last.get("base_realized_pnl", 0.0)) + float(last.get("base_unrealized_pnl", 0.0))
            rows.append({
                "strategy": run["strategy"],
                "symbol": symbol,
                "main_quantity": float(last.get("main_quantity", 0.0)),
                "base_quantity": float(last.get("base_quantity", 0.0)),
                "exchange_quantity": float(last.get("exchange_quantity", 0.0)),
                "exchange_quantity_after_est": float(last.get("exchange_quantity_after_est", last.get("exchange_quantity", 0.0))),
                "quantity_diff": sleeve_quantity_diff(last),
                "main_realized_pnl": float(last.get("main_realized_pnl", 0.0)),
                "main_unrealized_pnl": float(last.get("main_unrealized_pnl", 0.0)),
                "main_total_pnl": main_pnl,
                "base_realized_pnl": float(last.get("base_realized_pnl", 0.0)),
                "base_unrealized_pnl": float(last.get("base_unrealized_pnl", 0.0)),
                "base_total_pnl": base_pnl,
            })
    return pd.DataFrame(rows)


def composite_from_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    pieces = []
    pos = []
    for symbol, group in equity.groupby("symbol"):
        group = group.sort_values("timestamp")
        if group.empty:
            continue
        pieces.append(pd.DataFrame({
            "timestamp": group["timestamp"],
            symbol: group["total_value"].astype(float) / float(group["total_value"].iloc[0]),
        }))
        value_col = f"{symbol}_value"
        if value_col in group.columns:
            pos.append(pd.DataFrame({
                "timestamp": group["timestamp"],
                symbol: group[value_col].astype(float) / group["total_value"].astype(float),
            }))
    merged = merge_timestamp(pieces)
    if merged.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    cols = [col for col in merged.columns if col != "timestamp"]
    out["strategy_equity"] = merged[cols].mean(axis=1)
    if pos:
        pos_merged = merge_timestamp(pos)
        pos_cols = [col for col in pos_merged.columns if col != "timestamp"]
        out["avg_position_pct"] = pos_merged[pos_cols].mean(axis=1)
    return out


def merge_timestamp(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="timestamp", how="inner")
    return out.sort_values("timestamp").reset_index(drop=True)


def filter_window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame.columns:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out


def return_metrics(timestamps: pd.Series, equity: pd.Series) -> dict:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").loc[values.index]
    if len(values) < 2:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}
    total = float(values.iloc[-1] / values.iloc[0] - 1.0)
    years = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    annual = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
    drawdown = values / values.cummax() - 1.0
    return {"total_return": total, "annual_return": annual, "max_drawdown": float(drawdown.min())}


def summarize_sleeve_final(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {}
    out = {}
    for symbol, group in daily.sort_values("timestamp").groupby("symbol"):
        last = group.iloc[-1]
        clean = symbol.split("/")[0].lower()
        out[f"{clean}_main_total_pnl"] = float(last.get("main_realized_pnl", 0.0)) + float(last.get("main_unrealized_pnl", 0.0))
        out[f"{clean}_base_total_pnl"] = float(last.get("base_realized_pnl", 0.0)) + float(last.get("base_unrealized_pnl", 0.0))
        out[f"{clean}_quantity_diff"] = sleeve_quantity_diff(last)
    return out


def sleeve_quantity_diff(row: pd.Series) -> float:
    exchange_quantity = float(row.get("exchange_quantity_after_est", row.get("exchange_quantity", 0.0)))
    return exchange_quantity - float(row.get("main_quantity", 0.0)) - float(row.get("base_quantity", 0.0))


def diagnostic_totals(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    out = {}
    for col in frame.columns:
        if str(col).startswith("v4_2_"):
            out[col] = int(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum())
    return out


def prefix_dict(values: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in values.items()}


def first_action_hash(runs: list[dict], strategy: str) -> str:
    for run in runs:
        if run["strategy"] == strategy:
            return action_hash(run["actions"])
    return ""


def action_hash(actions: pd.DataFrame) -> str:
    if actions.empty:
        return hashlib.sha256(b"").hexdigest()
    cols = [col for col in ACTION_HASH_COLUMNS if col in actions.columns]
    canonical = actions[cols].copy()
    canonical = canonical.sort_values(cols[:2] if len(cols) >= 2 else cols).reset_index(drop=True)
    payload = canonical.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
