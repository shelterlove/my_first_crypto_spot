#!/usr/bin/env python3
"""Run path-dependent execution realism and robustness attribution for V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from futures_v1.benchmark import V1BenchmarkRunner  # noqa: E402
from scripts.render_strategy_review_chart import (  # noqa: E402
    SYMBOL_ORDER,
    _as_utc,
    _load_review_data,
    build_metrics,
    run_full_window,
    symbol_release_metrics,
)
from scripts.research_btc_regime import delay_btc  # noqa: E402
from scripts.research_funding_stress import attach_daily_funding  # noqa: E402


SCENARIOS = {
    "a_realistic": {"trend_mult": 1.75, "trend_cap": 1.75, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 0},
    "trend_1_9": {"trend_mult": 1.90, "trend_cap": 1.90, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 0},
    "b_trend_2_0": {"trend_mult": 2.00, "trend_cap": 2.00, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 0},
    "trend_2_1": {"trend_mult": 2.10, "trend_cap": 2.10, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 0},
    "b_stress": {"trend_mult": 2.00, "trend_cap": 2.00, "funding": 2.0, "slippage_bps": 10.0, "fill_ratio": 0.95, "btc_delay": 0},
    "b_btc_delay_1d": {"trend_mult": 2.00, "trend_cap": 2.00, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 1},
    "b_btc_delay_3d": {"trend_mult": 2.00, "trend_cap": 2.00, "funding": 1.0, "slippage_bps": 5.0, "fill_ratio": 1.0, "btc_delay": 3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--output-dir", default="results/research/realism_robustness")
    return parser.parse_args()


def stats_from_returns(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "total_return": 0.0}
    curve = (1.0 + clean).cumprod()
    total = float(curve.iloc[-1] - 1.0)
    annual = float(curve.iloc[-1] ** (365.25 / len(clean)) - 1.0)
    drawdown = curve / curve.cummax() - 1.0
    std = float(clean.std(ddof=1))
    return {
        "annual_return": annual,
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(clean.mean() / std * np.sqrt(365.25)) if std > 0.0 else 0.0,
        "total_return": total,
    }


def grouped_return_stats(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {"mean_daily_return": 0.0, "positive_rate": 0.0, "sum_log_return": 0.0, "sharpe": 0.0}
    std = float(clean.std(ddof=1))
    return {
        "mean_daily_return": float(clean.mean()),
        "positive_rate": float((clean > 0.0).mean()),
        "sum_log_return": float(np.log1p(clean.clip(lower=-0.999999)).sum()),
        "sharpe": float(clean.mean() / std * np.sqrt(365.25)) if std > 0.0 else 0.0,
    }


def curve_attribution(report: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    composite = report["composite"].copy()
    composite["timestamp"] = pd.to_datetime(composite["timestamp"], utc=True)
    composite["return"] = composite["strategy_equity"].pct_change()
    non_2021 = stats_from_returns(composite.loc[composite["timestamp"].dt.year != 2021, "return"])
    worst = []
    for days in (30, 90, 365):
        rolling = (1.0 + composite["return"]).rolling(days).apply(np.prod, raw=True) - 1.0
        if rolling.notna().any():
            idx = rolling.idxmin()
            worst.append({
                "window_days": days,
                "return": float(rolling.loc[idx]),
                "end": composite.loc[idx, "timestamp"].date().isoformat(),
            })
    calendar = []
    for year, rows in composite.groupby(composite["timestamp"].dt.year):
        values = rows["strategy_equity"]
        if len(values) > 1:
            calendar.append({"year": int(year), "return": float(values.iloc[-1] / values.iloc[0] - 1.0)})
    return non_2021, worst, calendar


def regime_attribution(report: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for symbol in SYMBOL_ORDER:
        equity = report["equity"].loc[report["equity"]["symbol"] == symbol, ["timestamp", "total_value"]].copy()
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity["return"] = equity["total_value"].pct_change()
        regimes = data[symbol][["timestamp", "btc_regime"]].copy()
        regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)
        merged = equity.merge(regimes, on="timestamp", how="left")
        for regime, rows in merged.groupby("btc_regime", dropna=False):
            stats = grouped_return_stats(rows["return"])
            out.append({"symbol": symbol, "regime": str(regime), "days": len(rows), **stats})
    return out


def gross_attribution(report: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    audit = report["execution_transform_audit"][["timestamp", "symbol", "gross_position"]].copy()
    audit["timestamp"] = pd.to_datetime(audit["timestamp"], utc=True)
    out = []
    bins = [-np.inf, 1.0, 1.5, 2.0, np.inf]
    labels = ["<=1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
    for symbol in SYMBOL_ORDER:
        equity = report["equity"].loc[report["equity"]["symbol"] == symbol, ["timestamp", "total_value"]].copy()
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity["return"] = equity["total_value"].pct_change()
        risk = audit.loc[audit["symbol"] == symbol].sort_values("timestamp")
        merged = equity.merge(risk, on="timestamp", how="left")
        merged["prior_gross"] = merged["gross_position"].shift(1)
        merged["gross_bucket"] = pd.cut(merged["prior_gross"], bins=bins, labels=labels)
        for bucket, rows in merged.groupby("gross_bucket", observed=True):
            out.append({"symbol": symbol, "gross_bucket": str(bucket), "days": len(rows), **grouped_return_stats(rows["return"])})
    return out


def weighted_portfolio_metrics(report: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    curves = {}
    for symbol in SYMBOL_ORDER:
        rows = report["equity"].loc[report["equity"]["symbol"] == symbol, ["timestamp", "equity_norm"]].copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
        curves[symbol] = rows.set_index("timestamp")["equity_norm"]
    merged = pd.concat(curves, axis=1).sort_index().ffill().fillna(1.0)
    out = []
    for eth_weight in (0.50, 0.65, 0.75):
        curve = eth_weight * merged["ETH/USDT"] + (1.0 - eth_weight) * merged["BNB/USDT"]
        stats = stats_from_returns(curve.pct_change())
        out.append({"eth_weight": eth_weight, "bnb_weight": 1.0 - eth_weight, **stats})
    return out


def main() -> None:
    args = parse_args()
    start, end = _as_utc(args.start), _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    base_data = _load_review_data(runner, SYMBOL_ORDER + ["BTC/USDT"], start, end)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, calendar_rows, worst_rows, symbol_rows = [], [], [], []
    regime_rows, gross_rows, weight_rows = [], [], []

    for name, scenario in SCENARIOS.items():
        data = attach_daily_funding(base_data, start, end, scenario["funding"])
        if scenario["btc_delay"]:
            data = delay_btc(data, scenario["btc_delay"])
        report = run_full_window(
            "eth_bnb_futures_v1",
            data,
            runner,
            start,
            end,
            target_gross_cap=2.0,
            execution_overrides={"trend_mult": scenario["trend_mult"], "trend_cap": scenario["trend_cap"]},
            strategy_overrides={
                "RESEARCH_SLIPPAGE_BPS": scenario["slippage_bps"],
                "RESEARCH_FILL_RATIO": scenario["fill_ratio"],
                "RESEARCH_ACTUAL_GROSS_HARD_CAP": 2.20,
                "RESEARCH_ACTUAL_GROSS_REDUCE_TO": 2.00,
                "RESEARCH_INTRADAY_LIQUIDATION": True,
                "RESEARCH_MAINTENANCE_MARGIN_RATE": 0.005,
                "RESEARCH_LIQUIDATION_FEE_RATE": 0.01,
            },
        )
        metrics = build_metrics(report, start, end)
        non_2021, worst, calendar = curve_attribution(report)
        actions = report["actions"]
        summary_rows.append({
            "scenario": name,
            **scenario,
            "annual_return": metrics["strategy_annual_return"],
            "max_drawdown": metrics["strategy_max_drawdown"],
            "sharpe_daily": metrics["strategy_sharpe_daily"],
            "observed_max_gross": metrics["execution_transform_max_gross_position"],
            "trade_count": metrics["trade_count"],
            "financing_cost": metrics["execution_transform_financing_cost"],
            "hard_gross_reductions": int(actions["reason"].astype(str).str.contains("research-hard-gross", regex=False).sum()),
            "intraday_liquidations": int(actions["reason"].astype(str).str.contains("research-intraday-liquidation", regex=False).sum()),
            **{f"ex_2021_{key}": value for key, value in non_2021.items()},
        })
        calendar_rows.extend({"scenario": name, **row} for row in calendar)
        worst_rows.extend({"scenario": name, **row} for row in worst)
        for symbol, values in symbol_release_metrics(report).items():
            symbol_rows.append({"scenario": name, "symbol": symbol, **values})
        if name in {"a_realistic", "b_trend_2_0"}:
            regime_rows.extend({"scenario": name, **row} for row in regime_attribution(report, data))
            gross_rows.extend({"scenario": name, **row} for row in gross_attribution(report))
            weight_rows.extend({"scenario": name, **row} for row in weighted_portfolio_metrics(report))
        print(
            f"{name}: annual={metrics['strategy_annual_return']:.2%} mdd={metrics['strategy_max_drawdown']:.2%} "
            f"sharpe={metrics['strategy_sharpe_daily']:.3f} maxGross={metrics['execution_transform_max_gross_position']:.3f}",
            flush=True,
        )

    tables = {
        "summary.csv": summary_rows,
        "calendar_returns.csv": calendar_rows,
        "worst_windows.csv": worst_rows,
        "symbol_metrics.csv": symbol_rows,
        "regime_attribution.csv": regime_rows,
        "gross_attribution.csv": gross_rows,
        "weight_sensitivity.csv": weight_rows,
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
    manifest = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": {"start": str(start.date()), "end": str(end.date())},
        "scenarios": SCENARIOS,
        "common": {
            "target_gross_cap": 2.0,
            "actual_gross_hard_cap": 2.2,
            "actual_gross_reduce_to": 2.0,
            "maintenance_margin_rate": 0.005,
            "liquidation_fee_rate": 0.01,
            "intraday_liquidation_check": "daily OHLC low; single-symbol isolated sleeve approximation",
            "funding": "historical daily aggregation, debited from cash before subsequent sizing",
            "strategy_defaults_changed": False,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"realism_robustness={output_dir}")


if __name__ == "__main__":
    main()
