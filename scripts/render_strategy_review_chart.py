#!/usr/bin/env python3
"""Render a strategy review chart for full-window inspection."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import importlib.metadata
import io
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_v1.backtest_event_driven import run_rebalance_backtest  # noqa: E402
from futures_v1.benchmark import V1BenchmarkRunner, build_strategy  # noqa: E402
from futures_v1.strategy_core.execution_engine import ExecutionEngine  # noqa: E402


SYMBOL_ORDER = ["ETH/USDT", "BNB/USDT"]
COLORS = {
    "BTC/USDT": "#f59e0b",
    "ETH/USDT": "#2563eb",
    "BNB/USDT": "#10b981",
    "SOL/USDT": "#7c3aed",
}
FALLBACK_COLORS = ["#7c3aed", "#dc2626", "#0891b2", "#9333ea", "#16a34a", "#ea580c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="eth_bnb_futures_v1")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument(
        "--symbols",
        default=",".join(SYMBOL_ORDER),
        help="Comma-separated traded symbols to review. BTC/USDT is loaded automatically as a reference.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "strategy_review" / "official_v1_full_20200101_20260518"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    global SYMBOL_ORDER
    SYMBOL_ORDER = symbols
    _ensure_symbol_colors(symbols)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    load_symbols = list(dict.fromkeys(symbols + ([] if "BTC/USDT" in symbols else ["BTC/USDT"])))
    start_ts = _as_utc(args.start)
    end_ts = _as_utc(args.end)
    data = _load_review_data(runner, load_symbols, start_ts, end_ts)

    report = run_full_window(args.strategy, data, runner, start_ts, end_ts)
    metrics = build_metrics(report, start_ts, end_ts)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    manifest = build_release_manifest(
        strategy_name=args.strategy,
        runner=runner,
        data=data,
        report=report,
        metrics=metrics,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    report["equity"].to_csv(output_dir / "equity_curves.csv", index=False)
    report["actions"].to_csv(output_dir / "actions.csv", index=False)
    report["prices"].to_csv(output_dir / "prices.csv", index=False)
    report.get("diagnostics", pd.DataFrame()).to_csv(output_dir / "diagnostics.csv", index=False)
    report.get("defense_episodes", pd.DataFrame()).to_csv(output_dir / "defense_episodes.csv", index=False)
    report.get("risk_cycles", pd.DataFrame()).to_csv(output_dir / "risk_cycles.csv", index=False)
    report.get("sleeve_events", pd.DataFrame()).to_csv(output_dir / "sleeve_events.csv", index=False)
    report.get("sleeve_daily", pd.DataFrame()).to_csv(output_dir / "sleeve_daily.csv", index=False)
    report.get("base_lot_events", pd.DataFrame()).to_csv(output_dir / "base_lot_events.csv", index=False)
    report.get("base_deferred_candidates", pd.DataFrame()).to_csv(output_dir / "base_deferred_candidates.csv", index=False)
    report.get("decision_trace", pd.DataFrame()).to_csv(output_dir / "decision_trace.csv", index=False)
    report.get("candidate_orders", pd.DataFrame()).to_csv(output_dir / "candidate_orders.csv", index=False)
    report.get("risk_assessment_shadow", pd.DataFrame()).to_csv(output_dir / "risk_assessment_shadow.csv", index=False)
    report.get("intent_plan_shadow", pd.DataFrame()).to_csv(output_dir / "intent_plan_shadow.csv", index=False)
    report.get("target_vector_shadow", pd.DataFrame()).to_csv(output_dir / "target_vector_shadow.csv", index=False)
    report.get("budget_ledger_shadow", pd.DataFrame()).to_csv(output_dir / "budget_ledger_shadow.csv", index=False)
    report.get("order_arbiter_shadow", pd.DataFrame()).to_csv(output_dir / "order_arbiter_shadow.csv", index=False)
    report.get("symbol_policy_shadow", pd.DataFrame()).to_csv(output_dir / "symbol_policy_shadow.csv", index=False)
    report.get("recovery_state_machine_shadow", pd.DataFrame()).to_csv(output_dir / "recovery_state_machine_shadow.csv", index=False)
    report.get("lifecycle_state_shadow", pd.DataFrame()).to_csv(output_dir / "lifecycle_state_shadow.csv", index=False)
    report.get("recovery_credit_events", pd.DataFrame()).to_csv(output_dir / "recovery_credit_events.csv", index=False)
    report.get("recovery_credit_checks", pd.DataFrame()).to_csv(output_dir / "recovery_credit_checks.csv", index=False)
    report.get("protected_recovery_events", pd.DataFrame()).to_csv(output_dir / "protected_recovery_events.csv", index=False)
    report.get("outer_overlay_events", pd.DataFrame()).to_csv(output_dir / "outer_overlay_events.csv", index=False)
    report.get("execution_transform_audit", pd.DataFrame()).to_csv(output_dir / "execution_transform_audit.csv", index=False)
    report.get("defense_sell_quality", pd.DataFrame()).to_csv(output_dir / "defense_sell_quality.csv", index=False)

    chart_path = output_dir / "strategy_review.svg"
    render_chart(report, metrics, args.strategy, start_ts, end_ts, chart_path)
    print(f"Wrote {chart_path}")
    print(f"Wrote {output_dir / 'metrics.csv'}")
    print(f"Wrote {output_dir / 'release_manifest.json'}")
    print(_metric_line(metrics))


def _parse_symbols(raw: str) -> list[str]:
    symbols = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
    return symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT"]


def _ensure_symbol_colors(symbols: list[str]) -> None:
    fallback_idx = 0
    for symbol in symbols:
        if symbol in COLORS:
            continue
        COLORS[symbol] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
        fallback_idx += 1


def _load_review_data(
    runner: V1BenchmarkRunner,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    for symbol in symbols:
        try:
            runner.load_data(symbol)
        except ValueError as exc:
            if "No candles loaded" not in str(exc):
                raise
            cache_dir = PROJECT_ROOT / "results" / "data_cache" / "binance_um_futures_1d"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = (
                f"{symbol.replace('/', '')}_{start_ts.strftime('%Y%m%d')}_"
                f"{end_ts.strftime('%Y%m%d')}.csv.gz"
            )
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
                runner._data_cache[symbol] = cached
            else:
                downloaded = _load_binance_vision_daily(symbol, start_ts, end_ts)
                downloaded.to_csv(cache_path, index=False, compression="gzip")
                runner._data_cache[symbol] = downloaded
    return runner._inject_btc_regime()


def _load_binance_vision_daily(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    binance_symbol = symbol.replace("/", "")
    start_period = start_ts.tz_convert(None).to_period("M")
    end_period = end_ts.tz_convert(None).to_period("M")
    months = pd.period_range(start_period, end_period, freq="M")
    rows: list[pd.DataFrame] = []
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for month in months:
        url = (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{binance_symbol}/1d/{binance_symbol}-1d-{month}.zip"
        )
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            name = zf.namelist()[0]
            frame = pd.read_csv(zf.open(name), header=None, names=columns)
            rows.append(frame)
    if not rows:
        raise ValueError(f"No Binance Vision candles loaded for {symbol} 1d.")
    out = pd.concat(rows, ignore_index=True)
    open_time = pd.to_numeric(out["open_time"], errors="coerce")
    out = out[open_time.notna()].copy()
    open_time = open_time[open_time.notna()]
    open_time_ms = np.where(open_time > 1e14, open_time / 1000.0, open_time)
    out["timestamp"] = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    out = out[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    out.insert(0, "timeframe", "1d")
    out.insert(0, "symbol", symbol)
    out.insert(0, "exchange", "binance_um_futures")
    return out.sort_values("timestamp").reset_index(drop=True)


def apply_execution_overrides(strategy, overrides: dict[str, float]) -> None:
    config = getattr(strategy, "_core_config", None)
    if config is None or not hasattr(strategy, "_core_execution_engine"):
        raise ValueError("Strategy does not expose the V1 execution config boundary.")
    unknown = set(overrides) - set(config.execution.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown execution config overrides: {sorted(unknown)}")
    execution = replace(config.execution, **{name: float(value) for name, value in overrides.items()})
    strategy._core_config = replace(config, execution=execution)
    strategy._core_execution_engine = ExecutionEngine(strategy._core_config)


def run_full_window(
    strategy_name: str,
    data: dict[str, pd.DataFrame],
    runner: V1BenchmarkRunner,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    target_gross_cap: float | None = None,
    strategy_overrides: dict[str, float] | None = None,
    execution_overrides: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    config = runner.config
    capital = float(config["capital"]["initial"])
    reserve = float(config["capital"]["reserve"])
    fee = float(config["cost"]["fee_rate"])
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")

    equity_rows: list[pd.DataFrame] = []
    action_rows: list[pd.DataFrame] = []
    price_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict] = []
    episode_rows: list[pd.DataFrame] = []
    risk_cycle_rows: list[pd.DataFrame] = []
    sleeve_event_rows: list[pd.DataFrame] = []
    sleeve_daily_rows: list[pd.DataFrame] = []
    base_lot_event_rows: list[pd.DataFrame] = []
    base_deferred_rows: list[pd.DataFrame] = []
    decision_trace_rows: list[pd.DataFrame] = []
    candidate_order_rows: list[pd.DataFrame] = []
    risk_assessment_rows: list[pd.DataFrame] = []
    intent_plan_rows: list[pd.DataFrame] = []
    target_vector_rows: list[pd.DataFrame] = []
    budget_ledger_rows: list[pd.DataFrame] = []
    order_arbiter_rows: list[pd.DataFrame] = []
    symbol_policy_rows: list[pd.DataFrame] = []
    recovery_state_rows: list[pd.DataFrame] = []
    lifecycle_state_rows: list[pd.DataFrame] = []
    recovery_credit_rows: list[pd.DataFrame] = []
    recovery_credit_check_rows: list[pd.DataFrame] = []
    protected_recovery_rows: list[pd.DataFrame] = []
    outer_overlay_rows: list[pd.DataFrame] = []
    execution_transform_audit_rows: list[pd.DataFrame] = []

    for symbol in SYMBOL_ORDER:
        df = data.get(symbol)
        if df is None or df.empty:
            continue
        starts = df.index[df["timestamp"] >= start_ts].tolist()
        ends = df.index[df["timestamp"] <= end_ts].tolist()
        if not starts or not ends:
            continue
        eval_start = starts[0]
        eval_end = ends[-1] + 1
        backtest_start = max(0, eval_start - 1 if execution_mode != "same_close" else eval_start)
        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)

        strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
        setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
        for name, value in (strategy_overrides or {}).items():
            if not hasattr(strategy, name):
                raise ValueError(f"Unknown strategy override: {name}")
            setattr(strategy, name, float(value))
        if execution_overrides:
            apply_execution_overrides(strategy, execution_overrides)
        if target_gross_cap is not None:
            setattr(strategy, "RESEARCH_TARGET_GROSS_CAP", float(target_gross_cap))
        result = run_rebalance_backtest(
            {symbol: backtest_df},
            strategy,
            initial_capital=capital,
            reserve=reserve,
            fee_rate=fee,
            execution_mode=execution_mode,
        )
        diagnostics = result.attrs.get("strategy_diagnostics")
        if isinstance(diagnostics, dict):
            row = {"strategy": strategy_name, "symbol": symbol}
            row.update(diagnostics)
            diagnostics_rows.append(row)
        defense_episodes = result.attrs.get("strategy_defense_episodes")
        defense_episodes = pd.DataFrame() if defense_episodes is None else defense_episodes.copy()
        if not defense_episodes.empty:
            defense_episodes.insert(0, "strategy", strategy_name)
            episode_rows.append(defense_episodes)
        risk_cycles = result.attrs.get("strategy_risk_cycles")
        risk_cycles = pd.DataFrame() if risk_cycles is None else risk_cycles.copy()
        if not risk_cycles.empty:
            risk_cycles.insert(0, "strategy", strategy_name)
            risk_cycle_rows.append(risk_cycles)
        sleeve_events = result.attrs.get("sleeve_events")
        sleeve_events = pd.DataFrame() if sleeve_events is None else sleeve_events.copy()
        if not sleeve_events.empty:
            sleeve_events = _filter_timestamp_start(sleeve_events, start_ts)
            sleeve_events.insert(0, "strategy", strategy_name)
            sleeve_event_rows.append(sleeve_events)
        sleeve_daily = result.attrs.get("sleeve_daily")
        sleeve_daily = pd.DataFrame() if sleeve_daily is None else sleeve_daily.copy()
        if not sleeve_daily.empty:
            sleeve_daily = _filter_timestamp_start(sleeve_daily, start_ts)
            sleeve_daily.insert(0, "strategy", strategy_name)
            sleeve_daily_rows.append(sleeve_daily)
        base_lot_events = result.attrs.get("base_lot_events")
        base_lot_events = pd.DataFrame() if base_lot_events is None else base_lot_events.copy()
        if not base_lot_events.empty:
            base_lot_events = _filter_timestamp_start(base_lot_events, start_ts)
            base_lot_events.insert(0, "strategy", strategy_name)
            base_lot_event_rows.append(base_lot_events)
        base_deferred = result.attrs.get("base_deferred_candidates")
        base_deferred = pd.DataFrame() if base_deferred is None else base_deferred.copy()
        if not base_deferred.empty:
            base_deferred = _filter_timestamp_start(base_deferred, start_ts)
            base_deferred.insert(0, "strategy", strategy_name)
            base_deferred_rows.append(base_deferred)
        decision_trace = result.attrs.get("decision_trace")
        decision_trace = pd.DataFrame() if decision_trace is None else decision_trace.copy()
        if not decision_trace.empty:
            decision_trace = _filter_timestamp_start(decision_trace, start_ts)
            decision_trace.insert(0, "strategy", strategy_name)
            decision_trace_rows.append(decision_trace)
        candidate_orders = result.attrs.get("candidate_orders")
        candidate_orders = pd.DataFrame() if candidate_orders is None else candidate_orders.copy()
        if not candidate_orders.empty:
            candidate_orders = _filter_timestamp_start(candidate_orders, start_ts)
            candidate_orders.insert(0, "strategy", strategy_name)
            candidate_order_rows.append(candidate_orders)
        risk_assessment = result.attrs.get("risk_assessment_shadow")
        risk_assessment = pd.DataFrame() if risk_assessment is None else risk_assessment.copy()
        if not risk_assessment.empty:
            risk_assessment = _filter_timestamp_start(risk_assessment, start_ts)
            risk_assessment.insert(0, "strategy", strategy_name)
            risk_assessment_rows.append(risk_assessment)
        intent_plan = result.attrs.get("intent_plan_shadow")
        intent_plan = pd.DataFrame() if intent_plan is None else intent_plan.copy()
        if not intent_plan.empty:
            intent_plan = _filter_timestamp_start(intent_plan, start_ts)
            intent_plan.insert(0, "strategy", strategy_name)
            intent_plan_rows.append(intent_plan)
        target_vector = result.attrs.get("target_vector_shadow")
        target_vector = pd.DataFrame() if target_vector is None else target_vector.copy()
        if not target_vector.empty:
            target_vector = _filter_timestamp_start(target_vector, start_ts)
            target_vector.insert(0, "strategy", strategy_name)
            target_vector_rows.append(target_vector)
        budget_ledger = result.attrs.get("budget_ledger_shadow")
        budget_ledger = pd.DataFrame() if budget_ledger is None else budget_ledger.copy()
        if not budget_ledger.empty:
            budget_ledger = _filter_timestamp_start(budget_ledger, start_ts)
            budget_ledger.insert(0, "strategy", strategy_name)
            budget_ledger_rows.append(budget_ledger)
        order_arbiter = result.attrs.get("order_arbiter_shadow")
        order_arbiter = pd.DataFrame() if order_arbiter is None else order_arbiter.copy()
        if not order_arbiter.empty:
            order_arbiter = _filter_timestamp_start(order_arbiter, start_ts)
            order_arbiter.insert(0, "strategy", strategy_name)
            order_arbiter_rows.append(order_arbiter)
        symbol_policy = result.attrs.get("symbol_policy_shadow")
        symbol_policy = pd.DataFrame() if symbol_policy is None else symbol_policy.copy()
        if not symbol_policy.empty:
            symbol_policy = _filter_timestamp_start(symbol_policy, start_ts)
            symbol_policy.insert(0, "strategy", strategy_name)
            symbol_policy_rows.append(symbol_policy)
        recovery_state = result.attrs.get("recovery_state_machine_shadow")
        recovery_state = pd.DataFrame() if recovery_state is None else recovery_state.copy()
        if not recovery_state.empty:
            recovery_state = _filter_timestamp_start(recovery_state, start_ts)
            recovery_state.insert(0, "strategy", strategy_name)
            recovery_state_rows.append(recovery_state)
        lifecycle_state = result.attrs.get("lifecycle_state_shadow")
        lifecycle_state = pd.DataFrame() if lifecycle_state is None else lifecycle_state.copy()
        if not lifecycle_state.empty:
            lifecycle_state = _filter_timestamp_start(lifecycle_state, start_ts)
            lifecycle_state.insert(0, "strategy", strategy_name)
            lifecycle_state_rows.append(lifecycle_state)
        recovery_credit = result.attrs.get("recovery_credit_events")
        recovery_credit = pd.DataFrame() if recovery_credit is None else recovery_credit.copy()
        if not recovery_credit.empty:
            recovery_credit = _filter_timestamp_start(recovery_credit, start_ts)
            recovery_credit.insert(0, "strategy", strategy_name)
            recovery_credit_rows.append(recovery_credit)
        recovery_credit_checks = result.attrs.get("recovery_credit_checks")
        recovery_credit_checks = pd.DataFrame() if recovery_credit_checks is None else recovery_credit_checks.copy()
        if not recovery_credit_checks.empty:
            recovery_credit_checks = _filter_timestamp_start(recovery_credit_checks, start_ts)
            recovery_credit_checks.insert(0, "strategy", strategy_name)
            recovery_credit_check_rows.append(recovery_credit_checks)
        protected_recovery = result.attrs.get("protected_recovery_events")
        protected_recovery = pd.DataFrame() if protected_recovery is None else protected_recovery.copy()
        if not protected_recovery.empty:
            protected_recovery = _filter_timestamp_start(protected_recovery, start_ts)
            protected_recovery.insert(0, "strategy", strategy_name)
            protected_recovery_rows.append(protected_recovery)
        outer_overlay = result.attrs.get("outer_overlay_events")
        outer_overlay = pd.DataFrame() if outer_overlay is None else outer_overlay.copy()
        if not outer_overlay.empty:
            outer_overlay = _filter_timestamp_start(outer_overlay, start_ts)
            outer_overlay.insert(0, "strategy", strategy_name)
            outer_overlay_rows.append(outer_overlay)
        execution_transform_audit = result.attrs.get("execution_transform_audit")
        execution_transform_audit = pd.DataFrame() if execution_transform_audit is None else execution_transform_audit.copy()
        if not execution_transform_audit.empty:
            execution_transform_audit = _filter_timestamp_start(execution_transform_audit, start_ts)
            execution_transform_audit.insert(0, "strategy", strategy_name)
            execution_transform_audit_rows.append(execution_transform_audit)
        actions = result.attrs.get("action_log")
        actions = pd.DataFrame() if actions is None else actions.copy()
        result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
        result.attrs = {}
        if not actions.empty:
            actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
            actions.insert(0, "strategy", strategy_name)
        result.insert(0, "symbol", symbol)
        value_col = f"{symbol}_value"
        result["position_pct"] = np.where(
            result["total_value"].astype(float) > 0,
            result[value_col].astype(float) / result["total_value"].astype(float),
            np.nan,
        )
        result["equity_norm"] = result["total_value"].astype(float) / capital
        price = window_df[["timestamp", "open", "close"]].copy()
        price.insert(0, "symbol", symbol)
        price["price_norm"] = price["close"].astype(float) / float(price["close"].iloc[0])
        price_rows.append(price)
        equity_rows.append(result)
        if not actions.empty:
            action_rows.append(actions)

    if not equity_rows:
        raise SystemExit("No backtest rows generated for the requested window.")
    equity = pd.concat(equity_rows, ignore_index=True)
    actions = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame()
    prices = pd.concat(price_rows, ignore_index=True)
    composite = build_composite(equity, prices)
    diagnostics = pd.DataFrame(diagnostics_rows)
    defense_episodes = _concat_frames(episode_rows)
    risk_cycles = _concat_frames(risk_cycle_rows)
    sleeve_events = _concat_frames(sleeve_event_rows)
    sleeve_daily = _concat_frames(sleeve_daily_rows)
    base_lot_events = _concat_frames(base_lot_event_rows)
    base_deferred = _concat_frames(base_deferred_rows)
    decision_trace = _concat_frames(decision_trace_rows)
    candidate_orders = _concat_frames(candidate_order_rows)
    risk_assessment = _concat_frames(risk_assessment_rows)
    intent_plan = _concat_frames(intent_plan_rows)
    target_vector = _concat_frames(target_vector_rows)
    budget_ledger = _concat_frames(budget_ledger_rows)
    order_arbiter = _concat_frames(order_arbiter_rows)
    symbol_policy = _concat_frames(symbol_policy_rows)
    recovery_state = _concat_frames(recovery_state_rows)
    lifecycle_state = _concat_frames(lifecycle_state_rows)
    recovery_credit = _concat_frames(recovery_credit_rows)
    recovery_credit_checks = _concat_frames(recovery_credit_check_rows)
    protected_recovery = _concat_frames(protected_recovery_rows)
    outer_overlay = _concat_frames(outer_overlay_rows)
    execution_transform_audit = _concat_frames(execution_transform_audit_rows)
    defense_sell_quality = build_defense_sell_quality(actions, prices)
    return {
        "equity": equity,
        "actions": actions,
        "prices": prices,
        "composite": composite,
        "diagnostics": diagnostics,
        "defense_episodes": defense_episodes,
        "risk_cycles": risk_cycles,
        "sleeve_events": sleeve_events,
        "sleeve_daily": sleeve_daily,
        "base_lot_events": base_lot_events,
        "base_deferred_candidates": base_deferred,
        "decision_trace": decision_trace,
        "candidate_orders": candidate_orders,
        "risk_assessment_shadow": risk_assessment,
        "intent_plan_shadow": intent_plan,
        "target_vector_shadow": target_vector,
        "budget_ledger_shadow": budget_ledger,
        "order_arbiter_shadow": order_arbiter,
        "symbol_policy_shadow": symbol_policy,
        "recovery_state_machine_shadow": recovery_state,
        "lifecycle_state_shadow": lifecycle_state,
        "recovery_credit_events": recovery_credit,
        "recovery_credit_checks": recovery_credit_checks,
        "protected_recovery_events": protected_recovery,
        "outer_overlay_events": outer_overlay,
        "execution_transform_audit": execution_transform_audit,
        "defense_sell_quality": defense_sell_quality,
    }


def build_defense_sell_quality(actions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if actions.empty or prices.empty or "setup" not in actions.columns:
        return pd.DataFrame()
    protective = actions[
        (actions["side"] == "sell")
        & actions["setup"].isin(["defense-sell", "structural-exit-sell"])
    ].copy()
    if protective.empty:
        return pd.DataFrame()
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    protective["timestamp"] = pd.to_datetime(protective["timestamp"], utc=True)
    rows: list[dict] = []
    horizons = [7, 30, 60, 90, 180, 365]
    for _, action in protective.iterrows():
        symbol_prices = prices[prices["symbol"] == action["symbol"]].sort_values("timestamp").reset_index(drop=True)
        idx = symbol_prices.index[symbol_prices["timestamp"] >= action["timestamp"]]
        if len(idx) == 0:
            continue
        start_idx = int(idx[0])
        row = {
            "strategy": action.get("strategy", ""),
            "timestamp": action["timestamp"],
            "symbol": action["symbol"],
            "setup": action["setup"],
            "price": float(action.get("price", float("nan"))),
            "quantity": float(action.get("quantity", 0.0) or 0.0),
            "notional": float(action.get("notional", 0.0) or 0.0),
            "actual_step_pct": float(action.get("actual_step_pct", 0.0) or 0.0),
            "actual_position_before": float(action.get("actual_position_before", 0.0) or 0.0),
            "actual_position_after": float(action.get("actual_position_after", 0.0) or 0.0),
            "risk_score": action.get("risk_score"),
            "trend_risk": action.get("trend_risk"),
            "drawdown_risk": action.get("drawdown_risk"),
            "raw_state": action.get("raw_state", ""),
            "confirmed_state": action.get("confirmed_state", ""),
            "target_pct": action.get("target_pct"),
            "mature_target": action.get("mature_target"),
            "phase_target": action.get("phase_target"),
            "execution_target_today": action.get("execution_target_today"),
            "main_intent": action.get("main_intent", ""),
            "base_intent": action.get("base_intent", ""),
            "primary_sleeve": action.get("primary_sleeve", ""),
            "guards": action.get("guards", ""),
            "reason": action.get("reason", ""),
        }
        for horizon in horizons:
            end_idx = start_idx + horizon
            if end_idx >= len(symbol_prices):
                row[f"fwd_{horizon}d"] = float("nan")
                row[f"price_{horizon}d"] = float("nan")
                continue
            future_price = float(symbol_prices.loc[end_idx, "close"])
            row[f"price_{horizon}d"] = future_price
            row[f"fwd_{horizon}d"] = future_price / row["price"] - 1.0 if row["price"] > 0.0 else float("nan")
        row["sell_helped_30d"] = bool(row.get("fwd_30d", float("nan")) < 0.0)
        row["sell_helped_90d"] = bool(row.get("fwd_90d", float("nan")) < 0.0)
        row["missed_upside_90d"] = bool(row.get("fwd_90d", float("nan")) > 0.20)
        rows.append(row)
    return pd.DataFrame(rows)


def build_composite(equity: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    sleeves = []
    bh_sleeves = []
    position_sleeves = []
    for symbol in SYMBOL_ORDER:
        eq = equity[equity["symbol"] == symbol][["timestamp", "equity_norm"]].copy()
        px = prices[prices["symbol"] == symbol][["timestamp", "price_norm"]].copy()
        pos = equity[equity["symbol"] == symbol][["timestamp", "position_pct"]].copy()
        if eq.empty or px.empty:
            continue
        sleeves.append(eq.rename(columns={"equity_norm": symbol}))
        bh_sleeves.append(px.rename(columns={"price_norm": symbol}))
        position_sleeves.append(pos.rename(columns={"position_pct": symbol}))

    # A sleeve whose contract has not listed yet remains in cash. Outer alignment
    # preserves the requested portfolio start instead of silently shortening the
    # annualization window to the latest symbol listing date.
    merged = _merge_on_timestamp(sleeves, fill_value=1.0)
    bh = _merge_on_timestamp(bh_sleeves, fill_value=1.0)
    pos = _merge_on_timestamp(position_sleeves, fill_value=0.0)
    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    symbol_cols = [col for col in merged.columns if col != "timestamp"]
    out["strategy_equity"] = merged[symbol_cols].mean(axis=1)
    out["buy_hold_equity"] = bh[[col for col in bh.columns if col != "timestamp"]].mean(axis=1)
    out["avg_position_pct"] = pos[[col for col in pos.columns if col != "timestamp"]].mean(axis=1)
    out["strategy_drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["buy_hold_drawdown"] = out["buy_hold_equity"] / out["buy_hold_equity"].cummax() - 1.0
    return out


def build_metrics(report: dict[str, pd.DataFrame], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, float | str | int]:
    composite = report["composite"]
    actions = report["actions"]
    equity = report["equity"]
    actual_start, actual_end, years = _actual_metric_span(composite, start_ts, end_ts)
    strategy_total = float(composite["strategy_equity"].iloc[-1] - 1.0)
    bh_total = float(composite["buy_hold_equity"].iloc[-1] - 1.0)
    strategy_annual = _annualized(strategy_total, years)
    bh_annual = _annualized(bh_total, years)
    returns = composite["strategy_equity"].pct_change().dropna()
    bh_returns = composite["buy_hold_equity"].pct_change().dropna()
    metrics = {
        "start": _metric_date(actual_start),
        "end": _metric_date(actual_end),
        "requested_start": str(start_ts.date()),
        "requested_end": str(end_ts.date()),
        "years": years,
        "strategy_total_return": strategy_total,
        "strategy_annual_return": strategy_annual,
        "strategy_max_drawdown": float(composite["strategy_drawdown"].min()),
        "strategy_sharpe_daily": _sharpe(returns),
        "buy_hold_total_return": bh_total,
        "buy_hold_annual_return": bh_annual,
        "buy_hold_max_drawdown": float(composite["buy_hold_drawdown"].min()),
        "buy_hold_sharpe_daily": _sharpe(bh_returns),
        "excess_total_return": strategy_total - bh_total,
        "excess_annual_return": strategy_annual - bh_annual,
        "avg_position_pct": float(composite["avg_position_pct"].mean()),
        "trade_count": int(len(actions)),
    }
    metrics.update(_position_distribution_metrics(equity))
    metrics.update(_reversal_metrics(actions))
    metrics.update(_monthly_churn_metrics(report["prices"], actions, years))
    metrics.update(_setup_count_metrics(actions))
    metrics.update(_v4_guard_metrics(actions, report.get("diagnostics", pd.DataFrame())))
    metrics.update(_execution_transform_metrics(report.get("execution_transform_audit", pd.DataFrame())))
    return metrics


def build_release_manifest(
    *,
    strategy_name: str,
    runner: V1BenchmarkRunner,
    data: dict[str, pd.DataFrame],
    report: dict[str, pd.DataFrame],
    metrics: dict[str, float | str | int],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    config_bytes = runner.config_path.read_bytes()
    strategy = build_strategy(
        strategy_name,
        float(runner.config["capital"]["initial"]),
        float(runner.config["capital"]["reserve"]),
        float(runner.config["cost"]["fee_rate"]),
        min_notional=runner.config.get("cost", {}).get("min_notional"),
    )
    return {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "strategy": strategy_name,
        "git": git_identity(),
        "environment": {
            "python": sys.version.split()[0],
            "packages": dependency_versions(),
        },
        "config": {
            "path": str(runner.config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "content": runner.config,
        },
        "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "execution_assumptions": {
            "mode": runner.config.get("execution", {}).get("mode", "next_open"),
            "intraday_shock_ladder": bool(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_SHOCK_LADDER_V1", False)),
            "fee_rate": float(runner.config["cost"]["fee_rate"]),
            "financing_model": "fixed_borrow_apr_proxy",
            "funding_rates_included": False,
            "exchange_liquidation_model": False,
        },
        "data": {
            symbol: frame_fingerprint(frame, start_ts=start_ts, end_ts=end_ts)
            for symbol, frame in sorted(data.items())
        },
        "portfolio_metrics": metrics,
        "symbol_metrics": symbol_release_metrics(report),
    }


def git_identity() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def dependency_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "SQLAlchemy", "psycopg2-binary", "python-dotenv", "requests")
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def frame_fingerprint(frame: pd.DataFrame, *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict:
    columns = [column for column in ["timestamp", "open", "high", "low", "close", "volume"] if column in frame.columns]
    raw = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)][columns].copy()
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).map(lambda value: value.isoformat())
    payload = raw.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    return {
        "rows": int(len(raw)),
        "start": None if raw.empty else str(raw["timestamp"].iloc[0]),
        "end": None if raw.empty else str(raw["timestamp"].iloc[-1]),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def symbol_release_metrics(report: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int]]:
    equity = report["equity"]
    actions = report["actions"]
    out = {}
    for symbol in SYMBOL_ORDER:
        rows = equity[equity["symbol"] == symbol].sort_values("timestamp")
        if rows.empty:
            continue
        curve = rows["equity_norm"].astype(float)
        drawdown = curve / curve.cummax() - 1.0
        symbol_actions = actions[actions["symbol"] == symbol] if not actions.empty else actions
        out[symbol] = {
            "total_return": float(curve.iloc[-1] - 1.0),
            "max_drawdown": float(drawdown.min()),
            "average_gross": float(rows["position_pct"].astype(float).mean()),
            "max_gross": float(rows["position_pct"].astype(float).max()),
            "trade_count": int(len(symbol_actions)),
        }
    return out


def _execution_transform_metrics(audit: pd.DataFrame) -> dict[str, float | int]:
    if audit.empty:
        return {
            "execution_transform_financing_cost": 0.0,
            "execution_transform_margin_call_count": 0,
            "execution_transform_gross_warning_count": 0,
            "execution_transform_max_gross_position": 0.0,
            "execution_transform_max_debt_to_equity": 0.0,
            "execution_transform_min_margin_buffer": 0.0,
        }
    out: dict[str, float | int] = {}
    if "financing_cost_today" in audit.columns:
        out["execution_transform_financing_cost"] = float(audit["financing_cost_today"].astype(float).sum())
    else:
        out["execution_transform_financing_cost"] = 0.0
    out["execution_transform_margin_call_count"] = int(audit.get("margin_call_est", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    out["execution_transform_gross_warning_count"] = int(audit.get("gross_warning", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    out["execution_transform_max_gross_position"] = float(audit.get("gross_position", pd.Series([0.0])).astype(float).max())
    out["execution_transform_max_debt_to_equity"] = float(audit.get("debt_to_equity", pd.Series([0.0])).astype(float).max())
    out["execution_transform_min_margin_buffer"] = float(audit.get("margin_buffer", pd.Series([0.0])).astype(float).min())
    return out


def _actual_metric_span(
    composite: pd.DataFrame,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    if composite.empty or "timestamp" not in composite.columns:
        years = max((requested_end - requested_start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
        return requested_start, requested_end, years
    timestamps = pd.to_datetime(composite["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        years = max((requested_end - requested_start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
        return requested_start, requested_end, years
    actual_start = timestamps.iloc[0]
    actual_end = timestamps.iloc[-1]
    years = max((actual_end - actual_start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    return actual_start, actual_end, years


def _metric_date(timestamp: pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return str(ts.tz_convert("Asia/Singapore").date())


def render_chart(
    report: dict[str, pd.DataFrame],
    metrics: dict[str, float | str | int],
    strategy: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    output: Path,
) -> None:
    prices = report["prices"].copy()
    actions = report["actions"].copy()
    composite = report["composite"].copy()
    equity = report["equity"].copy()
    for frame in (prices, actions, composite, equity):
        if not frame.empty and "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    width = 1600
    height = 1080
    left = 86
    right = 42
    top = 98
    gap = 34
    panel_heights = [320, 210, 180, 170]
    panel_titles = [
        "Price path with buys (up markers) and sells (down markers), log scale",
        "Total amount curve",
        "Position curve",
        "Drawdown",
    ]
    panels = []
    y0 = top
    for ph in panel_heights:
        panels.append((left, y0, width - left - right, ph))
        y0 += ph + gap

    x_min = pd.Timestamp(start_ts).timestamp()
    x_max = pd.Timestamp(end_ts).timestamp()

    def sx(ts: pd.Series | pd.Timestamp, panel: tuple[int, int, int, int]) -> np.ndarray | float:
        x0, _, w, _ = panel
        if isinstance(ts, pd.Series):
            values = pd.to_datetime(ts, utc=True).map(pd.Timestamp.timestamp).to_numpy(dtype=float)
            return x0 + (values - x_min) / (x_max - x_min) * w
        return x0 + (pd.Timestamp(ts).timestamp() - x_min) / (x_max - x_min) * w

    def sy(values: pd.Series | np.ndarray, panel: tuple[int, int, int, int], ymin: float, ymax: float, *, log: bool = False) -> np.ndarray:
        _, y, _, h = panel
        vals = np.asarray(values, dtype=float)
        if log:
            vals = np.log(np.maximum(vals, 1e-9))
            ymin_l = math.log(max(ymin, 1e-9))
            ymax_l = math.log(max(ymax, 1e-9))
            return y + h - (vals - ymin_l) / (ymax_l - ymin_l) * h
        return y + h - (vals - ymin) / (ymax - ymin) * h

    price_vals = prices["price_norm"].astype(float)
    price_ymin = max(float(price_vals.min()) * 0.82, 0.03)
    price_ymax = float(price_vals.max()) * 1.18
    equity_ymin = 0.85
    equity_ymax = max(float(composite["strategy_equity"].max()), float(composite["buy_hold_equity"].max())) * 1.08
    dd_ymin = min(float(composite["strategy_drawdown"].min()), float(composite["buy_hold_drawdown"].min())) * 1.08

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:23px;font-weight:700;fill:#111827}.sub{font-size:14px;fill:#475569}.panel-title{font-size:15px;font-weight:700;fill:#111827}.axis{font-size:12px;fill:#64748b}.grid{stroke:#e2e8f0;stroke-width:1}.frame{fill:#ffffff;stroke:#cbd5e1;stroke-width:1}.legend{font-size:12px;fill:#334155}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="54" y="38" class="title">{html.escape(strategy)} full-window strategy review</text>',
    ]
    subtitle = (
        f"{start_ts.date()} to {end_ts.date()} | "
        f"Annual {metrics['strategy_annual_return']:.2%} vs BH {metrics['buy_hold_annual_return']:.2%} | "
        f"MDD {metrics['strategy_max_drawdown']:.2%} vs BH {metrics['buy_hold_max_drawdown']:.2%} | "
        f"Avg position {metrics['avg_position_pct']:.1%} | Trades {metrics['trade_count']}"
    )
    parts.append(f'<text x="54" y="65" class="sub">{html.escape(subtitle)}</text>')

    for idx, panel in enumerate(panels):
        x, y, w, h = panel
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="0" class="frame"/>')
        parts.append(f'<text x="{x + 10}" y="{y + 22}" class="panel-title">{html.escape(panel_titles[idx])}</text>')
        for frac in [0.25, 0.5, 0.75]:
            gy = y + h * frac
            parts.append(f'<line x1="{x}" y1="{gy:.1f}" x2="{x + w}" y2="{gy:.1f}" class="grid"/>')
    for tick in _year_ticks(start_ts, end_ts):
        tx = sx(tick, panels[-1])
        for panel in panels:
            x, y, _, h = panel
            parts.append(f'<line x1="{tx:.1f}" y1="{y}" x2="{tx:.1f}" y2="{y + h}" class="grid"/>')
        parts.append(f'<text x="{tx - 14:.1f}" y="{panels[-1][1] + panels[-1][3] + 20}" class="axis">{tick.year}</text>')

    price_panel = panels[0]
    for symbol in SYMBOL_ORDER:
        px = prices[prices["symbol"] == symbol]
        if px.empty:
            continue
        parts.append(_polyline(sx(px["timestamp"], price_panel), sy(px["price_norm"], price_panel, price_ymin, price_ymax, log=True), COLORS[symbol], 2.0))
        if not actions.empty:
            acts = actions[actions["symbol"] == symbol].copy()
            if not acts.empty:
                acts["price_norm"] = acts["price"].astype(float) / float(px["close"].iloc[0])
                for _, row in acts.iterrows():
                    x = float(sx(row["timestamp"], price_panel))
                    y = float(sy(np.array([row["price_norm"]]), price_panel, price_ymin, price_ymax, log=True)[0])
                    parts.append(_marker(x, y, COLORS[symbol], up=(row["side"] == "buy")))

    equity_panel = panels[1]
    parts.append(_polyline(sx(composite["timestamp"], equity_panel), sy(composite["strategy_equity"], equity_panel, equity_ymin, equity_ymax), "#111827", 2.4))
    parts.append(_polyline(sx(composite["timestamp"], equity_panel), sy(composite["buy_hold_equity"], equity_panel, equity_ymin, equity_ymax), "#64748b", 2.0, dash=True))

    pos_panel = panels[2]
    pos_ymax = max(1.0, float(equity["position_pct"].astype(float).max()) * 1.08)
    for symbol in SYMBOL_ORDER:
        eq = equity[equity["symbol"] == symbol]
        if not eq.empty:
            parts.append(_polyline(sx(eq["timestamp"], pos_panel), sy(eq["position_pct"], pos_panel, 0.0, pos_ymax), COLORS[symbol], 1.7))
    parts.append(_polyline(sx(composite["timestamp"], pos_panel), sy(composite["avg_position_pct"], pos_panel, 0.0, pos_ymax), "#111827", 2.1))

    dd_panel = panels[3]
    dd_x = sx(composite["timestamp"], dd_panel)
    dd_y = sy(composite["strategy_drawdown"], dd_panel, dd_ymin, 0.0)
    zero_y = float(sy(np.array([0.0]), dd_panel, dd_ymin, 0.0)[0])
    fill_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in zip(dd_x, dd_y)])
    fill_points += f" {dd_x[-1]:.1f},{zero_y:.1f} {dd_x[0]:.1f},{zero_y:.1f}"
    parts.append(f'<polygon points="{fill_points}" fill="#ef4444" opacity="0.22"/>')
    parts.append(_polyline(dd_x, dd_y, "#b91c1c", 1.8))
    parts.append(_polyline(sx(composite["timestamp"], dd_panel), sy(composite["buy_hold_drawdown"], dd_panel, dd_ymin, 0.0), "#64748b", 1.7, dash=True))

    legend_items = [(symbol.split("/")[0], COLORS[symbol]) for symbol in SYMBOL_ORDER]
    legend_items.extend([
        ("Strategy", "#111827"),
        ("Buy-hold", "#64748b"),
    ])
    lx = width - 520
    ly = 36
    for label, color in legend_items:
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 36}" y="{ly + 4}" class="legend">{label}</text>')
        lx += 100

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def _merge_on_timestamp(frames: list[pd.DataFrame], *, fill_value: float) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on="timestamp", how="outer")
    out = out.sort_values("timestamp").reset_index(drop=True)
    value_columns = [column for column in out.columns if column != "timestamp"]
    out[value_columns] = out[value_columns].ffill().fillna(fill_value)
    return out


def _filter_timestamp_start(frame: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame.columns:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out[out["timestamp"] >= start_ts].reset_index(drop=True)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = [
        frame.dropna(axis=1, how="all")
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not cleaned:
        return pd.DataFrame()
    return pd.concat(cleaned, ignore_index=True)


def _polyline(xs: np.ndarray, ys: np.ndarray, color: str, width: float, *, dash: bool = False) -> str:
    points = " ".join(
        f"{float(x):.1f},{float(y):.1f}"
        for x, y in zip(xs, ys)
        if np.isfinite(x) and np.isfinite(y)
    )
    dash_attr = ' stroke-dasharray="8 6"' if dash else ""
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'


def _marker(x: float, y: float, color: str, *, up: bool) -> str:
    size = 7.0
    if up:
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
    else:
        points = [(x, y + size), (x - size, y - size), (x + size, y - size)]
    point_text = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    return f'<polygon points="{point_text}" fill="{color}" stroke="#ffffff" stroke-width="1" opacity="0.95"/>'


def _year_ticks(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
    first = pd.Timestamp(year=start_ts.year, month=1, day=1, tz="UTC")
    if first < start_ts:
        first = pd.Timestamp(year=start_ts.year + 1, month=1, day=1, tz="UTC")
    ticks = []
    current = first
    while current <= end_ts:
        ticks.append(current)
        current = pd.Timestamp(year=current.year + 1, month=1, day=1, tz="UTC")
    return ticks


def _as_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _annualized(total_return: float, years: float) -> float:
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(returns.mean() / std * math.sqrt(365.0))


def _position_distribution_metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty or "position_pct" not in equity.columns:
        return {}
    pct = equity["position_pct"].astype(float).dropna()
    total = max(len(pct), 1)
    return {
        "position_pct_le_20": float((pct <= 0.20).sum() / total),
        "position_pct_20_40": float(((pct > 0.20) & (pct <= 0.40)).sum() / total),
        "position_pct_40_70": float(((pct > 0.40) & (pct <= 0.70)).sum() / total),
        "position_pct_70_90": float(((pct > 0.70) & (pct <= 0.90)).sum() / total),
        "position_pct_gt_90": float((pct > 0.90).sum() / total),
    }


def _reversal_metrics(actions: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "reversal_rate_7d": 0.0,
        "reversal_rate_14d": 0.0,
        "reversal_rate_30d": 0.0,
        "target_gap_to_reduce_30d": 0,
        "target_reduce_to_gap_30d": 0,
    }
    if actions.empty or not {"timestamp", "symbol", "side"}.issubset(actions.columns):
        return out
    acts = actions.copy()
    acts["timestamp"] = pd.to_datetime(acts["timestamp"], utc=True)
    acts = acts.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for days in (7, 14, 30):
        reversed_count = 0
        total = 0
        for _, group in acts.groupby("symbol"):
            rows = group.reset_index(drop=True)
            for idx, row in rows.iterrows():
                total += 1
                later = rows.iloc[idx + 1 :]
                later = later[later["timestamp"] <= row["timestamp"] + pd.Timedelta(days=days)]
                if not later.empty and (later["side"] != row["side"]).any():
                    reversed_count += 1
        out[f"reversal_rate_{days}d"] = float(reversed_count / total) if total else 0.0
    reasons = acts.get("reason", pd.Series("", index=acts.index)).astype(str)
    acts["setup"] = np.select(
        [reasons.str.contains("_buy_target-gap_", regex=False), reasons.str.contains("_sell_target-reduce_", regex=False)],
        ["target-gap", "target-reduce"],
        default="",
    )
    gap_to_reduce = 0
    reduce_to_gap = 0
    for _, group in acts.groupby("symbol"):
        rows = group.reset_index(drop=True)
        for idx, row in rows.iterrows():
            if row["setup"] not in {"target-gap", "target-reduce"}:
                continue
            later = rows.iloc[idx + 1 :]
            later = later[later["timestamp"] <= row["timestamp"] + pd.Timedelta(days=30)]
            if row["setup"] == "target-gap" and (later["setup"] == "target-reduce").any():
                gap_to_reduce += 1
            if row["setup"] == "target-reduce" and (later["setup"] == "target-gap").any():
                reduce_to_gap += 1
    out["target_gap_to_reduce_30d"] = int(gap_to_reduce)
    out["target_reduce_to_gap_30d"] = int(reduce_to_gap)
    return out


def _monthly_churn_metrics(prices: pd.DataFrame, actions: pd.DataFrame, years: float) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "monthly_trade_count_mean": 0.0,
        "monthly_trade_count_max": 0,
        "monthly_symbol_trade_count_max": 0,
        "rolling_30d_trade_count_max": 0,
        "rolling_30d_symbol_trade_count_max": 0,
        "turnover_proxy_annual": 0.0,
        "low_net_move_high_trade_months": 0,
        "low_net_move_month_trade_count_max": 0,
    }
    if actions.empty:
        return out
    acts = actions.copy()
    acts["timestamp"] = pd.to_datetime(acts["timestamp"], utc=True)
    acts["month"] = acts["timestamp"].dt.tz_convert(None).dt.to_period("M")
    monthly_trades = acts.groupby("month").size()
    out["monthly_trade_count_mean"] = float(monthly_trades.mean()) if not monthly_trades.empty else 0.0
    out["monthly_trade_count_max"] = int(monthly_trades.max()) if not monthly_trades.empty else 0
    monthly_symbol_trades = acts.groupby(["symbol", "month"]).size() if "symbol" in acts.columns else pd.Series(dtype=int)
    out["monthly_symbol_trade_count_max"] = int(monthly_symbol_trades.max()) if not monthly_symbol_trades.empty else 0
    out["rolling_30d_trade_count_max"] = _rolling_trade_count_max(acts)
    out["rolling_30d_symbol_trade_count_max"] = _rolling_trade_count_max(acts, by_symbol=True)
    if {"quantity", "price"}.issubset(acts.columns):
        out["turnover_proxy_annual"] = float((acts["quantity"].astype(float) * acts["price"].astype(float)).sum() / max(years, 1e-9))
    if prices.empty:
        return out
    px = prices.copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
    px["month"] = px["timestamp"].dt.tz_convert(None).dt.to_period("M")
    low_trade_counts = []
    for (symbol, month), group in px.groupby(["symbol", "month"]):
        if group.empty:
            continue
        start_price = float(group["close"].iloc[0])
        end_price = float(group["close"].iloc[-1])
        if start_price <= 0 or abs(end_price / start_price - 1.0) >= 0.12:
            continue
        count = int(len(acts[(acts["symbol"] == symbol) & (acts["month"] == month)]))
        if count >= 4:
            low_trade_counts.append(count)
    out["low_net_move_high_trade_months"] = int(len(low_trade_counts))
    out["low_net_move_month_trade_count_max"] = int(max(low_trade_counts) if low_trade_counts else 0)
    return out


def _rolling_trade_count_max(actions: pd.DataFrame, *, by_symbol: bool = False) -> int:
    if actions.empty or "timestamp" not in actions.columns:
        return 0
    groups = [("", actions)]
    if by_symbol and "symbol" in actions.columns:
        groups = list(actions.groupby("symbol", sort=False))
    max_count = 0
    for _, group in groups:
        times = pd.to_datetime(group["timestamp"], utc=True).sort_values().reset_index(drop=True)
        left = 0
        for right, ts in enumerate(times):
            cutoff = ts - pd.Timedelta(days=30)
            while left <= right and times.iloc[left] < cutoff:
                left += 1
            max_count = max(max_count, right - left + 1)
    return int(max_count)


def _setup_count_metrics(actions: pd.DataFrame) -> dict[str, int]:
    setups = [
        "target-gap",
        "target-reduce",
        "risk-reduce",
        "starter-buy",
        "repair-add",
        "value-recovery",
        "trend-add",
        "trend-cont",
        "opportunity-add",
        "recovery-probe-buy",
        "defense-sell",
        "structural-exit-sell",
        "soft-defense-sell",
        "distribution-sell",
        "defense-release-buy",
    ]
    out = {f"buy_{setup}_count": 0 for setup in setups}
    out.update({f"sell_{setup}_count": 0 for setup in setups})
    if actions.empty or not {"side", "setup"}.issubset(actions.columns):
        return out
    for setup in setups:
        out[f"buy_{setup}_count"] = int(((actions["side"] == "buy") & (actions["setup"] == setup)).sum())
        out[f"sell_{setup}_count"] = int(((actions["side"] == "sell") & (actions["setup"] == setup)).sum())
    return out


def _v4_guard_metrics(actions: pd.DataFrame, diagnostics: pd.DataFrame | None = None) -> dict[str, int]:
    """Return current Official V1 diagnostic counters without historical V4.0 noise."""
    out: dict[str, int] = {}
    if diagnostics is not None and not diagnostics.empty:
        for key in sorted(diagnostics.columns):
            if str(key).startswith("core_"):
                out[str(key)] = int(pd.to_numeric(diagnostics[key], errors="coerce").fillna(0).sum())
    if actions.empty or "reason" not in actions.columns:
        return out
    reason_text = actions["reason"].fillna("").astype(str)
    current_tags = [
        "core_bear_base",
        "core_bear_base_exit",
        "core_deep_base_recovery",
        "core_intent_accumulate",
        "core_intent_defend",
        "core_intent_distribute",
        "core_intent_hold",
        "core_limited_recovery_overlay",
        "core_post_crash_recoil",
        "core_recovery_credit_soft",
        "core_regime_bear",
        "core_regime_bull",
        "core_regime_range",
        "core_regime_transition",
        "core_staged_recovery",
    ]
    for tag in current_tags:
        out[f"{tag}_count"] = int(reason_text.str.contains(tag, regex=False).sum())
    return out
def _metric_line(metrics: dict[str, float | str | int]) -> str:
    return (
        f"strategy annual={metrics['strategy_annual_return']:.2%}, "
        f"total={metrics['strategy_total_return']:.2%}, "
        f"mdd={metrics['strategy_max_drawdown']:.2%}; "
        f"buy_hold annual={metrics['buy_hold_annual_return']:.2%}, "
        f"total={metrics['buy_hold_total_return']:.2%}, "
        f"mdd={metrics['buy_hold_max_drawdown']:.2%}"
    )


if __name__ == "__main__":
    main()

