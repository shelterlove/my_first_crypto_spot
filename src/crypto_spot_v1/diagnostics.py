"""Per-bar diagnostic logging for V1 evaluation runs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import build_strategy
from .strategy_rebalance import PositionState
from .strategy_utils import compute_indicators, detect_market_state


def write_diagnostics_outputs(
    *,
    run_dir: Path,
    runner,
    raw_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
    candidate_name: str,
    mode: str,
) -> dict[str, pd.DataFrame]:
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)

    per_bar = build_per_bar_diagnostics(
        runner=runner,
        raw_df=raw_df,
        actions_df=actions_df,
        equity_df=equity_df,
        metadata=metadata,
        config=config,
        candidate_name=candidate_name,
        mode=mode,
    )
    summary = build_diagnostic_summary(per_bar)
    quality = build_diagnostic_quality_report(per_bar, summary, actions_df, equity_df, raw_df, candidate_name)
    risk_attr = build_risk_score_attribution_report(per_bar, runner)
    exposure = build_exposure_diagnostics_report(per_bar)
    buy_blocked = build_buy_blocked_report(per_bar, runner)
    sell_early = build_sell_too_early_report(per_bar, runner, config)

    per_bar.to_csv(diagnostics_dir / "per_bar_diagnostics.csv.gz", index=False, compression="gzip")
    summary.to_csv(diagnostics_dir / "diagnostic_summary.csv", index=False)
    quality.to_csv(diagnostics_dir / "diagnostic_quality_report.csv", index=False)
    risk_attr.to_csv(run_dir / "risk_score_attribution_report.csv", index=False)
    exposure.to_csv(run_dir / "exposure_diagnostics_report.csv", index=False)
    buy_blocked.to_csv(run_dir / "buy_blocked_report.csv", index=False)
    sell_early.to_csv(run_dir / "sell_too_early_report.csv", index=False)

    return {
        "per_bar_diagnostics": per_bar,
        "diagnostic_summary": summary,
        "diagnostic_quality_report": quality,
        "risk_score_attribution_report": risk_attr,
        "exposure_diagnostics_report": exposure,
        "buy_blocked_report": buy_blocked,
        "sell_too_early_report": sell_early,
    }


def build_per_bar_diagnostics(
    *,
    runner,
    raw_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    metadata: dict[str, Any],
    config: dict[str, Any],
    candidate_name: str,
    mode: str,
) -> pd.DataFrame:
    candidate = raw_df[raw_df["strategy_name"] == candidate_name].copy()
    if candidate.empty:
        return pd.DataFrame(columns=PER_BAR_COLUMNS)

    warmup_bars = int(config.get("warmup_bars", 200))
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    initial_cash = float(config.get("capital", {}).get("initial", 100.0))
    reserve = float(config.get("capital", {}).get("reserve", 0.0))
    fee_rate = float(config.get("cost", {}).get("fee_rate", 0.0))
    strategy = build_strategy(candidate_name, initial_cash, reserve, fee_rate)
    all_dfs = runner._inject_btc_regime()  # Evaluation-only reuse of runner data preparation.

    actions = actions_df.copy()
    equity = equity_df.copy()
    if not actions.empty:
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)

    rows: list[dict[str, Any]] = []
    for _, window in candidate.iterrows():
        symbol = window["symbol"]
        df = all_dfs[symbol].copy()
        if "ema168" not in df.columns:
            df = compute_indicators(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        window_start = pd.to_datetime(window["window_start"], utc=True)
        window_end = pd.to_datetime(window["window_end"], utc=True)
        eval_matches = df.index[df["timestamp"] == window_start].tolist()
        if not eval_matches:
            continue
        eval_start = eval_matches[0]
        eval_end = int(df.index[df["timestamp"] == window_end][0]) + 1
        warmup_start = max(0, eval_start - warmup_bars)
        window_id = window["window_label"]

        equity_window = _window_equity(equity, symbol, window_id)
        actions_window = _window_actions(actions, symbol, window_id)
        equity_by_ts = {pd.Timestamp(r["timestamp"]): r for r in equity_window.to_dict("records")}
        actions_by_ts = _actions_by_timestamp(actions_window)

        confirmed_states = _confirmed_state_sequence(df, eval_start, eval_end, strategy)
        peak_price = 0.0
        last_buy_call = -48
        call_count = 0
        prev_risk_score = np.nan
        raw_streak = 0
        confirmed_streak = 0
        last_raw_state = None
        last_confirmed_state = None

        for pos in range(warmup_start, eval_end):
            is_warmup = pos < eval_start
            is_trading = not is_warmup
            bar = df.iloc[pos]
            ts = pd.Timestamp(bar["timestamp"])
            signal_pos = pos - 1 if is_trading and execution_mode != "same_close" else pos
            signal = df.iloc[signal_pos] if signal_pos >= 0 else bar
            equity_row = equity_by_ts.get(ts)
            action_row = actions_by_ts.get(ts)
            action_side = action_row.get("side") if action_row else "none"
            reason = action_row.get("reason") if action_row else ""
            setup = _extract_setup(reason)

            actual_pct, qty, cash, pos_value, total_value, avg_cost = _position_from_equity(equity_row, symbol)
            if is_trading:
                call_count += 1
                if actual_pct < 0.20:
                    peak_price = float(bar["close"])
                elif qty > 1e-12:
                    peak_price = max(peak_price, float(bar["close"]))
            else:
                peak_price = np.nan

            raw_state = detect_market_state(signal) if is_trading else detect_market_state(bar)
            confirmed_state = confirmed_states.get(pos) if is_trading else np.nan
            raw_prev = detect_market_state(df.iloc[signal_pos - 1]) if is_trading and signal_pos > 0 else np.nan
            conf_prev = confirmed_states.get(pos - 1) if is_trading else np.nan
            raw_streak = raw_streak + 1 if raw_state == last_raw_state else 1
            last_raw_state = raw_state
            if is_trading and not pd.isna(confirmed_state):
                confirmed_streak = confirmed_streak + 1 if confirmed_state == last_confirmed_state else 1
                last_confirmed_state = confirmed_state
            else:
                confirmed_streak = np.nan

            trend_risk = strategy._calculate_trend_risk(signal, float(bar["open"])) if is_trading else np.nan
            pos_state = PositionState(quantity=qty if not math.isnan(qty) else 0.0, avg_cost=avg_cost if not math.isnan(avg_cost) else 0.0)
            if is_trading and not math.isnan(peak_price):
                strategy._peak_price = peak_price
            drawdown_risk = strategy._calculate_drawdown_risk(signal, pos_state, float(bar["open"])) if is_trading else np.nan
            risk_score = min(int(trend_risk + drawdown_risk), 5) if is_trading else np.nan

            targets = _target_diagnostics(strategy, symbol, signal, raw_state, confirmed_state, risk_score, float(bar["open"])) if is_trading else {}
            cooldown_remaining = max(0, _buy_cooldown(strategy, confirmed_state, risk_score) - (call_count - last_buy_call)) if is_trading else np.nan
            if action_side == "buy":
                last_buy_call = call_count

            can_buy = bool(is_trading and actual_pct < targets.get("target_position_pct_final", np.nan) - strategy.MIN_ADJUST_THRESHOLD)
            can_sell = bool(is_trading and actual_pct > targets.get("target_position_pct_final", np.nan) + strategy.MIN_ADJUST_THRESHOLD)
            blocked = _blocked_flags(signal, confirmed_state, trend_risk, cooldown_remaining, reason, can_buy)
            no_trade_reason = _no_trade_reason(action_side, can_buy, can_sell, blocked)

            btc_regime_ts = signal.get("btc_regime_timestamp") if is_trading else bar.get("btc_regime_timestamp")
            indicator_ts = signal.get("timestamp")
            signal_ts = action_row.get("signal_timestamp") if action_row else (indicator_ts if is_trading else np.nan)
            execution_ts = action_row.get("timestamp") if action_row else (ts if is_trading else np.nan)
            ts_pass, ts_error = _timestamp_check(indicator_ts, signal_ts, execution_ts, btc_regime_ts)
            missing = _missing_fields({
                "confirmed_state": confirmed_state,
                "cash": cash,
                "target_position_pct_final": targets.get("target_position_pct_final"),
                "btc_regime_timestamp": btc_regime_ts,
            })

            rows.append({
                "run_id": metadata["run_id"],
                "strategy_name": candidate_name,
                "candidate_name": candidate_name,
                "mode": mode,
                "symbol": symbol,
                "window_id": window_id,
                "timestamp": ts,
                "bar_index": pos - warmup_start,
                "is_warmup": is_warmup,
                "is_trading_bar": is_trading,
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
                "ema24": signal.get("ema24"),
                "ema72": signal.get("ema72"),
                "ema168": signal.get("ema168"),
                "ema24_gt_ema72": _gt(signal.get("ema24"), signal.get("ema72")),
                "ema72_gt_ema168": _gt(signal.get("ema72"), signal.get("ema168")),
                "price_gt_ema24": _gt(bar.get("open"), signal.get("ema24")),
                "price_gt_ema72": _gt(bar.get("open"), signal.get("ema72")),
                "price_gt_ema168": _gt(bar.get("open"), signal.get("ema168")),
                "ema168_slope": signal.get("ema168_slope"),
                "ema168_slope_pct": signal.get("ema168_slope"),
                "atr": signal.get("atr14"),
                "atr_pct_rank": signal.get("atr_pct_rank"),
                "natr": signal.get("atr_pct"),
                "donchian_high": signal.get("high_120"),
                "donchian_low": _donchian_low(signal),
                "donchian_pos": signal.get("donchian_pos"),
                "donchian_width": _donchian_width(signal),
                "raw_state": raw_state,
                "confirmed_state": confirmed_state,
                "raw_state_prev": raw_prev,
                "confirmed_state_prev": conf_prev,
                "raw_state_changed": raw_state != raw_prev if not pd.isna(raw_prev) else False,
                "confirmed_state_changed": confirmed_state != conf_prev if not pd.isna(conf_prev) else False,
                "raw_state_bars": raw_streak,
                "confirmed_state_bars": confirmed_streak,
                "btc_regime": signal.get("btc_regime"),
                "btc_regime_prev": df.iloc[signal_pos - 1].get("btc_regime") if is_trading and signal_pos > 0 else np.nan,
                "btc_regime_changed": signal.get("btc_regime") != (df.iloc[signal_pos - 1].get("btc_regime") if is_trading and signal_pos > 0 else signal.get("btc_regime")),
                "btc_regime_timestamp": btc_regime_ts,
                "btc_regime_timestamp_missing": pd.isna(btc_regime_ts),
                "btc_regime_adjustment": targets.get("btc_regime_adjustment"),
                "btc_regime_target_gap_mult": strategy.BTC_BEAR_TARGET_GAP_MULT if signal.get("btc_regime") == "BEAR" else 1.0,
                "trend_risk": trend_risk,
                "drawdown_risk": drawdown_risk,
                "risk_score": risk_score,
                "risk_score_prev": prev_risk_score,
                "risk_score_changed": risk_score != prev_risk_score if not pd.isna(prev_risk_score) and not pd.isna(risk_score) else False,
                "target_position_pct_raw": targets.get("target_position_pct_raw"),
                "target_position_pct_after_risk": targets.get("target_position_pct_after_risk"),
                "target_position_pct_after_btc": targets.get("target_position_pct_after_btc"),
                "target_position_pct_final": targets.get("target_position_pct_final"),
                "actual_position_pct": actual_pct,
                "position_qty": qty,
                "cash": cash,
                "position_value": pos_value,
                "total_value": total_value,
                "reserve": reserve,
                "avg_cost": avg_cost,
                "unrealized_pnl_pct": (float(bar["close"]) / avg_cost - 1) if avg_cost and avg_cost > 0 else np.nan,
                "peak_price_since_entry": peak_price,
                "drawdown_from_peak_pct": (1 - float(bar["close"]) / peak_price) if peak_price and peak_price > 0 else np.nan,
                "action": action_side,
                "action_type": action_side,
                "buy_setup": setup if action_side == "buy" else "",
                "sell_reason": setup if action_side == "sell" else "",
                "action_reason": reason,
                "target_gap": targets.get("target_position_pct_final", np.nan) - actual_pct if not math.isnan(actual_pct) else np.nan,
                "trade_qty": action_row.get("quantity") if action_row else np.nan,
                "trade_notional": action_row.get("notional") if action_row else np.nan,
                "fee_cost": action_row.get("fee") if action_row else 0.0,
                "execution_price": action_row.get("price") if action_row else (bar.get("open") if is_trading else np.nan),
                "signal_timestamp": signal_ts,
                "execution_timestamp": execution_ts,
                "indicator_timestamp": indicator_ts,
                "cooldown_remaining": cooldown_remaining,
                "can_buy": can_buy,
                "can_sell": can_sell,
                **blocked,
                "no_trade_reason": no_trade_reason,
                "timestamp_check_pass": ts_pass,
                "timestamp_check_error": ts_error,
                "accounting_check_pass": _accounting_check(equity_row, symbol),
                "diagnostics_missing_fields": ";".join(missing),
            })
            if is_trading:
                prev_risk_score = risk_score
    return pd.DataFrame(rows, columns=PER_BAR_COLUMNS)


def build_diagnostic_summary(per_bar: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "total_bars", "warmup_bars", "trading_bars",
        "action_count", "buy_count", "sell_count", "no_trade_count",
        "avg_actual_position_pct", "median_actual_position_pct",
        "avg_target_position_pct_final", "target_actual_gap_mean",
        "target_actual_gap_median", "target_actual_gap_max", "avg_risk_score",
        "max_risk_score", "raw_state_change_count", "confirmed_state_change_count",
        "btc_regime_change_count", "avg_atr_pct_rank", "high_vol_bar_ratio",
        "bull_bar_ratio", "bear_bar_ratio", "mixed_bar_ratio",
        "btc_bear_bar_ratio", "blocked_by_cooldown_count",
        "blocked_by_donchian_count", "blocked_by_volatility_count",
        "blocked_by_trend_risk_count", "blocked_by_btc_bear_count",
        "diagnostics_missing_field_count",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (symbol, window_id), group in per_bar.groupby(["symbol", "window_id"], dropna=False):
        trading = group[group["is_trading_bar"]]
        gap = trading["target_position_pct_final"] - trading["actual_position_pct"]
        rows.append({
            "symbol": symbol,
            "window_id": window_id,
            "total_bars": len(group),
            "warmup_bars": int(group["is_warmup"].sum()),
            "trading_bars": len(trading),
            "action_count": int((trading["action"] != "none").sum()),
            "buy_count": int((trading["action"] == "buy").sum()),
            "sell_count": int((trading["action"] == "sell").sum()),
            "no_trade_count": int((trading["action"] == "none").sum()),
            "avg_actual_position_pct": trading["actual_position_pct"].mean(),
            "median_actual_position_pct": trading["actual_position_pct"].median(),
            "avg_target_position_pct_final": trading["target_position_pct_final"].mean(),
            "target_actual_gap_mean": gap.mean(),
            "target_actual_gap_median": gap.median(),
            "target_actual_gap_max": gap.abs().max(),
            "avg_risk_score": trading["risk_score"].mean(),
            "max_risk_score": trading["risk_score"].max(),
            "raw_state_change_count": int(trading["raw_state_changed"].sum()),
            "confirmed_state_change_count": int(trading["confirmed_state_changed"].sum()),
            "btc_regime_change_count": int(trading["btc_regime_changed"].sum()),
            "avg_atr_pct_rank": trading["atr_pct_rank"].mean(),
            "high_vol_bar_ratio": (trading["atr_pct_rank"] >= 0.8).mean(),
            "bull_bar_ratio": (trading["confirmed_state"] == "BULL").mean(),
            "bear_bar_ratio": (trading["confirmed_state"] == "BEAR").mean(),
            "mixed_bar_ratio": (trading["confirmed_state"] == "MIXED").mean(),
            "btc_bear_bar_ratio": (trading["btc_regime"] == "BEAR").mean(),
            "blocked_by_cooldown_count": int(trading["blocked_by_cooldown"].sum()),
            "blocked_by_donchian_count": int(trading["blocked_by_donchian"].sum()),
            "blocked_by_volatility_count": int(trading["blocked_by_volatility"].sum()),
            "blocked_by_trend_risk_count": int(trading["blocked_by_trend_risk"].sum()),
            "blocked_by_btc_bear_count": int(trading["blocked_by_btc_bear"].sum()),
            "diagnostics_missing_field_count": int(trading["diagnostics_missing_fields"].fillna("").ne("").sum()),
        })
    return pd.DataFrame(rows, columns=columns)


def build_diagnostic_quality_report(
    per_bar: pd.DataFrame,
    summary: pd.DataFrame,
    actions_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    candidate_name: str,
) -> pd.DataFrame:
    rows = []
    _quality(rows, "per_bar_diagnostics_exists", not per_bar.empty, 0 if not per_bar.empty else 1, 0, "")
    candidate_windows = raw_df[raw_df["strategy_name"] == candidate_name][["symbol", "window_label"]].drop_duplicates()
    diag_windows = per_bar[["symbol", "window_id"]].drop_duplicates() if not per_bar.empty else pd.DataFrame(columns=["symbol", "window_id"])
    missing_windows = len(candidate_windows.merge(diag_windows, left_on=["symbol", "window_label"], right_on=["symbol", "window_id"], how="left", indicator=True).query("_merge == 'left_only'"))
    _quality(rows, "every_symbol_window_has_records", missing_windows == 0, missing_windows, 0, "")
    if not per_bar.empty:
        trading_counts = summary["trading_bars"].sum() if not summary.empty else 0
        equity_count = len(equity_df)
        _quality(rows, "trading_bars_match_equity_curve", abs(trading_counts - equity_count) <= len(summary), abs(trading_counts - equity_count), 0, f"diagnostics={trading_counts};equity={equity_count}")
        action_count = int((per_bar["action"] != "none").sum())
        _quality(rows, "action_count_matches_action_log", action_count == len(actions_df), abs(action_count - len(actions_df)), 0, "")
        monotonic_errors = sum(not g["timestamp"].is_monotonic_increasing for _, g in per_bar.groupby(["symbol", "window_id"]))
        _quality(rows, "timestamp_monotonic_increasing", monotonic_errors == 0, monotonic_errors, 0, "")
        btc_missing = int(per_bar.loc[per_bar["is_trading_bar"], "btc_regime_timestamp_missing"].sum())
        _quality(rows, "btc_regime_timestamp_present", btc_missing == 0, btc_missing, 0, "")
        ts_fail = int((~per_bar.loc[per_bar["is_trading_bar"], "timestamp_check_pass"]).sum())
        _quality(rows, "signal_indicator_execution_order", ts_fail == 0, ts_fail, 0, "")
        acct_fail = int((~per_bar.loc[per_bar["is_trading_bar"], "accounting_check_pass"]).sum())
        _quality(rows, "total_value_aligns_with_equity", acct_fail == 0, acct_fail, 0, "")
        pos_bad = int(((per_bar["actual_position_pct"] < -1e-8) | (per_bar["actual_position_pct"] > 1.10)).sum())
        _quality(rows, "actual_position_pct_reasonable", pos_bad == 0, pos_bad, 0, "")
        tgt_bad = int(((per_bar["target_position_pct_final"] < -1e-8) | (per_bar["target_position_pct_final"] > 1.05)).sum())
        _quality(rows, "target_position_pct_final_reasonable", tgt_bad == 0, tgt_bad, 0, "")
        trading = per_bar[per_bar["is_trading_bar"]]
        missing_rate = trading["diagnostics_missing_fields"].fillna("").ne("").mean() if len(trading) else 0.0
        _quality(rows, "critical_field_missing_rate", missing_rate < 0.20, int(missing_rate * len(trading)), 0 if missing_rate < 0.10 else 1, f"trading_missing_rate={missing_rate:.4f}")
    return pd.DataFrame(rows, columns=["check_name", "pass", "error_count", "warning_count", "details"])


def build_state_transition_report_from_diagnostics(per_bar: pd.DataFrame, runner) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "state_type", "state_from", "state_to",
        "transition_timestamp", "count", "avg_next_5d_return",
        "avg_next_10d_return", "avg_next_20d_return", "avg_next_60d_return",
        "avg_next_20d_drawdown", "avg_next_60d_drawdown",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    data = per_bar[per_bar["is_trading_bar"]].copy()
    data["risk_score_bucket"] = data["risk_score"].map(lambda x: f"r{int(x)}" if not pd.isna(x) else np.nan)
    for state_type in ["raw_state", "confirmed_state", "btc_regime", "risk_score_bucket"]:
        for (symbol, window_id), group in data.groupby(["symbol", "window_id"], dropna=False):
            group = group.sort_values("timestamp")
            prev = group[state_type].shift(1)
            changed = group[state_type].ne(prev) & prev.notna()
            for _, row in group[changed].iterrows():
                fwd = _future_from_timestamp(symbol, row["timestamp"], row["close"], runner)
                rows.append({
                    "symbol": symbol,
                    "window_id": window_id,
                    "state_type": state_type,
                    "state_from": prev.loc[row.name],
                    "state_to": row[state_type],
                    "transition_timestamp": row["timestamp"],
                    "count": 1,
                    "avg_next_5d_return": fwd["next_5d_return"],
                    "avg_next_10d_return": fwd["next_10d_return"],
                    "avg_next_20d_return": fwd["next_20d_return"],
                    "avg_next_60d_return": fwd["next_60d_return"],
                    "avg_next_20d_drawdown": fwd["mae_20d"],
                    "avg_next_60d_drawdown": fwd["mae_60d"],
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_risk_score_attribution_report(per_bar: pd.DataFrame, runner) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "risk_score", "bar_count", "avg_next_5d_return",
        "avg_next_10d_return", "avg_next_20d_return", "avg_next_60d_return",
        "avg_next_20d_drawdown", "avg_next_60d_drawdown",
        "avg_actual_position_pct", "avg_target_position_pct_final", "trade_count",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    trading = per_bar[per_bar["is_trading_bar"]]
    for (symbol, window_id, risk_score), group in trading.groupby(["symbol", "window_id", "risk_score"], dropna=False):
        fwd = pd.DataFrame([_future_from_timestamp(symbol, r["timestamp"], r["close"], runner) for _, r in group.iterrows()])
        rows.append({
            "symbol": symbol,
            "window_id": window_id,
            "risk_score": risk_score,
            "bar_count": len(group),
            "avg_next_5d_return": fwd["next_5d_return"].mean(),
            "avg_next_10d_return": fwd["next_10d_return"].mean(),
            "avg_next_20d_return": fwd["next_20d_return"].mean(),
            "avg_next_60d_return": fwd["next_60d_return"].mean(),
            "avg_next_20d_drawdown": fwd["mae_20d"].mean(),
            "avg_next_60d_drawdown": fwd["mae_60d"].mean(),
            "avg_actual_position_pct": group["actual_position_pct"].mean(),
            "avg_target_position_pct_final": group["target_position_pct_final"].mean(),
            "trade_count": int((group["action"] != "none").sum()),
        })
    return pd.DataFrame(rows, columns=columns)


def build_exposure_diagnostics_report(per_bar: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "avg_actual_position_pct",
        "avg_target_position_pct_final", "target_actual_gap_mean",
        "target_actual_gap_abs_mean", "underexposed_bar_ratio",
        "overexposed_bar_ratio", "full_exposure_bar_ratio",
        "low_exposure_bar_ratio", "cash_drag_estimate",
        "bull_underexposure_ratio", "bear_overexposure_ratio",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    trading = per_bar[per_bar["is_trading_bar"]].copy()
    gap = trading["target_position_pct_final"] - trading["actual_position_pct"]
    trading["gap"] = gap
    for (symbol, window_id), group in trading.groupby(["symbol", "window_id"], dropna=False):
        bull = group["confirmed_state"].eq("BULL") | group["btc_regime"].isin(["BULL", "STRONG_BULL"])
        bear = group["confirmed_state"].eq("BEAR") | group["btc_regime"].eq("BEAR")
        rows.append({
            "symbol": symbol,
            "window_id": window_id,
            "avg_actual_position_pct": group["actual_position_pct"].mean(),
            "avg_target_position_pct_final": group["target_position_pct_final"].mean(),
            "target_actual_gap_mean": group["gap"].mean(),
            "target_actual_gap_abs_mean": group["gap"].abs().mean(),
            "underexposed_bar_ratio": (group["gap"] > 0.05).mean(),
            "overexposed_bar_ratio": (group["gap"] < -0.05).mean(),
            "full_exposure_bar_ratio": (group["actual_position_pct"] >= 0.95).mean(),
            "low_exposure_bar_ratio": (group["actual_position_pct"] <= 0.25).mean(),
            "cash_drag_estimate": (1 - group["actual_position_pct"]).mean(),
            "bull_underexposure_ratio": (group.loc[bull, "gap"] > 0.05).mean() if bull.any() else np.nan,
            "bear_overexposure_ratio": (group.loc[bear, "gap"] < -0.05).mean() if bear.any() else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


def build_buy_blocked_report(per_bar: pd.DataFrame, runner) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "block_reason", "count", "avg_next_5d_return",
        "avg_next_10d_return", "avg_next_20d_return", "avg_next_60d_return",
        "missed_upside_estimate",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    flags = {
        "cooldown": "blocked_by_cooldown",
        "donchian": "blocked_by_donchian",
        "volatility": "blocked_by_volatility",
        "trend_risk": "blocked_by_trend_risk",
        "btc_bear": "blocked_by_btc_bear",
        "min_notional": "blocked_by_min_notional",
    }
    rows = []
    trading = per_bar[per_bar["is_trading_bar"]]
    for reason, col in flags.items():
        subset = trading[trading[col].fillna(False)]
        for (symbol, window_id), group in subset.groupby(["symbol", "window_id"], dropna=False):
            fwd = pd.DataFrame([_future_from_timestamp(symbol, r["timestamp"], r["close"], runner) for _, r in group.iterrows()])
            rows.append({
                "symbol": symbol,
                "window_id": window_id,
                "block_reason": reason,
                "count": len(group),
                "avg_next_5d_return": fwd["next_5d_return"].mean(),
                "avg_next_10d_return": fwd["next_10d_return"].mean(),
                "avg_next_20d_return": fwd["next_20d_return"].mean(),
                "avg_next_60d_return": fwd["next_60d_return"].mean(),
                "missed_upside_estimate": fwd["next_20d_return"].clip(lower=0).mean(),
            })
    return pd.DataFrame(rows, columns=columns)


def build_sell_too_early_report(per_bar: pd.DataFrame, runner, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "symbol", "window_id", "sell_reason", "count",
        "avg_next_5d_return_after_sell", "avg_next_10d_return_after_sell",
        "avg_next_20d_return_after_sell", "avg_next_60d_return_after_sell",
        "sell_too_early_rate_20d", "sell_too_early_rate_60d",
        "avg_missed_upside_20d", "avg_missed_upside_60d",
    ]
    if per_bar.empty:
        return pd.DataFrame(columns=columns)
    thresholds = config.get("evaluation", {}).get("diagnostics", {}).get("sell_too_early_thresholds", {})
    t20 = thresholds.get("20d", 0.05)
    t60 = thresholds.get("60d", 0.10)
    sells = per_bar[(per_bar["is_trading_bar"]) & (per_bar["action"] == "sell")]
    rows = []
    for (symbol, window_id, reason), group in sells.groupby(["symbol", "window_id", "sell_reason"], dropna=False):
        fwd = pd.DataFrame([_future_from_timestamp(symbol, r["timestamp"], r["close"], runner) for _, r in group.iterrows()])
        rows.append({
            "symbol": symbol,
            "window_id": window_id,
            "sell_reason": reason,
            "count": len(group),
            "avg_next_5d_return_after_sell": fwd["next_5d_return"].mean(),
            "avg_next_10d_return_after_sell": fwd["next_10d_return"].mean(),
            "avg_next_20d_return_after_sell": fwd["next_20d_return"].mean(),
            "avg_next_60d_return_after_sell": fwd["next_60d_return"].mean(),
            "sell_too_early_rate_20d": (fwd["next_20d_return"] > t20).mean(),
            "sell_too_early_rate_60d": (fwd["next_60d_return"] > t60).mean(),
            "avg_missed_upside_20d": fwd["next_20d_return"].clip(lower=0).mean(),
            "avg_missed_upside_60d": fwd["next_60d_return"].clip(lower=0).mean(),
        })
    return pd.DataFrame(rows, columns=columns)


def _confirmed_state_sequence(df: pd.DataFrame, eval_start: int, eval_end: int, strategy) -> dict[int, str]:
    current = "MIXED"
    pending = None
    pending_streak = 0
    states = {}
    for pos in range(eval_start, eval_end):
        signal_pos = pos - 1
        raw = detect_market_state(df.iloc[signal_pos])
        if raw == current:
            pending = None
            pending_streak = 0
        elif raw == pending:
            pending_streak += 1
        else:
            pending = raw
            pending_streak = 1
        if pending and pending_streak >= strategy.CONFIRM_BARS.get(raw, 3):
            current = raw
            pending = None
            pending_streak = 0
        states[pos] = current
    return states


def _target_diagnostics(strategy, symbol: str, signal: pd.Series, raw_state: str, confirmed_state: str, risk_score: int, price: float) -> dict[str, float]:
    raw_target = strategy._lookup_target(confirmed_state, risk_score)
    vol = strategy._get_directional_vol_multiplier(signal, price)
    after_risk = raw_target
    after_vol = max(0.0, min(1.0, raw_target * vol))
    btc_adjust = strategy._get_btc_adjust(signal, symbol)
    after_btc = max(0.0, min(1.0, after_vol + btc_adjust))
    final = strategy._compose_target(symbol, after_btc, raw_state, 0, 0, signal, price, "buy")
    return {
        "target_position_pct_raw": raw_target,
        "target_position_pct_after_risk": after_risk,
        "target_position_pct_after_btc": after_btc,
        "target_position_pct_final": max(0.0, min(strategy._target_cap(), final)),
        "btc_regime_adjustment": btc_adjust,
    }


def _buy_cooldown(strategy, confirmed_state: str, risk_score: int) -> int:
    if confirmed_state not in strategy.STATE_CONFIG or pd.isna(risk_score):
        return 0
    cfg = strategy.STATE_CONFIG[confirmed_state]
    return strategy._compute_buy_cooldown(confirmed_state, cfg, int(risk_score))


def _blocked_flags(signal: pd.Series, confirmed_state: str, trend_risk: float, cooldown_remaining: float, reason: str, can_buy: bool) -> dict[str, bool]:
    donchian_pos = signal.get("donchian_pos")
    atr_rank = signal.get("atr_pct_rank")
    return {
        "blocked_by_cooldown": bool(can_buy and cooldown_remaining > 0 and not reason),
        "blocked_by_donchian": bool(can_buy and not pd.isna(donchian_pos) and donchian_pos >= 0.92 and not reason),
        "blocked_by_volatility": bool(can_buy and not pd.isna(atr_rank) and atr_rank >= 0.90 and not reason),
        "blocked_by_trend_risk": bool(can_buy and not pd.isna(trend_risk) and trend_risk >= 2 and not reason),
        "blocked_by_btc_bear": bool(can_buy and signal.get("btc_regime") == "BEAR" and not reason),
        "blocked_by_min_notional": False,
    }


def _no_trade_reason(action: str, can_buy: bool, can_sell: bool, blocked: dict[str, bool]) -> str:
    if action != "none":
        return ""
    for key, value in blocked.items():
        if value:
            return key.replace("blocked_by_", "")
    if not can_buy and not can_sell:
        return "inside_rebalance_band"
    return "other"


def _timestamp_check(indicator_ts, signal_ts, execution_ts, btc_ts) -> tuple[bool, str]:
    errors = []
    i = pd.to_datetime(indicator_ts, utc=True, errors="coerce")
    s = pd.to_datetime(signal_ts, utc=True, errors="coerce")
    e = pd.to_datetime(execution_ts, utc=True, errors="coerce")
    b = pd.to_datetime(btc_ts, utc=True, errors="coerce")
    if pd.isna(i) or pd.isna(s) or pd.isna(e):
        errors.append("missing_indicator_signal_or_execution_timestamp")
    elif not (i <= s < e):
        errors.append("indicator_signal_execution_order_failed")
    if pd.isna(b):
        errors.append("btc_regime_timestamp_missing")
    elif not (b <= s < e):
        errors.append("btc_regime_signal_execution_order_failed")
    hard = [x for x in errors if x != "btc_regime_timestamp_missing"]
    return len(hard) == 0, ";".join(errors)


def _future_from_timestamp(symbol: str, timestamp, price: float, runner) -> dict[str, float]:
    cache = getattr(runner, "_diagnostic_future_cache", None)
    if cache is None:
        cache = {}
        setattr(runner, "_diagnostic_future_cache", cache)
    if symbol not in cache:
        df = runner.load_data(symbol).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cache[symbol] = df
    df = cache[symbol]
    ts = pd.to_datetime(timestamp, utc=True)
    pos = df["timestamp"].searchsorted(ts, side="right")
    closes = df["close"].astype(float)
    out = {}
    for horizon in [5, 10, 20, 60]:
        idx = pos + horizon - 1
        out[f"next_{horizon}d_return"] = float(closes.iloc[idx] / price - 1) if idx < len(closes) and price > 0 else np.nan
    future20 = closes.iloc[pos:min(pos + 20, len(closes))]
    future60 = closes.iloc[pos:min(pos + 60, len(closes))]
    out["mae_20d"] = float(future20.min() / price - 1) if len(future20) and price > 0 else np.nan
    out["mae_60d"] = float(future60.min() / price - 1) if len(future60) and price > 0 else np.nan
    return out


def _window_equity(equity: pd.DataFrame, symbol: str, window_id: str) -> pd.DataFrame:
    if equity.empty:
        return equity
    return equity[(equity["symbol"] == symbol) & (equity["window_label"] == window_id)].copy()


def _window_actions(actions: pd.DataFrame, symbol: str, window_id: str) -> pd.DataFrame:
    if actions.empty:
        return actions
    return actions[(actions["symbol"] == symbol) & (actions["window_label"] == window_id)].copy()


def _actions_by_timestamp(actions: pd.DataFrame) -> dict[pd.Timestamp, dict[str, Any]]:
    if actions.empty:
        return {}
    return {pd.Timestamp(row["timestamp"]): row for row in actions.to_dict("records")}


def _position_from_equity(equity_row: dict[str, Any] | None, symbol: str) -> tuple[float, float, float, float, float, float]:
    if equity_row is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    value = float(equity_row.get(f"{symbol}_value", np.nan))
    total = float(equity_row.get("total_value", np.nan))
    qty = float(equity_row.get(f"{symbol}_qty", np.nan))
    cash = float(equity_row.get("cash", np.nan))
    avg_cost = float(equity_row.get(f"{symbol}_avg_cost", np.nan))
    actual = value / total if total and total > 0 else np.nan
    return actual, qty, cash, value, total, avg_cost


def _accounting_check(equity_row: dict[str, Any] | None, symbol: str) -> bool:
    if equity_row is None:
        return False
    value = float(equity_row.get(f"{symbol}_value", 0.0))
    total = float(equity_row.get("total_value", np.nan))
    cash = float(equity_row.get("cash", np.nan))
    qty = float(equity_row.get(f"{symbol}_qty", 0.0))
    return bool(abs((cash + value) - total) < 1e-8 and cash >= -1e-8 and qty >= -1e-12)


def _quality(rows: list[dict[str, Any]], name: str, passed: bool, errors: int, warnings: int, details: str) -> None:
    rows.append({
        "check_name": name,
        "pass": bool(passed),
        "error_count": int(errors),
        "warning_count": int(warnings),
        "details": details,
    })


def _extract_setup(reason: str | None) -> str:
    if not reason:
        return ""
    match = re.search(r"_(?:buy|sell)_(.*?)_r\d+", str(reason))
    return match.group(1) if match else ""


def _missing_fields(values: dict[str, Any]) -> list[str]:
    return [k for k, v in values.items() if pd.isna(v)]


def _gt(a, b):
    return bool(a > b) if not pd.isna(a) and not pd.isna(b) else np.nan


def _donchian_low(row: pd.Series) -> float:
    high = row.get("high_120")
    pos = row.get("donchian_pos")
    close = row.get("close")
    if pd.isna(high) or pd.isna(pos) or pos == 1 or pd.isna(close):
        return np.nan
    return (close - pos * high) / (1 - pos)


def _donchian_width(row: pd.Series) -> float:
    high = row.get("high_120")
    low = _donchian_low(row)
    return high - low if not pd.isna(high) and not pd.isna(low) else np.nan


PER_BAR_COLUMNS = [
    "run_id", "strategy_name", "candidate_name", "mode", "symbol", "window_id",
    "timestamp", "bar_index", "is_warmup", "is_trading_bar",
    "open", "high", "low", "close", "volume",
    "ema24", "ema72", "ema168", "ema24_gt_ema72", "ema72_gt_ema168",
    "price_gt_ema24", "price_gt_ema72", "price_gt_ema168",
    "ema168_slope", "ema168_slope_pct", "atr", "atr_pct_rank", "natr",
    "donchian_high", "donchian_low", "donchian_pos", "donchian_width",
    "raw_state", "confirmed_state", "raw_state_prev", "confirmed_state_prev",
    "raw_state_changed", "confirmed_state_changed", "raw_state_bars",
    "confirmed_state_bars", "btc_regime", "btc_regime_prev",
    "btc_regime_changed", "btc_regime_timestamp", "btc_regime_timestamp_missing",
    "btc_regime_adjustment", "btc_regime_target_gap_mult",
    "trend_risk", "drawdown_risk", "risk_score", "risk_score_prev",
    "risk_score_changed", "target_position_pct_raw",
    "target_position_pct_after_risk", "target_position_pct_after_btc",
    "target_position_pct_final", "actual_position_pct", "position_qty",
    "cash", "position_value", "total_value", "reserve", "avg_cost",
    "unrealized_pnl_pct", "peak_price_since_entry", "drawdown_from_peak_pct",
    "action", "action_type", "buy_setup", "sell_reason", "action_reason",
    "target_gap", "trade_qty", "trade_notional", "fee_cost",
    "execution_price", "signal_timestamp", "execution_timestamp",
    "indicator_timestamp", "cooldown_remaining", "can_buy", "can_sell",
    "blocked_by_cooldown", "blocked_by_donchian", "blocked_by_volatility",
    "blocked_by_trend_risk", "blocked_by_btc_bear", "blocked_by_min_notional",
    "no_trade_reason", "timestamp_check_pass", "timestamp_check_error",
    "accounting_check_pass", "diagnostics_missing_fields",
]
