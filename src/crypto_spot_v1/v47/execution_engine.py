"""V4.7 execution-layer composition.

This module owns the clean V4.7 execution boundary:

raw decision target + lifecycle shadow + fixed outer lots -> final target.

It deliberately does not compute raw strategy decisions.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..strategy_rebalance import PortfolioState, PositionState
from .config import V47Config
from .models import V47ExecutionDecision


class V47ExecutionOwner(Protocol):
    _call_count: int
    _outer_overlay_state_by_symbol: dict[str, dict]
    _outer_overlay_events: list[dict]
    _outer_qty_last_eval: dict[str, tuple[int, float, str]]

    def _latest_lifecycle_shadow_row(self, symbol: str) -> dict | None: ...


class V47ExecutionEngine:
    def __init__(self, config: V47Config | None = None):
        self.config = config or V47Config()

    def target_for_symbol(
        self,
        *,
        owner: V47ExecutionOwner,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        execution_portfolio: PortfolioState,
    ) -> V47ExecutionDecision:
        cached = owner._outer_qty_last_eval.get(symbol)
        if cached is not None and cached[0] == owner._call_count:
            return V47ExecutionDecision(
                symbol=symbol,
                raw_pct=max(0.0, float(raw_position_pct)),
                main_target_pct=0.0,
                outer_quantity=0.0,
                final_target_pct=float(cached[1]),
                reason=str(cached[2]),
            )

        raw = max(0.0, float(raw_position_pct))
        price = float(current_prices.get(symbol, 0.0) or 0.0)
        if price <= 0.0:
            return self._cache(owner, symbol, raw, raw, 0.0, raw, "outer_bad_price")

        execution_pos = execution_portfolio.positions.get(symbol, PositionState())
        execution_total = execution_portfolio.cash + execution_pos.quantity * price
        if execution_total <= 0.0:
            return self._cache(owner, symbol, raw, raw, 0.0, raw, "outer_bad_total")

        main_target, main_reason = self.main_transform(owner=owner, symbol=symbol, raw_pct=raw)
        main_qty = max(0.0, float(main_target)) * execution_total / price
        outer_qty, outer_reason = self.outer_fixed_quantity(
            owner=owner,
            symbol=symbol,
            raw_pct=raw,
            df=candles_by_symbol.get(symbol),
            price=price,
            execution_total=execution_total,
        )
        final_target = max(0.0, (main_qty + outer_qty) * price / execution_total)
        return self._cache(
            owner,
            symbol,
            raw,
            main_target,
            outer_qty,
            final_target,
            f"{main_reason}+{outer_reason}",
        )

    def main_transform(self, *, owner: V47ExecutionOwner, symbol: str, raw_pct: float) -> tuple[float, str]:
        raw = max(0.0, float(raw_pct))
        row = owner._latest_lifecycle_shadow_row(symbol)
        if row is None:
            return raw, "raw_no_lifecycle"

        risk = int(row.get("risk_score", 0) or 0)
        state = str(row.get("lifecycle_state", "") or "")
        main_intent = str(row.get("main_intent", "") or "")
        confirmed = str(row.get("confirmed_state", "") or "")
        structural = bool(row.get("structural_bear", False))
        distribution = bool(row.get("distribution_shadow", False))
        low_location = bool(row.get("low_location_shadow", False))
        recovery_active = bool(row.get("recovery_active_shadow", False))
        trend_confirmed = bool(row.get("trend_confirmed_shadow", False))

        risk_off = bool(
            state in {"DEFENSE", "DISTRIBUTION"}
            or main_intent in {"DEFEND", "EXIT", "DISTRIBUTE"}
            or structural
            or risk >= 3
            or distribution
            or raw <= 0.05
        )
        if risk_off:
            return raw, "risk_off"

        cfg = self.config.execution
        if low_location and recovery_active and risk <= 2 and raw >= 0.20:
            return min(cfg.low_recovery_cap, raw * cfg.low_recovery_mult), "low_recovery"
        if trend_confirmed and confirmed == "BULL" and risk <= 1 and raw >= 0.50:
            return min(cfg.trend_cap, raw * cfg.trend_mult), "trend_confirmed"
        return raw, "raw"

    def outer_fixed_quantity(
        self,
        *,
        owner: V47ExecutionOwner,
        symbol: str,
        raw_pct: float,
        df: pd.DataFrame | None,
        price: float,
        execution_total: float,
    ) -> tuple[float, str]:
        cfg = self.config.outer
        if raw_pct < cfg.min_raw:
            return 0.0, "outer_raw_too_low"
        if df is None or len(df) < cfg.min_history or price <= 0.0 or execution_total <= 0.0:
            return 0.0, "outer_insufficient_history"

        state = owner._outer_overlay_state_by_symbol.setdefault(
            symbol,
            {"state": "IDLE", "overlay": 0.0, "quantity": 0.0},
        )
        if str(state.get("state", "IDLE")) == "HELD":
            qty = max(0.0, float(state.get("quantity", 0.0) or 0.0))
            entry_low = float(state.get("entry_low", price) or price)
            age = owner._call_count - int(state.get("entry_call", owner._call_count) or owner._call_count)
            if entry_low > 0.0 and price <= entry_low * cfg.hard_stop:
                owner._outer_overlay_events.append(self._outer_event(owner, symbol, df, price, "sell", qty * price / execution_total, "hard_stop"))
                state.update({"state": "IDLE", "overlay": 0.0, "quantity": 0.0, "entry_price": 0.0, "entry_low": 0.0, "entry_call": 0})
                return 0.0, "outer_hard_stop"
            if age >= cfg.min_hold_calls and self._outer_high_exit(df, price, state):
                owner._outer_overlay_events.append(self._outer_event(owner, symbol, df, price, "sell", qty * price / execution_total, "high_exit"))
                state.update({"state": "IDLE", "overlay": 0.0, "quantity": 0.0, "entry_price": 0.0, "entry_low": 0.0, "entry_call": 0})
                return 0.0, "outer_high_exit"
            state["entry_low"] = min(entry_low, price)
            return qty, "outer_qty_hold"

        if self._outer_low_entry(df):
            overlay_pct = max(0.0, float(cfg.target_pct.get(symbol, 0.0)))
            qty = overlay_pct * execution_total / price
            if qty <= 1e-12:
                return 0.0, "outer_qty_zero"
            state.update({
                "state": "HELD",
                "overlay": overlay_pct,
                "quantity": qty,
                "entry_price": price,
                "entry_low": price,
                "entry_call": owner._call_count,
            })
            owner._outer_overlay_events.append(self._outer_event(owner, symbol, df, price, "buy", overlay_pct, "low_entry"))
            return qty, "outer_qty_low_entry"
        return 0.0, "outer_qty_idle"

    def _cache(
        self,
        owner: V47ExecutionOwner,
        symbol: str,
        raw_pct: float,
        main_target_pct: float,
        outer_quantity: float,
        final_target_pct: float,
        reason: str,
    ) -> V47ExecutionDecision:
        owner._outer_qty_last_eval[symbol] = (owner._call_count, float(final_target_pct), str(reason))
        return V47ExecutionDecision(
            symbol=symbol,
            raw_pct=float(raw_pct),
            main_target_pct=float(main_target_pct),
            outer_quantity=float(outer_quantity),
            final_target_pct=float(final_target_pct),
            reason=str(reason),
        )

    def _outer_low_entry(self, df: pd.DataFrame) -> bool:
        close = pd.to_numeric(df["close"], errors="coerce")
        rolling_pos = self._position_in_window(close, 365)
        dd_180 = self._drawdown_from_high(close, 180)
        dd_365 = self._drawdown_from_high(close, 365)
        rebound_20 = self._rebound_from_low(close, 20)
        roc_5 = self._series_ratio(close, 5)
        roc_20 = self._series_ratio(close, 20)
        latest = df.iloc[-1]
        btc_regime = str(latest.get("btc_regime", "") or "")
        btc_roc_20 = latest.get("btc_roc_20", 0.0)
        if btc_regime == "BEAR" and not pd.isna(btc_roc_20) and float(btc_roc_20) <= -0.12:
            return False
        cfg = self.config.outer
        if cfg.deep_only_entry:
            deep_low = bool(rolling_pos <= cfg.deep_rolling365_pos or dd_365 <= cfg.deep_dd365)
            waterfall = bool(roc_20 <= cfg.waterfall_roc20 and rebound_20 < cfg.waterfall_rebound20)
            return deep_low and not waterfall
        extreme_low = bool(
            rolling_pos <= cfg.entry_rolling365_pos
            or dd_365 <= cfg.entry_dd365
            or dd_180 <= cfg.entry_dd180
        )
        stabilizing = bool(
            rebound_20 >= cfg.entry_rebound20
            or (roc_5 >= cfg.entry_roc5 and roc_20 >= cfg.entry_roc20)
        )
        return extreme_low and stabilizing

    def _outer_high_exit(self, df: pd.DataFrame, price: float, state: dict) -> bool:
        close = pd.to_numeric(df["close"], errors="coerce")
        rolling_pos = self._position_in_window(close, 365)
        donchian_pos = self._position_in_window(close, 90)
        roc_20 = self._series_ratio(close, 20)
        entry = float(state.get("entry_price", 0.0) or 0.0)
        profit = price / entry - 1.0 if entry > 0.0 else 0.0
        return bool(
            profit >= 1.0
            and (rolling_pos >= 0.88 or donchian_pos >= 0.90)
            and roc_20 <= 0.12
        )

    def _outer_event(
        self,
        owner: V47ExecutionOwner,
        symbol: str,
        df: pd.DataFrame,
        price: float,
        event: str,
        overlay: float,
        reason: str,
    ) -> dict:
        close = pd.to_numeric(df["close"], errors="coerce")
        return {
            "timestamp": df.iloc[-1].get("timestamp"),
            "symbol": symbol,
            "event": event,
            "reason": reason,
            "price": float(price),
            "overlay_pct": float(overlay),
            "call_count": int(owner._call_count),
            "rolling365_pos": self._position_in_window(close, 365),
            "dd_180": self._drawdown_from_high(close, 180),
            "dd_365": self._drawdown_from_high(close, 365),
            "rebound_20": self._rebound_from_low(close, 20),
            "roc_20": self._series_ratio(close, 20),
        }

    @staticmethod
    def _position_in_window(series: pd.Series, window: int) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().tail(window)
        if values.empty:
            return 0.5
        low = float(values.min())
        high = float(values.max())
        if high <= low:
            return 0.5
        return float((values.iloc[-1] - low) / (high - low))

    @staticmethod
    def _series_ratio(series: pd.Series, periods: int) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) <= periods or values.iloc[-periods - 1] <= 0.0:
            return 0.0
        return float(values.iloc[-1] / values.iloc[-periods - 1] - 1.0)

    @staticmethod
    def _drawdown_from_high(series: pd.Series, window: int) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().tail(window)
        if values.empty or values.max() <= 0.0:
            return 0.0
        return float(values.iloc[-1] / values.max() - 1.0)

    @staticmethod
    def _rebound_from_low(series: pd.Series, window: int) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().tail(window)
        if values.empty or values.min() <= 0.0:
            return 0.0
        return float(values.iloc[-1] / values.min() - 1.0)
