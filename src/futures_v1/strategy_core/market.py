"""Market context and regime construction for V1."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .. import strategy_utils
from ..strategy_rebalance import PortfolioState, PositionState
from ..strategy_types import StrategyContext, StrategyRegime


class MarketOwner(Protocol):
    CONFIRM_BARS: dict[str, int]
    _diag: dict[str, int]
    _peak_price_by_symbol: dict[str, float]
    _state_by_symbol: dict[str, dict]

    def _refresh_base_ledger_market(self, context: StrategyContext) -> None:
        ...


class MarketEngine:
    """Builds the market view used by the V1 decision pipeline."""

    def build_context(
        self,
        owner: MarketOwner,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> StrategyContext | None:
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return None
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        price = float(current_prices.get(symbol, 0.0) or 0.0)
        if price <= 0.0:
            return None
        pos = portfolio.positions.get(symbol, PositionState())
        position_value = float(pos.quantity) * price
        total_value = float(portfolio.cash) + position_value
        current_pct = position_value / total_value if total_value > 0.0 else 0.0

        peak = owner._peak_price_by_symbol.get(symbol, price)
        if current_pct < 0.20 or pos.quantity <= 1e-12:
            peak = price
        else:
            peak = max(peak, price)
        owner._peak_price_by_symbol[symbol] = peak

        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self.apply_state_confirmation(owner, symbol, raw_state)
        trend_risk = self.calculate_trend_risk(latest, price)
        drawdown_risk = self.calculate_drawdown_risk(owner, symbol, latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        owner._diag["core_context_built_count"] += 1
        context = StrategyContext(
            symbol=symbol,
            df=df,
            latest=latest,
            price=price,
            pos=pos,
            total_value=total_value,
            current_pct=current_pct,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
        )
        owner._refresh_base_ledger_market(context)
        return context

    def build_regime(self, owner: MarketOwner, context: StrategyContext) -> StrategyRegime:
        latest = context.latest
        price = context.price
        raw_state = context.raw_state
        confirmed_state = context.confirmed_state
        trend_risk = context.trend_risk
        drawdown_risk = context.drawdown_risk
        risk_score = context.risk_score
        btc_regime = str(latest.get("btc_regime", "RANGE"))
        atr_rank = value(latest, "atr_pct_rank", 0.5)
        price_vs_ema72 = price_vs(latest, price, "ema72")
        price_vs_ema168 = price_vs(latest, price, "ema168")
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = value(latest, "ema168_slope", 0.0)

        structural_bear = bool(
            trend_risk >= 3
            or (raw_state == "BEAR" and confirmed_state == "BEAR" and risk_score >= 3)
            or (
                not pd.isna(ema24)
                and not pd.isna(ema72)
                and not pd.isna(ema168)
                and price < float(ema168)
                and float(ema24) < float(ema72) < float(ema168)
            )
            or (
                btc_regime == "BEAR"
                and not pd.isna(price_vs_ema168)
                and price_vs_ema168 <= -0.10
                and trend_risk >= 2
            )
        )
        if structural_bear:
            regime = "BEAR"
            reason = "structural_bear"
        elif (
            raw_state != confirmed_state
            or trend_risk >= 2
            or drawdown_risk > 0
            or btc_regime == "BEAR"
            or (not pd.isna(price_vs_ema72) and price_vs_ema72 < -0.04)
            or (not pd.isna(atr_rank) and atr_rank >= 0.92 and risk_score >= 2)
        ):
            regime = "TRANSITION"
            reason = "risk_transition"
        elif (
            confirmed_state == "BULL"
            and trend_risk == 0
            and btc_regime != "BEAR"
            and (pd.isna(ema168_slope) or ema168_slope >= 0.0)
        ):
            regime = "BULL"
            reason = "confirmed_bull"
        else:
            regime = "RANGE"
            reason = "range"

        owner._diag[f"core_regime_{regime.lower()}_count"] += 1
        return StrategyRegime(
            regime=regime,
            reason=reason,
            btc_regime=btc_regime,
            atr_rank=atr_rank,
            price_vs_ema72=price_vs_ema72,
            price_vs_ema168=price_vs_ema168,
            structural_bear=structural_bear,
        )

    @staticmethod
    def calculate_trend_risk(latest: pd.Series, price: float) -> int:
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ok24 = not pd.isna(ema24)
        ok72 = not pd.isna(ema72)
        ok168 = not pd.isna(ema168)
        if ok24 and ok72 and ok168 and price < float(ema168) and float(ema24) < float(ema72) < float(ema168):
            return 3
        if ok168 and price < float(ema168) and ok24 and ok72 and float(ema24) < float(ema72):
            return 2
        if ok72 and price < float(ema72) and ok24 and float(ema24) < float(ema72):
            return 1
        return 0

    @staticmethod
    def calculate_drawdown_risk(
        owner: MarketOwner,
        symbol: str,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        peak = owner._peak_price_by_symbol.get(symbol, price)
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0.0 or peak <= 0.0:
            return 0
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        profit_pct = price / pos.avg_cost - 1.0
        dd_from_peak = 1.0 - price / peak
        if profit_pct > 0.30 and dd_from_peak > 0.18 and not pd.isna(ema72) and price < float(ema72):
            return 2
        if profit_pct > 0.20 and dd_from_peak > 0.10 and not pd.isna(ema24) and price < float(ema24):
            return 1
        return 0

    @staticmethod
    def apply_state_confirmation(owner: MarketOwner, symbol: str, raw_state: str) -> str:
        state = owner._state_by_symbol.get(symbol, {
            "current": "MIXED",
            "state_streak": 1,
            "pending": None,
            "pending_streak": 0,
        })
        if raw_state == state["current"]:
            state["state_streak"] = int(state.get("state_streak", 1)) + 1
            state["pending"] = None
            state["pending_streak"] = 0
            owner._state_by_symbol[symbol] = state
            return str(state["current"])
        if raw_state == state.get("pending"):
            state["pending_streak"] = int(state.get("pending_streak", 0)) + 1
        else:
            state["pending"] = raw_state
            state["pending_streak"] = 1
        if int(state["pending_streak"]) >= owner.CONFIRM_BARS.get(raw_state, 3):
            state["current"] = raw_state
            state["state_streak"] = int(state["pending_streak"])
            state["pending"] = None
            state["pending_streak"] = 0
        owner._state_by_symbol[symbol] = state
        return str(state["current"])


def value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
    result = latest.get(column, default)
    if pd.isna(result):
        return default
    return float(result)


def price_vs(latest: pd.Series, price: float, column: str) -> float:
    den = value(latest, column)
    if pd.isna(den) or den <= 0.0:
        return float("nan")
    return price / den - 1.0
