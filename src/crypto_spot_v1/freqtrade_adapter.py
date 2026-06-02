"""Small adapter between native target-position decisions and Freqtrade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .benchmark import build_strategy
from .decision import build_decision_record
from .strategy_rebalance import PortfolioState, PositionState


@dataclass(frozen=True)
class TargetPositionDecision:
    pair: str
    timestamp: Any
    action: str
    target_pct: float | None
    current_pct: float
    delta_pct: float
    price: float
    order_notional: float
    reason: str
    state: str
    risk_score: int | None


def build_target_position_decision(
    *,
    pair: str,
    dataframe: pd.DataFrame,
    current_position_pct: float,
    strategy_name: str = "v2_19B",
    capital: float = 100.0,
    reserve: float = 20.0,
    fee_rate: float = 0.001,
    min_notional: float = 0.0,
) -> TargetPositionDecision:
    """Run the native strategy once and return a compact target-position decision."""
    if dataframe.empty:
        raise ValueError("dataframe must not be empty")

    latest = dataframe.iloc[-1]
    price = float(latest["close"])
    portfolio = _portfolio_for_pair(pair, price, capital, current_position_pct)
    strategy = build_strategy(strategy_name, capital, reserve, fee_rate, min_notional=min_notional)
    setattr(strategy, "TARGET_ALLOC", {pair: 1.0})

    actions = strategy.compute_actions({pair: dataframe}, portfolio, {pair: price})
    action = actions[0] if actions else None
    record = build_decision_record(
        timestamp=latest.get("timestamp", dataframe.index[-1]),
        symbol=pair,
        strategy_name=strategy_name,
        action=action,
        portfolio=portfolio,
        price=price,
        latest=latest,
        no_trade_reason="" if action else "target_or_cooldown_not_actionable",
    )
    delta_pct = _action_delta_pct(record["side"], record["notional"], capital)
    return TargetPositionDecision(
        pair=pair,
        timestamp=record["timestamp"],
        action=record["action"],
        target_pct=record["target_pct"],
        current_pct=float(record["current_pct"]),
        delta_pct=delta_pct,
        price=price,
        order_notional=float(record["notional"]),
        reason=record["reason"],
        state=record["confirmed_state"] or record["raw_state"],
        risk_score=record["risk_score"],
    )


def decision_as_dict(decision: TargetPositionDecision) -> dict[str, Any]:
    return asdict(decision)


def _portfolio_for_pair(pair: str, price: float, capital: float, pct: float) -> PortfolioState:
    pct = max(0.0, min(1.0, float(pct)))
    position_value = capital * pct
    quantity = position_value / price if price > 0 else 0.0
    return PortfolioState(
        cash=capital - position_value,
        positions={pair: PositionState(quantity=quantity, avg_cost=price if quantity > 0 else 0.0)},
    )


def _action_delta_pct(side: str, notional: float, capital: float) -> float:
    if capital <= 0 or not side:
        return 0.0
    sign = 1.0 if side == "buy" else -1.0
    return sign * float(notional) / capital
