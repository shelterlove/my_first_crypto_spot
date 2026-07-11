from decimal import Decimal

import pandas as pd

from futures_v1.backtest_event_driven import apply_execution_friction
from futures_v1.benchmark import build_strategy
from futures_v1.strategy_core.position_book import (
    build_hard_gross_reduction_actions,
    build_intraday_liquidation_actions,
    financing_cost_today,
)
from futures_v1.strategy_rebalance import Action, PortfolioState, PositionState


def strategy_with(**values):
    strategy = build_strategy("eth_bnb_futures_v1", 100.0, 20.0, 0.001)
    for name, value in values.items():
        setattr(strategy, name, value)
    return strategy


def test_funding_enters_financing_cash_flow_with_both_signs() -> None:
    strategy = strategy_with(EXECUTION_TRANSFORM_BORROW_APR=0.0)
    portfolio = PortfolioState(cash=-100.0, positions={"ETH/USDT": PositionState(quantity=2.0)})
    positive = financing_cost_today(
        strategy,
        portfolio,
        {"ETH/USDT": pd.Series({"close": 100.0, "funding_rate_daily": 0.001})},
    )
    negative = financing_cost_today(
        strategy,
        portfolio,
        {"ETH/USDT": pd.Series({"close": 100.0, "funding_rate_daily": -0.001})},
    )
    assert positive == 0.2
    assert negative == -0.2


def test_hard_gross_reduction_targets_lower_ratio() -> None:
    strategy = strategy_with(
        RESEARCH_ACTUAL_GROSS_HARD_CAP=2.2,
        RESEARCH_ACTUAL_GROSS_REDUCE_TO=2.0,
    )
    portfolio = PortfolioState(cash=-130.0, positions={"ETH/USDT": PositionState(quantity=2.3)})
    actions = build_hard_gross_reduction_actions(strategy, portfolio, {"ETH/USDT": 100.0}, 0.001)
    assert len(actions) == 1
    assert round(actions[0].quantity, 8) == 0.3


def test_intraday_low_can_trigger_liquidation() -> None:
    strategy = strategy_with(
        RESEARCH_INTRADAY_LIQUIDATION=True,
        RESEARCH_MAINTENANCE_MARGIN_RATE=0.005,
        RESEARCH_LIQUIDATION_FEE_RATE=0.01,
    )
    portfolio = PortfolioState(cash=-200.0, positions={"ETH/USDT": PositionState(quantity=3.0)})
    actions = build_intraday_liquidation_actions(
        strategy,
        portfolio,
        {"ETH/USDT": pd.Series({"low": 60.0, "high": 100.0})},
        0.001,
    )
    assert len(actions) == 1
    assert actions[0].diagnostics["liquidation_trigger_price"] > 60.0


def test_slippage_and_partial_fill_are_recorded() -> None:
    strategy = strategy_with(RESEARCH_SLIPPAGE_BPS=10.0, RESEARCH_FILL_RATIO=0.9)
    action = Action("ETH/USDT", "buy", 1.0, 100.0, "test")
    filled = apply_execution_friction(action, strategy)
    assert Decimal(str(filled.price)) == Decimal("100.1000")
    assert filled.quantity == 0.9
    assert filled.diagnostics["requested_quantity"] == 1.0
