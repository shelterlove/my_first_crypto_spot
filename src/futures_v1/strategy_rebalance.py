"""Portfolio state and action primitives used by V1 strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class Action:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    reason: str
    order_type: str = "market"
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PositionState:
    quantity: float = 0.0
    avg_cost: float = 0.0


@dataclass
class PortfolioState:
    cash: float = 0.0
    positions: dict[str, PositionState] = field(default_factory=dict)


class PortfolioStrategyBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        ...
