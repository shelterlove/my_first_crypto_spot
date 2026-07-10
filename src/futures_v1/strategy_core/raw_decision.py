"""Raw decision orchestration for V1.

This module owns the V1 decision pipeline order.  The individual decision
policies are migrated behind this interface in later steps; for now the engine
calls the verified policy methods on the strategy owner to preserve behavior.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..strategy_rebalance import Action, PortfolioState


class RawDecisionOwner(Protocol):
    _call_count: int

    def _build_decision_snapshot(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ):
        ...

    def _compute_sizing(self, context, regime, episode: dict, signals, decision):
        ...

    def _build_action(self, context, regime, decision, sizing) -> Action | None:
        ...

    def _record_architecture_trace(self, snapshot, sizing, action: Action | None) -> None:
        ...

    def _record_event(self, context, regime, episode: dict, decision, sizing, action: Action | None) -> None:
        ...


class RawDecisionEngine:
    """Runs the raw decision pipeline for one event-driven backtest step."""

    def compute_actions(
        self,
        owner: RawDecisionOwner,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        owner._call_count += 1
        snapshot = owner._build_decision_snapshot(candles_by_symbol, portfolio, current_prices)
        if snapshot is None:
            return []

        sizing = owner._compute_sizing(
            snapshot.context,
            snapshot.regime,
            snapshot.episode,
            snapshot.signals,
            snapshot.decision,
        )
        action = owner._build_action(snapshot.context, snapshot.regime, snapshot.decision, sizing)
        if bool(getattr(owner, "ENABLE_ARCH_AUDIT", False)):
            owner._record_architecture_trace(snapshot, sizing, action)
        owner._record_event(snapshot.context, snapshot.regime, snapshot.episode, snapshot.decision, sizing, action)
        return [] if action is None else [action]
