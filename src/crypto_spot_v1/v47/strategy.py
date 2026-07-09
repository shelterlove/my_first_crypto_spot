"""Clean V4.7 strategy entry.

This first extraction keeps the verified raw decision path from the legacy
chain and moves V4.7's execution composition into ``v47.execution_engine``.
"""

from __future__ import annotations

import pandas as pd

from ..strategy_legacy import V47Strategy as LegacyV47Strategy
from ..strategy_rebalance import Action, PortfolioState
from .accounting import V47AccountingEngine
from .action import V47ActionEngine
from .bear_base import V47BearBaseEngine
from .config import V47Config, V47ExecutionConfig, V47OuterConfig
from .credit import V47RecoveryCreditEngine
from .episode import V47EpisodeEngine, cycle_recovered_pct, cycle_recovery_budget
from .events import V47EventEngine
from .execution_engine import V47ExecutionEngine
from .floor import V47FloorEngine
from .lifecycle import V47LifecycleEngine
from .market import V47MarketEngine, price_vs, value
from .raw_decision import V47RawDecisionEngine
from .recovery import V47RecoveryEngine
from .signals import V47SignalEngine
from .sleeve import V47SleeveEngine
from .sizing import V47SizingEngine
from .target import V47TargetEngine


class V47CleanStrategy(LegacyV47Strategy):
    """V4.7 clean candidate: legacy raw decisions plus clean execution engine."""

    @property
    def name(self) -> str:
        return "v4_7_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v47_config = V47Config()
        self._v47_market_engine = V47MarketEngine()
        self._v47_episode_engine = V47EpisodeEngine()
        self._v47_target_engine = V47TargetEngine()
        self._v47_signal_engine = V47SignalEngine()
        self._v47_recovery_engine = V47RecoveryEngine()
        self._v47_sleeve_engine = V47SleeveEngine()
        self._v47_sizing_engine = V47SizingEngine()
        self._v47_action_engine = V47ActionEngine()
        self._v47_event_engine = V47EventEngine()
        self._v47_accounting_engine = V47AccountingEngine()
        self._v47_bear_base_engine = V47BearBaseEngine()
        self._v47_floor_engine = V47FloorEngine()
        self._v47_credit_engine = V47RecoveryCreditEngine()
        self._v47_lifecycle_engine = V47LifecycleEngine()
        self._v47_raw_decision_engine = V47RawDecisionEngine()
        self._v47_execution_engine = V47ExecutionEngine(self._v47_config)

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        return self._v47_raw_decision_engine.compute_actions(
            owner=self,
            candles_by_symbol=candles_by_symbol,
            portfolio=portfolio,
            current_prices=current_prices,
        )

    def _build_context(self, candles_by_symbol, portfolio, current_prices):
        return self._v47_market_engine.build_context(
            owner=self,
            candles_by_symbol=candles_by_symbol,
            portfolio=portfolio,
            current_prices=current_prices,
        )

    def _build_regime(self, context):
        return self._v47_market_engine.build_regime(owner=self, context=context)

    def _calculate_trend_risk(self, latest: pd.Series, price: float) -> int:
        return self._v47_market_engine.calculate_trend_risk(latest, price)

    def _calculate_drawdown_risk(self, symbol: str, latest: pd.Series, pos, price: float) -> int:
        return self._v47_market_engine.calculate_drawdown_risk(self, symbol, latest, pos, price)

    def _apply_state_confirmation(self, symbol: str, raw_state: str) -> str:
        return self._v47_market_engine.apply_state_confirmation(self, symbol, raw_state)

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        return value(latest, column, default)

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        return price_vs(latest, price, column)

    def _build_signals(self, context, regime, episode: dict):
        return self._v47_signal_engine.build_signals(self, context, regime, episode)

    def _accumulation_signal(self, context, regime, episode: dict, signals) -> bool:
        return self._v47_signal_engine.accumulation_signal(episode, signals)

    def _starter_signal(self, context, regime) -> bool:
        return self._v47_signal_engine.starter_signal(context, regime)

    def _value_recovery(self, context, regime) -> bool:
        return self._v47_signal_engine.value_recovery(self, context, regime)

    def _trend_continuation(self, context, regime) -> bool:
        return self._v47_signal_engine.trend_continuation(self, context, regime)

    def _late_trend_continuation_risk(self, context, regime) -> bool:
        return self._v47_signal_engine.late_trend_continuation_risk(self, context, regime)

    def _distribution_exhaustion(self, context, regime) -> bool:
        return self._v47_signal_engine.distribution_exhaustion(self, context, regime)

    def _recovery_signal_from_parts(self, *, context, regime, value_recovery: bool, trend_continuation: bool) -> bool:
        return self._v47_signal_engine.recovery_signal_from_parts(
            self,
            context=context,
            regime=regime,
            value_recovery=value_recovery,
            trend_continuation=trend_continuation,
        )

    def _recovery_signal(self, context, regime) -> bool:
        return self._v47_signal_engine.recovery_signal(self, context, regime)

    def _strong_recovery_signal(self, context, regime) -> bool:
        return self._v47_signal_engine.strong_recovery_signal(self, context, regime)

    def _recovery_quality_ok(self, context, regime) -> bool:
        return self._v47_signal_engine.recovery_quality_ok(self, context, regime)

    def _episode_recovery_add_cap(self, episode: dict) -> float:
        return self._v47_recovery_engine.episode_recovery_add_cap(episode)

    def _episode_recovery_plan(self, context, regime, episode: dict, signals):
        return self._v47_recovery_engine.episode_recovery_plan(self, context, regime, episode, signals)

    def _episode_reentry_plan(self, context, regime, episode: dict, signals):
        return self._v47_recovery_engine.episode_reentry_plan(self, context, regime, episode, signals)

    def _episode_recovery_target(self, context, regime, episode: dict, signals, base_target: float) -> float:
        return self._v47_recovery_engine.episode_recovery_target(self, context, regime, episode, signals, base_target)

    def _episode_recovery_max_buy(self, context, regime, episode: dict, setup: str, signals=None) -> float:
        return self._v47_recovery_engine.episode_recovery_max_buy(self, context, regime, episode, setup, signals)

    @staticmethod
    def _episode_recovery_budget(episode: dict) -> float:
        return V47RecoveryEngine.episode_recovery_budget(episode)

    @staticmethod
    def _episode_recovered_pct(episode: dict) -> float:
        return V47RecoveryEngine.episode_recovered_pct(episode)

    def _record_episode_recovery_buy(self, context, episode: dict, action) -> None:
        self._v47_recovery_engine.record_episode_recovery_buy(self, context, episode, action)

    @staticmethod
    def _setup_from_action(action) -> str:
        return V47RecoveryEngine.setup_from_action(action)

    @staticmethod
    def _is_bear_base_buy(decision, sizing) -> bool:
        return V47RecoveryEngine.is_bear_base_buy(decision, sizing)

    def _should_record_episode_recovery_buy(self, bear_base_buy: bool) -> bool:
        return self._v47_recovery_engine.should_record_episode_recovery_buy(self, bear_base_buy)

    @staticmethod
    def _recovery_fraction_from_drawdown(low_drop: float, *, shallow: float, normal: float, deep: float, crash: float) -> float:
        return V47RecoveryEngine.recovery_fraction_from_drawdown(
            low_drop,
            shallow=shallow,
            normal=normal,
            deep=deep,
            crash=crash,
        )

    @staticmethod
    def _recovery_step_max_buy(add_pct: float, low_drop: float, strong: bool, *, deep_base: bool = False) -> float:
        return V47RecoveryEngine.recovery_step_max_buy(add_pct, low_drop, strong, deep_base=deep_base)

    @staticmethod
    def _recovery_drawdown_guard(low_drop: float) -> str:
        return V47RecoveryEngine.recovery_drawdown_guard(low_drop)

    def _recovery_permission(self, context, regime, signals, low_drop: float):
        return self._v47_recovery_engine.recovery_permission(self, context, regime, signals, low_drop)

    def _staged_recovery_plan(
        self,
        *,
        context,
        regime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
        lowest: float,
    ):
        return self._v47_recovery_engine.staged_recovery_plan(
            self,
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
            lowest=lowest,
        )

    def _deep_base_recovery_plan(
        self,
        *,
        context,
        regime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
    ):
        return self._v47_recovery_engine.deep_base_recovery_plan(
            self,
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
        )

    def _post_crash_recoil_plan(
        self,
        *,
        context,
        regime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
        lowest: float,
    ):
        return self._v47_recovery_engine.post_crash_recoil_plan(
            self,
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
            lowest=lowest,
        )

    @staticmethod
    def _structural_recovery_ready(context, regime, signals) -> bool:
        return V47RecoveryEngine.structural_recovery_ready(context, regime, signals)

    @staticmethod
    def _episode_recovery_cap(episode: dict) -> float:
        return V47RecoveryEngine.episode_recovery_cap(episode)

    def _base_recovery_established(self, context) -> bool:
        return self._v47_recovery_engine.base_recovery_established(self, context)

    def _base_recovery_stabilized(self, context, regime, signals) -> bool:
        return self._v47_recovery_engine.base_recovery_stabilized(self, context, regime, signals)

    def _base_recovery_target(self, context, regime, signals) -> float:
        return self._v47_recovery_engine.base_recovery_target(self, context, regime, signals)

    def _base_recovery_should_accumulate(self, context, regime, signals) -> bool:
        return self._v47_recovery_engine.base_recovery_should_accumulate(self, context, regime, signals)

    def _base_led_recovery_base_sizing(self, context, regime, episode: dict, signals, decision, primary):
        return self._v47_recovery_engine.base_led_recovery_base_sizing(
            self,
            context,
            regime,
            episode,
            signals,
            decision,
            primary,
        )

    def _base_led_recovery_allowed(self, context, regime, episode: dict, signals) -> bool:
        return self._v47_recovery_engine.base_led_recovery_allowed(self, context, regime, episode, signals)

    def _base_quantity(self, context) -> float:
        return self._v47_bear_base_engine.base_quantity(self, context)

    def _main_quantity(self, context) -> float:
        return self._v47_bear_base_engine.main_quantity(self, context)

    def _base_pct_from_quantity(self, context, quantity: float) -> float:
        return self._v47_bear_base_engine.base_pct_from_quantity(self, context, quantity)

    def _new_recovery_ledger(self) -> dict:
        return self._v47_bear_base_engine.new_recovery_ledger(self._call_count)

    @staticmethod
    def _new_base_ledger() -> dict:
        return V47BearBaseEngine.new_base_ledger()

    def _refresh_base_ledger_market(self, context) -> None:
        self._v47_bear_base_engine.refresh_base_ledger_market(self, context)

    def _record_ledger_sell(self, context, sizing) -> None:
        self._v47_bear_base_engine.record_ledger_sell(self, context, sizing)

    def _bear_base_floor(self, context) -> float:
        return self._v47_bear_base_engine.bear_base_floor(self, context)

    def _bear_base_target(self, context, regime) -> float:
        return self._v47_bear_base_engine.bear_base_target(self, context, regime)

    def _bear_base_buy_proposal(self, context, recovery_ledger: dict, base_ledger: dict, location_depth: float):
        return self._v47_bear_base_engine.bear_base_buy_proposal(
            self,
            context,
            recovery_ledger,
            base_ledger,
            location_depth,
        )

    def _bear_base_asset_cap(self, symbol: str) -> float:
        return self._v47_bear_base_engine.bear_base_asset_cap(self, symbol)

    @staticmethod
    def _ledger_base_budget_used_pct(ledger: dict) -> float:
        return V47BearBaseEngine.ledger_base_budget_used_pct(ledger)

    def _bear_base_exit_target(self, context, regime, signals) -> float:
        return self._v47_bear_base_engine.bear_base_exit_target(self, context, regime, signals)

    def _bear_base_maturity_score(self, context, regime, signals, profit: float) -> int:
        return self._v47_bear_base_engine.bear_base_maturity_score(self, context, regime, signals, profit)

    @staticmethod
    def _bear_base_layer(context, location_depth: float) -> int:
        return V47BearBaseEngine.bear_base_layer(location_depth)

    @staticmethod
    def _bear_base_current_discount(context, recovery_ledger: dict) -> float:
        return V47BearBaseEngine.bear_base_current_discount(context, recovery_ledger)

    def _bear_base_low_location(self, context, regime) -> bool:
        return self._v47_bear_base_engine.bear_base_low_location(self, context, regime)

    def _bear_base_location_depth(self, context) -> float:
        return self._v47_bear_base_engine.bear_base_location_depth(self, context)

    def _bear_base_path_accelerating(self, context, regime) -> bool:
        return self._v47_bear_base_engine.bear_base_path_accelerating(self, context, regime)

    def _bear_base_path_stabilizing(self, context, regime) -> bool:
        return self._v47_bear_base_engine.bear_base_path_stabilizing(self, context, regime)

    def _opportunity_floor_sizing(self, context, regime, signals, prior_sizing):
        return self._v47_floor_engine.opportunity_floor_sizing(self, context, regime, signals, prior_sizing)

    def _opportunity_floor_low_location(self, context, regime) -> bool:
        return self._v47_floor_engine.opportunity_floor_low_location(self, context, regime)

    def _opportunity_floor_stabilizing(self, context, regime, signals) -> bool:
        return self._v47_floor_engine.opportunity_floor_stabilizing(self, context, regime, signals)

    def _protected_floor_quantity(self, context) -> float:
        return self._v47_floor_engine.protected_floor_quantity(self, context)

    def _protected_floor_exit_allowed(self, context, regime) -> bool:
        return self._v47_floor_engine.protected_floor_exit_allowed(self, context, regime)

    def _record_protected_floor_buy(self, context, regime, action) -> None:
        self._v47_floor_engine.record_protected_floor_buy(self, context, regime, action)

    def _consume_protected_floor_for_sell(self, context, regime, action) -> None:
        self._v47_floor_engine.consume_protected_floor_for_sell(self, context, regime, action)

    def _protected_floor_exit_sizing(self, context, regime, signals, decision):
        return self._v47_floor_engine.protected_floor_exit_sizing(self, context, regime, signals, decision)

    def _protected_floor_exit_event(self, context, regime, signals, profit: float) -> bool:
        return self._v47_floor_engine.protected_floor_exit_event(self, context, regime, signals, profit)

    def _protected_floor_new_release_event(self, context, ledger: dict) -> bool:
        return self._v47_floor_engine.protected_floor_new_release_event(self, context, ledger)

    def _lifecycle_low_base_sizing(self, context, regime, episode: dict, signals, decision, prior_sizing):
        return self._v47_floor_engine.lifecycle_low_base_sizing(
            self,
            context,
            regime,
            episode,
            signals,
            decision,
            prior_sizing,
        )

    def _lifecycle_protected_base_exit_sizing(self, context, regime, signals, decision):
        return self._v47_floor_engine.lifecycle_protected_base_exit_sizing(self, context, regime, signals, decision)

    def _lifecycle_base_exit_high_location(self, context, regime) -> bool:
        return self._v47_floor_engine.lifecycle_base_exit_high_location(self, context, regime)

    def _lifecycle_low_base_entry_location(self, context, regime) -> bool:
        return self._v47_floor_engine.lifecycle_low_base_entry_location(self, context, regime)

    def _lifecycle_low_base_stabilizing(self, context, regime, signals) -> bool:
        return self._v47_floor_engine.lifecycle_low_base_stabilizing(self, context, regime, signals)

    def _lifecycle_protected_floor_portfolio_pct(self, context) -> float:
        return self._v47_floor_engine.lifecycle_protected_floor_portfolio_pct(self, context)

    def _update_episode(self, context, regime):
        return self._v47_episode_engine.update_episode(self, context, regime)

    def _close_episode(self, symbol: str, reason: str) -> None:
        self._v47_episode_engine.close_episode(self, symbol, reason)

    def _append_episode_sell_leg(self, context, episode: dict, sizing) -> None:
        self._v47_episode_engine.append_episode_sell_leg(self, context, episode, sizing)

    @staticmethod
    def _cycle_recovery_budget(episode: dict) -> float:
        return cycle_recovery_budget(episode)

    @staticmethod
    def _cycle_recovered_pct(episode: dict) -> float:
        return cycle_recovered_pct(episode)

    def _open_recovery_credit(self, symbol: str, episode: dict, open_budget: float, reason: str) -> None:
        self._v47_episode_engine.open_recovery_credit(self, symbol, episode, open_budget, reason)

    def _decay_recovery_credit(self, context) -> None:
        self._v47_episode_engine.decay_recovery_credit(self, context)

    def _record_recovery_credit_event(
        self,
        *,
        symbol: str,
        event: str,
        episode: dict,
        source_close_reason: str,
        credit_before: float,
        credit_delta: float,
        credit_after: float,
        anchor_price: float,
        guard: str,
        blocked_reason: str,
    ) -> None:
        self._v47_episode_engine.record_recovery_credit_event(
            self,
            symbol=symbol,
            event=event,
            episode=episode,
            source_close_reason=source_close_reason,
            credit_before=credit_before,
            credit_delta=credit_delta,
            credit_after=credit_after,
            anchor_price=anchor_price,
            guard=guard,
            blocked_reason=blocked_reason,
        )

    def _record_recovery_credit_check(
        self,
        context,
        regime,
        signals,
        ledger: dict,
        allowed: bool,
        blocked_reason: str,
        current_drop: float,
        rolling_pos: float,
        donchian_pos: float,
        drop_min: float,
        rolling_max: float,
        donchian_max: float,
    ) -> None:
        self._v47_episode_engine.record_recovery_credit_check(
            self,
            context,
            regime,
            signals,
            ledger,
            allowed,
            blocked_reason,
            current_drop,
            rolling_pos,
            donchian_pos,
            drop_min,
            rolling_max,
            donchian_max,
        )

    def _recovery_credit_total(self) -> float:
        return self._v47_episode_engine.recovery_credit_total(self)

    def _recovery_credit_portfolio_cap(self) -> float:
        return self._v47_episode_engine.recovery_credit_portfolio_cap(self)

    def _recovery_credit_plan(self, context, regime, signals):
        return self._v47_credit_engine.recovery_credit_plan(self, context, regime, signals)

    def _recovery_credit_sizing(self, context, regime, signals):
        return self._v47_credit_engine.recovery_credit_sizing(self, context, regime, signals)

    def _track_main_buy_for_credit(self, context, sizing) -> None:
        self._v47_credit_engine.track_main_buy_for_credit(self, context, sizing)

    def _consume_recovery_credit_from_current_buy(self, context, regime, signals, sizing) -> None:
        self._v47_credit_engine.consume_recovery_credit_from_current_buy(self, context, regime, signals, sizing)

    def _consume_recovery_credit_from_recent_buy(self, context, regime) -> None:
        self._v47_credit_engine.consume_recovery_credit_from_recent_buy(self, context, regime)

    def _consume_recovery_credit_by_main_buy(
        self,
        *,
        context,
        regime,
        signals,
        buy_call: int,
        buy_setup: str,
        buy_step_pct: float,
        guard_suffix: str,
    ) -> None:
        self._v47_credit_engine.consume_recovery_credit_by_main_buy(
            self,
            context=context,
            regime=regime,
            signals=signals,
            buy_call=buy_call,
            buy_setup=buy_setup,
            buy_step_pct=buy_step_pct,
            guard_suffix=guard_suffix,
        )

    def _release_recovery_credit(self, context, sizing) -> None:
        self._v47_credit_engine.release_recovery_credit(self, context, sizing)

    def _allow_btc_deep_credit_overlay(self, context) -> bool:
        return self._v47_credit_engine.allow_btc_deep_credit_overlay(self, context)

    def _compose_target_from_plan(self, context, plan):
        return self._v47_target_engine.compose_target_from_plan(self, context, plan)

    def _build_target_plan(self, context, regime, episode: dict, signals, intent: str):
        return self._v47_target_engine.build_target_plan(self, context, regime, episode, signals, intent)

    def _main_tactical_target(self, context, regime) -> float:
        return self._v47_target_engine.main_tactical_target(self, context, regime)

    def _combine_base_tactical_target(self, base_target: float, tactical_target: float) -> float:
        return self._v47_target_engine.combine_base_tactical_target(self, base_target, tactical_target)

    def _vol_multiplier(self, context, regime) -> float:
        return self._v47_target_engine.vol_multiplier(context, regime)

    def _apply_core_floor(self, symbol: str, tactical_target: float) -> float:
        return self._v47_target_engine.apply_core_floor(self, symbol, tactical_target)

    def _build_sleeve_plans(self, context, regime, episode: dict, signals, intent: str, target_plan, target: float):
        return self._v47_sleeve_engine.build_sleeve_plans(self, context, regime, episode, signals, intent, target_plan, target)

    def _select_primary_sleeve(self, sleeve_plans):
        return self._v47_sleeve_engine.select_primary_sleeve(sleeve_plans)

    def _sell_setup(self, intent: str) -> str:
        return self._v47_sleeve_engine.sell_setup(intent)

    def _buy_setup(self, context, regime, episode: dict, signals) -> str:
        return self._v47_sleeve_engine.buy_setup(self, context, regime, episode, signals)

    def _buy_setup_from_plan(self, context, regime, episode: dict, signals, target_plan) -> str:
        return self._v47_sleeve_engine.buy_setup_from_plan(self, context, regime, episode, signals, target_plan)

    def _bear_base_accumulate_needed(self, context, regime, episode: dict, signals) -> bool:
        return self._v47_sleeve_engine.bear_base_accumulate_needed(self, context, regime, episode, signals)

    def _compute_sizing(self, context, regime, episode: dict, signals, decision):
        return self._v47_sizing_engine.compute_sizing(self, context, regime, episode, signals, decision)

    def _hold_recovery_overlay_sizing(self, context, regime, episode: dict, signals):
        return self._v47_sizing_engine.hold_recovery_overlay_sizing(self, context, regime, episode, signals)

    def _sell_sizing(self, context, regime, episode: dict, signals, decision):
        return self._v47_sizing_engine.sell_sizing(self, context, regime, episode, signals, decision)

    def _structural_exit_max_sell(self, context, regime, episode: dict) -> float:
        return self._v47_sizing_engine.structural_exit_max_sell(context, regime, episode)

    def _sell_block_reason(self, context, regime, episode: dict, signals, intent: str, setup: str, base_exit_sell: bool = False) -> str:
        return self._v47_sizing_engine.sell_block_reason(self, context, regime, episode, signals, intent, setup, base_exit_sell)

    def _recovery_test_sell_block_reason(self, context, regime, episode: dict, signals) -> str:
        return self._v47_sizing_engine.recovery_test_sell_block_reason(self, context, regime, episode, signals)

    def _buy_block_reason(self, context, regime, episode: dict, signals, setup: str, target_plan=None) -> str:
        return self._v47_sizing_engine.buy_block_reason(self, context, regime, episode, signals, setup, target_plan)

    @staticmethod
    def _advance_failed_recovery_signal_count(episode: dict) -> int:
        return V47SizingEngine.advance_failed_recovery_signal_count(episode)

    def _buy_cooldown(self, context, regime, setup: str) -> int:
        return self._v47_sizing_engine.buy_cooldown(self, context, regime, setup)

    def _max_buy_pct(self, context, regime, episode: dict, setup: str, signals=None) -> float:
        return self._v47_sizing_engine.max_buy_pct(self, context, regime, episode, setup, signals)

    @staticmethod
    def _setup_base_buy_pct(setup: str) -> float:
        return V47SizingEngine.setup_base_buy_pct(setup)

    def _buy_sizing_guard(self, context, regime, episode: dict, signals, setup: str, target_plan=None, sleeve_guard: str = "") -> str:
        return self._v47_sizing_engine.buy_sizing_guard(self, context, regime, episode, signals, setup, target_plan, sleeve_guard)

    def _sell_sizing_guard(self, context, regime, episode: dict, signals, setup: str, base_exit_sell: bool = False, sleeve_guard: str = "") -> str:
        return self._v47_sizing_engine.sell_sizing_guard(self, context, regime, episode, signals, setup, base_exit_sell, sleeve_guard)

    def _sizing_guard(self, context, regime, episode: dict, setup: str) -> str:
        return self._v47_sizing_engine.sizing_guard(self, context, regime, episode, setup)

    @staticmethod
    def _setup_intent_name(setup: str) -> str:
        return V47SizingEngine.setup_intent_name(setup)

    def _build_action(self, context, regime, decision, sizing):
        return self._v47_action_engine.build_action(self, context, regime, decision, sizing)

    def _build_action_reason(
        self,
        side: str,
        setup: str,
        risk_score: int,
        trend_risk: int,
        drawdown_risk: int,
        raw_state: str,
        confirmed_state: str,
        target: float,
        guard: str = "",
    ) -> str:
        return self._v47_action_engine.build_action_reason(
            side=side,
            setup=setup,
            risk_score=risk_score,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            target=target,
            guard=guard,
        )

    def _record_event(self, context, regime, episode: dict, decision, sizing, action) -> None:
        self._v47_event_engine.record_event(
            self,
            context,
            regime,
            episode,
            decision,
            sizing,
            action,
        )

    def _record_sleeve_accounting(self, context, decision, sizing, action) -> None:
        self._v47_accounting_engine.record_sleeve_accounting(
            self,
            context,
            decision,
            sizing,
            action,
        )

    def _source_ledger(self, symbol: str) -> dict:
        return self._v47_accounting_engine.source_ledger(self, symbol)

    def _sync_base_from_sources(self, context) -> None:
        self._v47_accounting_engine.sync_base_from_sources(self, context)

    def _add_base_source(self, context, action, source: str, layer: int = 1) -> None:
        self._v47_accounting_engine.add_base_source(self, context, action, source, layer)

    def _source_quantity(self, context, source: str) -> float:
        return self._v47_accounting_engine.source_quantity(self, context, source)

    def _consume_base_sources(self, context, quantity: float, sources_allowed: tuple[str, ...]) -> float:
        return self._v47_accounting_engine.consume_base_sources(self, context, quantity, sources_allowed)

    def _record_protected_floor_base_buy(self, context, action) -> None:
        self._v47_accounting_engine.record_protected_floor_base_buy(self, context, action)

    def _record_bear_base_buy(self, context, action) -> None:
        self._v47_accounting_engine.record_bear_base_buy(self, context, action)

    def _record_base_led_recovery_buy(self, context, action) -> None:
        self._v47_accounting_engine.record_base_led_recovery_buy(self, context, action)

    def _record_protected_floor_base_exit(self, context, action) -> None:
        self._v47_accounting_engine.record_protected_floor_base_exit(self, context, action)

    def _record_base_exit_sell(self, context, action) -> None:
        self._v47_accounting_engine.record_base_exit_sell(self, context, action)

    def _record_lifecycle_state_shadow(self, context, regime, episode: dict, decision, sizing, action) -> None:
        self._v47_lifecycle_engine.record_lifecycle_state_shadow(
            self,
            context,
            regime,
            episode,
            decision,
            sizing,
            action,
        )

    def _lifecycle_source_quantities(self, context) -> dict[str, float]:
        return self._v47_lifecycle_engine.lifecycle_source_quantities(self, context)

    def _action_sleeve(self, decision, sizing) -> str:
        return self._v47_event_engine.action_sleeve(decision, sizing)

    def _latest_lifecycle_shadow_row(self, symbol: str) -> dict | None:
        return self._v47_event_engine.latest_lifecycle_shadow_row(self, symbol)

    def execution_target_for_symbol_with_portfolios(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        decision_portfolio: PortfolioState,
        execution_portfolio: PortfolioState,
    ) -> float:
        decision = self._v47_execution_engine.target_for_symbol(
            owner=self,
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            execution_portfolio=execution_portfolio,
        )
        return decision.final_target_pct

    def execution_transform_diagnostics_for_symbol_with_portfolios(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        decision_portfolio: PortfolioState,
        execution_portfolio: PortfolioState,
        ) -> tuple[float, str]:
        decision = self._v47_execution_engine.target_for_symbol(
            owner=self,
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            execution_portfolio=execution_portfolio,
        )
        return decision.final_target_pct, decision.reason

    def _outer_low_entry(self, df: pd.DataFrame, price: float | None = None) -> bool:
        return self._v47_execution_engine._outer_low_entry(df)

    def _outer_high_exit(self, df: pd.DataFrame, price: float, state: dict) -> bool:
        return self._v47_execution_engine._outer_high_exit(df, price, state)

    def _outer_overlay_target(self, symbol: str, raw: float, df: pd.DataFrame | None, price: float) -> tuple[float, str]:
        if df is None or price <= 0.0:
            return 0.0, "outer_insufficient_history"
        state = self._outer_overlay_state_by_symbol.get(symbol, {})
        if str(state.get("state", "IDLE")) == "HELD":
            return max(0.0, float(state.get("overlay", 0.0) or 0.0)), "outer_compat_hold"
        if self._outer_low_entry(df, price):
            target = max(0.0, float(self._v47_config.outer.target_pct.get(symbol, 0.0)))
            return target, "outer_compat_low_entry"
        return 0.0, "outer_compat_idle"


class V47CleanEventExecV1Strategy(V47CleanStrategy):
    """V4.7 clean with event-triggered execution transform.

    The raw decision portfolio is unchanged.  The execution portfolio only
    rebalances on raw action days or when fixed-quantity outer lots enter/exit.
    Daily target drift is still audited but no longer traded.
    """

    EXECUTION_TRANSFORM_EVENT_TRIGGERED = True

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_v1"


class V47CleanEventExecDrift2V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.02

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift2_v1"


class V47CleanEventExecDrift5V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.05

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift5_v1"


class V47CleanEventExecDrift10V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.10

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_v1"


class V47CleanEventExecDrift10OuterDeepV1Strategy(V47CleanEventExecDrift10V1Strategy):
    """V4.7 drift10 with a sparse, deep-value-only outer layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v47_config = V47Config(
            outer=V47OuterConfig(deep_only_entry=True),
        )
        self._v47_execution_engine = V47ExecutionEngine(self._v47_config)

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_outer_deep_v1"


class V47CleanEventExecDrift10OuterRelaxedV1Strategy(V47CleanEventExecDrift10V1Strategy):
    """V4.7 drift10 with a simply relaxed outer low-entry threshold."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v47_config = V47Config(
            outer=V47OuterConfig(
                entry_rolling365_pos=0.25,
                entry_dd365=-0.52,
                entry_dd180=-0.35,
                entry_rebound20=0.06,
                entry_roc5=-0.06,
                entry_roc20=-0.22,
            ),
        )
        self._v47_execution_engine = V47ExecutionEngine(self._v47_config)

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_outer_relaxed_v1"


class V47CleanEventExecDrift10IntradayShockLadderV7Strategy(V47CleanEventExecDrift10V1Strategy):
    """V4.7 drift10 with v5 shock ladder and stronger panic add tiers."""

    EXECUTION_TRANSFORM_INTRADAY_SHOCK_LADDER_V1 = True
    EXECUTION_TRANSFORM_INTRADAY_LADDER_SELL_DROP = -0.10
    EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_DROP = -0.15
    EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_DROP = -0.20
    EXECUTION_TRANSFORM_INTRADAY_LADDER_MIN_POSITION = 0.80
    EXECUTION_TRANSFORM_INTRADAY_LADDER_FLOOR_POSITION = 0.80
    EXECUTION_TRANSFORM_INTRADAY_LADDER_REDUCE_STEP = 0.35
    EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_STEP = 0.20
    EXECUTION_TRANSFORM_INTRADAY_LADDER_EXTRA_ADD_TIERS = [
        (-0.25, 0.20, "add_25"),
        (-0.30, 0.30, "add_30"),
    ]
    EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_POSITION = 2.70
    EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_CLOSE = False

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v7"


class V48EthBnbStrategy(V47CleanEventExecDrift10IntradayShockLadderV7Strategy):
    """V4.8 deployment candidate: trade ETH/BNB only, keep BTC as regime input.

    The fixed review selected no ETH/BNB parameter changes over V4.7-v7. BTC is
    removed from the traded universe in config, not from the market-regime data.
    """

    @property
    def name(self) -> str:
        return "v4_8_eth_bnb"


class V47CleanEventExecDrift10IntradayShockLadderV11Strategy(V47CleanEventExecDrift10V1Strategy):
    """V4.7 drift10 with v7 panic tiers capped by portfolio gross at 2.60."""

    EXECUTION_TRANSFORM_INTRADAY_SHOCK_LADDER_V1 = True
    EXECUTION_TRANSFORM_INTRADAY_LADDER_SELL_DROP = -0.10
    EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_DROP = -0.15
    EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_DROP = -0.20
    EXECUTION_TRANSFORM_INTRADAY_LADDER_MIN_POSITION = 0.80
    EXECUTION_TRANSFORM_INTRADAY_LADDER_FLOOR_POSITION = 0.80
    EXECUTION_TRANSFORM_INTRADAY_LADDER_REDUCE_STEP = 0.35
    EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_STEP = 0.20
    EXECUTION_TRANSFORM_INTRADAY_LADDER_EXTRA_ADD_TIERS = [
        (-0.25, 0.20, "add_25"),
        (-0.30, 0.30, "add_30"),
    ]
    EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_POSITION = 2.70
    EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_GROSS = 2.60
    EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_CLOSE = False

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v11"


class V47CleanEventExecDrift10IntradayShockLadderV12Strategy(V47CleanEventExecDrift10IntradayShockLadderV7Strategy):
    """V7 shock ladder with a moderately greedier confirmed-trend transform."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v47_config = V47Config(
            execution=V47ExecutionConfig(trend_mult=1.90, trend_cap=1.90),
        )
        self._v47_execution_engine = V47ExecutionEngine(self._v47_config)

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v12"


class V47CleanEventExecDrift15V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.15

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift15_v1"


class V47CleanEventExecDrift20V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.20

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift20_v1"


class V47CleanEventExecDrift30V1Strategy(V47CleanEventExecV1Strategy):
    EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.30

    @property
    def name(self) -> str:
        return "v4_7_clean_event_exec_drift30_v1"
