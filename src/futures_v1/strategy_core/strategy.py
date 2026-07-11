"""Official ETH/BNB futures strategy entry."""

from __future__ import annotations

import pandas as pd

from .. import strategy_utils
from ..strategy_rebalance import Action, PortfolioState, PortfolioStrategyBase
from ..sleeve_accounting import StrategySleeveAccountingMixin
from ..bear_base_rules import StrategyBearBaseMixin
from ..decision_rules import StrategyDecisionMixin
from ..episode_rules import StrategyEpisodeMixin
from ..execution_rules import StrategyExecutionMixin
from ..market_rules import StrategyMarketMixin
from ..recovery_rules import StrategyRecoveryMixin
from ..risk_rules import StrategyRiskMixin
from ..signal_rules import StrategySignalMixin
from ..sizing_rules import StrategySizingMixin
from ..sleeve_rules import StrategySleeveMixin
from ..target_rules import StrategyTargetMixin
from ..strategy_types import StrategyDecisionSnapshot
from .accounting import AccountingEngine
from .action import ActionEngine
from .bear_base import BearBaseEngine
from .config import Config
from .constants import StrategyConstants
from .credit import RecoveryCreditEngine
from .episode import EpisodeEngine, cycle_recovered_pct, cycle_recovery_budget
from .events import EventEngine
from .execution_engine import ExecutionEngine
from .floor import FloorEngine
from .lifecycle import LifecycleEngine
from .market import MarketEngine, price_vs, value
from .raw_decision import RawDecisionEngine
from .recovery import RecoveryEngine
from .signals import SignalEngine
from .sleeve import SleeveEngine
from .sizing import SizingEngine
from .state import StrategyStateMixin
from .target import TargetEngine


class StrategyCore(
    StrategyConstants,
    StrategyStateMixin,
    StrategyMarketMixin,
    StrategySignalMixin,
    StrategyEpisodeMixin,
    StrategyRiskMixin,
    StrategyRecoveryMixin,
    StrategyBearBaseMixin,
    StrategyTargetMixin,
    StrategySleeveMixin,
    StrategySizingMixin,
    StrategyExecutionMixin,
    StrategyDecisionMixin,
    StrategySleeveAccountingMixin,
    PortfolioStrategyBase,
):
    """Composable strategy shell used by the official ETH/BNB futures strategy."""

    compute_indicators = staticmethod(strategy_utils.compute_indicators)

    @property
    def name(self) -> str:
        return "eth_bnb_futures_core"

    def __init__(self, initial_capital: float = 100.0, reserve: float = 20.0, fee_rate: float = 0.001):
        self._init_state(initial_capital, reserve, fee_rate)
        self._core_config = Config()
        self._core_market_engine = MarketEngine()
        self._core_episode_engine = EpisodeEngine()
        self._core_target_engine = TargetEngine()
        self._core_signal_engine = SignalEngine()
        self._core_recovery_engine = RecoveryEngine()
        self._core_sleeve_engine = SleeveEngine()
        self._core_sizing_engine = SizingEngine()
        self._core_action_engine = ActionEngine()
        self._core_event_engine = EventEngine()
        self._core_accounting_engine = AccountingEngine()
        self._core_bear_base_engine = BearBaseEngine()
        self._core_floor_engine = FloorEngine()
        self._core_credit_engine = RecoveryCreditEngine()
        self._core_lifecycle_engine = LifecycleEngine()
        self._core_raw_decision_engine = RawDecisionEngine()
        self._core_execution_engine = ExecutionEngine(self._core_config)

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        return self._core_raw_decision_engine.compute_actions(
            owner=self,
            candles_by_symbol=candles_by_symbol,
            portfolio=portfolio,
            current_prices=current_prices,
        )

    def _build_decision_snapshot(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> StrategyDecisionSnapshot | None:
        context = self._build_context(candles_by_symbol, portfolio, current_prices)
        if context is None:
            return None
        regime = self._build_regime(context)
        episode = self._update_episode(context, regime)
        lifecycle = self._build_lifecycle(context, regime, episode)
        risk_gate = self._build_risk_gate(context, regime, lifecycle)
        risk_assessment = self._build_risk_assessment_shadow(context, regime, lifecycle, risk_gate, episode)
        signals = self._build_signals(context, regime, episode)
        decision = self._build_decision_plan(context, regime, episode, signals)
        self._current_context_by_symbol[context.symbol] = context
        return StrategyDecisionSnapshot(
            context=context,
            regime=regime,
            episode=episode,
            lifecycle=lifecycle,
            risk_gate=risk_gate,
            risk_assessment=risk_assessment,
            signals=signals,
            decision=decision,
        )

    def _record_architecture_trace(self, snapshot, sizing, action: Action | None) -> None:
        return None

    def _build_context(self, candles_by_symbol, portfolio, current_prices):
        return self._core_market_engine.build_context(
            owner=self,
            candles_by_symbol=candles_by_symbol,
            portfolio=portfolio,
            current_prices=current_prices,
        )

    def _build_regime(self, context):
        return self._core_market_engine.build_regime(owner=self, context=context)

    def _calculate_trend_risk(self, latest: pd.Series, price: float) -> int:
        return self._core_market_engine.calculate_trend_risk(latest, price)

    def _calculate_drawdown_risk(self, symbol: str, latest: pd.Series, pos, price: float) -> int:
        return self._core_market_engine.calculate_drawdown_risk(self, symbol, latest, pos, price)

    def _apply_state_confirmation(self, symbol: str, raw_state: str) -> str:
        return self._core_market_engine.apply_state_confirmation(self, symbol, raw_state)

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        return value(latest, column, default)

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        return price_vs(latest, price, column)

    def _build_signals(self, context, regime, episode: dict):
        return self._core_signal_engine.build_signals(self, context, regime, episode)

    def _accumulation_signal(self, context, regime, episode: dict, signals) -> bool:
        return self._core_signal_engine.accumulation_signal(episode, signals)

    def _starter_signal(self, context, regime) -> bool:
        return self._core_signal_engine.starter_signal(context, regime)

    def _value_recovery(self, context, regime) -> bool:
        return self._core_signal_engine.value_recovery(self, context, regime)

    def _trend_continuation(self, context, regime) -> bool:
        return self._core_signal_engine.trend_continuation(self, context, regime)

    def _late_trend_continuation_risk(self, context, regime) -> bool:
        return self._core_signal_engine.late_trend_continuation_risk(self, context, regime)

    def _distribution_exhaustion(self, context, regime) -> bool:
        return self._core_signal_engine.distribution_exhaustion(self, context, regime)

    def _recovery_signal_from_parts(self, *, context, regime, value_recovery: bool, trend_continuation: bool) -> bool:
        return self._core_signal_engine.recovery_signal_from_parts(
            self,
            context=context,
            regime=regime,
            value_recovery=value_recovery,
            trend_continuation=trend_continuation,
        )

    def _recovery_signal(self, context, regime) -> bool:
        return self._core_signal_engine.recovery_signal(self, context, regime)

    def _strong_recovery_signal(self, context, regime) -> bool:
        return self._core_signal_engine.strong_recovery_signal(self, context, regime)

    def _recovery_quality_ok(self, context, regime) -> bool:
        return self._core_signal_engine.recovery_quality_ok(self, context, regime)

    def _episode_recovery_add_cap(self, episode: dict) -> float:
        return self._core_recovery_engine.episode_recovery_add_cap(episode)

    def _episode_recovery_plan(self, context, regime, episode: dict, signals):
        return self._core_recovery_engine.episode_recovery_plan(self, context, regime, episode, signals)

    def _episode_reentry_plan(self, context, regime, episode: dict, signals):
        return self._core_recovery_engine.episode_reentry_plan(self, context, regime, episode, signals)

    def _episode_recovery_target(self, context, regime, episode: dict, signals, base_target: float) -> float:
        return self._core_recovery_engine.episode_recovery_target(self, context, regime, episode, signals, base_target)

    def _episode_recovery_max_buy(self, context, regime, episode: dict, setup: str, signals=None) -> float:
        return self._core_recovery_engine.episode_recovery_max_buy(self, context, regime, episode, setup, signals)

    @staticmethod
    def _episode_recovery_budget(episode: dict) -> float:
        return RecoveryEngine.episode_recovery_budget(episode)

    @staticmethod
    def _episode_recovered_pct(episode: dict) -> float:
        return RecoveryEngine.episode_recovered_pct(episode)

    def _record_episode_recovery_buy(self, context, episode: dict, action) -> None:
        self._core_recovery_engine.record_episode_recovery_buy(self, context, episode, action)

    @staticmethod
    def _setup_from_action(action) -> str:
        return RecoveryEngine.setup_from_action(action)

    @staticmethod
    def _is_bear_base_buy(decision, sizing) -> bool:
        return RecoveryEngine.is_bear_base_buy(decision, sizing)

    def _should_record_episode_recovery_buy(self, bear_base_buy: bool) -> bool:
        return self._core_recovery_engine.should_record_episode_recovery_buy(self, bear_base_buy)

    @staticmethod
    def _recovery_fraction_from_drawdown(low_drop: float, *, shallow: float, normal: float, deep: float, crash: float) -> float:
        return RecoveryEngine.recovery_fraction_from_drawdown(
            low_drop,
            shallow=shallow,
            normal=normal,
            deep=deep,
            crash=crash,
        )

    @staticmethod
    def _recovery_step_max_buy(add_pct: float, low_drop: float, strong: bool, *, deep_base: bool = False) -> float:
        return RecoveryEngine.recovery_step_max_buy(add_pct, low_drop, strong, deep_base=deep_base)

    @staticmethod
    def _recovery_drawdown_guard(low_drop: float) -> str:
        return RecoveryEngine.recovery_drawdown_guard(low_drop)

    def _recovery_permission(self, context, regime, signals, low_drop: float):
        return self._core_recovery_engine.recovery_permission(self, context, regime, signals, low_drop)

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
        return self._core_recovery_engine.staged_recovery_plan(
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
        return self._core_recovery_engine.deep_base_recovery_plan(
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
        return self._core_recovery_engine.post_crash_recoil_plan(
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
        return RecoveryEngine.structural_recovery_ready(context, regime, signals)

    @staticmethod
    def _episode_recovery_cap(episode: dict) -> float:
        return RecoveryEngine.episode_recovery_cap(episode)

    def _base_recovery_established(self, context) -> bool:
        return self._core_recovery_engine.base_recovery_established(self, context)

    def _base_recovery_stabilized(self, context, regime, signals) -> bool:
        return self._core_recovery_engine.base_recovery_stabilized(self, context, regime, signals)

    def _base_recovery_target(self, context, regime, signals) -> float:
        return self._core_recovery_engine.base_recovery_target(self, context, regime, signals)

    def _base_recovery_should_accumulate(self, context, regime, signals) -> bool:
        return self._core_recovery_engine.base_recovery_should_accumulate(self, context, regime, signals)

    def _base_led_recovery_base_sizing(self, context, regime, episode: dict, signals, decision, primary):
        return self._core_recovery_engine.base_led_recovery_base_sizing(
            self,
            context,
            regime,
            episode,
            signals,
            decision,
            primary,
        )

    def _base_led_recovery_allowed(self, context, regime, episode: dict, signals) -> bool:
        return self._core_recovery_engine.base_led_recovery_allowed(self, context, regime, episode, signals)

    def _base_quantity(self, context) -> float:
        return self._core_bear_base_engine.base_quantity(self, context)

    def _main_quantity(self, context) -> float:
        return self._core_bear_base_engine.main_quantity(self, context)

    def _base_pct_from_quantity(self, context, quantity: float) -> float:
        return self._core_bear_base_engine.base_pct_from_quantity(self, context, quantity)

    def _new_recovery_ledger(self) -> dict:
        return self._core_bear_base_engine.new_recovery_ledger(self._call_count)

    @staticmethod
    def _new_base_ledger() -> dict:
        return BearBaseEngine.new_base_ledger()

    def _refresh_base_ledger_market(self, context) -> None:
        self._core_bear_base_engine.refresh_base_ledger_market(self, context)

    def _record_ledger_sell(self, context, sizing) -> None:
        self._core_bear_base_engine.record_ledger_sell(self, context, sizing)

    def _bear_base_floor(self, context) -> float:
        return self._core_bear_base_engine.bear_base_floor(self, context)

    def _bear_base_target(self, context, regime) -> float:
        return self._core_bear_base_engine.bear_base_target(self, context, regime)

    def _bear_base_buy_proposal(self, context, recovery_ledger: dict, base_ledger: dict, location_depth: float):
        return self._core_bear_base_engine.bear_base_buy_proposal(
            self,
            context,
            recovery_ledger,
            base_ledger,
            location_depth,
        )

    def _bear_base_asset_cap(self, symbol: str) -> float:
        return self._core_bear_base_engine.bear_base_asset_cap(self, symbol)

    @staticmethod
    def _ledger_base_budget_used_pct(ledger: dict) -> float:
        return BearBaseEngine.ledger_base_budget_used_pct(ledger)

    def _bear_base_exit_target(self, context, regime, signals) -> float:
        return self._core_bear_base_engine.bear_base_exit_target(self, context, regime, signals)

    def _bear_base_maturity_score(self, context, regime, signals, profit: float) -> int:
        return self._core_bear_base_engine.bear_base_maturity_score(self, context, regime, signals, profit)

    @staticmethod
    def _bear_base_layer(context, location_depth: float) -> int:
        return BearBaseEngine.bear_base_layer(location_depth)

    @staticmethod
    def _bear_base_current_discount(context, recovery_ledger: dict) -> float:
        return BearBaseEngine.bear_base_current_discount(context, recovery_ledger)

    def _bear_base_low_location(self, context, regime) -> bool:
        return self._core_bear_base_engine.bear_base_low_location(self, context, regime)

    def _bear_base_location_depth(self, context) -> float:
        return self._core_bear_base_engine.bear_base_location_depth(self, context)

    def _bear_base_path_accelerating(self, context, regime) -> bool:
        return self._core_bear_base_engine.bear_base_path_accelerating(self, context, regime)

    def _bear_base_path_stabilizing(self, context, regime) -> bool:
        return self._core_bear_base_engine.bear_base_path_stabilizing(self, context, regime)

    def _opportunity_floor_sizing(self, context, regime, signals, prior_sizing):
        return self._core_floor_engine.opportunity_floor_sizing(self, context, regime, signals, prior_sizing)

    def _opportunity_floor_low_location(self, context, regime) -> bool:
        return self._core_floor_engine.opportunity_floor_low_location(self, context, regime)

    def _opportunity_floor_stabilizing(self, context, regime, signals) -> bool:
        return self._core_floor_engine.opportunity_floor_stabilizing(self, context, regime, signals)

    def _protected_floor_quantity(self, context) -> float:
        return self._core_floor_engine.protected_floor_quantity(self, context)

    def _protected_floor_exit_allowed(self, context, regime) -> bool:
        return self._core_floor_engine.protected_floor_exit_allowed(self, context, regime)

    def _record_protected_floor_buy(self, context, regime, action) -> None:
        self._core_floor_engine.record_protected_floor_buy(self, context, regime, action)

    def _consume_protected_floor_for_sell(self, context, regime, action) -> None:
        self._core_floor_engine.consume_protected_floor_for_sell(self, context, regime, action)

    def _protected_floor_exit_sizing(self, context, regime, signals, decision):
        return self._core_floor_engine.protected_floor_exit_sizing(self, context, regime, signals, decision)

    def _protected_floor_exit_event(self, context, regime, signals, profit: float) -> bool:
        return self._core_floor_engine.protected_floor_exit_event(self, context, regime, signals, profit)

    def _protected_floor_new_release_event(self, context, ledger: dict) -> bool:
        return self._core_floor_engine.protected_floor_new_release_event(self, context, ledger)

    def _lifecycle_low_base_sizing(self, context, regime, episode: dict, signals, decision, prior_sizing):
        return self._core_floor_engine.lifecycle_low_base_sizing(
            self,
            context,
            regime,
            episode,
            signals,
            decision,
            prior_sizing,
        )

    def _lifecycle_protected_base_exit_sizing(self, context, regime, signals, decision):
        return self._core_floor_engine.lifecycle_protected_base_exit_sizing(self, context, regime, signals, decision)

    def _lifecycle_base_exit_high_location(self, context, regime) -> bool:
        return self._core_floor_engine.lifecycle_base_exit_high_location(self, context, regime)

    def _lifecycle_low_base_entry_location(self, context, regime) -> bool:
        return self._core_floor_engine.lifecycle_low_base_entry_location(self, context, regime)

    def _lifecycle_low_base_stabilizing(self, context, regime, signals) -> bool:
        return self._core_floor_engine.lifecycle_low_base_stabilizing(self, context, regime, signals)

    def _lifecycle_protected_floor_portfolio_pct(self, context) -> float:
        return self._core_floor_engine.lifecycle_protected_floor_portfolio_pct(self, context)

    def _update_episode(self, context, regime):
        return self._core_episode_engine.update_episode(self, context, regime)

    def _close_episode(self, symbol: str, reason: str) -> None:
        self._core_episode_engine.close_episode(self, symbol, reason)

    def _append_episode_sell_leg(self, context, episode: dict, sizing) -> None:
        self._core_episode_engine.append_episode_sell_leg(self, context, episode, sizing)

    @staticmethod
    def _cycle_recovery_budget(episode: dict) -> float:
        return cycle_recovery_budget(episode)

    @staticmethod
    def _cycle_recovered_pct(episode: dict) -> float:
        return cycle_recovered_pct(episode)

    def _open_recovery_credit(self, symbol: str, episode: dict, open_budget: float, reason: str) -> None:
        self._core_episode_engine.open_recovery_credit(self, symbol, episode, open_budget, reason)

    def _decay_recovery_credit(self, context) -> None:
        self._core_episode_engine.decay_recovery_credit(self, context)

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
        self._core_episode_engine.record_recovery_credit_event(
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
        self._core_episode_engine.record_recovery_credit_check(
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
        return self._core_episode_engine.recovery_credit_total(self)

    def _recovery_credit_portfolio_cap(self) -> float:
        return self._core_episode_engine.recovery_credit_portfolio_cap(self)

    def _recovery_credit_plan(self, context, regime, signals):
        return self._core_credit_engine.recovery_credit_plan(self, context, regime, signals)

    def _recovery_credit_sizing(self, context, regime, signals):
        return self._core_credit_engine.recovery_credit_sizing(self, context, regime, signals)

    def _track_main_buy_for_credit(self, context, sizing) -> None:
        self._core_credit_engine.track_main_buy_for_credit(self, context, sizing)

    def _consume_recovery_credit_from_current_buy(self, context, regime, signals, sizing) -> None:
        self._core_credit_engine.consume_recovery_credit_from_current_buy(self, context, regime, signals, sizing)

    def _consume_recovery_credit_from_recent_buy(self, context, regime) -> None:
        self._core_credit_engine.consume_recovery_credit_from_recent_buy(self, context, regime)

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
        self._core_credit_engine.consume_recovery_credit_by_main_buy(
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
        self._core_credit_engine.release_recovery_credit(self, context, sizing)

    def _allow_btc_deep_credit_overlay(self, context) -> bool:
        return self._core_credit_engine.allow_btc_deep_credit_overlay(self, context)

    def _compose_target_from_plan(self, context, plan):
        return self._core_target_engine.compose_target_from_plan(self, context, plan)

    def _build_target_plan(self, context, regime, episode: dict, signals, intent: str):
        return self._core_target_engine.build_target_plan(self, context, regime, episode, signals, intent)

    def _main_tactical_target(self, context, regime) -> float:
        return self._core_target_engine.main_tactical_target(self, context, regime)

    def _combine_base_tactical_target(self, base_target: float, tactical_target: float) -> float:
        return self._core_target_engine.combine_base_tactical_target(self, base_target, tactical_target)

    def _vol_multiplier(self, context, regime) -> float:
        return self._core_target_engine.vol_multiplier(context, regime)

    def _apply_core_floor(self, symbol: str, tactical_target: float) -> float:
        return self._core_target_engine.apply_core_floor(self, symbol, tactical_target)

    def _build_sleeve_plans(self, context, regime, episode: dict, signals, intent: str, target_plan, target: float):
        return self._core_sleeve_engine.build_sleeve_plans(self, context, regime, episode, signals, intent, target_plan, target)

    def _select_primary_sleeve(self, sleeve_plans):
        return self._core_sleeve_engine.select_primary_sleeve(sleeve_plans)

    def _sell_setup(self, intent: str) -> str:
        return self._core_sleeve_engine.sell_setup(intent)

    def _buy_setup(self, context, regime, episode: dict, signals) -> str:
        return self._core_sleeve_engine.buy_setup(self, context, regime, episode, signals)

    def _buy_setup_from_plan(self, context, regime, episode: dict, signals, target_plan) -> str:
        return self._core_sleeve_engine.buy_setup_from_plan(self, context, regime, episode, signals, target_plan)

    def _bear_base_accumulate_needed(self, context, regime, episode: dict, signals) -> bool:
        return self._core_sleeve_engine.bear_base_accumulate_needed(self, context, regime, episode, signals)

    def _compute_sizing(self, context, regime, episode: dict, signals, decision):
        return self._core_sizing_engine.compute_sizing(self, context, regime, episode, signals, decision)

    def _hold_recovery_overlay_sizing(self, context, regime, episode: dict, signals):
        return self._core_sizing_engine.hold_recovery_overlay_sizing(self, context, regime, episode, signals)

    def _sell_sizing(self, context, regime, episode: dict, signals, decision):
        return self._core_sizing_engine.sell_sizing(self, context, regime, episode, signals, decision)

    def _structural_exit_max_sell(self, context, regime, episode: dict) -> float:
        return self._core_sizing_engine.structural_exit_max_sell(context, regime, episode)

    def _sell_block_reason(self, context, regime, episode: dict, signals, intent: str, setup: str, base_exit_sell: bool = False) -> str:
        return self._core_sizing_engine.sell_block_reason(self, context, regime, episode, signals, intent, setup, base_exit_sell)

    def _recovery_test_sell_block_reason(self, context, regime, episode: dict, signals) -> str:
        return self._core_sizing_engine.recovery_test_sell_block_reason(self, context, regime, episode, signals)

    def _buy_block_reason(self, context, regime, episode: dict, signals, setup: str, target_plan=None) -> str:
        return self._core_sizing_engine.buy_block_reason(self, context, regime, episode, signals, setup, target_plan)

    @staticmethod
    def _advance_failed_recovery_signal_count(episode: dict) -> int:
        return SizingEngine.advance_failed_recovery_signal_count(episode)

    def _buy_cooldown(self, context, regime, setup: str) -> int:
        return self._core_sizing_engine.buy_cooldown(self, context, regime, setup)

    def _max_buy_pct(self, context, regime, episode: dict, setup: str, signals=None) -> float:
        return self._core_sizing_engine.max_buy_pct(self, context, regime, episode, setup, signals)

    @staticmethod
    def _setup_base_buy_pct(setup: str) -> float:
        return SizingEngine.setup_base_buy_pct(setup)

    def _buy_sizing_guard(self, context, regime, episode: dict, signals, setup: str, target_plan=None, sleeve_guard: str = "") -> str:
        return self._core_sizing_engine.buy_sizing_guard(self, context, regime, episode, signals, setup, target_plan, sleeve_guard)

    def _sell_sizing_guard(self, context, regime, episode: dict, signals, setup: str, base_exit_sell: bool = False, sleeve_guard: str = "") -> str:
        return self._core_sizing_engine.sell_sizing_guard(self, context, regime, episode, signals, setup, base_exit_sell, sleeve_guard)

    def _sizing_guard(self, context, regime, episode: dict, setup: str) -> str:
        return self._core_sizing_engine.sizing_guard(self, context, regime, episode, setup)

    @staticmethod
    def _setup_intent_name(setup: str) -> str:
        return SizingEngine.setup_intent_name(setup)

    def _route_eth_recovery_buy_to_base(self, context, regime, episode: dict, signals, sizing) -> bool:
        if context.symbol != "ETH/USDT":
            return False
        if sizing.side != "buy" or sizing.quantity <= 1e-12:
            return False
        if sizing.setup not in {"recovery-probe-buy", "value-recovery"}:
            return False
        if str(episode.get("state", "NORMAL")) not in {
            "DEFENSE_LOCK",
            "RECOVERY_TEST",
            "FAILED_RECOVERY_LOCK",
            "STRUCTURAL_BEAR_LOCK",
        }:
            return False
        if self._base_pct_from_quantity(context, self._base_quantity(context)) < 0.015:
            return False
        if regime.structural_bear or signals.distribution_exhaustion:
            return False
        if context.trend_risk >= 3 or context.risk_score >= 4:
            return False
        ledger = self._base_ledger_by_symbol.get(context.symbol, {})
        avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry > 0.0 and context.price > avg_entry * self.RECOVERY_ROUTE_MAX_ENTRY_MULTIPLE:
            return False
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(context.latest, "donchian_pos", 0.5)
        return bool(
            (not pd.isna(rolling_pos) and rolling_pos <= 0.35)
            or (not pd.isna(donchian_pos) and donchian_pos <= 0.45)
            or regime.price_vs_ema168 <= 0.02
        )

    def _build_action(self, context, regime, decision, sizing):
        return self._core_action_engine.build_action(self, context, regime, decision, sizing)

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
        return self._core_action_engine.build_action_reason(
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
        self._core_event_engine.record_event(
            self,
            context,
            regime,
            episode,
            decision,
            sizing,
            action,
        )

    def _record_sleeve_accounting(self, context, decision, sizing, action) -> None:
        self._core_accounting_engine.record_sleeve_accounting(
            self,
            context,
            decision,
            sizing,
            action,
        )

    def _source_ledger(self, symbol: str) -> dict:
        return self._core_accounting_engine.source_ledger(self, symbol)

    def _sync_base_from_sources(self, context) -> None:
        self._core_accounting_engine.sync_base_from_sources(self, context)

    def _add_base_source(self, context, action, source: str, layer: int = 1) -> None:
        self._core_accounting_engine.add_base_source(self, context, action, source, layer)

    def _source_quantity(self, context, source: str) -> float:
        return self._core_accounting_engine.source_quantity(self, context, source)

    def _consume_base_sources(self, context, quantity: float, sources_allowed: tuple[str, ...]) -> float:
        return self._core_accounting_engine.consume_base_sources(self, context, quantity, sources_allowed)

    def _record_protected_floor_base_buy(self, context, action) -> None:
        self._core_accounting_engine.record_protected_floor_base_buy(self, context, action)

    def _record_bear_base_buy(self, context, action) -> None:
        self._core_accounting_engine.record_bear_base_buy(self, context, action)

    def _record_base_led_recovery_buy(self, context, action) -> None:
        self._core_accounting_engine.record_base_led_recovery_buy(self, context, action)

    def _record_protected_floor_base_exit(self, context, action) -> None:
        self._core_accounting_engine.record_protected_floor_base_exit(self, context, action)

    def _record_base_exit_sell(self, context, action) -> None:
        self._core_accounting_engine.record_base_exit_sell(self, context, action)

    def _record_lifecycle_state_shadow(self, context, regime, episode: dict, decision, sizing, action) -> None:
        self._core_lifecycle_engine.record_lifecycle_state_shadow(
            self,
            context,
            regime,
            episode,
            decision,
            sizing,
            action,
        )

    def _lifecycle_source_quantities(self, context) -> dict[str, float]:
        return self._core_lifecycle_engine.lifecycle_source_quantities(self, context)

    def _action_sleeve(self, decision, sizing) -> str:
        return self._core_event_engine.action_sleeve(decision, sizing)

    def _latest_lifecycle_shadow_row(self, symbol: str) -> dict | None:
        return self._core_event_engine.latest_lifecycle_shadow_row(self, symbol)

    def execution_target_for_symbol(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
    ) -> float:
        target, _reason = self.execution_transform_diagnostics_for_symbol(
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
        )
        return target

    def execution_transform_diagnostics_for_symbol(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
    ) -> tuple[float, str]:
        raw = max(0.0, float(raw_position_pct))
        row = self._latest_lifecycle_shadow_row(symbol)
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
        if low_location and recovery_active and risk <= 2 and raw >= 0.20:
            return min(self.EXEC_LOW_AWARE_LOW_CAP, raw * self.EXEC_LOW_AWARE_LOW_MULT), "low_recovery"
        if trend_confirmed and confirmed == "BULL" and risk <= 1 and raw >= 0.50:
            return min(self.EXEC_LOW_AWARE_TREND_CAP, raw * self.EXEC_LOW_AWARE_TREND_MULT), "trend_confirmed"
        return raw, "raw"

    def execution_target_for_symbol_with_portfolios(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        decision_portfolio: PortfolioState,
        execution_portfolio: PortfolioState,
    ) -> float:
        decision = self._core_execution_engine.target_for_symbol(
            owner=self,
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            execution_portfolio=execution_portfolio,
        )
        target = self._apply_research_target_gross_cap(decision.final_target_pct)
        return self._apply_research_financing_gate(target, candles_by_symbol.get(symbol))[0]

    def execution_transform_diagnostics_for_symbol_with_portfolios(
        self,
        symbol: str,
        raw_position_pct: float,
        candles_by_symbol: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        decision_portfolio: PortfolioState,
        execution_portfolio: PortfolioState,
        ) -> tuple[float, str]:
        decision = self._core_execution_engine.target_for_symbol(
            owner=self,
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            execution_portfolio=execution_portfolio,
        )
        target = self._apply_research_target_gross_cap(decision.final_target_pct)
        target, gate_reason = self._apply_research_financing_gate(target, candles_by_symbol.get(symbol))
        return target, f"{decision.reason}+{gate_reason}" if gate_reason else decision.reason

    def _apply_research_target_gross_cap(self, target: float) -> float:
        cap = float(getattr(self, "RESEARCH_TARGET_GROSS_CAP", 0.0) or 0.0)
        return min(float(target), cap) if cap > 0.0 else float(target)

    def _apply_research_financing_gate(
        self,
        target: float,
        df: pd.DataFrame | None,
    ) -> tuple[float, str]:
        if not bool(getattr(self, "RESEARCH_FINANCING_GATE_ENABLED", False)) or df is None or df.empty:
            return float(target), ""
        latest = df.iloc[-1]
        regime = str(latest.get("btc_regime", "RANGE") or "RANGE")
        strong_cap = float(getattr(self, "RESEARCH_FINANCING_GATE_STRONG_BULL_CAP", 0.0) or 0.0)
        non_strong_cap = float(getattr(self, "RESEARCH_FINANCING_GATE_NON_STRONG_BULL_CAP", 0.0) or 0.0)
        cap = strong_cap if regime == "STRONG_BULL" else non_strong_cap
        reasons = [f"financing-gate_{regime.lower()}"]

        vol_target = float(getattr(self, "RESEARCH_FINANCING_GATE_VOL_TARGET", 0.0) or 0.0)
        if regime != "STRONG_BULL" and vol_target > 0.0 and len(df) >= 31:
            returns = pd.to_numeric(df["close"], errors="coerce").pct_change().tail(30)
            realized_vol = float(returns.std(ddof=1) * (365.25 ** 0.5))
            if realized_vol > 0.0:
                min_cap = float(getattr(self, "RESEARCH_FINANCING_GATE_VOL_MIN_CAP", 0.8) or 0.8)
                max_cap = float(getattr(self, "RESEARCH_FINANCING_GATE_VOL_MAX_CAP", 1.2) or 1.2)
                cap = min(cap, max(min_cap, min(max_cap, vol_target / realized_vol)))
                reasons.append("vol")

        funding_quantile = float(getattr(self, "RESEARCH_FINANCING_GATE_FUNDING_QUANTILE", 0.0) or 0.0)
        if funding_quantile > 0.0 and "funding_rate_daily" in df.columns and len(df) >= 31:
            funding = pd.to_numeric(df["funding_rate_daily"], errors="coerce").dropna()
            history = funding.iloc[:-1].tail(180)
            latest_funding = float(funding.iloc[-1]) if not funding.empty else 0.0
            if len(history) >= 30:
                threshold = float(history.quantile(funding_quantile))
                if latest_funding > max(0.0, threshold):
                    cap = min(cap, float(getattr(self, "RESEARCH_FINANCING_GATE_FUNDING_CAP", 1.0) or 1.0))
                    reasons.append("funding-high")

        if cap <= 0.0:
            return float(target), "+".join(reasons)
        return min(float(target), cap), "+".join(reasons)

    def _outer_low_entry(self, df: pd.DataFrame, price: float | None = None) -> bool:
        return self._core_execution_engine._outer_low_entry(df)

    def _outer_high_exit(self, df: pd.DataFrame, price: float, state: dict) -> bool:
        return self._core_execution_engine._outer_high_exit(df, price, state)

    def _outer_overlay_target(self, symbol: str, raw: float, df: pd.DataFrame | None, price: float) -> tuple[float, str]:
        if df is None or price <= 0.0:
            return 0.0, "outer_insufficient_history"
        state = self._outer_overlay_state_by_symbol.get(symbol, {})
        if str(state.get("state", "IDLE")) == "HELD":
            return max(0.0, float(state.get("overlay", 0.0) or 0.0)), "outer_compat_hold"
        if self._outer_low_entry(df, price):
            target = max(0.0, float(self._core_config.outer.target_pct.get(symbol, 0.0)))
            return target, "outer_compat_low_entry"
        return 0.0, "outer_compat_idle"

class FuturesV1Strategy(StrategyCore):
    """Official V1 deployment candidate: trade ETH/BNB only, keep BTC as regime input.

    The fixed review selected no ETH/BNB parameter changes over V1-v7. BTC is
    removed from the traded universe in config, not from the market-regime data.
    """

    @property
    def name(self) -> str:
        return "eth_bnb_futures_v1"
