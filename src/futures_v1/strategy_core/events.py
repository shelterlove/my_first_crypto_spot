"""Event and lifecycle lookup helpers for V1."""

from __future__ import annotations

from ..strategy_rebalance import Action
from ..strategy_types import StrategyContext, StrategyDecisionPlan, StrategyRegime, StrategySizing


class EventEngine:
    def record_event(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        if action is not None and action.side == "buy" and sizing.setup == "base-led-recovery-buy":
            owner._add_base_source(context, action, "base_led_recovery", layer=2)

        self._record_event_protected_floor_v10(owner, context, regime, episode, decision, sizing, action)

        if action is not None and action.side == "buy" and "v4_5_lifecycle_low_base" in str(sizing.guard):
            owner._last_lifecycle_low_base_buy_call_by_symbol[context.symbol] = owner._call_count
        owner._record_lifecycle_state_shadow(context, regime, episode, decision, sizing, action)

    def _record_event_protected_floor_v10(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        self._record_event_protected_floor_v6(owner, context, regime, episode, decision, sizing, action)
        if action is not None and action.side == "sell" and sizing.setup == "protected-floor-exit":
            owner._record_protected_floor_base_exit(context, action)

    def _record_event_protected_floor_v6(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        self._record_event_protected_floor_v1(owner, context, regime, episode, decision, sizing, action)
        if action is not None and action.side == "buy" and sizing.setup == "opportunity-floor-buy":
            owner._record_protected_floor_base_buy(context, action)

    def _record_event_protected_floor_v1(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        if action is not None and action.side == "sell":
            owner._consume_protected_floor_for_sell(context, regime, action)
        self._record_event_opportunity_floor_v3(owner, context, regime, episode, decision, sizing, action)
        if action is not None and action.side == "buy" and sizing.setup == "opportunity-floor-buy":
            owner._record_protected_floor_buy(context, regime, action)

    def _record_event_opportunity_floor_v3(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        if action is None or sizing.setup != "opportunity-floor-buy" or action.side != "buy":
            self._record_event_recovery_credit_consume(owner, context, regime, episode, decision, sizing, action)
            return

        symbol = context.symbol
        owner._diag["core_action_buy_count"] += 1
        owner._diag["v4_3_opportunity_floor_executed_count"] = (
            owner._diag.get("v4_3_opportunity_floor_executed_count", 0) + 1
        )
        owner._last_opportunity_floor_buy_call_by_symbol[symbol] = owner._call_count
        if hasattr(owner, "_record_sleeve_accounting"):
            owner._record_sleeve_accounting(context, decision, sizing, action)

    def _record_event_recovery_credit_consume(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        self._record_event_recovery_credit_soft(owner, context, regime, episode, decision, sizing, action)
        if action is not None and action.side == "buy":
            owner._track_main_buy_for_credit(context, sizing)
            owner._consume_recovery_credit_from_current_buy(context, regime, signals=None, sizing=sizing)
            return
        if (
            action is None
            and sizing.side == "buy"
            and sizing.setup == "recovery-probe-buy"
            and sizing.blocked_reason == "cooldown"
        ):
            owner._consume_recovery_credit_from_recent_buy(context, regime)

    def _record_event_recovery_credit_soft(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        self._record_event_execution_rules(owner, context, regime, episode, decision, sizing, action)
        if action is not None and action.side == "buy" and "core_recovery_credit_soft" in str(sizing.guard):
            owner._release_recovery_credit(context, sizing)

    @staticmethod
    def _record_event_execution_rules(
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        symbol = context.symbol
        blocked = sizing.blocked_reason
        if blocked and blocked != "target_reached":
            owner._diag["core_sizing_blocked_count"] += 1
            active = owner._episodes_by_symbol.get(symbol)
            if active is not None:
                active["blocked_reason"] = owner._join_guard(str(active.get("blocked_reason", "")), blocked)
        if action is None:
            if hasattr(owner, "_record_sleeve_accounting"):
                owner._record_sleeve_accounting(context, decision, sizing, action)
            return

        owner._diag[f"core_action_{action.side}_count"] += 1
        setup_key = sizing.setup.replace("-", "_")
        diag_key = f"core_{setup_key}_count"
        if diag_key in owner._diag:
            owner._diag[diag_key] += 1
        if action.side == "buy":
            owner._last_buy_call_by_symbol[symbol] = owner._call_count
            bear_base_buy = owner._is_bear_base_buy(decision, sizing)
            if bear_base_buy:
                owner._diag["core_bear_base_buy_count"] += 1
                owner._record_bear_base_buy(context, action)
            if "core_deep_base_recovery" in str(sizing.guard):
                owner._diag["core_deep_base_recovery_count"] += 1
            if "core_post_crash_recoil" in str(sizing.guard):
                owner._diag["core_post_crash_recoil_count"] += 1
            active = owner._episodes_by_symbol.get(symbol)
            if active is not None and owner._should_record_episode_recovery_buy(bear_base_buy):
                owner._record_episode_recovery_buy(context, active, action)
            if active is not None and sizing.setup == "value-recovery":
                active["had_value_recovery"] = True
            if hasattr(owner, "_record_sleeve_accounting"):
                owner._record_sleeve_accounting(context, decision, sizing, action)
            return

        owner._last_sell_call_by_symbol[symbol] = owner._call_count
        primary_sleeve = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        if action.side == "sell" and (primary_sleeve == "bear-base-exit" or "core_bear_base_exit" in str(sizing.guard)):
            owner._record_base_exit_sell(context, action)
        if action.side == "sell" and sizing.setup in {"defense-sell", "structural-exit-sell", "distribution-sell"}:
            if sizing.setup in {"defense-sell", "structural-exit-sell"}:
                owner._record_ledger_sell(context, sizing)
                if owner._base_quantity(context) > 1e-12:
                    owner._diag["core_would_force_base_exit_count"] += 1
            owner._start_episode(context, sizing)
        if hasattr(owner, "_record_sleeve_accounting"):
            owner._record_sleeve_accounting(context, decision, sizing, action)

    @staticmethod
    def action_sleeve(decision: StrategyDecisionPlan, sizing: StrategySizing) -> str:
        if sizing.setup in {"base-led-recovery-buy", "opportunity-floor-buy", "protected-floor-exit"}:
            return "base"
        primary = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        if primary in {"bear-base", "bear-base-exit"}:
            return "base"
        if sizing.setup == "bear-base-exit" or "core_bear_base" in str(sizing.guard):
            return "base"
        return "main"

    @staticmethod
    def latest_lifecycle_shadow_row(owner, symbol: str) -> dict | None:
        for row in reversed(getattr(owner, "_lifecycle_state_shadow", [])):
            if row.get("symbol") == symbol:
                return row
        return None
