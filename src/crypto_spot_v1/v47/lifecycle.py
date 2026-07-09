"""Lifecycle shadow boundary for V4.7."""

from __future__ import annotations

from ..strategy_rebalance import Action
from ..v42_types import V42Context, V42DecisionPlan, V42Regime, V42Sizing


class V47LifecycleEngine:
    def record_lifecycle_state_shadow(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action | None,
    ) -> None:
        signals = owner._build_signals(context, regime, episode)
        lifecycle = owner._v45_lifecycle_policy.build_view(
            context=context,
            regime=regime,
            episode=episode,
            signals=signals,
            decision=decision,
            sizing=sizing,
        )
        source_qty = self.lifecycle_source_quantities(owner, context)
        target_plan = decision.target_plan
        primary = decision.primary_sleeve
        owner._lifecycle_state_shadow.append({
            "timestamp": context.latest.get("timestamp"),
            "symbol": context.symbol,
            "call_count": int(owner._call_count),
            "lifecycle_state": lifecycle.state,
            "lifecycle_reason": lifecycle.reason,
            "regime": regime.regime,
            "regime_reason": regime.reason,
            "btc_regime": regime.btc_regime,
            "raw_state": context.raw_state,
            "confirmed_state": context.confirmed_state,
            "episode_id": str(episode.get("id", "")),
            "episode_state": str(episode.get("state", "NORMAL")),
            "episode_setup": str(episode.get("setup", "")),
            "episode_cumulative_sold_pct": float(episode.get("cumulative_sold_pct", 0.0) or 0.0),
            "episode_cumulative_recovered_pct": float(episode.get("cumulative_recovered_pct", 0.0) or 0.0),
            "episode_open_recovery_budget_pct": max(
                0.0,
                float(episode.get("recovery_budget_pct", episode.get("cumulative_sold_pct", 0.0)) or 0.0)
                - float(episode.get("cumulative_recovered_pct", episode.get("recovery_bought_pct", 0.0)) or 0.0),
            ),
            "risk_score": int(context.risk_score),
            "trend_risk": int(context.trend_risk),
            "drawdown_risk": int(context.drawdown_risk),
            "structural_bear": bool(regime.structural_bear),
            "current_pct": float(context.current_pct),
            "mature_target": float(decision.mature_target),
            "phase_target": float(decision.phase_target),
            "execution_target_today": float(decision.execution_target_today),
            "decision_target": float(decision.target),
            "risk_ceiling_shadow": float(lifecycle.risk_ceiling),
            "lifecycle_target_shadow": float(lifecycle.target),
            "target_gap_to_lifecycle_shadow": max(0.0, float(lifecycle.target) - float(context.current_pct)),
            "base_target_shadow": float(target_plan.desired_base if target_plan is not None else 0.0),
            "bear_base_floor": float(target_plan.bear_base_floor if target_plan is not None else owner._bear_base_floor(context)),
            "main_quantity": float(owner._main_quantity(context)),
            "base_quantity": float(owner._base_quantity(context)),
            "source_protected_floor_qty": source_qty["protected_floor"],
            "source_strategic_base_qty": source_qty["strategic_base"],
            "source_base_led_recovery_qty": source_qty["base_led_recovery"],
            "source_other_base_qty": source_qty["other_base"],
            "buy_allowed_shadow": bool(sizing.side == "buy" and not sizing.blocked_reason and sizing.quantity > 1e-12),
            "sell_allowed_sources": lifecycle.sell_allowed_sources,
            "low_location_shadow": bool(lifecycle.low_location),
            "recovery_active_shadow": bool(
                str(episode.get("state", "NORMAL"))
                in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}
            ),
            "trend_confirmed_shadow": bool(regime.regime == "BULL" and context.trend_risk <= 1),
            "distribution_shadow": bool(
                signals.distribution_exhaustion
                or sizing.setup in {"distribution-sell", "bear-base-exit", "protected-floor-exit"}
            ),
            "starter_signal": bool(signals.starter),
            "value_recovery_signal": bool(signals.value_recovery),
            "trend_continuation_signal": bool(signals.trend_continuation),
            "recovery_signal": bool(signals.recovery_signal),
            "strong_recovery_signal": bool(signals.strong_recovery_signal),
            "recovery_quality_ok": bool(signals.recovery_quality_ok),
            "selected_sleeve": primary.sleeve if primary is not None else "",
            "selected_setup": primary.setup if primary is not None else "",
            "selected_priority": int(primary.priority) if primary is not None else 0,
            "sizing_side": sizing.side,
            "sizing_setup": sizing.setup,
            "sizing_quantity": float(sizing.quantity),
            "sizing_target": float(sizing.target),
            "sizing_guard": sizing.guard,
            "sizing_blocked_reason": sizing.blocked_reason,
            "actual_position_before": float(sizing.actual_position_before),
            "actual_position_after": float(sizing.actual_position_after),
            "actual_step_pct": float(sizing.actual_step_pct),
            "action_side": action.side if action is not None else "",
            "action_setup": sizing.setup if action is not None else "",
            "action_quantity": float(action.quantity) if action is not None else 0.0,
            "action_reason": action.reason if action is not None else "",
        })

    @staticmethod
    def lifecycle_source_quantities(owner, context: V42Context) -> dict[str, float]:
        sources = owner._base_ledger_by_symbol.get(context.symbol, {}).get("source_ledger", {})
        protected = max(0.0, float(sources.get("protected_floor", {}).get("quantity", 0.0) or 0.0))
        strategic = max(0.0, float(sources.get("strategic_base", {}).get("quantity", 0.0) or 0.0))
        recovery = max(0.0, float(sources.get("base_led_recovery", {}).get("quantity", 0.0) or 0.0))
        total = max(0.0, float(owner._base_quantity(context)))
        return {
            "protected_floor": protected,
            "strategic_base": strategic,
            "base_led_recovery": recovery,
            "other_base": max(0.0, total - protected - strategic - recovery),
        }
