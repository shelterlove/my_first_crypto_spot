from __future__ import annotations

from .strategy_rebalance import Action
from .strategy_types import StrategyContext, StrategyDecisionPlan, StrategyRegime, StrategySizing


class StrategyExecutionMixin:
    def _build_action(self, context: StrategyContext, regime: StrategyRegime, decision: StrategyDecisionPlan, sizing: StrategySizing) -> Action | None:
        if not sizing.side or sizing.quantity <= 1e-12:
            return None
        return Action(
            symbol=context.symbol,
            side=sizing.side,
            quantity=sizing.quantity,
            price=context.price,
            reason=self._build_action_reason(
                side=sizing.side,
                setup=sizing.setup,
                risk_score=context.risk_score,
                trend_risk=context.trend_risk,
                drawdown_risk=context.drawdown_risk,
                raw_state=context.raw_state,
                confirmed_state=context.confirmed_state,
                target=float(sizing.get("target", decision.target)),
                guard=sizing.guard,
            ),
            diagnostics={
                "mature_target": float(decision.mature_target),
                "phase_target": float(decision.phase_target),
                "execution_target_today": float(decision.execution_target_today),
                "main_intent": str(getattr(decision, "main_intent", decision.intent)),
                "base_intent": str(getattr(decision, "base_intent", "")),
                "main_delta": float(getattr(decision, "main_delta", 0.0)),
                "base_delta": float(getattr(decision, "base_delta", 0.0)),
                "primary_sleeve": decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else "",
                "actual_position_before": float(sizing.actual_position_before),
                "actual_position_after": float(sizing.actual_position_after),
                "target_gap_before": float(sizing.target_gap_before),
                "actual_step_pct": float(sizing.actual_step_pct),
                "remaining_gap_after": float(sizing.remaining_gap_after),
            },
        )

    def _record_event(self, context: StrategyContext, regime: StrategyRegime, episode: dict, decision: StrategyDecisionPlan, sizing: StrategySizing, action: Action | None) -> None:
        symbol = context.symbol
        blocked = sizing.blocked_reason
        if blocked and blocked != "target_reached":
            self._diag["core_sizing_blocked_count"] += 1
            active = self._episodes_by_symbol.get(symbol)
            if active is not None:
                active["blocked_reason"] = self._join_guard(str(active.get("blocked_reason", "")), blocked)
        if action is None:
            if hasattr(self, "_record_sleeve_accounting"):
                self._record_sleeve_accounting(context, decision, sizing, action)
            return

        self._diag[f"core_action_{action.side}_count"] += 1
        setup_key = sizing.setup.replace("-", "_")
        diag_key = f"core_{setup_key}_count"
        if diag_key in self._diag:
            self._diag[diag_key] += 1
        if action.side == "buy":
            self._last_buy_call_by_symbol[symbol] = self._call_count
            primary_sleeve = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
            bear_base_buy = self._is_bear_base_buy(decision, sizing)
            if bear_base_buy:
                self._diag["core_bear_base_buy_count"] += 1
                self._record_bear_base_buy(context, action)
            if "core_deep_base_recovery" in str(sizing.guard):
                self._diag["core_deep_base_recovery_count"] += 1
            if "core_post_crash_recoil" in str(sizing.guard):
                self._diag["core_post_crash_recoil_count"] += 1
            active = self._episodes_by_symbol.get(symbol)
            if active is not None and self._should_record_episode_recovery_buy(bear_base_buy):
                self._record_episode_recovery_buy(context, active, action)
            if active is not None and sizing.setup == "value-recovery":
                active["had_value_recovery"] = True
            if hasattr(self, "_record_sleeve_accounting"):
                self._record_sleeve_accounting(context, decision, sizing, action)
            return

        self._last_sell_call_by_symbol[symbol] = self._call_count
        primary_sleeve = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        if action.side == "sell" and (primary_sleeve == "bear-base-exit" or "core_bear_base_exit" in str(sizing.guard)):
            self._record_base_exit_sell(context, action)
        if action.side == "sell" and sizing.setup in {"defense-sell", "structural-exit-sell", "distribution-sell"}:
            if sizing.setup in {"defense-sell", "structural-exit-sell"}:
                self._record_ledger_sell(context, sizing)
                if self._base_quantity(context) > 1e-12:
                    self._diag["core_would_force_base_exit_count"] += 1
            self._start_episode(context, sizing)
        if hasattr(self, "_record_sleeve_accounting"):
            self._record_sleeve_accounting(context, decision, sizing, action)

    @staticmethod
    def _join_guard(existing: str, addition: str) -> str:
        if not existing:
            return addition
        if not addition:
            return existing
        parts = [part for part in existing.split("-") if part]
        if addition not in parts:
            parts.append(addition)
        return "-".join(parts)

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
        reason = (
            f"core_{side}_{setup}"
            f"_r{risk_score}_tr{trend_risk}_dd{drawdown_risk}"
            f"_raw{raw_state}_conf{confirmed_state}_t{target:.0%}"
        )
        if guard:
            reason = f"{reason}_{guard}"
        return reason
