"""Raw action construction for V4.7."""

from __future__ import annotations

from ..strategy_rebalance import Action
from ..v42_types import V42Context, V42DecisionPlan, V42Regime, V42Sizing


class V47ActionEngine:
    def build_action(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
    ) -> Action | None:
        if not sizing.side or sizing.quantity <= 1e-12:
            return None
        action = Action(
            symbol=context.symbol,
            side=sizing.side,
            quantity=sizing.quantity,
            price=context.price,
            reason=self.build_action_reason(
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
        diagnostics = dict(getattr(action, "diagnostics", {}) or {})
        for key in (
            "recovery_credit_before",
            "recovery_credit_used",
            "recovery_credit_after",
            "recovery_credit_anchor_price",
        ):
            if hasattr(sizing, key):
                diagnostics[key] = float(getattr(sizing, key) or 0.0)
        action.diagnostics = diagnostics
        return action

    @staticmethod
    def build_action_reason(
        *,
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
            f"v4_2_{side}_{setup}"
            f"_r{risk_score}_tr{trend_risk}_dd{drawdown_risk}"
            f"_raw{raw_state}_conf{confirmed_state}_t{target:.0%}"
        )
        if guard:
            reason = f"{reason}_{guard}"
        return reason
