from __future__ import annotations

from .strategy_types import StrategyContext, StrategyLifecycle, StrategyRegime, StrategyRiskAssessment, StrategyRiskGate


class StrategyRiskMixin:
    def _build_lifecycle(self, context: StrategyContext, regime: StrategyRegime, episode: dict) -> StrategyLifecycle:
        state = str(episode.get("state", "NORMAL"))
        phase = {
            "DEFENSE_LOCK": "RISK_DEFENSE",
            "RECOVERY_TEST": "RECOVERY",
            "FAILED_RECOVERY_LOCK": "FAILED_RECOVERY",
            "DISTRIBUTION_LOCK": "DISTRIBUTION",
            "STRUCTURAL_BEAR_LOCK": "STRUCTURAL_BEAR",
        }.get(state, "NORMAL")
        if phase == "NORMAL" and regime.regime == "BEAR":
            phase = "STRUCTURAL_BEAR" if regime.structural_bear else "BEAR"
        return StrategyLifecycle(
            phase=phase,
            episode_state=state,
            reason=self._lifecycle_reason(phase, regime),
        )

    @staticmethod
    def _lifecycle_reason(phase: str, regime: StrategyRegime) -> str:
        if phase == "NORMAL":
            return "normal"
        if phase == "BEAR":
            return "bear_regime"
        if phase == "STRUCTURAL_BEAR" and regime.structural_bear:
            return "structural_bear"
        return phase.lower()

    def _build_risk_gate(self, context: StrategyContext, regime: StrategyRegime, lifecycle: StrategyLifecycle) -> StrategyRiskGate:
        force_exit = bool(regime.structural_bear or lifecycle.phase == "STRUCTURAL_BEAR")
        force_defense = bool(
            not force_exit
            and (
                context.trend_risk >= 2
                or context.drawdown_risk > 0
                or (regime.btc_regime == "BEAR" and regime.regime in {"RANGE", "TRANSITION"})
            )
            and context.current_pct > 0.08
        )
        reason = "force_exit" if force_exit else "force_defense" if force_defense else "normal"
        return StrategyRiskGate(
            allow_main_buy=not force_exit and regime.regime != "BEAR",
            allow_recovery_buy=not force_exit or lifecycle.phase in {"STRUCTURAL_BEAR", "FAILED_RECOVERY", "RECOVERY"},
            allow_base_buy=not force_exit,
            allow_main_sell=True,
            allow_base_sell=True,
            allow_distribution_sell=True,
            force_defense=force_defense,
            force_exit=force_exit,
            reason=reason,
        )

    def _build_risk_assessment_shadow(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        lifecycle: StrategyLifecycle,
        risk_gate: StrategyRiskGate,
        episode: dict,
    ) -> StrategyRiskAssessment:
        state = str(episode.get("state", "NORMAL"))
        episode_override = ""
        if lifecycle.phase in {"RECOVERY", "FAILED_RECOVERY"}:
            episode_override = lifecycle.phase.lower()
        elif state in {"DEFENSE_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            episode_override = state.lower()

        if risk_gate.force_exit:
            risk_mode = "STRUCTURAL_EXIT"
            expected_intent = "EXIT"
        elif risk_gate.force_defense:
            risk_mode = "HARD_DEFENSE" if context.risk_score >= 4 or context.trend_risk >= 3 else "SOFT_DEFENSE"
            expected_intent = "DEFEND"
        elif lifecycle.phase in {"RECOVERY", "FAILED_RECOVERY"}:
            risk_mode = "RECOVERY"
            expected_intent = "ALLOW_RECOVERY"
        elif lifecycle.phase == "DISTRIBUTION":
            risk_mode = "DISTRIBUTION"
            expected_intent = "DISTRIBUTE"
        elif regime.regime == "BEAR":
            risk_mode = "BEAR_WATCH"
            expected_intent = "HOLD"
        elif regime.regime == "TRANSITION":
            risk_mode = "TRANSITION_WATCH"
            expected_intent = "HOLD"
        else:
            risk_mode = "NORMAL"
            expected_intent = "HOLD"

        reason_parts = [risk_gate.reason, lifecycle.reason, regime.reason]
        if state != "NORMAL":
            reason_parts.append(state.lower())
        reason = "|".join(part for part in reason_parts if part)
        severity = max(0, min(5, int(context.risk_score)))
        if risk_gate.force_exit:
            severity = max(severity, 5)
        elif risk_gate.force_defense:
            severity = max(severity, 3)

        return StrategyRiskAssessment(
            risk_mode=risk_mode,
            severity=severity,
            expected_intent=expected_intent,
            allow_main_buy=bool(risk_gate.allow_main_buy),
            allow_recovery_buy=bool(risk_gate.allow_recovery_buy),
            allow_base_buy=bool(risk_gate.allow_base_buy),
            allow_main_sell=bool(risk_gate.allow_main_sell),
            allow_base_sell=bool(risk_gate.allow_base_sell),
            force_defense=bool(risk_gate.force_defense),
            force_exit=bool(risk_gate.force_exit),
            episode_override=episode_override,
            reason=reason,
        )
