from __future__ import annotations

from .strategy_types import StrategyContext, StrategyDecisionPlan, StrategyRegime, StrategySignals


class StrategyDecisionMixin:
    def _choose_intent(self, context: StrategyContext, regime: StrategyRegime, episode: dict, signals: StrategySignals) -> str:
        state = str(episode.get("state", "NORMAL"))
        current_pct = context.current_pct
        bear_base_exit_target = self._bear_base_exit_target(context, regime, signals)
        base_accumulate_needed = self._bear_base_accumulate_needed(context, regime, episode, signals)
        if bear_base_exit_target < self._bear_base_floor(context) - self.RECOVERY_MIN_STEP:
            intent = "DISTRIBUTE"
        elif base_accumulate_needed and not bool(getattr(self, "ENABLE_HOLD_ONLY_BASE_INTENT", False)):
            intent = "ACCUMULATE"
        elif state == "RECOVERY_TEST":
            sell_blocked = self._recovery_test_sell_block_reason(context, regime, episode, signals) != ""
            if not sell_blocked and (regime.structural_bear or context.trend_risk >= 3):
                intent = "EXIT"
            elif not sell_blocked and signals.distribution_exhaustion:
                intent = "DISTRIBUTE"
            elif not sell_blocked and (
                context.trend_risk >= 2
                or context.drawdown_risk > 0
                or (regime.btc_regime == "BEAR" and regime.regime in {"RANGE", "TRANSITION"})
            ) and current_pct > 0.08:
                intent = "DEFEND"
            elif signals.recovery_signal or self._accumulation_signal(context, regime, episode, signals):
                intent = "ACCUMULATE"
            else:
                intent = "HOLD"
        elif state == "STRUCTURAL_BEAR_LOCK" and self._structural_recovery_ready(context, regime, signals):
            intent = "ACCUMULATE"
        elif state in {"DEFENSE_LOCK", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"} and self._staged_recovery_ready(context, regime, episode):
            intent = "ACCUMULATE"
        elif regime.structural_bear or state == "STRUCTURAL_BEAR_LOCK":
            intent = "EXIT"
        elif state in {"DEFENSE_LOCK", "FAILED_RECOVERY_LOCK"} and signals.recovery_signal:
            intent = "ACCUMULATE"
        elif signals.distribution_exhaustion:
            intent = "DISTRIBUTE"
        elif (
            context.trend_risk >= 2
            or context.drawdown_risk > 0
            or (regime.btc_regime == "BEAR" and regime.regime in {"RANGE", "TRANSITION"})
        ) and current_pct > 0.08:
            intent = "DEFEND"
        elif self._accumulation_signal(context, regime, episode, signals):
            intent = "ACCUMULATE"
        else:
            intent = "HOLD"
        if bool(getattr(self, "ENABLE_HOLD_ONLY_BASE_INTENT", False)) and base_accumulate_needed:
            if intent == "HOLD" and self._base_intent_hard_gate_allows(context, regime):
                intent = "ACCUMULATE"
                self._diag["core_base_intent_accumulate_count"] += 1
            elif intent == "DEFEND":
                self._diag["core_base_deferred_by_main_defense_count"] += 1
                self._record_base_deferred_candidate(
                    context,
                    regime,
                    main_intent=intent,
                    base_target=self._bear_base_target(context, regime),
                    blocked_reason="main_defense",
                )
            elif intent != "ACCUMULATE":
                self._diag["core_base_intent_deferred_count"] += 1
                self._record_base_deferred_candidate(
                    context,
                    regime,
                    main_intent=intent,
                    base_target=self._bear_base_target(context, regime),
                    blocked_reason="hard_gate" if intent == "HOLD" else f"main_{intent.lower()}",
                )
        self._diag[f"core_intent_{intent.lower()}_count"] += 1
        return intent

    def _build_decision_plan(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
    ) -> StrategyDecisionPlan:
        intent = self._choose_intent(context, regime, episode, signals)
        target_plan = self._build_target_plan(context, regime, episode, signals, intent)
        target = self._compose_target_from_plan(context, target_plan)
        target_plan.execution_target_today = target
        sleeve_plans = self._build_sleeve_plans(context, regime, episode, signals, intent, target_plan, target)
        plan = StrategyDecisionPlan(
            intent=intent,
            target=target,
            mature_target=target_plan.mature_target,
            phase_target=target_plan.phase_target,
            execution_target_today=target,
            target_plan=target_plan,
            primary_sleeve=self._select_primary_sleeve(sleeve_plans),
            sleeve_plans=sleeve_plans,
        )
        self._annotate_main_base_intent(plan, context)
        return plan

    def _base_intent_hard_gate_allows(self, context: StrategyContext, regime: StrategyRegime) -> bool:
        return bool(
            not regime.structural_bear
            and context.trend_risk < 3
            and context.risk_score < 4
            and regime.btc_regime != "BEAR"
        )

    def _record_base_deferred_candidate(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        main_intent: str,
        base_target: float,
        blocked_reason: str,
    ) -> None:
        rows = getattr(self, "_base_deferred_candidates", None)
        if rows is None:
            self._base_deferred_candidates = []
            rows = self._base_deferred_candidates
        rows.append({
            "timestamp": context.latest.get("timestamp"),
            "symbol": context.symbol,
            "main_intent": main_intent,
            "regime": regime.regime,
            "btc_regime": regime.btc_regime,
            "risk_score": int(context.risk_score),
            "base_floor": float(self._bear_base_floor(context)),
            "base_target": float(base_target),
            "blocked_reason": blocked_reason,
        })

    def _annotate_main_base_intent(self, plan: StrategyDecisionPlan, context: StrategyContext) -> None:
        main_intent = plan.intent
        base_intent = "HOLD"
        if plan.target_plan is not None:
            if plan.target_plan.base_accumulate_needed:
                base_intent = "BASE_ACCUMULATE"
                main_intent = "HOLD" if bool(getattr(self, "ENABLE_HOLD_ONLY_BASE_INTENT", False)) else plan.intent
            elif plan.target_plan.base_exit_distribute:
                base_intent = "BASE_EXIT"
                main_intent = "HOLD"
        setattr(plan, "main_intent", main_intent)
        setattr(plan, "base_intent", base_intent)
        setattr(plan, "main_delta", float(plan.target - context.current_pct))
        if plan.target_plan is None:
            setattr(plan, "base_delta", 0.0)
            return
        base_delta = float(plan.target_plan.desired_base - plan.target_plan.bear_base_floor)
        setattr(plan, "base_delta", base_delta)
