from __future__ import annotations

from .strategy_types import StrategyContext, StrategyRegime, StrategySignals, StrategySleevePlan, StrategyTargetPlan


class StrategySleeveMixin:
    def _build_sleeve_plans(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
        intent: str,
        target_plan: StrategyTargetPlan,
        target: float,
    ) -> tuple[StrategySleevePlan, ...]:
        plans: list[StrategySleevePlan] = []
        if target_plan.base_exit_distribute:
            base_exit_setup = "bear-base-exit" if bool(getattr(self, "USE_INDEPENDENT_BEAR_BASE_EXIT_SETUP", False)) else "distribution-sell"
            plans.append(StrategySleevePlan(
                sleeve="bear-base-exit",
                side="sell",
                setup=base_exit_setup,
                target=target,
                guard="core_bear_base_exit",
                priority=100,
                allowed=True,
            ))
        if target_plan.base_accumulate_needed:
            plans.append(StrategySleevePlan(
                sleeve="bear-base",
                side="buy",
                setup="recovery-probe-buy",
                target=target,
                guard="core_bear_base",
                priority=80,
                allowed=True,
            ))
        if intent in {"EXIT", "DEFEND", "DISTRIBUTE"}:
            plans.append(StrategySleevePlan(
                sleeve="main",
                side="sell",
                setup=self._sell_setup(intent),
                target=target,
                priority={"EXIT": 90, "DEFEND": 70, "DISTRIBUTE": 60}.get(intent, 50),
                allowed=True,
            ))
        if intent == "ACCUMULATE":
            setup = self._buy_setup_from_plan(context, regime, episode, signals, target_plan)
            sleeve = "recovery" if setup == "recovery-probe-buy" and str(episode.get("state", "NORMAL")) != "NORMAL" else "main"
            guard = ""
            if sleeve == "recovery":
                recovery_plan = self._episode_recovery_plan(context, regime, episode, signals)
                if bool(recovery_plan.get("allowed", False)):
                    guard = str(recovery_plan.get("guard", ""))
            plans.append(StrategySleevePlan(
                sleeve=sleeve,
                side="buy",
                setup=setup,
                target=target,
                guard=guard,
                priority=50,
                allowed=True,
            ))
        if intent == "HOLD":
            overlay_plan = self._hold_recovery_overlay_plan(context, regime, episode, signals)
            if bool(overlay_plan.get("allowed", False)):
                plans.append(StrategySleevePlan(
                    sleeve="recovery-overlay",
                    side="buy",
                    setup="recovery-probe-buy",
                    target=float(overlay_plan.get("target", target) or target),
                    guard="core_limited_recovery_overlay",
                    priority=40,
                    allowed=True,
                ))
        if not plans:
            plans.append(StrategySleevePlan(sleeve="main", target=target, allowed=False, blocked_reason="hold"))
        return tuple(plans)

    @staticmethod
    def _select_primary_sleeve(sleeve_plans: tuple[StrategySleevePlan, ...]) -> StrategySleevePlan:
        allowed = [plan for plan in sleeve_plans if plan.allowed]
        if not allowed:
            return sleeve_plans[0] if sleeve_plans else StrategySleevePlan(blocked_reason="no_sleeve_plan")
        return sorted(allowed, key=lambda plan: plan.priority, reverse=True)[0]

    def _sell_setup(self, intent: str) -> str:
        if intent == "EXIT":
            return "structural-exit-sell"
        if intent == "DISTRIBUTE":
            return "distribution-sell"
        return "defense-sell"

    def _buy_setup(self, context: StrategyContext, regime: StrategyRegime, episode: dict, signals: StrategySignals) -> str:
        return self._buy_setup_from_plan(context, regime, episode, signals, None)

    def _buy_setup_from_plan(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
        target_plan: StrategyTargetPlan | None,
    ) -> str:
        state = str(episode.get("state", "NORMAL"))
        base_accumulate_needed = bool(
            target_plan.base_accumulate_needed if target_plan is not None else self._bear_base_accumulate_needed(context, regime, episode, signals)
        )
        if base_accumulate_needed:
            return "recovery-probe-buy"
        if state in {"DEFENSE_LOCK", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            return "recovery-probe-buy"
        if state == "RECOVERY_TEST":
            if signals.trend_continuation and signals.recovery_quality_ok:
                return "trend-cont"
            if signals.value_recovery or signals.recovery_signal:
                return "value-recovery"
            return "recovery-probe-buy"
        if signals.starter:
            return "starter-buy"
        if signals.value_recovery:
            return "value-recovery"
        if signals.trend_continuation:
            return "trend-cont"
        return "value-recovery"

    def _bear_base_accumulate_needed(self, context: StrategyContext, regime: StrategyRegime, episode: dict, signals: StrategySignals) -> bool:
        return bool(
            self._bear_base_target(context, regime) > self._bear_base_floor(context) + self.RECOVERY_MIN_STEP
            and not signals.accumulation
            and not signals.recovery_signal
            and not self._structural_recovery_ready(context, regime, signals)
        )
