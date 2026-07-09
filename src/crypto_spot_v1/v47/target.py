"""Target-vector construction for V4.7 raw decisions."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..v42_types import V42Context, V42Regime, V42Signals, V42TargetPlan


class V47TargetOwner(Protocol):
    TARGET_CAP: float
    RECOVERY_MIN_STEP: float
    TREND_CONTINUATION_BOOST: float
    CORE_FLOOR: dict[str, float]
    TARGET_TABLE: dict[str, dict[int, float]]
    BTC_ADJUST: dict[str, float]
    _diag: dict[str, int]

    def _bear_base_target(self, context: V42Context, regime: V42Regime) -> float:
        ...

    def _bear_base_floor(self, context: V42Context) -> float:
        ...

    def _bear_base_exit_target(self, context: V42Context, regime: V42Regime, signals: V42Signals) -> float:
        ...

    def _bear_base_accumulate_needed(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
    ) -> bool:
        ...

    def _episode_recovery_target(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        base_target: float,
    ) -> float:
        ...


class V47TargetEngine:
    """Builds mature, phase, and executable raw target vectors."""

    def compose_target_from_plan(self, owner: V47TargetOwner, context: V42Context, plan: V42TargetPlan) -> float:
        if plan.base_accumulate_needed:
            add_base = max(0.0, plan.desired_base - plan.bear_base_floor)
            return max(0.0, min(owner.TARGET_CAP, context.current_pct + add_base))
        if plan.base_exit_distribute:
            reduce_base = max(0.0, plan.bear_base_floor - plan.desired_base)
            return max(0.0, min(owner.TARGET_CAP, context.current_pct - reduce_base))
        return self.combine_base_tactical_target(owner, plan.desired_base, plan.tactical_target)

    def build_target_plan(
        self,
        owner: V47TargetOwner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        intent: str,
    ) -> V42TargetPlan:
        symbol = context.symbol
        bear_base_target = owner._bear_base_target(context, regime)
        bear_base_floor = owner._bear_base_floor(context)
        bear_base_exit_target = owner._bear_base_exit_target(context, regime, signals)
        base_exit_distribute = intent == "DISTRIBUTE" and bear_base_exit_target < bear_base_floor - owner.RECOVERY_MIN_STEP
        base_accumulate_needed = intent == "ACCUMULATE" and owner._bear_base_accumulate_needed(context, regime, episode, signals)
        tactical_capacity = max(owner.RECOVERY_MIN_STEP, 1.0 - bear_base_floor)
        tactical_current = max(0.0, context.current_pct - bear_base_floor)
        tactical_current_ratio = max(0.0, min(1.0, tactical_current / tactical_capacity))
        desired_base = max(bear_base_floor, bear_base_target) if base_accumulate_needed else bear_base_floor
        if base_exit_distribute:
            desired_base = max(0.0, bear_base_exit_target)
        tactical_target = self.main_tactical_target(owner, context, regime)

        if intent == "EXIT":
            tactical_target = 0.0 if context.trend_risk >= 3 else min(tactical_target, owner.CORE_FLOOR.get(symbol, 0.2) * 0.35)
        elif intent == "DEFEND":
            tactical_target = min(tactical_target, max(0.0, tactical_current_ratio - 0.12))
            if tactical_target <= owner.RECOVERY_MIN_STEP and bear_base_floor > 0.0:
                owner._diag["v4_2_bear_base_floor_sell_protected_count"] += 1
        elif intent == "DISTRIBUTE":
            tactical_target = min(tactical_target, 0.78)
            if base_exit_distribute:
                tactical_target = tactical_current_ratio
        elif intent == "ACCUMULATE":
            if base_accumulate_needed:
                tactical_target = tactical_current_ratio
            else:
                if signals.trend_continuation:
                    tactical_target = min(owner.TARGET_CAP, tactical_target + owner.TREND_CONTINUATION_BOOST)
                if signals.value_recovery:
                    tactical_target = max(tactical_target, 0.42)
                if signals.starter:
                    tactical_target = max(tactical_target, 0.28)
                if str(episode.get("state", "NORMAL")) in {
                    "DEFENSE_LOCK",
                    "FAILED_RECOVERY_LOCK",
                    "RECOVERY_TEST",
                    "STRUCTURAL_BEAR_LOCK",
                }:
                    tactical_target = owner._episode_recovery_target(context, regime, episode, signals, tactical_target)
        else:
            tactical_target = tactical_current_ratio

        if intent not in {"EXIT", "DEFEND"} and not base_accumulate_needed and not base_exit_distribute:
            tactical_target = self.apply_core_floor(owner, symbol, tactical_target)
        mature_tactical_target = self.apply_core_floor(owner, symbol, self.main_tactical_target(owner, context, regime))
        mature_target = self.combine_base_tactical_target(owner, bear_base_floor, mature_tactical_target)
        phase_target = self.combine_base_tactical_target(owner, desired_base, tactical_target)
        return V42TargetPlan(
            bear_base_target=bear_base_target,
            bear_base_floor=bear_base_floor,
            bear_base_exit_target=bear_base_exit_target,
            base_accumulate_needed=base_accumulate_needed,
            base_exit_distribute=base_exit_distribute,
            tactical_capacity=tactical_capacity,
            tactical_current=tactical_current,
            tactical_current_ratio=tactical_current_ratio,
            desired_base=desired_base,
            tactical_target=tactical_target,
            mature_target=mature_target,
            phase_target=phase_target,
        )

    def main_tactical_target(self, owner: V47TargetOwner, context: V42Context, regime: V42Regime) -> float:
        target = owner.TARGET_TABLE[regime.regime].get(context.risk_score, 0.0)
        target = max(0.0, min(1.0, target * self.vol_multiplier(context, regime)))
        if context.symbol != "BTC/USDT":
            target = max(0.0, min(1.0, target + owner.BTC_ADJUST.get(regime.btc_regime, 0.0)))
        return target

    @staticmethod
    def combine_base_tactical_target(owner: V47TargetOwner, base_target: float, tactical_target: float) -> float:
        base = max(0.0, min(owner.TARGET_CAP, base_target))
        tactical = max(0.0, min(1.0, tactical_target))
        total = base + tactical * max(0.0, 1.0 - base)
        return max(0.0, min(owner.TARGET_CAP, total))

    @staticmethod
    def vol_multiplier(context: V42Context, regime: V42Regime) -> float:
        atr_rank = regime.atr_rank
        if pd.isna(atr_rank) or float(atr_rank) <= 0.80:
            return 1.0
        excess = float(atr_rank) - 0.80
        if regime.regime == "BULL":
            return max(0.78, 1.0 - excess * 0.6)
        if regime.regime == "BEAR":
            return max(0.60, 1.0 - excess * 1.8)
        return max(0.68, 1.0 - excess * 1.25)

    @staticmethod
    def apply_core_floor(owner: V47TargetOwner, symbol: str, tactical_target: float) -> float:
        core = owner.CORE_FLOOR.get(symbol, 0.22)
        return core + tactical_target * (1.0 - core)
