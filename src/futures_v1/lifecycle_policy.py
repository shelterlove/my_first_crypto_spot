from __future__ import annotations

from dataclasses import dataclass
import math

from .strategy_types import StrategyContext, StrategyDecisionPlan, StrategyRegime, StrategySignals, StrategySizing


@dataclass(frozen=True)
class LifecycleView:
    state: str
    reason: str
    low_location: bool
    sell_allowed_sources: str
    risk_ceiling: float
    target: float


class LifecyclePolicy:
    """Pure lifecycle classification and audit policy for V4.5."""

    def __init__(self, *, target_cap: float, target_table: dict[str, dict[int, float]], recovery_min_step: float):
        self.target_cap = float(target_cap)
        self.target_table = target_table
        self.recovery_min_step = float(recovery_min_step)

    def build_view(
        self,
        *,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
    ) -> LifecycleView:
        state, reason = self.classify_state(
            context=context,
            regime=regime,
            episode=episode,
            signals=signals,
            decision=decision,
            sizing=sizing,
        )
        risk_ceiling = self.risk_ceiling(context=context, regime=regime, lifecycle_state=state)
        return LifecycleView(
            state=state,
            reason=reason,
            low_location=self.low_location(context=context, regime=regime),
            sell_allowed_sources=self.sell_allowed_sources(context=context, sizing=sizing),
            risk_ceiling=risk_ceiling,
            target=min(float(decision.target), risk_ceiling),
        )

    def classify_state(
        self,
        *,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
    ) -> tuple[str, str]:
        setup = sizing.setup
        episode_state = str(episode.get("state", "NORMAL"))
        if setup in {"distribution-sell", "bear-base-exit", "protected-floor-exit"} or signals.distribution_exhaustion:
            return "DISTRIBUTION", "distribution_or_base_exit"
        if setup in {"defense-sell", "structural-exit-sell"} or decision.intent in {"DEFEND", "EXIT"}:
            return "DEFENSE", "risk_reduction"
        if setup in {"opportunity-floor-buy", "bear-base-buy", "low-base-buy"}:
            return "LOW_BASE", "low_location_base_entry"
        if setup == "base-led-recovery-buy":
            return "RECOVERY", "base_led_recovery_entry"
        if episode_state in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            return "RECOVERY", "risk_cycle_recovery"
        if setup == "trend-cont" or (regime.regime == "BULL" and context.trend_risk <= 1):
            return "TREND", "trend_confirmed"
        recovery_gap = float(decision.execution_target_today) - float(context.current_pct)
        if setup in {"recovery-probe-buy", "value-recovery"}:
            return "RECOVERY", "recovery_signal"
        if signals.recovery_signal and decision.intent == "ACCUMULATE" and recovery_gap >= self.recovery_min_step:
            return "RECOVERY", "recovery_gap_open"
        if self.low_location(context=context, regime=regime):
            return "LOW_BASE", "low_location_watch"
        if context.current_pct <= 1e-9:
            return "CASH", "no_exposure"
        return "HOLD", "no_active_transition"

    def low_location(self, *, context: StrategyContext, regime: StrategyRegime) -> bool:
        rolling_pos = self._series_float(context, "rolling_365d_pos", 0.5)
        donchian_pos = self._series_float(context, "donchian_pos", 0.5)
        price_vs_ema168 = self._finite_float(regime.price_vs_ema168)
        return bool(
            (rolling_pos is not None and rolling_pos <= 0.30)
            or (donchian_pos is not None and donchian_pos <= 0.35)
            or (price_vs_ema168 is not None and price_vs_ema168 <= -0.15)
        )

    @staticmethod
    def sell_allowed_sources(*, context: StrategyContext, sizing: StrategySizing) -> str:
        if sizing.side != "sell":
            return ""
        if sizing.setup in {"defense-sell", "structural-exit-sell", "distribution-sell"}:
            return "main"
        if sizing.setup == "protected-floor-exit":
            return "protected_floor"
        if sizing.setup == "bear-base-exit":
            if context.symbol == "BNB/USDT":
                return "strategic_base"
            return "strategic_base|protected_floor|base_led_recovery"
        return "main"

    def risk_ceiling(self, *, context: StrategyContext, regime: StrategyRegime, lifecycle_state: str) -> float:
        if lifecycle_state == "DISTRIBUTION":
            return min(self.target_cap, max(0.0, float(context.current_pct)))
        if regime.structural_bear and context.trend_risk >= 3:
            return 0.08
        if lifecycle_state == "LOW_BASE":
            return min(0.18, self.target_cap)
        if lifecycle_state == "RECOVERY":
            if context.symbol == "ETH/USDT":
                return 0.62
            if context.symbol == "BNB/USDT":
                return 0.38
            return 0.30
        if lifecycle_state == "TREND":
            return self.target_cap
        base = float(self.target_table.get(regime.regime, {}).get(int(context.risk_score), 0.0))
        return min(self.target_cap, max(0.0, base))

    @classmethod
    def _series_float(cls, context: StrategyContext, key: str, default: float | None = None) -> float | None:
        try:
            value = context.latest.get(key, default)
        except Exception:
            value = default
        return cls._finite_float(value)

    @staticmethod
    def _finite_float(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result) or math.isinf(result):
            return None
        return result
