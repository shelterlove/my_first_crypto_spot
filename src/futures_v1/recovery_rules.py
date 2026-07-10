from __future__ import annotations

import pandas as pd

from .strategy_rebalance import Action
from .strategy_types import StrategyContext, StrategyRecoveryPlan, StrategyRegime, StrategySignals


class StrategyRecoveryMixin:
    def _episode_recovery_add_cap(self, episode: dict) -> float:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        lowest = float(episode.get("lowest_price", sell_price) or sell_price)
        if sell_price <= 0.0 or lowest <= 0.0:
            return 0.03
        drawdown = 1.0 - lowest / sell_price
        if drawdown >= 0.20:
            return 0.12
        if drawdown >= 0.10:
            return 0.08
        return 0.04

    def _episode_recovery_plan(self, context: StrategyContext, regime: StrategyRegime, episode: dict, signals: StrategySignals) -> StrategyRecoveryPlan:
        state = str(episode.get("state", "NORMAL"))
        setup = str(episode.get("setup", ""))
        if state not in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            return StrategyRecoveryPlan(blocked_reason="not_recovery_episode")

        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return StrategyRecoveryPlan(blocked_reason="missing_sell_anchor")
        age = self._call_count - int(episode.get("start_call", self._call_count))
        if setup == "defense-sell" and age < self.DEFENSE_MIN_RECOVERY_CALLS:
            return StrategyRecoveryPlan(blocked_reason="defense_cooldown")

        current_drop = max(0.0, 1.0 - context.price / sell_price)
        lowest = float(episode.get("lowest_price", context.price) or context.price)
        low_drop = max(current_drop, 1.0 - lowest / sell_price if lowest > 0.0 else 0.0)
        budget = self._episode_recovery_budget(episode)
        recovered = self._episode_recovered_pct(episode)
        remaining = max(0.0, budget - recovered)
        if remaining < self.RECOVERY_MIN_STEP:
            return StrategyRecoveryPlan(blocked_reason="recovery_budget_filled")

        permission = self._recovery_permission(context, regime, signals, low_drop)
        staged_plan = self._staged_recovery_plan(
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
            lowest=lowest,
        )
        strong = bool(signals.strong_recovery_signal or signals.recovery_quality_ok)
        if setup == "structural-exit-sell" or state == "STRUCTURAL_BEAR_LOCK":
            if bool(staged_plan.get("allowed", False)):
                fraction = float(staged_plan.get("fraction", 0.0) or 0.0)
                strong = False
            elif not self._structural_recovery_ready(context, regime, signals):
                return StrategyRecoveryPlan(blocked_reason="structural_bear_active")
            else:
                fraction = self._recovery_fraction_from_drawdown(low_drop, shallow=0.18, normal=0.34, deep=0.52, crash=0.70)
        else:
            if bool(staged_plan.get("allowed", False)):
                fraction = float(staged_plan.get("fraction", 0.0) or 0.0)
                strong = False
            elif low_drop < 0.04:
                return StrategyRecoveryPlan(blocked_reason="insufficient_sell_to_low_drawdown")
            elif current_drop < 0.02 and not strong:
                return StrategyRecoveryPlan(blocked_reason="current_discount_missing")
            elif not bool(permission.get("allowed", False)):
                return StrategyRecoveryPlan(blocked_reason=str(permission.get("blocked_reason", "recovery_risk_not_stabilized")))
            else:
                fraction = self._recovery_fraction_from_drawdown(low_drop, shallow=0.36, normal=0.58, deep=0.80, crash=1.00)

        if strong:
            fraction = min(1.0, fraction + 0.12)
        elif signals.recovery_signal:
            fraction = min(1.0, fraction + 0.06)
        desired_recovered = min(budget, budget * fraction)
        add_pct = max(0.0, desired_recovered - recovered)
        if add_pct < self.RECOVERY_MIN_STEP:
            return StrategyRecoveryPlan(blocked_reason="recovery_step_too_small")
        target = max(context.current_pct + add_pct, float(episode.get("post_sell_pct", context.current_pct)) + desired_recovered)
        return StrategyRecoveryPlan(
            allowed=True,
            target=min(self.TARGET_CAP, target),
            max_buy=self._recovery_step_max_buy(
                add_pct,
                low_drop,
                strong,
                deep_base=bool(staged_plan.get("allowed", False)),
            ),
            drawdown=low_drop,
            remaining_budget=remaining,
            guard=str(staged_plan.get("guard", "")),
        )

    def _episode_reentry_plan(self, context: StrategyContext, regime: StrategyRegime, episode: dict, signals: StrategySignals) -> StrategyRecoveryPlan:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return StrategyRecoveryPlan(blocked_reason="missing_sell_anchor")
        age = self._call_count - int(episode.get("start_call", self._call_count))
        current_drop = max(0.0, 1.0 - context.price / sell_price)
        breakout = bool(signals.trend_continuation and signals.recovery_quality_ok and context.price >= sell_price * 1.03)
        value_reset = bool(current_drop >= 0.10 and signals.recovery_signal)
        if age < self.DISTRIBUTION_MIN_REENTRY_CALLS and not breakout:
            return StrategyRecoveryPlan(blocked_reason="distribution_reentry_cooldown")
        if not breakout and not value_reset:
            return StrategyRecoveryPlan(blocked_reason="distribution_reentry_no_edge")
        return StrategyRecoveryPlan(allowed=True)

    def _episode_recovery_target(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        signals: StrategySignals,
        base_target: float,
    ) -> float:
        plan = self._episode_recovery_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return min(max(base_target, context.current_pct), self._episode_recovery_cap(episode))
        path_target = float(plan.get("target", context.current_pct))
        return min(max(base_target, path_target), self._episode_recovery_cap(episode))

    def _episode_recovery_max_buy(
        self,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        setup: str,
        signals: StrategySignals | None = None,
    ) -> float:
        if signals is None:
            signals = self._build_signals(context, regime, episode)
        plan = self._episode_recovery_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return 0.0
        max_buy = float(plan.get("max_buy", 0.0) or 0.0)
        if setup == "recovery-probe-buy":
            return max(0.04, max_buy)
        return max_buy

    def _episode_recovery_budget(self, episode: dict) -> float:
        budget = float(episode.get("recovery_budget_pct", 0.0) or 0.0)
        if budget > 0.0:
            return budget
        sold = float(episode.get("sold_pct", 0.0) or 0.0)
        if sold > 0.0:
            return sold
        sell_position = float(episode.get("sell_position_pct", 0.0) or 0.0)
        sell_target = float(episode.get("sell_target_pct", 0.0) or 0.0)
        return max(0.0, sell_position - sell_target)

    @staticmethod
    def _episode_recovered_pct(episode: dict) -> float:
        return max(0.0, float(episode.get("recovery_bought_pct", 0.0) or 0.0))

    def _record_episode_recovery_buy(self, context: StrategyContext, episode: dict, action: Action) -> None:
        buy_pct = action.quantity * action.price / context.total_value if context.total_value > 0.0 else 0.0
        self._diag["core_recovery_path_buy_count"] += 1
        episode["recovery_buy_count"] = int(episode.get("recovery_buy_count", 0)) + 1
        episode["recovery_buy_notional"] = float(episode.get("recovery_buy_notional", 0.0)) + action.quantity * action.price
        recovered = self._episode_recovered_pct(episode) + max(0.0, buy_pct)
        cumulative_recovered = float(episode.get("cumulative_recovered_pct", 0.0) or 0.0) + max(0.0, buy_pct)
        episode["recovery_bought_pct"] = recovered
        episode["cumulative_recovered_pct"] = cumulative_recovered
        cumulative_budget = float(episode.get("cumulative_sold_pct", self._episode_recovery_budget(episode)) or 0.0)
        episode["unrecovered_budget_pct"] = max(0.0, cumulative_budget - cumulative_recovered)
        episode["last_recovery_buy_call"] = self._call_count
        episode["last_recovery_buy_price"] = context.price
        position_after = context.current_pct + max(0.0, buy_pct)
        episode["max_recovery_position_pct"] = max(float(episode.get("max_recovery_position_pct", position_after) or position_after), position_after)
        for threshold, key in ((0.30, "recovered_to_30_call"), (0.50, "recovered_to_50_call"), (0.80, "recovered_to_80_call")):
            if episode.get(key) is None and position_after >= threshold:
                episode[key] = self._call_count
        legs = list(episode.get("recovery_legs", []))
        legs.append({
            "call": self._call_count,
            "setup": self._setup_from_action(action),
            "price": context.price,
            "buy_pct": max(0.0, buy_pct),
            "position_before_pct": context.current_pct,
            "position_after_pct": position_after,
        })
        episode["recovery_legs"] = legs
        sell_price = float(episode.get("sell_price", action.price) or action.price)
        episode["episode_contribution_notional"] = float(episode.get("episode_contribution_notional", 0.0)) + (sell_price - action.price) * action.quantity
        if str(episode.get("state")) in {"DEFENSE_LOCK", "FAILED_RECOVERY_LOCK"}:
            episode["state"] = "RECOVERY_TEST"
            episode["recovery_start_call"] = self._call_count

    def _setup_from_action(self, action: Action) -> str:
        marker = "_buy_"
        reason = str(action.reason)
        if marker in reason:
            tail = reason.split(marker, 1)[1]
            return tail.split("_r", 1)[0]
        marker = "_sell_"
        if marker in reason:
            tail = reason.split(marker, 1)[1]
            return tail.split("_r", 1)[0]
        return ""

    def _is_bear_base_buy(self, decision: StrategyDecisionPlan, sizing: StrategySizing) -> bool:
        primary_sleeve = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        return bool(primary_sleeve == "bear-base" or "core_bear_base" in str(sizing.guard))

    def _should_record_episode_recovery_buy(self, bear_base_buy: bool) -> bool:
        return bool(not bear_base_buy or self.COUNT_BEAR_BASE_BUYS_AS_EPISODE_RECOVERY)

    @staticmethod
    def _recovery_fraction_from_drawdown(low_drop: float, *, shallow: float, normal: float, deep: float, crash: float) -> float:
        if low_drop >= 0.25:
            return crash
        if low_drop >= 0.15:
            return deep
        if low_drop >= 0.08:
            return normal
        if low_drop >= 0.04:
            return shallow
        return 0.0

    @staticmethod
    def _recovery_step_max_buy(add_pct: float, low_drop: float, strong: bool, *, deep_base: bool = False) -> float:
        if deep_base:
            cap = 0.05
            if low_drop >= 0.45:
                cap = 0.08
            elif low_drop >= 0.35:
                cap = 0.06
            return min(max(add_pct, 0.0), cap)
        cap = 0.08
        if low_drop >= 0.15:
            cap = 0.14
        elif low_drop >= 0.08:
            cap = 0.11
        if strong:
            cap += 0.03
        return min(max(add_pct, 0.0), cap)

    @staticmethod
    def _recovery_drawdown_guard(low_drop: float) -> str:
        if low_drop >= 0.25:
            return "core_recovery_crash_discount"
        if low_drop >= 0.15:
            return "core_recovery_deep_discount"
        if low_drop >= 0.08:
            return "core_recovery_normal_discount"
        return "core_recovery_shallow_discount"


    def _recovery_permission(self, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals, low_drop: float) -> StrategyRecoveryPlan:
        if regime.structural_bear:
            return StrategyRecoveryPlan(blocked_reason="structural_bear_active")
        if regime.btc_regime == "BEAR" and not signals.strong_recovery_signal:
            return StrategyRecoveryPlan(blocked_reason="btc_bear_recovery_unconfirmed")
        if context.trend_risk >= 3:
            return StrategyRecoveryPlan(blocked_reason="trend_risk_severe")
        if context.raw_state == "BEAR" and context.confirmed_state == "BEAR" and context.trend_risk >= 2:
            return StrategyRecoveryPlan(blocked_reason="confirmed_bear_recovery_unconfirmed")
        if signals.recovery_signal or signals.strong_recovery_signal:
            return StrategyRecoveryPlan(allowed=True, quality="confirmed")
        if low_drop >= 0.08 and context.trend_risk <= 2 and regime.btc_regime != "BEAR":
            return StrategyRecoveryPlan(allowed=True, quality="discount")
        return StrategyRecoveryPlan(blocked_reason="recovery_signal_missing")

    def _staged_recovery_ready(self, context: StrategyContext, regime: StrategyRegime, episode: dict) -> bool:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return False
        age = self._call_count - int(episode.get("start_call", self._call_count))
        lowest = float(episode.get("lowest_price", context.price) or context.price)
        current_drop = max(0.0, 1.0 - context.price / sell_price)
        low_drop = max(current_drop, 1.0 - lowest / sell_price if lowest > 0.0 else 0.0)
        budget = self._episode_recovery_budget(episode)
        recovered = self._episode_recovered_pct(episode)
        remaining = max(0.0, budget - recovered)
        return bool(self._staged_recovery_plan(
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
            lowest=lowest,
        ).get("allowed", False))

    def _staged_recovery_plan(
        self,
        *,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
        lowest: float,
    ) -> StrategyRecoveryPlan:
        deep_base = self._deep_base_recovery_plan(
            context=context,
            regime=regime,
            episode=episode,
            age=age,
            current_drop=current_drop,
            low_drop=low_drop,
            remaining=remaining,
        )
        if bool(deep_base.get("allowed", False)):
            return deep_base
        return self._post_crash_recoil_plan(
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
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
    ) -> StrategyRecoveryPlan:
        if remaining < self.RECOVERY_MIN_STEP:
            return StrategyRecoveryPlan()
        if age < self.DEEP_BASE_MIN_CALLS:
            return StrategyRecoveryPlan()
        if low_drop < self.DEEP_BASE_MIN_DRAWDOWN:
            return StrategyRecoveryPlan()
        if current_drop < self.DEEP_BASE_MIN_CURRENT_DISCOUNT:
            return StrategyRecoveryPlan()
        if context.trend_risk >= 4:
            return StrategyRecoveryPlan()
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(context.latest, "donchian_pos", 0.5)
        low_location = bool(
            (pd.isna(rolling_pos) or rolling_pos <= 0.42)
            and (pd.isna(donchian_pos) or donchian_pos <= 0.45)
        )
        if not low_location:
            return StrategyRecoveryPlan()
        fraction = 0.18
        if low_drop >= 0.45 and age >= self.DEEP_BASE_MIN_CALLS + 45:
            fraction = 0.30
        elif low_drop >= 0.35:
            fraction = 0.24
        return StrategyRecoveryPlan(
            allowed=True,
            fraction=fraction,
            guard="core_deep_base_recovery",
        )

    def _post_crash_recoil_plan(
        self,
        *,
        context: StrategyContext,
        regime: StrategyRegime,
        episode: dict,
        age: int,
        current_drop: float,
        low_drop: float,
        remaining: float,
        lowest: float,
    ) -> StrategyRecoveryPlan:
        if remaining < self.RECOVERY_MIN_STEP or age < 12:
            return StrategyRecoveryPlan()
        if low_drop < 0.12:
            return StrategyRecoveryPlan()
        if context.raw_state == "BEAR" and context.confirmed_state == "BEAR" and context.trend_risk == 0:
            return StrategyRecoveryPlan()
        if regime.btc_regime == "BEAR" and low_drop < 0.22:
            return StrategyRecoveryPlan()
        roc_10 = self._value(context.latest, "roc_10", 0.0)
        roc_20 = self._value(context.latest, "roc_20", 0.0)
        atr_rank = regime.atr_rank
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(context.latest, "donchian_pos", 0.5)
        rebound_from_low = context.price / lowest - 1.0 if lowest > 0.0 else 0.0
        if not pd.isna(roc_20) and roc_20 <= -0.18:
            return StrategyRecoveryPlan()
        if not pd.isna(roc_10) and roc_10 <= -0.12:
            return StrategyRecoveryPlan()
        if context.trend_risk >= 3 and current_drop < 0.18:
            return StrategyRecoveryPlan()
        if not pd.isna(rolling_pos) and rolling_pos > 0.55:
            return StrategyRecoveryPlan()
        if not pd.isna(atr_rank) and atr_rank >= 0.96 and rebound_from_low < 0.10:
            return StrategyRecoveryPlan()
        low_or_reclaim = bool(
            current_drop >= 0.06
            or (low_drop >= 0.22 and rebound_from_low >= 0.10)
            or (not pd.isna(donchian_pos) and 0.18 <= donchian_pos <= 0.55 and low_drop >= 0.18)
        )
        if not low_or_reclaim:
            return StrategyRecoveryPlan()
        fraction = 0.12
        if low_drop >= 0.35 and age >= 30:
            fraction = 0.22
        elif low_drop >= 0.22:
            fraction = 0.18
        return StrategyRecoveryPlan(
            allowed=True,
            fraction=fraction,
            guard="core_post_crash_recoil",
        )

    def _structural_recovery_ready(self, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals) -> bool:
        return bool(
            not regime.structural_bear
            and regime.regime in {"RANGE", "TRANSITION", "BULL"}
            and regime.btc_regime != "BEAR"
            and context.trend_risk <= 2
            and signals.strong_recovery_signal
        )

    def _episode_recovery_cap(self, episode: dict) -> float:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        lowest = float(episode.get("lowest_price", sell_price) or sell_price)
        if sell_price <= 0.0 or lowest <= 0.0:
            return 0.54
        drawdown = 1.0 - lowest / sell_price
        if drawdown >= 0.20:
            return 0.72
        if drawdown >= 0.10:
            return 0.64
        return 0.54

