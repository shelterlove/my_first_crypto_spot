"""Recovery path helpers for V4.7."""

from __future__ import annotations

import pandas as pd

from ..strategy_rebalance import Action
from ..v42_types import V42Context, V42DecisionPlan, V42RecoveryPlan, V42Regime, V42Signals, V42Sizing, V42SleevePlan


class V47RecoveryEngine:
    @staticmethod
    def episode_recovery_add_cap(episode: dict) -> float:
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

    def episode_recovery_plan(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> V42RecoveryPlan:
        return self.base_episode_recovery_plan(owner, context, regime, episode, signals)

    def base_episode_recovery_plan(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> V42RecoveryPlan:
        state = str(episode.get("state", "NORMAL"))
        setup = str(episode.get("setup", ""))
        if state not in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            return V42RecoveryPlan(blocked_reason="not_recovery_episode")

        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return V42RecoveryPlan(blocked_reason="missing_sell_anchor")
        age = owner._call_count - int(episode.get("start_call", owner._call_count))
        if setup == "defense-sell" and age < owner.DEFENSE_MIN_RECOVERY_CALLS:
            return V42RecoveryPlan(blocked_reason="defense_cooldown")

        current_drop = max(0.0, 1.0 - context.price / sell_price)
        lowest = float(episode.get("lowest_price", context.price) or context.price)
        low_drop = max(current_drop, 1.0 - lowest / sell_price if lowest > 0.0 else 0.0)
        budget = self.episode_recovery_budget(episode)
        recovered = self.episode_recovered_pct(episode)
        remaining = max(0.0, budget - recovered)
        if remaining < owner.RECOVERY_MIN_STEP:
            return V42RecoveryPlan(blocked_reason="recovery_budget_filled")

        permission = self.recovery_permission(owner, context, regime, signals, low_drop)
        staged_plan = self.staged_recovery_plan(
            owner,
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
            elif not self.structural_recovery_ready(context, regime, signals):
                return V42RecoveryPlan(blocked_reason="structural_bear_active")
            else:
                fraction = self.recovery_fraction_from_drawdown(low_drop, shallow=0.18, normal=0.34, deep=0.52, crash=0.70)
        else:
            if bool(staged_plan.get("allowed", False)):
                fraction = float(staged_plan.get("fraction", 0.0) or 0.0)
                strong = False
            elif low_drop < 0.04:
                return V42RecoveryPlan(blocked_reason="insufficient_sell_to_low_drawdown")
            elif current_drop < 0.02 and not strong:
                return V42RecoveryPlan(blocked_reason="current_discount_missing")
            elif not bool(permission.get("allowed", False)):
                return V42RecoveryPlan(blocked_reason=str(permission.get("blocked_reason", "recovery_risk_not_stabilized")))
            else:
                fraction = self.recovery_fraction_from_drawdown(low_drop, shallow=0.36, normal=0.58, deep=0.80, crash=1.00)

        if strong:
            fraction = min(1.0, fraction + 0.12)
        elif signals.recovery_signal:
            fraction = min(1.0, fraction + 0.06)
        desired_recovered = min(budget, budget * fraction)
        add_pct = max(0.0, desired_recovered - recovered)
        if add_pct < owner.RECOVERY_MIN_STEP:
            return V42RecoveryPlan(blocked_reason="recovery_step_too_small")
        target = max(context.current_pct + add_pct, float(episode.get("post_sell_pct", context.current_pct)) + desired_recovered)
        return V42RecoveryPlan(
            allowed=True,
            target=min(owner.TARGET_CAP, target),
            max_buy=self.recovery_step_max_buy(
                add_pct,
                low_drop,
                strong,
                deep_base=bool(staged_plan.get("allowed", False)),
            ),
            drawdown=low_drop,
            remaining_budget=remaining,
            guard=str(staged_plan.get("guard", "")),
        )

    @staticmethod
    def episode_reentry_plan(owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> V42RecoveryPlan:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return V42RecoveryPlan(blocked_reason="missing_sell_anchor")
        age = owner._call_count - int(episode.get("start_call", owner._call_count))
        current_drop = max(0.0, 1.0 - context.price / sell_price)
        breakout = bool(signals.trend_continuation and signals.recovery_quality_ok and context.price >= sell_price * 1.03)
        value_reset = bool(current_drop >= 0.10 and signals.recovery_signal)
        if age < owner.DISTRIBUTION_MIN_REENTRY_CALLS and not breakout:
            return V42RecoveryPlan(blocked_reason="distribution_reentry_cooldown")
        if not breakout and not value_reset:
            return V42RecoveryPlan(blocked_reason="distribution_reentry_no_edge")
        return V42RecoveryPlan(allowed=True)

    def episode_recovery_target(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals, base_target: float) -> float:
        plan = self.episode_recovery_plan(owner, context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return min(max(base_target, context.current_pct), self.episode_recovery_cap(episode))
        path_target = float(plan.get("target", context.current_pct))
        return min(max(base_target, path_target), self.episode_recovery_cap(episode))

    def episode_recovery_max_buy(self, owner, context: V42Context, regime: V42Regime, episode: dict, setup: str, signals: V42Signals | None = None) -> float:
        if signals is None:
            signals = owner._build_signals(context, regime, episode)
        plan = self.episode_recovery_plan(owner, context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return 0.0
        max_buy = float(plan.get("max_buy", 0.0) or 0.0)
        if setup == "recovery-probe-buy":
            return max(0.04, max_buy)
        return max_buy

    @staticmethod
    def episode_recovery_budget(episode: dict) -> float:
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
    def episode_recovered_pct(episode: dict) -> float:
        return max(0.0, float(episode.get("recovery_bought_pct", 0.0) or 0.0))

    def record_episode_recovery_buy(self, owner, context: V42Context, episode: dict, action: Action) -> None:
        buy_pct = action.quantity * action.price / context.total_value if context.total_value > 0.0 else 0.0
        owner._diag["v4_2_recovery_path_buy_count"] += 1
        episode["recovery_buy_count"] = int(episode.get("recovery_buy_count", 0)) + 1
        episode["recovery_buy_notional"] = float(episode.get("recovery_buy_notional", 0.0)) + action.quantity * action.price
        recovered = self.episode_recovered_pct(episode) + max(0.0, buy_pct)
        cumulative_recovered = float(episode.get("cumulative_recovered_pct", 0.0) or 0.0) + max(0.0, buy_pct)
        episode["recovery_bought_pct"] = recovered
        episode["cumulative_recovered_pct"] = cumulative_recovered
        cumulative_budget = float(episode.get("cumulative_sold_pct", self.episode_recovery_budget(episode)) or 0.0)
        episode["unrecovered_budget_pct"] = max(0.0, cumulative_budget - cumulative_recovered)
        episode["last_recovery_buy_call"] = owner._call_count
        episode["last_recovery_buy_price"] = context.price
        position_after = context.current_pct + max(0.0, buy_pct)
        episode["max_recovery_position_pct"] = max(float(episode.get("max_recovery_position_pct", position_after) or position_after), position_after)
        for threshold, key in ((0.30, "recovered_to_30_call"), (0.50, "recovered_to_50_call"), (0.80, "recovered_to_80_call")):
            if episode.get(key) is None and position_after >= threshold:
                episode[key] = owner._call_count
        legs = list(episode.get("recovery_legs", []))
        legs.append({
            "call": owner._call_count,
            "setup": self.setup_from_action(action),
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
            episode["recovery_start_call"] = owner._call_count

    @staticmethod
    def setup_from_action(action: Action) -> str:
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

    @staticmethod
    def is_bear_base_buy(decision: V42DecisionPlan, sizing: V42Sizing) -> bool:
        primary_sleeve = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        return bool(primary_sleeve == "bear-base" or "v4_2_bear_base" in str(sizing.guard))

    @staticmethod
    def should_record_episode_recovery_buy(owner, bear_base_buy: bool) -> bool:
        return bool(not bear_base_buy or owner.COUNT_BEAR_BASE_BUYS_AS_EPISODE_RECOVERY)

    @staticmethod
    def recovery_fraction_from_drawdown(low_drop: float, *, shallow: float, normal: float, deep: float, crash: float) -> float:
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
    def recovery_step_max_buy(add_pct: float, low_drop: float, strong: bool, *, deep_base: bool = False) -> float:
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
    def recovery_drawdown_guard(low_drop: float) -> str:
        if low_drop >= 0.25:
            return "v4_2_recovery_crash_discount"
        if low_drop >= 0.15:
            return "v4_2_recovery_deep_discount"
        if low_drop >= 0.08:
            return "v4_2_recovery_normal_discount"
        return "v4_2_recovery_shallow_discount"

    def recovery_permission(self, owner, context: V42Context, regime: V42Regime, signals: V42Signals, low_drop: float) -> V42RecoveryPlan:
        return self.base_recovery_permission(context, regime, signals, low_drop)

    @staticmethod
    def base_recovery_permission(context: V42Context, regime: V42Regime, signals: V42Signals, low_drop: float) -> V42RecoveryPlan:
        if regime.structural_bear:
            return V42RecoveryPlan(blocked_reason="structural_bear_active")
        if regime.btc_regime == "BEAR" and not signals.strong_recovery_signal:
            return V42RecoveryPlan(blocked_reason="btc_bear_recovery_unconfirmed")
        if context.trend_risk >= 3:
            return V42RecoveryPlan(blocked_reason="trend_risk_severe")
        if context.raw_state == "BEAR" and context.confirmed_state == "BEAR" and context.trend_risk >= 2:
            return V42RecoveryPlan(blocked_reason="confirmed_bear_recovery_unconfirmed")
        if signals.recovery_signal or signals.strong_recovery_signal:
            return V42RecoveryPlan(allowed=True, quality="confirmed")
        if low_drop >= 0.08 and context.trend_risk <= 2 and regime.btc_regime != "BEAR":
            return V42RecoveryPlan(allowed=True, quality="discount")
        return V42RecoveryPlan(blocked_reason="recovery_signal_missing")

    def staged_recovery_plan(self, owner, *, context: V42Context, regime: V42Regime, episode: dict, age: int, current_drop: float, low_drop: float, remaining: float, lowest: float) -> V42RecoveryPlan:
        deep_base = self.deep_base_recovery_plan(owner, context=context, regime=regime, episode=episode, age=age, current_drop=current_drop, low_drop=low_drop, remaining=remaining)
        if bool(deep_base.get("allowed", False)):
            return deep_base
        return self.post_crash_recoil_plan(owner, context=context, regime=regime, episode=episode, age=age, current_drop=current_drop, low_drop=low_drop, remaining=remaining, lowest=lowest)

    @staticmethod
    def deep_base_recovery_plan(owner, *, context: V42Context, regime: V42Regime, episode: dict, age: int, current_drop: float, low_drop: float, remaining: float) -> V42RecoveryPlan:
        if remaining < owner.RECOVERY_MIN_STEP or age < owner.DEEP_BASE_MIN_CALLS or low_drop < owner.DEEP_BASE_MIN_DRAWDOWN or current_drop < owner.DEEP_BASE_MIN_CURRENT_DISCOUNT:
            return V42RecoveryPlan()
        if context.trend_risk >= 4:
            return V42RecoveryPlan()
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        low_location = bool((pd.isna(rolling_pos) or rolling_pos <= 0.42) and (pd.isna(donchian_pos) or donchian_pos <= 0.45))
        if not low_location:
            return V42RecoveryPlan()
        fraction = 0.18
        if low_drop >= 0.45 and age >= owner.DEEP_BASE_MIN_CALLS + 45:
            fraction = 0.30
        elif low_drop >= 0.35:
            fraction = 0.24
        return V42RecoveryPlan(allowed=True, fraction=fraction, guard="v4_2_deep_base_recovery")

    @staticmethod
    def post_crash_recoil_plan(owner, *, context: V42Context, regime: V42Regime, episode: dict, age: int, current_drop: float, low_drop: float, remaining: float, lowest: float) -> V42RecoveryPlan:
        if remaining < owner.RECOVERY_MIN_STEP or age < 12 or low_drop < 0.12:
            return V42RecoveryPlan()
        if context.raw_state == "BEAR" and context.confirmed_state == "BEAR" and context.trend_risk == 0:
            return V42RecoveryPlan()
        if regime.btc_regime == "BEAR" and low_drop < 0.22:
            return V42RecoveryPlan()
        roc_10 = owner._value(context.latest, "roc_10", 0.0)
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        atr_rank = regime.atr_rank
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        rebound_from_low = context.price / lowest - 1.0 if lowest > 0.0 else 0.0
        if not pd.isna(roc_20) and roc_20 <= -0.18:
            return V42RecoveryPlan()
        if not pd.isna(roc_10) and roc_10 <= -0.12:
            return V42RecoveryPlan()
        if context.trend_risk >= 3 and current_drop < 0.18:
            return V42RecoveryPlan()
        if not pd.isna(rolling_pos) and rolling_pos > 0.55:
            return V42RecoveryPlan()
        if not pd.isna(atr_rank) and atr_rank >= 0.96 and rebound_from_low < 0.10:
            return V42RecoveryPlan()
        low_or_reclaim = bool(
            current_drop >= 0.06
            or (low_drop >= 0.22 and rebound_from_low >= 0.10)
            or (not pd.isna(donchian_pos) and 0.18 <= donchian_pos <= 0.55 and low_drop >= 0.18)
        )
        if not low_or_reclaim:
            return V42RecoveryPlan()
        fraction = 0.12
        if low_drop >= 0.35 and age >= 30:
            fraction = 0.22
        elif low_drop >= 0.22:
            fraction = 0.18
        return V42RecoveryPlan(allowed=True, fraction=fraction, guard="v4_2_post_crash_recoil")

    @staticmethod
    def structural_recovery_ready(context: V42Context, regime: V42Regime, signals: V42Signals) -> bool:
        return bool(
            not regime.structural_bear
            and regime.regime in {"RANGE", "TRANSITION", "BULL"}
            and regime.btc_regime != "BEAR"
            and context.trend_risk <= 2
            and signals.strong_recovery_signal
        )

    @staticmethod
    def episode_recovery_cap(episode: dict) -> float:
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

    @staticmethod
    def base_recovery_established(owner, context: V42Context) -> bool:
        if not hasattr(owner, "BASE_RECOVERY_MIN_BASE_PCT"):
            return False
        if context.symbol == "BTC/USDT":
            return False
        return bool(owner._base_pct_from_quantity(context, owner._base_quantity(context)) >= owner.BASE_RECOVERY_MIN_BASE_PCT)

    @staticmethod
    def base_recovery_stabilized(owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> bool:
        if not hasattr(owner, "BASE_RECOVERY_MAX_ENTRY_MULTIPLE"):
            return False
        if regime.structural_bear or signals.distribution_exhaustion or context.trend_risk >= 4 or context.risk_score >= 5:
            return False
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        roc_10 = owner._value(context.latest, "roc_10", 0.0)
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        low_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos <= 0.45)
            or (not pd.isna(donchian_pos) and donchian_pos <= 0.55)
            or regime.price_vs_ema168 <= -0.08
        )
        stabilizing = bool(
            signals.recovery_signal
            or signals.recovery_quality_ok
            or signals.value_recovery
            or (context.trend_risk <= 2 and roc_10 > -0.08 and roc_20 > -0.16)
        )
        if not (low_location and stabilizing):
            return False
        if regime.regime == "BULL" or context.confirmed_state == "BULL":
            return False
        ledger = owner._base_ledger_by_symbol.get(context.symbol, {})
        avg_entry = float(ledger.get("protected_floor_avg_entry_price", 0.0) or ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry > 0.0 and context.price > avg_entry * owner.BASE_RECOVERY_MAX_ENTRY_MULTIPLE:
            return False
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        return bool(
            (not pd.isna(rolling_pos) and rolling_pos <= 0.35)
            or (not pd.isna(donchian_pos) and donchian_pos <= 0.45)
            or regime.price_vs_ema168 <= 0.02
        )

    @staticmethod
    def base_recovery_target(owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> float:
        if not hasattr(owner, "BASE_RECOVERY_TARGET"):
            return 0.0
        base_target = float(owner.BASE_RECOVERY_TARGET.get(context.symbol, 0.0))
        if base_target <= 0.0:
            return 0.0
        if signals.strong_recovery_signal or signals.trend_continuation:
            base_target = max(base_target, 0.64)
        if regime.regime == "BULL" and context.risk_score <= 1:
            base_target = max(base_target, 0.70)
        cap = float(owner.BASE_RECOVERY_MAX_TARGET.get(context.symbol, base_target))
        return min(cap, base_target, owner.TARGET_CAP)

    def base_recovery_should_accumulate(self, owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> bool:
        if not self.base_recovery_established(owner, context):
            return False
        if not self.base_recovery_stabilized(owner, context, regime, signals):
            return False
        target = self.base_recovery_target(owner, context, regime, signals)
        if target <= 0.0 or context.current_pct >= target - owner.RECOVERY_MIN_STEP:
            return False
        return bool(context.trend_risk <= 3 and context.risk_score <= 4)

    def base_led_recovery_base_sizing(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals, decision: V42DecisionPlan, primary: V42Sizing) -> V42Sizing:
        if not hasattr(owner, "BASE_LED_RECOVERY_TARGET"):
            return V42Sizing()
        if not self.base_led_recovery_allowed(owner, context, regime, episode, signals):
            return V42Sizing()
        target = float(owner.BASE_LED_RECOVERY_TARGET.get(context.symbol, 0.0))
        base_pct = owner._base_pct_from_quantity(context, owner._base_quantity(context))
        if target <= 0.0 or base_pct >= target - owner.RECOVERY_MIN_STEP:
            return V42Sizing()
        last_buy = int(getattr(owner, "_last_base_led_recovery_buy_call_by_symbol", {}).get(context.symbol, -10_000))
        if owner._call_count - last_buy < owner.BASE_LED_RECOVERY_COOLDOWN_CALLS:
            return V42Sizing()
        gap = max(0.0, target - base_pct)
        buy_pct = min(gap, owner.BASE_LED_RECOVERY_MAX_BUY_PCT)
        if buy_pct < owner.RECOVERY_MIN_STEP:
            return V42Sizing()
        buy_qty = context.total_value * buy_pct / context.price if context.price > 0.0 else 0.0
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup="base-led-recovery-buy", blocked_reason="zero_quantity")
        if buy_qty * context.price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup="base-led-recovery-buy", blocked_reason="min_notional")
        owner._diag["v4_4_base_led_recovery_base_buy_count"] = owner._diag.get("v4_4_base_led_recovery_base_buy_count", 0) + 1
        guard = owner._join_guard(owner._sizing_guard(context, regime, episode, "recovery-probe-buy"), "v4_4_base_led_recovery_base")
        decision.primary_sleeve = V42SleevePlan(
            sleeve="base-led-recovery",
            side="buy",
            setup="base-led-recovery-buy",
            target=min(owner.TARGET_CAP, context.current_pct + buy_pct),
            guard=guard,
            priority=82,
            allowed=True,
        )
        return V42Sizing(
            side="buy",
            setup="base-led-recovery-buy",
            quantity=buy_qty,
            target=min(owner.TARGET_CAP, context.current_pct + buy_pct),
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=context.current_pct + buy_pct,
            target_gap_before=gap,
            actual_step_pct=buy_pct,
            remaining_gap_after=max(0.0, gap - buy_pct),
        )

    @staticmethod
    def base_led_recovery_allowed(owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> bool:
        if not hasattr(owner, "BASE_LED_RECOVERY_TARGET"):
            return False
        if context.symbol == "BTC/USDT":
            return False
        if str(episode.get("state", "NORMAL")) not in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            return False
        if owner._base_pct_from_quantity(context, owner._base_quantity(context)) < owner.BASE_RECOVERY_MIN_BASE_PCT:
            return False
        if regime.structural_bear or signals.distribution_exhaustion:
            return False
        if regime.regime == "BULL" or context.confirmed_state == "BULL":
            return False
        if context.trend_risk >= 3 or context.risk_score >= 4:
            return False
        ledger = owner._base_ledger_by_symbol.get(context.symbol, {})
        avg_entry = float(ledger.get("protected_floor_avg_entry_price", 0.0) or ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry > 0.0 and context.price > avg_entry * owner.BASE_LED_RECOVERY_MAX_ENTRY_MULTIPLE:
            return False
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        roc_10 = owner._value(context.latest, "roc_10", 0.0)
        return bool(
            (
                (not pd.isna(rolling_pos) and rolling_pos <= 0.32)
                or (not pd.isna(donchian_pos) and donchian_pos <= 0.42)
                or regime.price_vs_ema168 <= -0.04
            )
            and (signals.recovery_signal or signals.recovery_quality_ok or roc_10 > -0.06)
        )
