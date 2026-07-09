"""Protected low-floor and lifecycle base helpers for V4.7."""

from __future__ import annotations

import pandas as pd

from ..strategy_rebalance import Action
from ..v42_types import V42Context, V42DecisionPlan, V42Regime, V42Signals, V42Sizing, V42SleevePlan


class V47FloorEngine:
    def opportunity_floor_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        signals: V42Signals,
        prior_sizing: V42Sizing,
    ) -> V42Sizing:
        if context.symbol not in owner.OPPORTUNITY_FLOOR_SYMBOLS:
            return V42Sizing()
        target = float(owner.OPPORTUNITY_FLOOR_TARGET.get(context.symbol, 0.0))
        if target <= 0.0 or context.current_pct >= target - owner.RECOVERY_MIN_STEP:
            return V42Sizing(blocked_reason="opportunity_floor_filled")
        if not self.opportunity_floor_low_location(owner, context, regime):
            return V42Sizing()
        if not self.opportunity_floor_stabilizing(owner, context, regime, signals):
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="opportunity_floor_not_stabilizing")
        last_buy = owner._last_opportunity_floor_buy_call_by_symbol.get(context.symbol, -10_000)
        if owner._call_count - last_buy < owner.OPPORTUNITY_FLOOR_COOLDOWN_CALLS:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="opportunity_floor_cooldown")
        gap = max(0.0, target - context.current_pct)
        buy_pct = min(gap, owner.OPPORTUNITY_FLOOR_MAX_BUY_PCT)
        buy_qty = context.total_value * buy_pct / context.price if context.price > 0.0 else 0.0
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="zero_quantity")
        if buy_qty * context.price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="min_notional")
        owner._diag["v4_3_opportunity_floor_buy_count"] = owner._diag.get("v4_3_opportunity_floor_buy_count", 0) + 1
        guard = owner._join_guard(
            owner._sizing_guard(context, regime, {"state": "OPPORTUNITY_FLOOR"}, "opportunity-floor-buy"),
            "v4_3_opportunity_floor_independent",
        )
        return V42Sizing(
            side="buy",
            setup="opportunity-floor-buy",
            quantity=buy_qty,
            target=min(target, owner.TARGET_CAP),
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=context.current_pct + buy_pct,
            target_gap_before=gap,
            actual_step_pct=buy_pct,
            remaining_gap_after=max(0.0, gap - buy_pct),
        )

    @staticmethod
    def opportunity_floor_low_location(owner, context: V42Context, regime: V42Regime) -> bool:
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        if pd.isna(rolling_pos) or rolling_pos > 0.15:
            return False
        if not pd.isna(donchian_pos) and donchian_pos > 0.25:
            return False
        return True

    @staticmethod
    def opportunity_floor_stabilizing(owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> bool:
        if context.risk_score >= 5:
            return False
        recent = context.df.tail(20)
        if recent.empty or "close" not in recent.columns:
            return bool(signals.recovery_signal or signals.recovery_quality_ok)

        recent_low_idx = recent["close"].idxmin()
        recent_low = float(recent.loc[recent_low_idx, "close"])
        rebound = context.price / recent_low - 1.0 if recent_low > 0.0 else 0.0
        days_since_low = max(0, len(recent) - 1 - recent.index.get_loc(recent_low_idx))
        roc_5 = owner._value(context.latest, "roc_5", 0.0)
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        roc_60 = owner._value(context.latest, "roc_60", 0.0)

        waterfall = bool(not pd.isna(roc_20) and roc_20 <= -0.20 and rebound < 0.05)
        if waterfall:
            return False

        deep_collapse_too_fresh = bool(not pd.isna(roc_60) and roc_60 <= -0.40 and days_since_low < 10)
        if deep_collapse_too_fresh:
            return False

        panic = bool(regime.regime == "BEAR" and context.trend_risk >= 3)
        if panic:
            return bool(rebound >= 0.05 and days_since_low >= 2 and not pd.isna(roc_5) and roc_5 >= 0.0)
        return bool(
            rebound >= 0.03
            or (not pd.isna(roc_5) and roc_5 >= -0.02)
            or signals.recovery_signal
            or signals.recovery_quality_ok
        )

    @staticmethod
    def protected_floor_quantity(owner, context: V42Context) -> float:
        ledger = owner._protected_floor_ledger_by_symbol.get(context.symbol, {})
        qty = max(0.0, float(ledger.get("quantity", 0.0) or 0.0))
        return min(qty, max(0.0, float(context.pos.quantity)))

    @staticmethod
    def protected_floor_exit_allowed(owner, context: V42Context, regime: V42Regime) -> bool:
        ledger = owner._protected_floor_ledger_by_symbol.get(context.symbol)
        if not ledger or float(ledger.get("quantity", 0.0) or 0.0) <= 1e-12:
            return False
        avg_entry = float(ledger.get("avg_entry", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return False
        loss = 1.0 - context.price / avg_entry
        profit = context.price / avg_entry - 1.0
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        high_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos >= owner.PROTECTED_FLOOR_HIGH_ROLLING_POS)
            or (not pd.isna(donchian_pos) and donchian_pos >= owner.PROTECTED_FLOOR_HIGH_DONCHIAN_POS)
        )
        hard_stop = loss >= owner.PROTECTED_FLOOR_STOP_LOSS_PCT
        take_profit = profit >= owner.PROTECTED_FLOOR_TAKE_PROFIT_PCT and high_location
        if hard_stop:
            owner._diag["v4_3_protected_floor_stop_exit_allowed_count"] = (
                owner._diag.get("v4_3_protected_floor_stop_exit_allowed_count", 0) + 1
            )
        if take_profit:
            owner._diag["v4_3_protected_floor_profit_exit_allowed_count"] = (
                owner._diag.get("v4_3_protected_floor_profit_exit_allowed_count", 0) + 1
            )
        return bool(hard_stop or take_profit)

    @staticmethod
    def record_protected_floor_buy(owner, context: V42Context, regime: V42Regime, action: Action) -> None:
        ledger = owner._protected_floor_ledger_by_symbol.setdefault(
            context.symbol,
            {
                "quantity": 0.0,
                "avg_entry": 0.0,
                "entry_call": owner._call_count,
                "entry_price": context.price,
                "entry_structural_bear": regime.structural_bear,
            },
        )
        old_qty = max(0.0, float(ledger.get("quantity", 0.0) or 0.0))
        old_entry = float(ledger.get("avg_entry", context.price) or context.price)
        new_qty = old_qty + float(action.quantity)
        ledger["quantity"] = new_qty
        ledger["avg_entry"] = (old_entry * old_qty + context.price * float(action.quantity)) / new_qty if new_qty > 0.0 else 0.0
        ledger["entry_price"] = min(float(ledger.get("entry_price", context.price) or context.price), float(context.price))
        ledger["entry_call"] = min(int(ledger.get("entry_call", owner._call_count) or owner._call_count), owner._call_count)
        ledger["entry_structural_bear"] = bool(ledger.get("entry_structural_bear", False) or regime.structural_bear)
        ledger["peak_price"] = max(float(ledger.get("peak_price", context.price) or context.price), float(context.price))

    def consume_protected_floor_for_sell(self, owner, context: V42Context, regime: V42Regime, action: Action) -> None:
        if not self.protected_floor_exit_allowed(owner, context, regime):
            return
        floor_qty = self.protected_floor_quantity(owner, context)
        if floor_qty <= 1e-12:
            return
        non_floor_qty = max(0.0, owner._main_quantity(context) - floor_qty)
        consume = max(0.0, float(action.quantity) - non_floor_qty)
        if consume <= 1e-12:
            return
        ledger = owner._protected_floor_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return
        remaining = max(0.0, floor_qty - consume)
        ledger["quantity"] = remaining
        if remaining <= 1e-12:
            ledger["quantity"] = 0.0
            ledger["avg_entry"] = 0.0

    def protected_floor_exit_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        if context.symbol != "BNB/USDT":
            return V42Sizing()
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return V42Sizing()
        protected_qty = min(owner._source_quantity(context, "protected_floor"), owner._base_quantity(context))
        if protected_qty <= 1e-12:
            return V42Sizing()
        return self.protected_floor_event_exit_sizing(owner, context, regime, signals, decision, protected_qty)

    def protected_floor_event_exit_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        signals: V42Signals,
        decision: V42DecisionPlan,
        protected_qty: float,
    ) -> V42Sizing:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger or "protected_floor" not in str(ledger.get("base_source", "")):
            return V42Sizing()
        protected_qty = min(float(ledger.get("protected_floor_quantity", 0.0) or 0.0), protected_qty)
        if protected_qty <= 1e-12:
            return V42Sizing()
        entry_call = ledger.get("protected_floor_entry_call") or ledger.get("base_entry_call")
        if entry_call is None or owner._call_count - int(entry_call) < owner.PROTECTED_FLOOR_EXIT_MIN_HOLD_CALLS:
            return V42Sizing()
        avg_entry = float(ledger.get("protected_floor_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return V42Sizing()

        profit = context.price / avg_entry - 1.0
        if not self.protected_floor_exit_event(owner, context, regime, signals, profit):
            return V42Sizing()
        if not self.protected_floor_new_release_event(owner, context, ledger):
            return V42Sizing()

        sell_qty = min(protected_qty, protected_qty * owner.PROTECTED_FLOOR_EXIT_FRACTION)
        notional = sell_qty * context.price
        if sell_qty <= 1e-12 or notional < owner.min_notional:
            return V42Sizing()
        step_pct = notional / context.total_value if context.total_value > 0.0 else 0.0
        owner._diag["v4_3_protected_floor_event_exit_count"] = (
            owner._diag.get("v4_3_protected_floor_event_exit_count", 0) + 1
        )
        guard = owner._join_guard(
            owner._sizing_guard(context, regime, {"state": "PROTECTED_FLOOR_EVENT_EXIT"}, "protected-floor-exit"),
            "v4_3_protected_floor_event_release",
        )
        return V42Sizing(
            side="sell",
            setup="protected-floor-exit",
            quantity=sell_qty,
            target=max(0.0, context.current_pct - step_pct),
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=max(0.0, context.current_pct - step_pct),
            target_gap_before=step_pct,
            actual_step_pct=step_pct,
            remaining_gap_after=0.0,
        )

    @staticmethod
    def protected_floor_exit_event(owner, context: V42Context, regime: V42Regime, signals: V42Signals, profit: float) -> bool:
        latest = context.latest
        rolling_pos = owner._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(latest, "donchian_pos", 0.5)
        roc_10 = owner._value(latest, "roc_10", 0.0)
        roc_20 = owner._value(latest, "roc_20", 0.0)
        price_vs_ema168 = owner._value(latest, "price_vs_ema168", default=owner._price_vs(latest, context.price, "ema168"))
        strong_expansion = bool(roc_20 > 0.15 and not signals.distribution_exhaustion and context.risk_score == 0 and context.drawdown_risk == 0)
        if strong_expansion:
            return False

        high_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos >= 0.80)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.88)
            or price_vs_ema168 >= 0.35
        )
        exhaustion = bool(signals.distribution_exhaustion or context.risk_score >= 1 or (roc_10 <= 0.03 and roc_20 <= 0.10))
        bull_exhaustion_release = bool(profit >= owner.PROTECTED_FLOOR_EXIT_PROFIT and high_location and exhaustion)
        rebound_failure = bool(
            profit >= owner.PROTECTED_FLOOR_EXIT_REBOUND_PROFIT
            and not pd.isna(donchian_pos)
            and donchian_pos >= 0.75
            and price_vs_ema168 >= 0.12
            and (signals.distribution_exhaustion or roc_10 <= 0.0 or context.risk_score >= 2)
        )
        return bull_exhaustion_release or rebound_failure

    @staticmethod
    def protected_floor_new_release_event(owner, context: V42Context, ledger: dict) -> bool:
        last_exit_price = float(ledger.get("protected_floor_last_exit_price", 0.0) or 0.0)
        if last_exit_price <= 0.0:
            return True
        return bool(context.price >= last_exit_price * (1.0 + owner.PROTECTED_FLOOR_EXIT_MIN_NEW_HIGH))

    def lifecycle_low_base_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        decision: V42DecisionPlan,
        prior_sizing: V42Sizing,
    ) -> V42Sizing:
        cap = max(0.0, float(owner.LIFECYCLE_LOW_BASE_CAP.get(context.symbol, 0.0)))
        if cap <= 0.0:
            return V42Sizing()
        if context.current_pct > cap + owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_total_exposure_filled")
        if not self.lifecycle_low_base_entry_location(owner, context, regime):
            return V42Sizing()
        if not self.lifecycle_low_base_stabilizing(owner, context, regime, signals):
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_not_stabilizing")
        if signals.distribution_exhaustion:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_distribution")
        source_qty = owner._source_quantity(context, "protected_floor")
        source_pct = owner._base_pct_from_quantity(context, source_qty)
        if source_pct >= cap - owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_cap_filled")
        portfolio_room = max(0.0, owner.LIFECYCLE_LOW_BASE_PORTFOLIO_CAP - self.lifecycle_protected_floor_portfolio_pct(owner, context))
        if portfolio_room < owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_portfolio_cap")
        spot_room = max(0.0, owner.SPOT_EXPOSURE_CAP - float(context.current_pct))
        if spot_room < owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="spot_exposure_cap")
        last_buy = owner._last_lifecycle_low_base_buy_call_by_symbol.get(context.symbol, -10_000)
        if owner._call_count - last_buy < owner.LIFECYCLE_LOW_BASE_COOLDOWN_CALLS:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_cooldown")
        gap = min(max(0.0, cap - source_pct), portfolio_room, spot_room)
        buy_pct = min(gap, owner.LIFECYCLE_LOW_BASE_MAX_BUY_PCT)
        if buy_pct < owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="lifecycle_low_base_step_too_small")
        buy_qty = context.total_value * buy_pct / context.price if context.price > 0.0 else 0.0
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="zero_quantity")
        if buy_qty * context.price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup="opportunity-floor-buy", blocked_reason="min_notional")
        guard = owner._join_guard(owner._sizing_guard(context, regime, episode, "opportunity-floor-buy"), "v4_5_lifecycle_low_base")
        decision.primary_sleeve = V42SleevePlan(
            sleeve="lifecycle-low-base",
            side="buy",
            setup="opportunity-floor-buy",
            target=min(owner.SPOT_EXPOSURE_CAP, context.current_pct + buy_pct),
            guard=guard,
            priority=86,
            allowed=True,
        )
        owner._diag["v4_5_lifecycle_low_base_buy_count"] = owner._diag.get("v4_5_lifecycle_low_base_buy_count", 0) + 1
        return V42Sizing(
            side="buy",
            setup="opportunity-floor-buy",
            quantity=buy_qty,
            target=min(owner.SPOT_EXPOSURE_CAP, context.current_pct + buy_pct),
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=context.current_pct + buy_pct,
            target_gap_before=gap,
            actual_step_pct=buy_pct,
            remaining_gap_after=max(0.0, gap - buy_pct),
        )

    def lifecycle_protected_base_exit_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        source_qty = min(owner._source_quantity(context, "protected_floor"), owner._base_quantity(context))
        if source_qty <= 1e-12:
            return V42Sizing()
        sources = owner._base_ledger_by_symbol.get(context.symbol, {}).get("source_ledger", {})
        row = sources.get("protected_floor", {})
        avg_entry = float(row.get("avg_entry", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return V42Sizing()
        entry_call = int(row.get("entry_call", owner._call_count) or owner._call_count)
        if owner._call_count - entry_call < owner.LIFECYCLE_BASE_EXIT_MIN_HOLD_CALLS:
            return V42Sizing(side="sell", setup="protected-floor-exit", blocked_reason="lifecycle_base_exit_min_hold")
        profit = context.price / avg_entry - 1.0
        high_location = self.lifecycle_base_exit_high_location(owner, context, regime)
        distribution_exit = bool(signals.distribution_exhaustion and profit >= owner.LIFECYCLE_BASE_EXIT_DISTRIBUTION_PROFIT_PCT)
        normal_exit = bool(high_location and profit >= owner.LIFECYCLE_BASE_EXIT_PROFIT_PCT)
        if not (distribution_exit or normal_exit):
            return V42Sizing()
        sell_qty = source_qty
        if sell_qty <= 1e-12:
            return V42Sizing(side="sell", setup="protected-floor-exit", blocked_reason="zero_quantity")
        notional = sell_qty * context.price
        if notional < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="sell", setup="protected-floor-exit", blocked_reason="min_notional")
        sell_pct = notional / context.total_value if context.total_value > 0.0 else 0.0
        guard = owner._join_guard(
            owner._sizing_guard(context, regime, {"state": "LIFECYCLE_BASE_EXIT"}, "protected-floor-exit"),
            "v4_5_lifecycle_base_exit",
        )
        decision.primary_sleeve = V42SleevePlan(
            sleeve="lifecycle-base-exit",
            side="sell",
            setup="protected-floor-exit",
            target=max(0.0, context.current_pct - sell_pct),
            guard=guard,
            priority=84,
            allowed=True,
        )
        owner._diag["v4_5_lifecycle_base_exit_count"] = owner._diag.get("v4_5_lifecycle_base_exit_count", 0) + 1
        return V42Sizing(
            side="sell",
            setup="protected-floor-exit",
            quantity=sell_qty,
            target=max(0.0, context.current_pct - sell_pct),
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=max(0.0, context.current_pct - sell_pct),
            target_gap_before=sell_pct,
            actual_step_pct=-sell_pct,
            remaining_gap_after=0.0,
        )

    @staticmethod
    def lifecycle_base_exit_high_location(owner, context: V42Context, regime: V42Regime) -> bool:
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        return bool(
            (not pd.isna(rolling_pos) and rolling_pos >= 0.78)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.82)
            or (not pd.isna(regime.price_vs_ema168) and regime.price_vs_ema168 >= 0.28)
        )

    @staticmethod
    def lifecycle_low_base_entry_location(owner, context: V42Context, regime: V42Regime) -> bool:
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        deep_discount = bool(
            not pd.isna(regime.price_vs_ema168)
            and regime.price_vs_ema168 <= -0.20
            and (
                (not pd.isna(rolling_pos) and rolling_pos <= 0.35)
                or (not pd.isna(donchian_pos) and donchian_pos <= 0.35)
            )
        )
        return bool(
            (not pd.isna(rolling_pos) and rolling_pos <= 0.18)
            or (not pd.isna(donchian_pos) and donchian_pos <= 0.25)
            or deep_discount
        )

    @staticmethod
    def lifecycle_low_base_stabilizing(owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> bool:
        if context.risk_score >= 5:
            return False
        recent = context.df.tail(20)
        if recent.empty or "close" not in recent.columns:
            return bool(signals.recovery_signal or signals.recovery_quality_ok)
        recent_low = float(recent["close"].min())
        rebound = context.price / recent_low - 1.0 if recent_low > 0.0 else 0.0
        roc_5 = owner._value(context.latest, "roc_5", 0.0)
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        waterfall = bool(not pd.isna(roc_20) and roc_20 <= -0.25 and rebound < 0.04)
        if waterfall:
            return False
        return bool(
            rebound >= 0.025
            or (not pd.isna(roc_5) and roc_5 >= -0.03)
            or signals.recovery_signal
            or signals.recovery_quality_ok
        )

    @staticmethod
    def lifecycle_protected_floor_portfolio_pct(owner, context: V42Context) -> float:
        total = 0.0
        for symbol in owner.TARGET_ALLOC:
            if symbol == context.symbol:
                total += owner._base_pct_from_quantity(context, owner._source_quantity(context, "protected_floor"))
                continue
            ledger = owner._base_ledger_by_symbol.get(symbol, {})
            sources = ledger.get("source_ledger", {})
            protected = max(0.0, float(sources.get("protected_floor", {}).get("quantity", 0.0) or 0.0))
            base_qty = max(0.0, float(ledger.get("base_quantity", 0.0) or 0.0))
            base_pct = max(0.0, float(ledger.get("base_position_pct", 0.0) or 0.0))
            total += base_pct * protected / base_qty if base_qty > 1e-12 else 0.0
        return total
