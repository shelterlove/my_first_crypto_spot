"""Bear-base budget, location, and exit helpers for V4.7."""

from __future__ import annotations

import pandas as pd

from ..v42_types import V42BearBaseProposal, V42Context, V42Regime, V42Signals, V42Sizing


class V47BearBaseEngine:
    @staticmethod
    def base_quantity(owner, context: V42Context) -> float:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return 0.0
        quantity = max(0.0, float(ledger.get("base_quantity", 0.0) or 0.0))
        return min(quantity, max(0.0, float(context.pos.quantity)))

    def main_quantity(self, owner, context: V42Context) -> float:
        return max(0.0, float(context.pos.quantity) - self.base_quantity(owner, context))

    def base_pct_from_quantity(self, owner, context: V42Context, quantity: float) -> float:
        if context.total_value <= 0.0 or context.price <= 0.0:
            return 0.0
        return max(0.0, min(context.current_pct, float(quantity) * context.price / context.total_value))

    @staticmethod
    def new_recovery_ledger(call_count: int) -> dict:
        return {
            "defensive_sold_pct": 0.0,
            "recovery_bought_pct": 0.0,
            "avg_sell_price": 0.0,
            "first_sell_call": call_count,
            "last_sell_call": call_count,
        }

    @staticmethod
    def new_base_ledger() -> dict:
        return {
            "base_budget_used_pct": 0.0,
            "base_position_pct": 0.0,
            "base_quantity": 0.0,
            "base_avg_entry_price": 0.0,
            "base_entry_call": None,
            "base_layer": 0,
            "base_peak_price": 0.0,
            "base_peak_profit": 0.0,
        }

    def refresh_base_ledger_market(self, owner, context: V42Context) -> None:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger or self.base_quantity(owner, context) <= 0.0:
            return
        avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return
        peak_price = max(float(ledger.get("base_peak_price", 0.0) or 0.0), context.price)
        ledger["base_peak_price"] = peak_price
        ledger["base_peak_profit"] = max(0.0, peak_price / avg_entry - 1.0)
        ledger["base_position_pct"] = self.base_pct_from_quantity(owner, context, self.base_quantity(owner, context))

    def record_ledger_sell(self, owner, context: V42Context, sizing: V42Sizing) -> None:
        sold_pct = sizing.quantity * context.price / context.total_value if context.total_value > 0.0 else 0.0
        if sold_pct <= 0.0:
            return
        ledger = owner._recovery_ledger_by_symbol.setdefault(context.symbol, self.new_recovery_ledger(owner._call_count))
        old_sold = float(ledger.get("defensive_sold_pct", 0.0) or 0.0)
        new_sold = old_sold + sold_pct
        old_price = float(ledger.get("avg_sell_price", context.price) or context.price)
        ledger["avg_sell_price"] = (old_price * old_sold + context.price * sold_pct) / new_sold if new_sold > 0.0 else context.price
        ledger["defensive_sold_pct"] = new_sold
        ledger["first_sell_call"] = int(ledger.get("first_sell_call", owner._call_count) or owner._call_count)
        ledger["last_sell_call"] = owner._call_count

    def bear_base_floor(self, owner, context: V42Context) -> float:
        return self.base_pct_from_quantity(owner, context, self.base_quantity(owner, context))

    def bear_base_target(self, owner, context: V42Context, regime: V42Regime) -> float:
        recovery_ledger = owner._recovery_ledger_by_symbol.get(context.symbol)
        if not recovery_ledger:
            return 0.0
        base_ledger = owner._base_ledger_by_symbol.get(context.symbol, {})
        sold = float(recovery_ledger.get("defensive_sold_pct", 0.0) or 0.0)
        bought = self.ledger_base_budget_used_pct(base_ledger)
        unrecovered = max(0.0, sold - bought)
        if unrecovered < owner.RECOVERY_MIN_STEP:
            return 0.0

        avg_sell_price = float(recovery_ledger.get("avg_sell_price", 0.0) or 0.0)
        if avg_sell_price <= 0.0:
            return 0.0
        current_drop = self.bear_base_current_discount(context, recovery_ledger)
        location_depth = max(current_drop, self.bear_base_location_depth(owner, context))
        if self.bear_base_path_accelerating(owner, context, regime):
            owner._diag["v4_2_bear_base_accelerating_blocked_count"] += 1
            return 0.0
        if not self.bear_base_low_location(owner, context, regime):
            return 0.0
        proposal = self.bear_base_buy_proposal(owner, context, recovery_ledger, base_ledger, location_depth)
        if not proposal.allowed:
            return 0.0
        raw_target = proposal.target
        floor = self.bear_base_floor(owner, context)

        budget_cap = unrecovered * owner.BEAR_BASE_BUDGET_FRACTION
        asset_cap = self.bear_base_asset_cap(owner, context.symbol)
        target = min(raw_target, budget_cap, asset_cap)
        if target <= floor + owner.RECOVERY_MIN_STEP:
            return 0.0
        return max(0.0, target)

    def bear_base_buy_proposal(
        self,
        owner,
        context: V42Context,
        recovery_ledger: dict,
        base_ledger: dict,
        location_depth: float,
    ) -> V42BearBaseProposal:
        age = owner._call_count - int(recovery_ledger.get("first_sell_call", owner._call_count) or owner._call_count)
        current_drop = self.bear_base_current_discount(context, recovery_ledger)
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        price_vs_ema168 = owner._price_vs(context.latest, context.price, "ema168")
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        atr_rank = owner._value(context.latest, "atr_pct_rank", 0.5)
        held_layer = int(base_ledger.get("base_layer", 0) or 0)
        floor = self.bear_base_floor(owner, context)

        layer = 0
        target = 0.0
        if (
            location_depth >= 0.52
            and age >= 100
            and not pd.isna(rolling_pos)
            and rolling_pos <= 0.18
            and current_drop >= 0.22
            and (pd.isna(roc_20) or roc_20 > -0.20)
            and (pd.isna(atr_rank) or atr_rank <= 0.94)
        ):
            layer = 2
            target = 0.15
        elif (
            location_depth >= 0.38
            and age >= 60
            and (
                (not pd.isna(rolling_pos) and rolling_pos <= 0.30)
                or (not pd.isna(price_vs_ema168) and price_vs_ema168 <= -0.18)
            )
            and current_drop >= 0.12
            and (pd.isna(roc_20) or roc_20 > -0.25)
        ):
            layer = 1
            target = 0.06

        if layer <= 0:
            return V42BearBaseProposal(blocked_reason="no_layer")
        if layer < held_layer:
            return V42BearBaseProposal(blocked_reason="held_deeper_layer")
        if layer == held_layer and floor >= target - owner.RECOVERY_MIN_STEP:
            return V42BearBaseProposal(blocked_reason="layer_filled")
        return V42BearBaseProposal(allowed=True, layer=layer, target=target)

    @staticmethod
    def bear_base_asset_cap(owner, symbol: str) -> float:
        return max(0.0, float(owner.BEAR_BASE_CAP.get(symbol, owner.BEAR_BASE_DEFAULT_CAP)))

    @staticmethod
    def ledger_base_budget_used_pct(ledger: dict) -> float:
        if "base_budget_used_pct" in ledger:
            return max(0.0, float(ledger.get("base_budget_used_pct", 0.0) or 0.0))
        return max(0.0, float(ledger.get("recovery_bought_pct", 0.0) or 0.0))

    def bear_base_exit_target(self, owner, context: V42Context, regime: V42Regime, signals: V42Signals) -> float:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return 0.0
        base = self.bear_base_floor(owner, context)
        if base <= owner.RECOVERY_MIN_STEP:
            return 0.0
        entry_call = ledger.get("base_entry_call")
        if entry_call is None or owner._call_count - int(entry_call) < owner.BEAR_BASE_EXIT_MIN_HOLD_CALLS:
            return base
        avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return base

        profit = context.price / avg_entry - 1.0
        if profit < 1.50:
            return base
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        price_vs_ema168 = regime.price_vs_ema168
        high_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos >= 0.88)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.88)
            or (not pd.isna(price_vs_ema168) and price_vs_ema168 >= 0.42)
        )
        if not high_location:
            return base

        maturity = self.bear_base_maturity_score(owner, context, regime, signals, profit)
        if profit >= 2.50 and maturity >= 4:
            return 0.0
        if profit >= 1.60 and maturity >= 3:
            return max(0.0, min(base, 0.06))
        if maturity >= 4:
            return max(0.0, min(base, 0.07))
        return base

    @staticmethod
    def bear_base_maturity_score(owner, context: V42Context, regime: V42Regime, signals: V42Signals, profit: float) -> int:
        ledger = owner._base_ledger_by_symbol.get(context.symbol, {})
        latest = context.latest
        rolling_pos = owner._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(latest, "donchian_pos", 0.5)
        roc_10 = owner._value(latest, "roc_10", 0.0)
        roc_20 = owner._value(latest, "roc_20", 0.0)
        volume_strength = owner._value(latest, "volume_strength", 1.0)
        atr_rank = regime.atr_rank
        peak_profit = float(ledger.get("base_peak_profit", 0.0) or 0.0)
        entry_call = int(ledger.get("base_entry_call", owner._call_count) or owner._call_count)
        age = owner._call_count - entry_call

        score = 0
        if age >= 300:
            score += 1
        if profit >= 1.60:
            score += 1
        if (
            (not pd.isna(rolling_pos) and rolling_pos >= 0.90)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.92)
            or (not pd.isna(regime.price_vs_ema168) and regime.price_vs_ema168 >= 0.45)
        ):
            score += 1
        if peak_profit >= 1.60 and profit <= peak_profit * 0.82:
            score += 1
        if signals.distribution_exhaustion:
            score += 1
        if not pd.isna(roc_20) and roc_20 <= 0.08 and not pd.isna(donchian_pos) and donchian_pos >= 0.82:
            score += 1
        if not pd.isna(volume_strength) and not pd.isna(roc_10):
            if volume_strength >= 1.25 and roc_10 <= 0.06:
                score += 1
            elif volume_strength <= 0.85 and roc_10 <= 0.02:
                score += 1
        if not pd.isna(atr_rank) and atr_rank >= 0.92 and not pd.isna(donchian_pos) and donchian_pos >= 0.84:
            score += 1
        return score

    @staticmethod
    def bear_base_layer(location_depth: float) -> int:
        if location_depth >= 0.52:
            return 2
        if location_depth >= 0.38:
            return 1
        return 0

    @staticmethod
    def bear_base_current_discount(context: V42Context, recovery_ledger: dict) -> float:
        avg_sell_price = float(recovery_ledger.get("avg_sell_price", 0.0) or 0.0)
        if avg_sell_price <= 0.0:
            return 0.0
        return max(0.0, 1.0 - context.price / avg_sell_price)

    @staticmethod
    def bear_base_low_location(owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        rolling_pos = owner._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(latest, "donchian_pos", 0.5)
        price_vs_ema168 = regime.price_vs_ema168
        return bool(
            (pd.isna(rolling_pos) or rolling_pos <= 0.55)
            and (
                (not pd.isna(donchian_pos) and donchian_pos <= 0.60)
                or (not pd.isna(price_vs_ema168) and price_vs_ema168 <= -0.12)
            )
        )

    @staticmethod
    def bear_base_location_depth(owner, context: V42Context) -> float:
        rolling_pos = owner._value(context.latest, "rolling_365d_pos", 0.5)
        if pd.isna(rolling_pos):
            return 0.0
        return max(0.0, 1.0 - float(rolling_pos))

    @staticmethod
    def bear_base_path_accelerating(owner, context: V42Context, regime: V42Regime) -> bool:
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        atr_rank = regime.atr_rank
        return bool(
            (not pd.isna(roc_20) and roc_20 <= -0.22)
            or (
                not pd.isna(atr_rank)
                and atr_rank >= 0.96
                and not pd.isna(roc_20)
                and roc_20 <= -0.12
            )
        )

    @staticmethod
    def bear_base_path_stabilizing(owner, context: V42Context, regime: V42Regime) -> bool:
        roc_20 = owner._value(context.latest, "roc_20", 0.0)
        atr_rank = regime.atr_rank
        donchian_pos = owner._value(context.latest, "donchian_pos", 0.5)
        return bool(
            (pd.isna(roc_20) or roc_20 >= -0.12)
            and (pd.isna(atr_rank) or atr_rank <= 0.92)
            and (pd.isna(donchian_pos) or donchian_pos >= 0.18)
        )
