from __future__ import annotations

import pandas as pd

from .strategy_rebalance import Action
from .strategy_types import StrategyBearBaseProposal, StrategyContext, StrategyRegime, StrategySignals, StrategySizing


class StrategyBearBaseMixin:
    def _base_quantity(self, context: StrategyContext) -> float:
        ledger = self._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return 0.0
        quantity = max(0.0, float(ledger.get("base_quantity", 0.0) or 0.0))
        return min(quantity, max(0.0, float(context.pos.quantity)))

    def _main_quantity(self, context: StrategyContext) -> float:
        return max(0.0, float(context.pos.quantity) - self._base_quantity(context))

    def _base_pct_from_quantity(self, context: StrategyContext, quantity: float) -> float:
        if context.total_value <= 0.0 or context.price <= 0.0:
            return 0.0
        return max(0.0, min(context.current_pct, float(quantity) * context.price / context.total_value))

    def _refresh_base_ledger_market(self, context: StrategyContext) -> None:
        ledger = self._base_ledger_by_symbol.get(context.symbol)
        if not ledger or self._base_quantity(context) <= 0.0:
            return
        avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return
        peak_price = max(float(ledger.get("base_peak_price", 0.0) or 0.0), context.price)
        ledger["base_peak_price"] = peak_price
        ledger["base_peak_profit"] = max(0.0, peak_price / avg_entry - 1.0)
        ledger["base_position_pct"] = self._base_pct_from_quantity(context, self._base_quantity(context))

    def _record_ledger_sell(self, context: StrategyContext, sizing: StrategySizing) -> None:
        sold_pct = sizing.quantity * context.price / context.total_value if context.total_value > 0.0 else 0.0
        if sold_pct <= 0.0:
            return
        ledger = self._recovery_ledger_by_symbol.setdefault(context.symbol, self._new_recovery_ledger())
        old_sold = float(ledger.get("defensive_sold_pct", 0.0) or 0.0)
        new_sold = old_sold + sold_pct
        old_price = float(ledger.get("avg_sell_price", context.price) or context.price)
        ledger["avg_sell_price"] = (old_price * old_sold + context.price * sold_pct) / new_sold if new_sold > 0.0 else context.price
        ledger["defensive_sold_pct"] = new_sold
        ledger["first_sell_call"] = int(ledger.get("first_sell_call", self._call_count) or self._call_count)
        ledger["last_sell_call"] = self._call_count

    def _new_recovery_ledger(self) -> dict:
        return {
            "defensive_sold_pct": 0.0,
            "recovery_bought_pct": 0.0,
            "avg_sell_price": 0.0,
            "first_sell_call": self._call_count,
            "last_sell_call": self._call_count,
        }

    def _new_base_ledger(self) -> dict:
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

    def _record_bear_base_buy(self, context: StrategyContext, action: Action) -> None:
        recovery_ledger = self._recovery_ledger_by_symbol.get(context.symbol)
        if recovery_ledger is None:
            return
        ledger = self._base_ledger_by_symbol.setdefault(context.symbol, self._new_base_ledger())
        buy_pct = action.quantity * action.price / context.total_value if context.total_value > 0.0 else 0.0
        buy_pct = max(0.0, buy_pct)
        old_base_qty = float(ledger.get("base_quantity", 0.0) or 0.0)
        old_entry = float(ledger.get("base_avg_entry_price", context.price) or context.price)
        new_base_qty = old_base_qty + float(action.quantity)
        if new_base_qty > 0.0:
            ledger["base_avg_entry_price"] = (old_entry * old_base_qty + context.price * float(action.quantity)) / new_base_qty
        base_budget_used = self._ledger_base_budget_used_pct(ledger) + buy_pct
        ledger["base_budget_used_pct"] = base_budget_used
        ledger["base_quantity"] = new_base_qty
        ledger["base_position_pct"] = self._base_pct_from_quantity(context, new_base_qty)
        if ledger.get("base_entry_call") is None:
            ledger["base_entry_call"] = self._call_count
            ledger["base_peak_price"] = context.price
            ledger["base_peak_profit"] = 0.0
        proposal = self._bear_base_buy_proposal(context, self._recovery_ledger_by_symbol[context.symbol], ledger, self._bear_base_location_depth(context))
        layer = int(proposal.layer)
        ledger["base_layer"] = max(int(ledger.get("base_layer", 0) or 0), layer)
        if layer == 1:
            self._diag["core_bear_base_layer1_buy_count"] += 1
        elif layer == 2:
            self._diag["core_bear_base_layer2_buy_count"] += 1

    def _record_base_exit_sell(self, context: StrategyContext, action: Action) -> None:
        ledger = self._base_ledger_by_symbol.get(context.symbol)
        if ledger is None:
            return
        base_qty = self._base_quantity(context)
        if base_qty <= 0.0 or action.quantity <= 0.0:
            return
        layer = int(ledger.get("base_layer", 0) or 0)
        ledger["base_quantity"] = max(0.0, base_qty - float(action.quantity))
        ledger["base_position_pct"] = self._base_pct_from_quantity(context, float(ledger.get("base_quantity", 0.0) or 0.0))
        if float(ledger.get("base_quantity", 0.0) or 0.0) <= 1e-12:
            ledger["base_quantity"] = 0.0
            ledger["base_position_pct"] = 0.0
            ledger["base_avg_entry_price"] = 0.0
            ledger["base_entry_call"] = None
            ledger["base_layer"] = 0
            ledger["base_peak_price"] = 0.0
            ledger["base_peak_profit"] = 0.0
        self._diag["core_bear_base_exit_count"] += 1
        if layer == 1:
            self._diag["core_bear_base_layer1_exit_count"] += 1
        elif layer == 2:
            self._diag["core_bear_base_layer2_exit_count"] += 1

    def _bear_base_floor(self, context: StrategyContext) -> float:
        return self._base_pct_from_quantity(context, self._base_quantity(context))

    def _bear_base_target(self, context: StrategyContext, regime: StrategyRegime) -> float:
        recovery_ledger = self._recovery_ledger_by_symbol.get(context.symbol)
        if not recovery_ledger:
            return 0.0
        base_ledger = self._base_ledger_by_symbol.get(context.symbol, {})
        sold = float(recovery_ledger.get("defensive_sold_pct", 0.0) or 0.0)
        bought = self._ledger_base_budget_used_pct(base_ledger)
        unrecovered = max(0.0, sold - bought)
        if unrecovered < self.RECOVERY_MIN_STEP:
            return 0.0

        avg_sell_price = float(recovery_ledger.get("avg_sell_price", 0.0) or 0.0)
        if avg_sell_price <= 0.0:
            return 0.0
        current_drop = self._bear_base_current_discount(context, recovery_ledger)
        location_depth = max(current_drop, self._bear_base_location_depth(context))
        if self._bear_base_path_accelerating(context, regime):
            self._diag["core_bear_base_accelerating_blocked_count"] += 1
            return 0.0
        if not self._bear_base_low_location(context, regime):
            return 0.0
        proposal = self._bear_base_buy_proposal(context, recovery_ledger, base_ledger, location_depth)
        if not proposal.allowed:
            return 0.0
        layer = proposal.layer
        raw_target = proposal.target
        floor = self._bear_base_floor(context)

        budget_cap = unrecovered * self.BEAR_BASE_BUDGET_FRACTION
        asset_cap = self._bear_base_asset_cap(context.symbol)
        target = min(raw_target, budget_cap, asset_cap)
        if target <= floor + self.RECOVERY_MIN_STEP:
            return 0.0
        return max(0.0, target)

    def _bear_base_buy_proposal(
        self,
        context: StrategyContext,
        recovery_ledger: dict,
        base_ledger: dict,
        location_depth: float,
    ) -> StrategyBearBaseProposal:
        age = self._call_count - int(recovery_ledger.get("first_sell_call", self._call_count) or self._call_count)
        current_drop = self._bear_base_current_discount(context, recovery_ledger)
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        price_vs_ema168 = self._price_vs(context.latest, context.price, "ema168")
        roc_20 = self._value(context.latest, "roc_20", 0.0)
        atr_rank = self._value(context.latest, "atr_pct_rank", 0.5)
        held_layer = int(base_ledger.get("base_layer", 0) or 0)
        floor = self._bear_base_floor(context)

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
            return StrategyBearBaseProposal(blocked_reason="no_layer")
        if layer < held_layer:
            return StrategyBearBaseProposal(blocked_reason="held_deeper_layer")
        if layer == held_layer and floor >= target - self.RECOVERY_MIN_STEP:
            return StrategyBearBaseProposal(blocked_reason="layer_filled")
        return StrategyBearBaseProposal(allowed=True, layer=layer, target=target)

    def _bear_base_asset_cap(self, symbol: str) -> float:
        return max(0.0, float(self.BEAR_BASE_CAP.get(symbol, self.BEAR_BASE_DEFAULT_CAP)))

    @staticmethod
    def _ledger_base_budget_used_pct(ledger: dict) -> float:
        if "base_budget_used_pct" in ledger:
            return max(0.0, float(ledger.get("base_budget_used_pct", 0.0) or 0.0))
        return max(0.0, float(ledger.get("recovery_bought_pct", 0.0) or 0.0))

    def _bear_base_exit_target(self, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals) -> float:
        ledger = self._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return 0.0
        base = self._bear_base_floor(context)
        if base <= self.RECOVERY_MIN_STEP:
            return 0.0
        entry_call = ledger.get("base_entry_call")
        if entry_call is None or self._call_count - int(entry_call) < self.BEAR_BASE_EXIT_MIN_HOLD_CALLS:
            return base
        avg_entry = float(ledger.get("base_avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0.0:
            return base

        profit = context.price / avg_entry - 1.0
        if profit < 1.50:
            return base
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(context.latest, "donchian_pos", 0.5)
        price_vs_ema168 = regime.price_vs_ema168
        high_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos >= 0.88)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.88)
            or (not pd.isna(price_vs_ema168) and price_vs_ema168 >= 0.42)
        )
        if not high_location:
            return base

        maturity = self._bear_base_maturity_score(context, regime, signals, profit)
        if profit >= 2.50 and maturity >= 4:
            return 0.0
        if profit >= 1.60 and maturity >= 3:
            return max(0.0, min(base, 0.06))
        if maturity >= 4:
            return max(0.0, min(base, 0.07))
        return base

    def _bear_base_maturity_score(self, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals, profit: float) -> int:
        ledger = self._base_ledger_by_symbol.get(context.symbol, {})
        latest = context.latest
        rolling_pos = self._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(latest, "donchian_pos", 0.5)
        roc_10 = self._value(latest, "roc_10", 0.0)
        roc_20 = self._value(latest, "roc_20", 0.0)
        volume_strength = self._value(latest, "volume_strength", 1.0)
        atr_rank = regime.atr_rank
        peak_profit = float(ledger.get("base_peak_profit", 0.0) or 0.0)
        entry_call = int(ledger.get("base_entry_call", self._call_count) or self._call_count)
        age = self._call_count - entry_call

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
    def _bear_base_layer(context: StrategyContext, location_depth: float) -> int:
        if location_depth >= 0.52:
            return 2
        if location_depth >= 0.38:
            return 1
        return 0

    @staticmethod
    def _bear_base_current_discount(context: StrategyContext, recovery_ledger: dict) -> float:
        avg_sell_price = float(recovery_ledger.get("avg_sell_price", 0.0) or 0.0)
        if avg_sell_price <= 0.0:
            return 0.0
        return max(0.0, 1.0 - context.price / avg_sell_price)

    def _bear_base_low_location(self, context: StrategyContext, regime: StrategyRegime) -> bool:
        latest = context.latest
        rolling_pos = self._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(latest, "donchian_pos", 0.5)
        price_vs_ema168 = regime.price_vs_ema168
        return bool(
            (pd.isna(rolling_pos) or rolling_pos <= 0.55)
            and (
                (not pd.isna(donchian_pos) and donchian_pos <= 0.60)
                or (not pd.isna(price_vs_ema168) and price_vs_ema168 <= -0.12)
            )
        )

    def _bear_base_location_depth(self, context: StrategyContext) -> float:
        rolling_pos = self._value(context.latest, "rolling_365d_pos", 0.5)
        if pd.isna(rolling_pos):
            return 0.0
        return max(0.0, 1.0 - float(rolling_pos))

    def _bear_base_path_accelerating(self, context: StrategyContext, regime: StrategyRegime) -> bool:
        roc_20 = self._value(context.latest, "roc_20", 0.0)
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

    def _bear_base_path_stabilizing(self, context: StrategyContext, regime: StrategyRegime) -> bool:
        roc_20 = self._value(context.latest, "roc_20", 0.0)
        atr_rank = regime.atr_rank
        donchian_pos = self._value(context.latest, "donchian_pos", 0.5)
        return bool(
            (pd.isna(roc_20) or roc_20 >= -0.12)
            and (pd.isna(atr_rank) or atr_rank <= 0.92)
            and (pd.isna(donchian_pos) or donchian_pos >= 0.18)
        )

