"""Recovery-credit release and consumption helpers for V1."""

from __future__ import annotations

import pandas as pd

from ..strategy_types import StrategyContext, StrategyRecoveryPlan, StrategyRegime, StrategySignals, StrategySizing


class RecoveryCreditEngine:
    def recovery_credit_plan(self, owner, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals) -> StrategyRecoveryPlan:
        ledger = owner._recovery_credit_ledger.get(context.symbol)
        if not ledger or float(ledger.get("remaining", 0.0) or 0.0) <= owner.RECOVERY_MIN_STEP:
            return StrategyRecoveryPlan(blocked_reason="no_recovery_credit")
        remaining = float(ledger.get("remaining", 0.0) or 0.0)
        anchor = float(ledger.get("anchor_price", 0.0) or 0.0)
        current_drop = max(0.0, 1.0 - context.price / anchor) if anchor > 0.0 else 0.0
        rolling_pos = owner._value(context.latest, "rolling_365d_pos")
        donchian_pos = owner._value(context.latest, "donchian_pos")
        drop_min = owner.RECOVERY_CREDIT_DROP_MIN.get(context.symbol, 0.04)
        rolling_max = owner.RECOVERY_CREDIT_ROLLING_POS_MAX.get(context.symbol, 0.55)
        donchian_max = owner.RECOVERY_CREDIT_DONCHIAN_POS_MAX.get(context.symbol, 0.60)
        if regime.structural_bear:
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "structural_bear_active", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="structural_bear_active")
        if regime.regime not in {"BULL", "RANGE", "TRANSITION"}:
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "regime_not_allowed", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="regime_not_allowed")
        if regime.btc_regime == "BEAR":
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "btc_bear", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="btc_bear")
        if context.trend_risk > 2:
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "trend_risk_high", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="trend_risk_high")
        if signals.distribution_exhaustion:
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "distribution_exhaustion", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="distribution_exhaustion")
        if not (signals.recovery_signal or signals.recovery_quality_ok or signals.value_recovery):
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "recovery_signal_missing", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="recovery_signal_missing")
        location_filter_ok = bool((pd.isna(rolling_pos) or rolling_pos <= rolling_max) and (pd.isna(donchian_pos) or donchian_pos <= donchian_max))
        location_ok = bool(current_drop >= drop_min and location_filter_ok)
        if not location_ok:
            owner._record_recovery_credit_check(context, regime, signals, ledger, False, "credit_location_hot", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
            return StrategyRecoveryPlan(blocked_reason="credit_location_hot")
        owner._record_recovery_credit_check(context, regime, signals, ledger, True, "", current_drop, rolling_pos, donchian_pos, drop_min, rolling_max, donchian_max)
        return StrategyRecoveryPlan(
            allowed=True,
            target=anchor,
            remaining_budget=remaining,
            guard="core_recovery_credit_soft",
        )

    def recovery_credit_sizing(self, owner, context: StrategyContext, regime: StrategyRegime, signals: StrategySignals) -> StrategySizing:
        plan = self.recovery_credit_plan(owner, context, regime, signals)
        if not bool(plan.get("allowed", False)):
            return StrategySizing(blocked_reason=str(plan.get("blocked_reason", "")))
        credit_before = float(plan.get("remaining_budget", 0.0) or 0.0)
        cooldown = max(owner._buy_cooldown(context, regime, "recovery-probe-buy"), 8)
        last_buy = owner._last_buy_call_by_symbol.get(context.symbol, -10_000)
        cooldown_active = owner._call_count - last_buy < cooldown
        btc_deep_overlay = bool(cooldown_active and self.allow_btc_deep_credit_overlay(owner, context))
        if cooldown_active and not btc_deep_overlay:
            owner._diag["core_cooldown_blocked_count"] += 1
            return StrategySizing(side="buy", setup="recovery-probe-buy", blocked_reason="cooldown")
        release_cap = (
            owner.RECOVERY_CREDIT_BTC_DEEP_RELEASE_CAP
            if btc_deep_overlay
            else owner.RECOVERY_CREDIT_RELEASE_CAP.get(context.symbol, 0.04)
        )
        buy_pct = min(credit_before, release_cap)
        buy_qty = context.total_value * buy_pct / context.price
        if buy_qty <= 1e-12:
            return StrategySizing(side="buy", setup="recovery-probe-buy", blocked_reason="zero_quantity")
        if buy_qty * context.price < owner.min_notional:
            owner._diag["core_min_notional_blocked_count"] += 1
            return StrategySizing(side="buy", setup="recovery-probe-buy", blocked_reason="min_notional")
        guard = owner._join_guard(
            owner._sizing_guard(context, regime, {"state": "NORMAL"}, "recovery-probe-buy"),
            "core_recovery_credit_soft",
        )
        if btc_deep_overlay:
            guard = owner._join_guard(guard, "core_btc_deep_credit_overlay")
        target = min(owner.TARGET_CAP, context.current_pct + buy_pct)
        sizing = StrategySizing(
            side="buy",
            setup="recovery-probe-buy",
            quantity=buy_qty,
            target=target,
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=context.current_pct + buy_pct,
            target_gap_before=buy_pct,
            actual_step_pct=buy_pct,
            remaining_gap_after=0.0,
        )
        setattr(sizing, "recovery_credit_before", credit_before)
        setattr(sizing, "recovery_credit_used", buy_pct)
        setattr(sizing, "recovery_credit_after", max(0.0, credit_before - buy_pct))
        setattr(sizing, "recovery_credit_anchor_price", float(plan.get("target", 0.0) or 0.0))
        return sizing

    @staticmethod
    def track_main_buy_for_credit(owner, context: StrategyContext, sizing: StrategySizing) -> None:
        if sizing.setup not in owner.RECOVERY_CREDIT_CONSUME_MAIN_BUY_SET:
            return
        if "core_recovery_credit_soft" in str(sizing.guard):
            return
        owner._last_main_buy_for_credit_by_symbol[context.symbol] = {
            "call": owner._call_count,
            "setup": sizing.setup,
            "step_pct": max(0.0, float(sizing.actual_step_pct or 0.0)),
            "price": float(context.price),
        }

    def consume_recovery_credit_from_current_buy(
        self,
        owner,
        context: StrategyContext,
        regime: StrategyRegime,
        signals: StrategySignals | None,
        sizing: StrategySizing,
    ) -> None:
        if sizing.setup not in owner.RECOVERY_CREDIT_CONSUME_MAIN_BUY_SET:
            return
        if "core_recovery_credit_soft" in str(sizing.guard):
            return
        if sizing.actual_step_pct <= 0.0:
            return
        self.consume_recovery_credit_by_main_buy(
            owner,
            context=context,
            regime=regime,
            signals=signals,
            buy_call=owner._call_count,
            buy_setup=sizing.setup,
            buy_step_pct=float(sizing.actual_step_pct),
            guard_suffix="current_main_buy",
        )

    def consume_recovery_credit_from_recent_buy(self, owner, context: StrategyContext, regime: StrategyRegime) -> None:
        last = owner._last_main_buy_for_credit_by_symbol.get(context.symbol)
        if not last:
            return
        buy_call = int(last.get("call", -10_000) or -10_000)
        if owner._call_count - buy_call > owner.RECOVERY_CREDIT_CONSUME_LOOKBACK_CALLS:
            return
        self.consume_recovery_credit_by_main_buy(
            owner,
            context=context,
            regime=regime,
            signals=None,
            buy_call=buy_call,
            buy_setup=str(last.get("setup", "")),
            buy_step_pct=float(last.get("step_pct", 0.0) or 0.0),
            guard_suffix="recent_main_buy",
        )

    def consume_recovery_credit_by_main_buy(
        self,
        owner,
        *,
        context: StrategyContext,
        regime: StrategyRegime,
        signals: StrategySignals | None,
        buy_call: int,
        buy_setup: str,
        buy_step_pct: float,
        guard_suffix: str,
    ) -> None:
        if buy_step_pct <= 0.0:
            return
        key = (context.symbol, buy_call)
        if key in owner._credit_consumed_buy_calls:
            return
        active_signals = signals if signals is not None else owner._build_signals(context, regime, {"state": "NORMAL"})
        plan = self.recovery_credit_plan(owner, context, regime, active_signals)
        if not bool(plan.get("allowed", False)):
            return
        ledger = owner._recovery_credit_ledger.get(context.symbol)
        if not ledger:
            return
        before = float(ledger.get("remaining", 0.0) or 0.0)
        release_cap = owner.RECOVERY_CREDIT_RELEASE_CAP.get(context.symbol, 0.04)
        used = min(before, release_cap, buy_step_pct)
        if used <= owner.RECOVERY_MIN_STEP:
            return
        after = max(0.0, before - used)
        ledger["remaining"] = after
        owner._recovery_credit_ledger[context.symbol] = ledger
        owner._credit_consumed_buy_calls.add(key)
        owner._record_recovery_credit_event(
            symbol=context.symbol,
            event="credit_consumed_by_main_buy",
            episode={"episode_id": ledger.get("episode_id", "")},
            source_close_reason=str(ledger.get("source_close_reason", "")),
            credit_before=before,
            credit_delta=-used,
            credit_after=after,
            anchor_price=float(ledger.get("anchor_price", 0.0) or 0.0),
            guard=f"core_recovery_credit_consumed_by_{guard_suffix}_{buy_setup}",
            blocked_reason="",
        )

    def release_recovery_credit(self, owner, context: StrategyContext, sizing: StrategySizing) -> None:
        ledger = owner._recovery_credit_ledger.get(context.symbol)
        episode_id = str((ledger or {}).get("episode_id", ""))
        if not ledger:
            return
        before = float(ledger.get("remaining", 0.0) or 0.0)
        used = min(before, max(0.0, float(getattr(sizing, "recovery_credit_used", 0.0) or 0.0)))
        after = max(0.0, before - used)
        ledger["remaining"] = after
        owner._recovery_credit_ledger[context.symbol] = ledger
        owner._record_recovery_credit_event(
            symbol=context.symbol,
            event="credit_released",
            episode={"episode_id": ledger.get("episode_id", "")},
            source_close_reason=str(ledger.get("source_close_reason", "")),
            credit_before=before,
            credit_delta=-used,
            credit_after=after,
            anchor_price=float(ledger.get("anchor_price", 0.0) or 0.0),
            guard=str(sizing.guard),
            blocked_reason="",
        )
        if context.symbol == "BTC/USDT" and "core_btc_deep_credit_overlay" in str(sizing.guard) and episode_id:
            owner._btc_deep_overlay_used_episode_ids.add(episode_id)

    @staticmethod
    def allow_btc_deep_credit_overlay(owner, context: StrategyContext) -> bool:
        if context.symbol != "BTC/USDT":
            return False
        ledger = owner._recovery_credit_ledger.get(context.symbol)
        if not ledger:
            return False
        anchor = float(ledger.get("anchor_price", 0.0) or 0.0)
        if anchor <= 0.0:
            return False
        current_drop = max(0.0, 1.0 - context.price / anchor)
        if current_drop < owner.RECOVERY_CREDIT_BTC_DEEP_DROP_MIN:
            return False
        last = owner._last_main_buy_for_credit_by_symbol.get(context.symbol)
        if not last or str(last.get("setup", "")) != "value-recovery":
            return False
        buy_call = int(last.get("call", -10_000) or -10_000)
        if not bool(0 <= owner._call_count - buy_call <= owner.RECOVERY_CREDIT_BTC_DEEP_LOOKBACK_CALLS):
            return False
        episode_id = str(ledger.get("episode_id", ""))
        return bool(episode_id and episode_id not in owner._btc_deep_overlay_used_episode_ids)
