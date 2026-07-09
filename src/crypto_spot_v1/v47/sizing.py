"""Base sizing rules for V4.7 raw decisions."""

from __future__ import annotations

import pandas as pd

from ..v42_types import V42Context, V42DecisionPlan, V42Regime, V42Signals, V42Sizing, V42TargetPlan


class V47SizingEngine:
    def compute_sizing(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals, decision: V42DecisionPlan) -> V42Sizing:
        sizing = self._compute_protected_floor_stack(owner, context, regime, episode, signals, decision)
        if context.symbol == "BNB/USDT" and sizing.side == "sell" and sizing.setup == "bear-base-exit":
            strategic_qty = min(owner._source_quantity(context, "strategic_base"), owner._base_quantity(context))
            if strategic_qty <= 1e-12:
                return V42Sizing()
            if sizing.quantity > strategic_qty:
                sizing.quantity = strategic_qty
                actual_sell_pct = sizing.quantity * context.price / context.total_value if context.total_value > 0.0 else 0.0
                sizing.actual_step_pct = -actual_sell_pct
                sizing.actual_position_after = max(0.0, context.current_pct - actual_sell_pct)
                sizing.remaining_gap_after = sizing.target - sizing.actual_position_after
                sizing.guard = owner._join_guard(str(sizing.guard), "v4_4_source_limited_base_exit")

        if owner._route_eth_recovery_buy_to_base(context, regime, episode, signals, sizing):
            sizing.setup = "base-led-recovery-buy"
            sizing.guard = owner._join_guard(str(sizing.guard), "v4_4_eth_recovery_to_base")
            from ..v42_types import V42SleevePlan

            decision.primary_sleeve = V42SleevePlan(
                sleeve="base-led-recovery",
                side="buy",
                setup="base-led-recovery-buy",
                target=sizing.target,
                guard=sizing.guard,
                priority=82,
                allowed=True,
            )
            owner._diag["v4_4_eth_recovery_to_base_count"] = owner._diag.get("v4_4_eth_recovery_to_base_count", 0) + 1

        if sizing.side == "sell" and sizing.quantity > 1e-12:
            return sizing
        if bool(getattr(owner, "ENABLE_LIFECYCLE_BASE_EXIT_BEHAVIOR", False)):
            base_exit = owner._lifecycle_protected_base_exit_sizing(context, regime, signals, decision)
            if base_exit.side:
                return base_exit
        if sizing.side == "buy" and sizing.quantity > 1e-12:
            return sizing
        if bool(getattr(owner, "ENABLE_LIFECYCLE_LOW_BASE_BEHAVIOR", False)):
            low_base = owner._lifecycle_low_base_sizing(context, regime, episode, signals, decision, sizing)
            if low_base.side:
                return low_base
        return sizing

    def _compute_protected_floor_stack(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        protected_exit = owner._protected_floor_exit_sizing(context, regime, signals, decision)
        if protected_exit.side and protected_exit.quantity > 1e-12:
            return protected_exit

        sizing = self._compute_recovery_credit_stack(owner, context, regime, episode, signals, decision)
        if sizing.side == "sell":
            return sizing
        if sizing.side == "buy" and sizing.quantity > 1e-12:
            return sizing
        floor = owner._opportunity_floor_sizing(context, regime, signals, sizing)
        if floor.side or floor.blocked_reason:
            return floor if floor.side else sizing
        return sizing

    def _compute_recovery_credit_stack(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        owner._decay_recovery_credit(context)
        sizing = self.base_compute_sizing(owner, context, regime, episode, signals, decision)
        if sizing.side or str(sizing.blocked_reason or "") not in {"", "target_reached"}:
            return sizing
        if str(getattr(decision, "main_intent", decision.intent)) not in {"HOLD", "ACCUMULATE"}:
            return sizing
        credit_sizing = owner._recovery_credit_sizing(context, regime, signals)
        if credit_sizing.side or credit_sizing.blocked_reason:
            return credit_sizing if credit_sizing.side else sizing
        return sizing

    def base_compute_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        current_pct = context.current_pct
        total_value = context.total_value
        price = context.price
        symbol = context.symbol
        intent = decision.intent
        target = decision.target
        if price <= 0.0 or total_value <= 0.0:
            return V42Sizing(blocked_reason="bad_price_or_value")

        if intent in {"EXIT", "DEFEND", "DISTRIBUTE"}:
            return owner._sell_sizing(context, regime, episode, signals, decision)
        if intent == "HOLD":
            overlay = owner._hold_recovery_overlay_sizing(context, regime, episode, signals)
            if overlay.side or overlay.blocked_reason:
                return overlay
            return V42Sizing()
        if intent != "ACCUMULATE":
            return V42Sizing()
        buy_threshold = owner.RECOVERY_MIN_STEP if str(episode.get("state", "NORMAL")) != "NORMAL" else owner.MIN_ADJUST_THRESHOLD
        if current_pct >= target - buy_threshold:
            return V42Sizing(blocked_reason="target_reached")

        setup = str(decision.primary_sleeve.setup if decision.primary_sleeve is not None else "")
        if not setup:
            setup = owner._buy_setup(context, regime, episode, signals)
        blocked = owner._buy_block_reason(context, regime, episode, signals, setup, decision.target_plan)
        if blocked:
            return V42Sizing(side="buy", setup=setup, blocked_reason=blocked)

        cooldown = owner._buy_cooldown(context, regime, setup)
        last_buy = owner._last_buy_call_by_symbol.get(symbol, -10_000)
        if owner._call_count - last_buy < cooldown:
            owner._diag["v4_2_cooldown_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="cooldown")

        gap = max(0.0, target - current_pct)
        max_buy = owner._max_buy_pct(context, regime, episode, setup, signals)
        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup=setup, blocked_reason="zero_quantity")
        if buy_qty * price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="min_notional")
        return V42Sizing(
            side="buy",
            setup=setup,
            quantity=buy_qty,
            target=target,
            guard=owner._buy_sizing_guard(
                context,
                regime,
                episode,
                signals,
                setup,
                decision.target_plan,
                decision.primary_sleeve.guard if decision.primary_sleeve is not None else "",
            ),
            actual_position_before=current_pct,
            actual_position_after=current_pct + buy_pct,
            target_gap_before=target - current_pct,
            actual_step_pct=buy_pct,
            remaining_gap_after=target - current_pct - buy_pct,
        )

    def hold_recovery_overlay_sizing(self, owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> V42Sizing:
        plan = owner._hold_recovery_overlay_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            blocked = str(plan.get("blocked_reason", "limited_recovery_overlay_blocked"))
            if blocked not in {"insufficient_sell_to_low_drawdown", "recovery_signal_missing"}:
                owner._diag["v4_2_limited_recovery_overlay_blocked_count"] += 1
            return V42Sizing()

        setup = "recovery-probe-buy"
        target = min(float(plan.get("target", context.current_pct) or context.current_pct), owner._episode_recovery_cap(episode))
        if context.current_pct >= target - owner.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup=setup, blocked_reason="target_reached")

        cooldown = max(owner._buy_cooldown(context, regime, setup), 8)
        last_buy = owner._last_buy_call_by_symbol.get(context.symbol, -10_000)
        if owner._call_count - last_buy < cooldown:
            owner._diag["v4_2_cooldown_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="cooldown")

        gap = max(0.0, target - context.current_pct)
        max_buy = min(float(plan.get("max_buy", 0.0) or 0.0), 0.08)
        buy_pct = min(gap, max_buy)
        buy_qty = context.total_value * buy_pct / context.price
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup=setup, blocked_reason="zero_quantity")
        if buy_qty * context.price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="min_notional")

        owner._diag["v4_2_limited_recovery_overlay_count"] += 1
        overlay_guard = "v4_2_limited_recovery_overlay"
        guard = owner._join_guard(owner._sizing_guard(context, regime, episode, setup), overlay_guard)
        return V42Sizing(
            side="buy",
            setup=setup,
            quantity=buy_qty,
            target=target,
            guard=guard,
            actual_position_before=context.current_pct,
            actual_position_after=context.current_pct + buy_pct,
            target_gap_before=target - context.current_pct,
            actual_step_pct=buy_pct,
            remaining_gap_after=target - context.current_pct - buy_pct,
        )

    def sell_sizing(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        decision: V42DecisionPlan,
    ) -> V42Sizing:
        current_pct = context.current_pct
        total_value = context.total_value
        price = context.price
        pos = context.pos
        intent = decision.intent
        target = decision.target
        setup = str(decision.primary_sleeve.setup if decision.primary_sleeve is not None else "")
        if not setup:
            setup = owner._sell_setup(intent)
        base_exit_sell = bool(decision.target_plan is not None and decision.target_plan.base_exit_distribute)
        guard = owner._sell_sizing_guard(
            context,
            regime,
            episode,
            signals,
            setup,
            base_exit_sell,
            decision.primary_sleeve.guard if decision.primary_sleeve is not None else "",
        )
        blocked = owner._sell_block_reason(context, regime, episode, signals, intent, setup, base_exit_sell)
        if blocked:
            return V42Sizing(side="sell", setup=setup, blocked_reason=blocked)
        if current_pct <= target + owner.MIN_ADJUST_THRESHOLD:
            return V42Sizing(side="sell", setup=setup, blocked_reason="target_reached")
        gap = current_pct - target
        max_sell = {"EXIT": 0.65, "DEFEND": 0.28, "DISTRIBUTE": 0.10}.get(intent, 0.20)
        if intent == "EXIT":
            max_sell = owner._structural_exit_max_sell(context, regime, episode)
        if intent == "DEFEND" and context.risk_score >= 4:
            max_sell = 0.40
        sell_pct = min(gap, max_sell)
        sleeve_qty = owner._base_quantity(context) if base_exit_sell else owner._main_quantity(context)
        sell_qty = min(total_value * sell_pct / price, float(pos.quantity), sleeve_qty)
        if sell_qty <= 1e-12:
            blocked_reason = "base_quantity_unavailable" if base_exit_sell else "main_quantity_unavailable"
            return V42Sizing(side="sell", setup=setup, blocked_reason=blocked_reason)
        if sell_qty * price < owner.min_notional:
            owner._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="sell", setup=setup, blocked_reason="min_notional")
        actual_sell_pct = sell_qty * price / total_value if total_value > 0.0 else 0.0
        return V42Sizing(
            side="sell",
            setup=setup,
            quantity=sell_qty,
            target=target,
            guard=guard,
            actual_position_before=current_pct,
            actual_position_after=max(0.0, current_pct - actual_sell_pct),
            target_gap_before=target - current_pct,
            actual_step_pct=-actual_sell_pct,
            remaining_gap_after=target - max(0.0, current_pct - actual_sell_pct),
        )

    @staticmethod
    def structural_exit_max_sell(context: V42Context, regime: V42Regime, episode: dict) -> float:
        return 1.0 if context.trend_risk >= 3 else 0.65

    def sell_block_reason(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        intent: str,
        setup: str,
        base_exit_sell: bool = False,
    ) -> str:
        if base_exit_sell:
            return ""
        if str(episode.get("state", "NORMAL")) != "RECOVERY_TEST":
            return ""
        blocked = owner._recovery_test_sell_block_reason(context, regime, episode, signals)
        if not blocked:
            return ""
        owner._diag["v4_2_recovery_test_sell_blocked_count"] += 1
        return blocked

    @staticmethod
    def recovery_test_sell_block_reason(owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals) -> str:
        if str(episode.get("state", "NORMAL")) != "RECOVERY_TEST":
            return ""
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        buy_price = float(episode.get("last_recovery_buy_price", context.price) or context.price)
        recovery_start = int(episode.get("recovery_start_call", owner._call_count) or owner._call_count)
        age = owner._call_count - recovery_start
        failure = bool(
            regime.structural_bear
            or context.trend_risk >= 3
            or (regime.btc_regime == "BEAR" and context.price < buy_price * 0.94)
            or (sell_price > 0.0 and context.price < sell_price * 0.88 and context.trend_risk >= 2)
        )
        profit_take = bool(signals.distribution_exhaustion and buy_price > 0.0 and context.price >= buy_price * 1.10)
        if failure or profit_take:
            return ""
        if age < owner.RECOVERY_TEST_CALLS:
            return "recovery_test_observation"
        return ""

    def buy_block_reason(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        setup: str,
        target_plan: V42TargetPlan | None = None,
    ) -> str:
        state = str(episode.get("state", "NORMAL"))
        base_accumulate_needed = bool(
            target_plan.base_accumulate_needed if target_plan is not None else owner._bear_base_accumulate_needed(context, regime, episode, signals)
        )
        if base_accumulate_needed:
            return ""
        if regime.regime == "BEAR" and setup != "recovery-probe-buy":
            return "bear_regime"
        if state in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            plan = owner._episode_recovery_plan(context, regime, episode, signals)
            if not bool(plan.get("allowed", False)):
                if state in {"DEFENSE_LOCK", "RECOVERY_TEST"}:
                    owner._diag["v4_2_defense_recovery_blocked_count"] += 1
                return str(plan.get("blocked_reason", "recovery_path_blocked"))
        if state == "DEFENSE_LOCK":
            age = owner._call_count - int(episode.get("start_call", owner._call_count))
            if age < owner.DEFENSE_MIN_RECOVERY_CALLS:
                owner._diag["v4_2_defense_recovery_blocked_count"] += 1
                return "defense_cooldown"
            if not signals.recovery_signal:
                owner._diag["v4_2_defense_recovery_blocked_count"] += 1
                return "recovery_signal_missing"
        if state == "DISTRIBUTION_LOCK":
            plan = owner._episode_reentry_plan(context, regime, episode, signals)
            if not bool(plan.get("allowed", False)):
                owner._diag["v4_2_distribution_reentry_blocked_count"] += 1
                return str(plan.get("blocked_reason", "distribution_reentry_blocked"))
        if state == "FAILED_RECOVERY_LOCK":
            signal_count = self.advance_failed_recovery_signal_count(episode)
            if signal_count <= 0 and not signals.strong_recovery_signal:
                owner._diag["v4_2_failed_recovery_probe_blocked_count"] += 1
                return "first_failed_recovery_signal_blocked"
        return ""

    @staticmethod
    def advance_failed_recovery_signal_count(episode: dict) -> int:
        signal_count = int(episode.get("recovery_signal_count", 0))
        episode["recovery_signal_count"] = signal_count + 1
        return signal_count

    @staticmethod
    def buy_cooldown(owner, context: V42Context, regime: V42Regime, setup: str) -> int:
        base = owner.BUY_COOLDOWN.get(regime.regime, 8)
        if setup == "starter-buy":
            return max(2, base // 2)
        if setup == "recovery-probe-buy":
            return max(4, base)
        if setup == "trend-cont":
            return max(2, base // 2 + context.risk_score)
        return base + context.risk_score * 2

    def max_buy_pct(self, owner, context: V42Context, regime: V42Regime, episode: dict, setup: str, signals: V42Signals | None = None) -> float:
        if context.symbol == "BTC/USDT" and setup == "trend-cont":
            mode = str(getattr(owner, "BTC_TREND_CONT_EXPERIMENT", "baseline"))
            if mode == "off":
                return 0.0
            if mode == "after_value" and not bool(episode.get("had_value_recovery", False)):
                return 0.0
        base = self.setup_base_buy_pct(setup)
        if context.symbol == "BTC/USDT" and setup == "trend-cont" and str(getattr(owner, "BTC_TREND_CONT_EXPERIMENT", "baseline")) == "half":
            base *= 0.5
        if regime.regime == "TRANSITION":
            base = min(base, owner.TRANSITION_BUY_MAX_PCT)
        if regime.regime == "BEAR":
            base = min(base, owner.BEAR_BUY_MAX_PCT)
        if str(episode.get("state", "NORMAL")) in {
            "DEFENSE_LOCK",
            "RECOVERY_TEST",
            "FAILED_RECOVERY_LOCK",
            "STRUCTURAL_BEAR_LOCK",
        }:
            base = max(base, owner._episode_recovery_max_buy(context, regime, episode, setup, signals))
        elif setup == "recovery-probe-buy":
            base = min(base, owner._episode_recovery_add_cap(episode))
        atr_rank = regime.get("atr_rank")
        if not pd.isna(atr_rank) and float(atr_rank) >= 0.90:
            base *= 0.65
        if setup == "trend-cont" and owner.ENABLE_LATE_TREND_SOFT_CAP and owner._late_trend_continuation_risk(context, regime):
            base = min(base, owner.LATE_TREND_CONT_MAX_BUY_PCT)
        return max(0.0, base)

    @staticmethod
    def setup_base_buy_pct(setup: str) -> float:
        return {
            "starter-buy": 0.24,
            "value-recovery": 0.18,
            "trend-cont": 0.22,
            "recovery-probe-buy": 0.06,
        }.get(setup, 0.10)

    def buy_sizing_guard(
        self,
        owner,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        setup: str,
        target_plan: V42TargetPlan | None = None,
        sleeve_guard: str = "",
    ) -> str:
        guard = owner._sizing_guard(context, regime, episode, setup)
        base_accumulate_needed = bool(
            target_plan.base_accumulate_needed if target_plan is not None else owner._bear_base_accumulate_needed(context, regime, episode, signals)
        )
        if base_accumulate_needed:
            guard = owner._join_guard(guard, "v4_2_bear_base")
        if sleeve_guard and not base_accumulate_needed:
            return owner._join_guard(guard, sleeve_guard)
        if str(episode.get("state", "NORMAL")) not in {
            "DEFENSE_LOCK",
            "RECOVERY_TEST",
            "FAILED_RECOVERY_LOCK",
            "STRUCTURAL_BEAR_LOCK",
        }:
            return guard
        plan = owner._episode_recovery_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return guard
        return owner._join_guard(guard, str(plan.get("guard", "")))

    @staticmethod
    def sell_sizing_guard(owner, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals, setup: str, base_exit_sell: bool = False, sleeve_guard: str = "") -> str:
        guard = owner._sizing_guard(context, regime, episode, setup)
        if base_exit_sell:
            guard = owner._join_guard(guard, sleeve_guard or "v4_2_bear_base_exit")
        return guard

    @staticmethod
    def sizing_guard(owner, context: V42Context, regime: V42Regime, episode: dict, setup: str) -> str:
        parts = [
            f"v4_2_intent_{owner._setup_intent_name(setup)}",
            f"v4_2_regime_{regime.regime.lower()}",
        ]
        state = str(episode.get("state", "NORMAL"))
        if state != "NORMAL":
            parts.append(f"v4_2_episode_{state.lower()}")
        if setup == "recovery-probe-buy":
            parts.append("v4_2_staged_recovery")
        if state in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            sell_price = float(episode.get("sell_price", 0.0) or 0.0)
            lowest = float(episode.get("lowest_price", context.price) or context.price)
            if sell_price > 0.0 and lowest > 0.0:
                parts.append(owner._recovery_drawdown_guard(max(0.0, 1.0 - lowest / sell_price)))
        return "-".join(parts)

    @staticmethod
    def setup_intent_name(setup: str) -> str:
        if setup in {"defense-sell"}:
            return "defend"
        if setup == "structural-exit-sell":
            return "exit"
        if setup in {"distribution-sell", "bear-base-exit"}:
            return "distribute"
        return "accumulate"
