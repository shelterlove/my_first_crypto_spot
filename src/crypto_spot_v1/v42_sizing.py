from __future__ import annotations

import pandas as pd

from .v42_types import V42Context, V42DecisionPlan, V42RecoveryPlan, V42Regime, V42Signals, V42Sizing, V42TargetPlan


class V42SizingMixin:
    def _compute_sizing(self, context: V42Context, regime: V42Regime, episode: dict, signals: V42Signals, decision: V42DecisionPlan) -> V42Sizing:
        current_pct = context.current_pct
        total_value = context.total_value
        price = context.price
        symbol = context.symbol
        intent = decision.intent
        target = decision.target
        if price <= 0.0 or total_value <= 0.0:
            return V42Sizing(blocked_reason="bad_price_or_value")

        if intent in {"EXIT", "DEFEND", "DISTRIBUTE"}:
            return self._sell_sizing(context, regime, episode, signals, decision)
        if intent == "HOLD":
            overlay = self._hold_recovery_overlay_sizing(context, regime, episode, signals)
            if overlay.side or overlay.blocked_reason:
                return overlay
            return V42Sizing()
        if intent != "ACCUMULATE":
            return V42Sizing()
        buy_threshold = self.RECOVERY_MIN_STEP if str(episode.get("state", "NORMAL")) != "NORMAL" else self.MIN_ADJUST_THRESHOLD
        if current_pct >= target - buy_threshold:
            return V42Sizing(blocked_reason="target_reached")

        setup = str(decision.primary_sleeve.setup if decision.primary_sleeve is not None else "")
        if not setup:
            setup = self._buy_setup(context, regime, episode, signals)
        blocked = self._buy_block_reason(context, regime, episode, signals, setup, decision.target_plan)
        if blocked:
            return V42Sizing(side="buy", setup=setup, blocked_reason=blocked)

        cooldown = self._buy_cooldown(context, regime, setup)
        last_buy = self._last_buy_call_by_symbol.get(symbol, -10_000)
        if self._call_count - last_buy < cooldown:
            self._diag["v4_2_cooldown_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="cooldown")

        gap = max(0.0, target - current_pct)
        max_buy = self._max_buy_pct(context, regime, episode, setup, signals)
        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup=setup, blocked_reason="zero_quantity")
        if buy_qty * price < self.min_notional:
            self._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="min_notional")
        return V42Sizing(
            side="buy",
            setup=setup,
            quantity=buy_qty,
            target=target,
            guard=self._buy_sizing_guard(
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

    def _hold_recovery_overlay_sizing(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
    ) -> V42Sizing:
        plan = self._hold_recovery_overlay_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            blocked = str(plan.get("blocked_reason", "limited_recovery_overlay_blocked"))
            if blocked not in {"insufficient_sell_to_low_drawdown", "recovery_signal_missing"}:
                self._diag["v4_2_limited_recovery_overlay_blocked_count"] += 1
            return V42Sizing()

        setup = "recovery-probe-buy"
        target = min(float(plan.get("target", context.current_pct) or context.current_pct), self._episode_recovery_cap(episode))
        if context.current_pct >= target - self.RECOVERY_MIN_STEP:
            return V42Sizing(side="buy", setup=setup, blocked_reason="target_reached")

        cooldown = max(self._buy_cooldown(context, regime, setup), 8)
        last_buy = self._last_buy_call_by_symbol.get(context.symbol, -10_000)
        if self._call_count - last_buy < cooldown:
            self._diag["v4_2_cooldown_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="cooldown")

        gap = max(0.0, target - context.current_pct)
        max_buy = min(float(plan.get("max_buy", 0.0) or 0.0), 0.08)
        buy_pct = min(gap, max_buy)
        buy_qty = context.total_value * buy_pct / context.price
        if buy_qty <= 1e-12:
            return V42Sizing(side="buy", setup=setup, blocked_reason="zero_quantity")
        if buy_qty * context.price < self.min_notional:
            self._diag["v4_2_min_notional_blocked_count"] += 1
            return V42Sizing(side="buy", setup=setup, blocked_reason="min_notional")

        self._diag["v4_2_limited_recovery_overlay_count"] += 1
        overlay_guard = "v4_2_limited_recovery_overlay"
        guard = self._join_guard(self._sizing_guard(context, regime, episode, setup), overlay_guard)
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

    def _hold_recovery_overlay_plan(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
    ) -> V42RecoveryPlan:
        state = str(episode.get("state", "NORMAL"))
        if state not in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK"}:
            return V42RecoveryPlan(blocked_reason="not_overlay_episode")
        if signals.recovery_signal:
            return V42RecoveryPlan(blocked_reason="recovery_signal_present")
        return self._episode_recovery_plan(context, regime, episode, signals)

    def _sell_sizing(
        self,
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
            setup = self._sell_setup(intent)
        base_exit_sell = bool(decision.target_plan is not None and decision.target_plan.base_exit_distribute)
        guard = self._sell_sizing_guard(
            context,
            regime,
            episode,
            signals,
            setup,
            base_exit_sell,
            decision.primary_sleeve.guard if decision.primary_sleeve is not None else "",
        )
        blocked = self._sell_block_reason(context, regime, episode, signals, intent, setup, base_exit_sell)
        if blocked:
            return V42Sizing(side="sell", setup=setup, blocked_reason=blocked)
        if current_pct <= target + self.MIN_ADJUST_THRESHOLD:
            return V42Sizing(side="sell", setup=setup, blocked_reason="target_reached")
        gap = current_pct - target
        max_sell = {"EXIT": 0.65, "DEFEND": 0.28, "DISTRIBUTE": 0.10}.get(intent, 0.20)
        if intent == "EXIT":
            max_sell = self._structural_exit_max_sell(context, regime, episode)
        if intent == "DEFEND" and context.risk_score >= 4:
            max_sell = 0.40
        sell_pct = min(gap, max_sell)
        sleeve_qty = self._base_quantity(context) if base_exit_sell else self._main_quantity(context)
        sell_qty = min(total_value * sell_pct / price, float(pos.quantity), sleeve_qty)
        if sell_qty <= 1e-12:
            blocked_reason = "base_quantity_unavailable" if base_exit_sell else "main_quantity_unavailable"
            return V42Sizing(side="sell", setup=setup, blocked_reason=blocked_reason)
        if sell_qty * price < self.min_notional:
            self._diag["v4_2_min_notional_blocked_count"] += 1
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

    def _structural_exit_max_sell(self, context: V42Context, regime: V42Regime, episode: dict) -> float:
        base = 1.0 if context.trend_risk >= 3 else 0.65
        return base

    def _sell_block_reason(
        self,
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
        blocked = self._recovery_test_sell_block_reason(context, regime, episode, signals)
        if not blocked:
            return ""
        self._diag["v4_2_recovery_test_sell_blocked_count"] += 1
        return blocked

    def _recovery_test_sell_block_reason(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
    ) -> str:
        if str(episode.get("state", "NORMAL")) != "RECOVERY_TEST":
            return ""
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        buy_price = float(episode.get("last_recovery_buy_price", context.price) or context.price)
        recovery_start = int(episode.get("recovery_start_call", self._call_count) or self._call_count)
        age = self._call_count - recovery_start
        failure = bool(
            regime.structural_bear
            or context.trend_risk >= 3
            or (regime.btc_regime == "BEAR" and context.price < buy_price * 0.94)
            or (sell_price > 0.0 and context.price < sell_price * 0.88 and context.trend_risk >= 2)
        )
        profit_take = bool(
            signals.distribution_exhaustion
            and buy_price > 0.0
            and context.price >= buy_price * 1.10
        )
        if failure or profit_take:
            return ""
        if age < self.RECOVERY_TEST_CALLS:
            return "recovery_test_observation"
        return ""

    def _buy_block_reason(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        setup: str,
        target_plan: V42TargetPlan | None = None,
    ) -> str:
        state = str(episode.get("state", "NORMAL"))
        base_accumulate_needed = bool(
            target_plan.base_accumulate_needed if target_plan is not None else self._bear_base_accumulate_needed(context, regime, episode, signals)
        )
        if base_accumulate_needed:
            return ""
        if regime.regime == "BEAR" and setup != "recovery-probe-buy":
            return "bear_regime"
        if state in {"DEFENSE_LOCK", "RECOVERY_TEST", "FAILED_RECOVERY_LOCK", "STRUCTURAL_BEAR_LOCK"}:
            plan = self._episode_recovery_plan(context, regime, episode, signals)
            if not bool(plan.get("allowed", False)):
                if state in {"DEFENSE_LOCK", "RECOVERY_TEST"}:
                    self._diag["v4_2_defense_recovery_blocked_count"] += 1
                return str(plan.get("blocked_reason", "recovery_path_blocked"))
        if state == "DEFENSE_LOCK":
            age = self._call_count - int(episode.get("start_call", self._call_count))
            if age < self.DEFENSE_MIN_RECOVERY_CALLS:
                self._diag["v4_2_defense_recovery_blocked_count"] += 1
                return "defense_cooldown"
            if not signals.recovery_signal:
                self._diag["v4_2_defense_recovery_blocked_count"] += 1
                return "recovery_signal_missing"
        if state == "DISTRIBUTION_LOCK":
            plan = self._episode_reentry_plan(context, regime, episode, signals)
            if not bool(plan.get("allowed", False)):
                self._diag["v4_2_distribution_reentry_blocked_count"] += 1
                return str(plan.get("blocked_reason", "distribution_reentry_blocked"))
        if state == "FAILED_RECOVERY_LOCK":
            signal_count = self._advance_failed_recovery_signal_count(episode)
            if signal_count <= 0 and not signals.strong_recovery_signal:
                self._diag["v4_2_failed_recovery_probe_blocked_count"] += 1
                return "first_failed_recovery_signal_blocked"
        return ""

    @staticmethod
    def _advance_failed_recovery_signal_count(episode: dict) -> int:
        signal_count = int(episode.get("recovery_signal_count", 0))
        episode["recovery_signal_count"] = signal_count + 1
        return signal_count

    def _buy_cooldown(self, context: V42Context, regime: V42Regime, setup: str) -> int:
        base = self.BUY_COOLDOWN.get(regime.regime, 8)
        if setup == "starter-buy":
            return max(2, base // 2)
        if setup == "recovery-probe-buy":
            return max(4, base)
        if setup == "trend-cont":
            return max(2, base // 2 + context.risk_score)
        return base + context.risk_score * 2

    def _max_buy_pct(self, context: V42Context, regime: V42Regime, episode: dict, setup: str, signals: V42Signals | None = None) -> float:
        if context.symbol == "BTC/USDT" and setup == "trend-cont":
            mode = str(getattr(self, "BTC_TREND_CONT_EXPERIMENT", "baseline"))
            if mode == "off":
                return 0.0
            if mode == "after_value" and not bool(episode.get("had_value_recovery", False)):
                return 0.0
        base = self._setup_base_buy_pct(setup)
        if context.symbol == "BTC/USDT" and setup == "trend-cont" and str(getattr(self, "BTC_TREND_CONT_EXPERIMENT", "baseline")) == "half":
            base *= 0.5
        if regime.regime == "TRANSITION":
            base = min(base, self.TRANSITION_BUY_MAX_PCT)
        if regime.regime == "BEAR":
            base = min(base, self.BEAR_BUY_MAX_PCT)
        if str(episode.get("state", "NORMAL")) in {
            "DEFENSE_LOCK",
            "RECOVERY_TEST",
            "FAILED_RECOVERY_LOCK",
            "STRUCTURAL_BEAR_LOCK",
        }:
            base = max(base, self._episode_recovery_max_buy(context, regime, episode, setup, signals))
        elif setup == "recovery-probe-buy":
            base = min(base, self._episode_recovery_add_cap(episode))
        atr_rank = regime.get("atr_rank")
        if not pd.isna(atr_rank) and float(atr_rank) >= 0.90:
            base *= 0.65
        if setup == "trend-cont" and self.ENABLE_LATE_TREND_SOFT_CAP and self._late_trend_continuation_risk(context, regime):
            base = min(base, self.LATE_TREND_CONT_MAX_BUY_PCT)
        return max(0.0, base)

    @staticmethod
    def _setup_base_buy_pct(setup: str) -> float:
        return {
            "starter-buy": 0.24,
            "value-recovery": 0.18,
            "trend-cont": 0.22,
            "recovery-probe-buy": 0.06,
        }.get(setup, 0.10)

    def _buy_sizing_guard(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        setup: str,
        target_plan: V42TargetPlan | None = None,
        sleeve_guard: str = "",
    ) -> str:
        guard = self._sizing_guard(context, regime, episode, setup)
        base_accumulate_needed = bool(
            target_plan.base_accumulate_needed if target_plan is not None else self._bear_base_accumulate_needed(context, regime, episode, signals)
        )
        if base_accumulate_needed:
            guard = self._join_guard(guard, "v4_2_bear_base")
        if sleeve_guard and not base_accumulate_needed:
            return self._join_guard(guard, sleeve_guard)
        if str(episode.get("state", "NORMAL")) not in {
            "DEFENSE_LOCK",
            "RECOVERY_TEST",
            "FAILED_RECOVERY_LOCK",
            "STRUCTURAL_BEAR_LOCK",
        }:
            return guard
        plan = self._episode_recovery_plan(context, regime, episode, signals)
        if not bool(plan.get("allowed", False)):
            return guard
        return self._join_guard(guard, str(plan.get("guard", "")))

    def _sell_sizing_guard(
        self,
        context: V42Context,
        regime: V42Regime,
        episode: dict,
        signals: V42Signals,
        setup: str,
        base_exit_sell: bool = False,
        sleeve_guard: str = "",
    ) -> str:
        guard = self._sizing_guard(context, regime, episode, setup)
        if base_exit_sell:
            guard = self._join_guard(guard, sleeve_guard or "v4_2_bear_base_exit")
        return guard

    def _sizing_guard(self, context: V42Context, regime: V42Regime, episode: dict, setup: str) -> str:
        parts = [
            f"v4_2_intent_{self._setup_intent_name(setup)}",
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
                parts.append(self._recovery_drawdown_guard(max(0.0, 1.0 - lowest / sell_price)))
        return "-".join(parts)

    def _setup_intent_name(self, setup: str) -> str:
        if setup in {"defense-sell"}:
            return "defend"
        if setup == "structural-exit-sell":
            return "exit"
        if setup in {"distribution-sell", "bear-base-exit"}:
            return "distribute"
        return "accumulate"
