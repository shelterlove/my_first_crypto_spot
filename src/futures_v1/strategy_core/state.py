"""State bootstrap for the frozen Official V1 strategy."""

from __future__ import annotations

from ..lifecycle_policy import LifecyclePolicy


DIAGNOSTIC_KEYS = (
    "core_context_built_count",
    "core_regime_bull_count",
    "core_regime_range_count",
    "core_regime_transition_count",
    "core_regime_bear_count",
    "core_intent_accumulate_count",
    "core_intent_hold_count",
    "core_intent_defend_count",
    "core_intent_distribute_count",
    "core_intent_exit_count",
    "core_action_buy_count",
    "core_action_sell_count",
    "core_starter_buy_count",
    "core_value_recovery_count",
    "core_trend_cont_count",
    "core_recovery_probe_buy_count",
    "core_defense_sell_count",
    "core_structural_exit_sell_count",
    "core_distribution_sell_count",
    "core_sizing_blocked_count",
    "core_cooldown_blocked_count",
    "core_min_notional_blocked_count",
    "core_defense_recovery_blocked_count",
    "core_distribution_reentry_blocked_count",
    "core_failed_recovery_probe_blocked_count",
    "core_recovery_path_buy_count",
    "core_limited_recovery_overlay_count",
    "core_limited_recovery_overlay_blocked_count",
    "core_deep_base_recovery_count",
    "core_post_crash_recoil_count",
    "core_bear_base_buy_count",
    "core_bear_base_exit_count",
    "core_bear_base_layer1_buy_count",
    "core_bear_base_layer2_buy_count",
    "core_bear_base_layer1_exit_count",
    "core_bear_base_layer2_exit_count",
    "core_bear_base_accelerating_blocked_count",
    "core_bear_base_floor_sell_protected_count",
    "core_recovery_test_sell_blocked_count",
    "core_episode_started_count",
    "core_episode_superseded_count",
    "core_episode_closed_count",
    "core_base_intent_accumulate_count",
    "core_base_intent_deferred_count",
    "core_base_deferred_by_main_defense_count",
    "core_main_sell_base_buy_conflict_count",
    "core_would_force_base_exit_count",
)


class StrategyStateMixin:
    def _init_state(self, initial_capital: float, reserve: float, fee_rate: float) -> None:
        self.initial_capital = initial_capital
        self.reserve = reserve
        self.fee_rate = fee_rate
        self.min_notional = 10.0

        self._call_count = 0
        self._last_buy_call_by_symbol: dict[str, int] = {}
        self._last_sell_call_by_symbol: dict[str, int] = {}
        self._peak_price_by_symbol: dict[str, float] = {}
        self._state_by_symbol: dict[str, dict] = {}
        self._episodes_by_symbol: dict[str, dict] = {}
        self._recovery_ledger_by_symbol: dict[str, dict] = {}
        self._base_ledger_by_symbol: dict[str, dict] = {}
        self._init_sleeve_accounting()

        self._base_deferred_candidates: list[dict] = []
        self._decision_trace: list[dict] = []
        self._candidate_order_trace: list[dict] = []
        self._risk_assessment_shadow: list[dict] = []
        self._intent_plan_shadow: list[dict] = []
        self._target_vector_shadow: list[dict] = []
        self._budget_ledger_shadow: list[dict] = []
        self._order_arbiter_shadow: list[dict] = []
        self._symbol_policy_shadow: list[dict] = []
        self._recovery_state_machine_shadow: list[dict] = []
        self._episode_log: list[dict] = []
        self._next_episode_id = 1
        self._diag: dict[str, int] = {key: 0 for key in DIAGNOSTIC_KEYS}

        self._recovery_credit_ledger: dict[str, dict] = {}
        self._recovery_credit_events: list[dict] = []
        self._recovery_credit_checks: list[dict] = []
        self._current_context_by_symbol: dict = {}
        self._last_main_buy_for_credit_by_symbol: dict[str, dict] = {}
        self._credit_consumed_buy_calls: set[tuple[str, int]] = set()
        self._btc_deep_overlay_used_episode_ids: set[str] = set()

        self._protected_floor_ledger_by_symbol: dict[str, dict] = {}
        self._current_regime_for_protected_floor_by_symbol: dict = {}
        self._last_opportunity_floor_buy_call_by_symbol: dict[str, int] = {}
        self._last_base_led_recovery_buy_call_by_symbol: dict[str, int] = {}

        self._lifecycle_state_shadow: list[dict] = []
        self._last_lifecycle_low_base_buy_call_by_symbol: dict[str, int] = {}
        self._lifecycle_policy = LifecyclePolicy(
            target_cap=self.TARGET_CAP,
            target_table=self.TARGET_TABLE,
            recovery_min_step=self.RECOVERY_MIN_STEP,
        )

        self._outer_overlay_state_by_symbol: dict[str, dict] = {}
        self._outer_overlay_last_eval: dict[str, tuple[int, float, str]] = {}
        self._outer_overlay_events: list[dict] = []
        self._outer_qty_last_eval: dict[str, tuple[int, float, str]] = {}

    @property
    def deployable_capital(self) -> float:
        return self.initial_capital

    def strategy_diagnostics(self) -> dict[str, int]:
        return dict(self._diag)

    def strategy_defense_episodes(self) -> list[dict]:
        rows = [self._episode_row(row) for row in self._episode_log]
        rows.extend(self._episode_row(row) for row in self._episodes_by_symbol.values())
        return rows

    def strategy_risk_cycles(self) -> list[dict]:
        return [row for row in self.strategy_defense_episodes() if row.get("setup") in {"defense-sell", "structural-exit-sell"}]

    def strategy_base_deferred_candidates(self) -> list[dict]:
        return list(self._base_deferred_candidates)

    def strategy_decision_trace(self) -> list[dict]:
        return list(self._decision_trace)

    def strategy_candidate_orders(self) -> list[dict]:
        return list(self._candidate_order_trace)

    def strategy_risk_assessment_shadow(self) -> list[dict]:
        return list(self._risk_assessment_shadow)

    def strategy_intent_plan_shadow(self) -> list[dict]:
        return list(self._intent_plan_shadow)

    def strategy_target_vector_shadow(self) -> list[dict]:
        return list(self._target_vector_shadow)

    def strategy_budget_ledger_shadow(self) -> list[dict]:
        return list(self._budget_ledger_shadow)

    def strategy_order_arbiter_shadow(self) -> list[dict]:
        return list(self._order_arbiter_shadow)

    def strategy_symbol_policy_shadow(self) -> list[dict]:
        return list(self._symbol_policy_shadow)

    def strategy_recovery_state_machine_shadow(self) -> list[dict]:
        return list(self._recovery_state_machine_shadow)

    def strategy_recovery_credit_events(self) -> list[dict]:
        return list(self._recovery_credit_events)

    def strategy_recovery_credit_checks(self) -> list[dict]:
        return list(self._recovery_credit_checks)

    def strategy_lifecycle_state_shadow(self) -> list[dict]:
        return list(self._lifecycle_state_shadow)

    def strategy_outer_overlay_events(self) -> list[dict]:
        return list(self._outer_overlay_events)
