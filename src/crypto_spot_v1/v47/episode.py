"""Risk-cycle episode and recovery-credit ledger for V4.7."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..v42_types import V42Context, V42Regime, V42Signals, V42Sizing


class V47EpisodeOwner(Protocol):
    EPISODE_MAX_CALLS: int
    FAILED_RECOVERY_CALLS: int
    RECOVERY_TEST_CALLS: int
    DISTRIBUTION_LOCK_CALLS: int
    RECOVERY_MIN_STEP: float
    RECOVERY_CONFIRM_MIN_BUDGET_RATIO: float
    RECOVERY_CREDIT_ALLOWED_CLOSE_REASONS: set[str] | None
    RECOVERY_CREDIT_SYMBOL_CAP: float
    RECOVERY_CREDIT_PORTFOLIO_AVG_CAP: float
    RECOVERY_CREDIT_DECAY_CALLS: int
    RECOVERY_CREDIT_DECAY_FRACTION: float
    TARGET_ALLOC: dict[str, float]
    _call_count: int
    _episodes_by_symbol: dict[str, dict]
    _episode_log: list[dict]
    _diag: dict[str, int]
    _recovery_credit_ledger: dict[str, dict]
    _recovery_credit_events: list[dict]
    _recovery_credit_checks: list[dict]
    _current_context_by_symbol: dict[str, V42Context]

    def _recovery_quality_ok(self, context: V42Context, regime: V42Regime) -> bool:
        ...

    def _failed_recovery(self, context: V42Context, episode: dict) -> bool:
        ...

    def _distribution_exhaustion(self, context: V42Context, regime: V42Regime) -> bool:
        ...

    def _close_episode(self, symbol: str, reason: str) -> None:
        ...


class V47EpisodeEngine:
    """Owns risk-cycle state and recovery-credit ledger side effects."""

    def update_episode(self, owner: V47EpisodeOwner, context: V42Context, regime: V42Regime) -> dict:
        owner._current_context_by_symbol[context.symbol] = context
        symbol = context.symbol
        price = context.price
        episode = owner._episodes_by_symbol.get(symbol)
        if episode is not None:
            episode["lowest_price"] = min(float(episode.get("lowest_price", price)), price)
            episode["cycle_lowest_price"] = min(float(episode.get("cycle_lowest_price", episode.get("lowest_price", price)) or price), price)
            episode["last_call"] = owner._call_count
            if owner._call_count - int(episode.get("start_call", owner._call_count)) > owner.EPISODE_MAX_CALLS:
                owner._close_episode(symbol, "expired")
                episode = None

        if episode is None:
            return {"state": "NORMAL"}

        state = str(episode.get("state", "NORMAL"))
        age = owner._call_count - int(episode.get("start_call", owner._call_count))
        if state == "RECOVERY_TEST":
            recovery_start = int(episode.get("recovery_start_call", episode.get("start_call", owner._call_count)) or owner._call_count)
            recovery_age = owner._call_count - recovery_start
            if owner._recovery_quality_ok(context, regime) and recovery_age >= owner.RECOVERY_TEST_CALLS:
                owner._close_episode(symbol, "recovery_confirmed")
                return {"state": "NORMAL"}
            if regime.regime == "BEAR" or owner._failed_recovery(context, episode):
                episode["state"] = "FAILED_RECOVERY_LOCK"
                episode["blocked_reason"] = join_guard(str(episode.get("blocked_reason", "")), "failed_recovery")
        elif state == "FAILED_RECOVERY_LOCK":
            if age >= owner.FAILED_RECOVERY_CALLS and owner._recovery_quality_ok(context, regime):
                episode["state"] = "RECOVERY_TEST"
        elif state == "DISTRIBUTION_LOCK":
            if age >= owner.DISTRIBUTION_LOCK_CALLS and not owner._distribution_exhaustion(context, regime):
                owner._close_episode(symbol, "distribution_cooled")
                return {"state": "NORMAL"}
        elif state == "STRUCTURAL_BEAR_LOCK":
            if age >= owner.FAILED_RECOVERY_CALLS and owner._recovery_quality_ok(context, regime):
                episode["state"] = "RECOVERY_TEST"

        return episode

    def close_episode(self, owner: V47EpisodeOwner, symbol: str, reason: str) -> None:
        episode = owner._episodes_by_symbol.pop(symbol, None)
        if episode is None:
            return
        budget = cycle_recovery_budget(episode)
        recovered = cycle_recovered_pct(episode)
        open_budget = max(0.0, budget - recovered)
        ratio = recovered / budget if budget > 1e-12 else 1.0
        reconciled = bool(open_budget <= owner.RECOVERY_MIN_STEP or ratio >= owner.RECOVERY_CONFIRM_MIN_BUDGET_RATIO)
        if reason == "recovery_confirmed" and not reconciled:
            reason = "recovery_signal_confirmed_budget_open"
            owner._diag["v4_2_recovery_signal_confirmed_budget_open_count"] = (
                owner._diag.get("v4_2_recovery_signal_confirmed_budget_open_count", 0) + 1
            )
        episode["budget_recovery_ratio"] = ratio
        episode["budget_open_pct"] = open_budget
        episode["budget_reconciled"] = reconciled
        episode["close_reason"] = reason
        episode["close_call"] = owner._call_count
        episode["state"] = "CLOSED"
        allowed_reasons = owner.RECOVERY_CREDIT_ALLOWED_CLOSE_REASONS
        source_allowed = allowed_reasons is None or reason in allowed_reasons
        if open_budget > owner.RECOVERY_MIN_STEP and source_allowed:
            self.open_recovery_credit(owner, symbol, episode, open_budget, reason)
        owner._episode_log.append(dict(episode))
        owner._diag["v4_2_episode_closed_count"] += 1

    def append_episode_sell_leg(self, owner, context: V42Context, episode: dict, sizing: V42Sizing) -> None:
        prior_budget = float(episode.get("cumulative_sold_pct", episode.get("recovery_budget_pct", 0.0)) or 0.0)
        prior_recovered = float(episode.get("cumulative_recovered_pct", episode.get("recovery_bought_pct", 0.0)) or 0.0)
        prior_unrecovered = max(0.0, prior_budget - prior_recovered)
        prior_id = str(episode.get("prior_episode_id") or episode.get("episode_id", ""))

        sold_pct = sizing.quantity * context.price / context.total_value if context.total_value > 0.0 else 0.0
        post_sell_pct = max(0.0, context.current_pct - sold_pct)
        prior_sold = float(episode.get("cumulative_sold_pct", episode.get("sold_pct", 0.0)) or 0.0)
        cumulative_sold = prior_sold + sold_pct
        first_price = float(episode.get("first_sell_price", context.price) or context.price)
        avg_price = float(episode.get("avg_sell_price", context.price) or context.price)
        episode["setup"] = sizing.setup
        episode["state"] = {
            "defense-sell": "DEFENSE_LOCK",
            "structural-exit-sell": "STRUCTURAL_BEAR_LOCK",
        }.get(sizing.setup, str(episode.get("state", "DEFENSE_LOCK")))
        episode["start_call"] = owner._call_count
        episode["sell_price"] = context.price
        episode["lowest_price"] = context.price
        episode["sell_position_pct"] = context.current_pct
        episode["sell_target_pct"] = float(sizing.get("target", 0.0) or 0.0)
        episode["sell_risk_score"] = context.risk_score
        episode["sell_trend_risk"] = context.trend_risk
        episode["sell_drawdown_risk"] = context.drawdown_risk
        episode["last_sell_price"] = context.price
        episode["avg_sell_price"] = (avg_price * prior_sold + context.price * sold_pct) / cumulative_sold if cumulative_sold > 0.0 else context.price
        episode["first_sell_price"] = first_price
        episode["sell_count"] = int(episode.get("sell_count", 1) or 1) + 1
        episode["last_sell_call"] = owner._call_count
        episode["last_call"] = owner._call_count
        episode["sold_pct"] = cumulative_sold
        episode["cumulative_sold_pct"] = cumulative_sold
        episode["post_sell_pct"] = post_sell_pct
        episode["min_position_pct"] = min(float(episode.get("min_position_pct", post_sell_pct) or post_sell_pct), post_sell_pct)
        episode["cycle_lowest_price"] = min(float(episode.get("cycle_lowest_price", context.price) or context.price), context.price)
        episode["recovery_budget_pct"] = sold_pct
        episode["recovery_bought_pct"] = 0.0
        recovered = float(episode.get("cumulative_recovered_pct", episode.get("recovery_bought_pct", 0.0)) or 0.0)
        episode["unrecovered_budget_pct"] = max(0.0, cumulative_sold - recovered)
        legs = list(episode.get("sell_legs", []))
        legs.append({
            "call": owner._call_count,
            "setup": sizing.setup,
            "price": context.price,
            "sold_pct": sold_pct,
            "position_before_pct": context.current_pct,
            "position_after_pct": post_sell_pct,
            "target_pct": float(sizing.get("target", 0.0) or 0.0),
        })
        episode["sell_legs"] = legs

        episode["prior_episode_id"] = prior_id
        episode["prior_unrecovered_budget_pct"] = prior_unrecovered
        episode["prior_sold_pct"] = prior_budget
        episode["prior_recovery_bought_pct"] = prior_recovered

    def open_recovery_credit(self, owner: V47EpisodeOwner, symbol: str, episode: dict, open_budget: float, reason: str) -> None:
        before = float(owner._recovery_credit_ledger.get(symbol, {}).get("remaining", 0.0) or 0.0)
        symbol_room = max(0.0, owner.RECOVERY_CREDIT_SYMBOL_CAP - before)
        portfolio_room = max(0.0, self.recovery_credit_portfolio_cap(owner) - self.recovery_credit_total(owner))
        delta = min(open_budget, symbol_room, portfolio_room)
        anchor = float(episode.get("avg_sell_price", episode.get("sell_price", 0.0)) or 0.0)
        if delta > 0.0:
            ledger = dict(owner._recovery_credit_ledger.get(symbol, {}))
            prior_anchor = float(ledger.get("anchor_price", anchor) or anchor)
            prior_remaining = before
            after = before + delta
            ledger.update({
                "remaining": after,
                "anchor_price": (
                    (prior_anchor * prior_remaining + anchor * delta) / after
                    if after > 0.0 and prior_anchor > 0.0 and anchor > 0.0
                    else anchor or prior_anchor
                ),
                "opened_call": int(ledger.get("opened_call", owner._call_count) or owner._call_count),
                "last_decay_call": owner._call_count,
                "episode_id": episode.get("episode_id", ""),
                "source_close_reason": reason,
            })
            source_ids = list(ledger.get("source_episode_ids", []) or [])
            source_ids.append(episode.get("episode_id", ""))
            ledger["source_episode_ids"] = [item for item in source_ids if item]
            owner._recovery_credit_ledger[symbol] = ledger
            self.record_recovery_credit_event(owner, symbol=symbol, event="credit_opened", episode=episode, source_close_reason=reason, credit_before=before, credit_delta=delta, credit_after=after, anchor_price=float(ledger.get("anchor_price", 0.0) or 0.0), guard="", blocked_reason="")
        if open_budget - delta > 1e-12:
            self.record_recovery_credit_event(owner, symbol=symbol, event="credit_capped", episode=episode, source_close_reason=reason, credit_before=before + delta, credit_delta=0.0, credit_after=before + delta, anchor_price=anchor, guard="", blocked_reason="credit_cap")

    def decay_recovery_credit(self, owner: V47EpisodeOwner, context: V42Context) -> None:
        ledger = owner._recovery_credit_ledger.get(context.symbol)
        if not ledger:
            return
        remaining = float(ledger.get("remaining", 0.0) or 0.0)
        if remaining <= owner.RECOVERY_MIN_STEP:
            return
        last_decay = int(ledger.get("last_decay_call", ledger.get("opened_call", owner._call_count)) or owner._call_count)
        periods = (owner._call_count - last_decay) // owner.RECOVERY_CREDIT_DECAY_CALLS
        if periods <= 0:
            return
        before = remaining
        for _ in range(periods):
            remaining *= 1.0 - owner.RECOVERY_CREDIT_DECAY_FRACTION
        ledger["remaining"] = remaining
        ledger["last_decay_call"] = last_decay + periods * owner.RECOVERY_CREDIT_DECAY_CALLS
        owner._recovery_credit_ledger[context.symbol] = ledger
        self.record_recovery_credit_event(owner, symbol=context.symbol, event="credit_decayed", episode={"episode_id": ledger.get("episode_id", "")}, source_close_reason=str(ledger.get("source_close_reason", "")), credit_before=before, credit_delta=remaining - before, credit_after=remaining, anchor_price=float(ledger.get("anchor_price", 0.0) or 0.0), guard="", blocked_reason="")

    def record_recovery_credit_event(
        self,
        owner: V47EpisodeOwner,
        *,
        symbol: str,
        event: str,
        episode: dict,
        source_close_reason: str,
        credit_before: float,
        credit_delta: float,
        credit_after: float,
        anchor_price: float,
        guard: str,
        blocked_reason: str,
    ) -> None:
        context = owner._current_context_by_symbol.get(symbol)
        timestamp = context.latest.get("timestamp") if context is not None else None
        owner._recovery_credit_events.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "event": event,
            "episode_id": episode.get("episode_id", ""),
            "source_close_reason": source_close_reason,
            "credit_before": float(credit_before),
            "credit_delta": float(credit_delta),
            "credit_after": float(credit_after),
            "anchor_price": float(anchor_price),
            "guard": guard,
            "blocked_reason": blocked_reason,
        })

    def record_recovery_credit_check(
        self,
        owner: V47EpisodeOwner,
        context: V42Context,
        regime: V42Regime,
        signals: V42Signals,
        ledger: dict,
        allowed: bool,
        blocked_reason: str,
        current_drop: float,
        rolling_pos: float,
        donchian_pos: float,
        drop_min: float,
        rolling_max: float,
        donchian_max: float,
    ) -> None:
        owner._recovery_credit_checks.append({
            "timestamp": context.latest.get("timestamp"),
            "symbol": context.symbol,
            "allowed": bool(allowed),
            "blocked_reason": blocked_reason,
            "remaining": float(ledger.get("remaining", 0.0) or 0.0),
            "anchor_price": float(ledger.get("anchor_price", 0.0) or 0.0),
            "price": float(context.price),
            "current_drop": float(current_drop),
            "drop_min": float(drop_min),
            "rolling_365d_pos": float(rolling_pos) if not pd.isna(rolling_pos) else float("nan"),
            "rolling_pos_max": float(rolling_max),
            "donchian_pos": float(donchian_pos) if not pd.isna(donchian_pos) else float("nan"),
            "donchian_pos_max": float(donchian_max),
            "regime": regime.regime,
            "btc_regime": regime.btc_regime,
            "trend_risk": int(context.trend_risk),
            "risk_score": int(context.risk_score),
            "recovery_signal": bool(signals.recovery_signal),
            "recovery_quality_ok": bool(signals.recovery_quality_ok),
            "value_recovery": bool(signals.value_recovery),
            "distribution_exhaustion": bool(signals.distribution_exhaustion),
            "source_close_reason": str(ledger.get("source_close_reason", "")),
            "episode_id": str(ledger.get("episode_id", "")),
        })

    @staticmethod
    def recovery_credit_total(owner: V47EpisodeOwner) -> float:
        return sum(float(item.get("remaining", 0.0) or 0.0) for item in owner._recovery_credit_ledger.values())

    @staticmethod
    def recovery_credit_portfolio_cap(owner: V47EpisodeOwner) -> float:
        return owner.RECOVERY_CREDIT_PORTFOLIO_AVG_CAP * max(1, len(getattr(owner, "TARGET_ALLOC", {}) or {}))


def cycle_recovery_budget(episode: dict) -> float:
    cumulative = float(episode.get("cumulative_sold_pct", 0.0) or 0.0)
    if cumulative > 0.0:
        return cumulative
    budget = float(episode.get("recovery_budget_pct", 0.0) or 0.0)
    if budget > 0.0:
        return budget
    return max(
        0.0,
        float(episode.get("sell_position_pct", 0.0) or 0.0)
        - float(episode.get("sell_target_pct", 0.0) or 0.0),
    )


def cycle_recovered_pct(episode: dict) -> float:
    cumulative = float(episode.get("cumulative_recovered_pct", 0.0) or 0.0)
    if cumulative > 0.0:
        return cumulative
    return max(0.0, float(episode.get("recovery_bought_pct", 0.0) or 0.0))


def join_guard(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if not addition:
        return existing
    parts = [part for part in existing.split("-") if part]
    if addition not in parts:
        parts.append(addition)
    return "-".join(parts)
