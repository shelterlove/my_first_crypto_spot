from __future__ import annotations

from .v42_types import V42Context, V42Regime, V42Sizing


class V42EpisodeMixin:
    def _update_episode(self, context: V42Context, regime: V42Regime) -> dict:
        symbol = context.symbol
        price = context.price
        episode = self._episodes_by_symbol.get(symbol)
        if episode is not None:
            episode["lowest_price"] = min(float(episode.get("lowest_price", price)), price)
            episode["cycle_lowest_price"] = min(float(episode.get("cycle_lowest_price", episode.get("lowest_price", price)) or price), price)
            episode["last_call"] = self._call_count
            if self._call_count - int(episode.get("start_call", self._call_count)) > self.EPISODE_MAX_CALLS:
                self._close_episode(symbol, "expired")
                episode = None

        if episode is None:
            return {"state": "NORMAL"}

        state = str(episode.get("state", "NORMAL"))
        age = self._call_count - int(episode.get("start_call", self._call_count))
        if state == "RECOVERY_TEST":
            recovery_start = int(episode.get("recovery_start_call", episode.get("start_call", self._call_count)) or self._call_count)
            recovery_age = self._call_count - recovery_start
            if self._recovery_quality_ok(context, regime) and recovery_age >= self.RECOVERY_TEST_CALLS:
                self._close_episode(symbol, "recovery_confirmed")
                return {"state": "NORMAL"}
            if regime.regime == "BEAR" or self._failed_recovery(context, episode):
                episode["state"] = "FAILED_RECOVERY_LOCK"
                episode["blocked_reason"] = self._join_guard(str(episode.get("blocked_reason", "")), "failed_recovery")
        elif state == "FAILED_RECOVERY_LOCK":
            if age >= self.FAILED_RECOVERY_CALLS and self._recovery_quality_ok(context, regime):
                episode["state"] = "RECOVERY_TEST"
        elif state == "DISTRIBUTION_LOCK":
            if age >= self.DISTRIBUTION_LOCK_CALLS and not self._distribution_exhaustion(context, regime):
                self._close_episode(symbol, "distribution_cooled")
                return {"state": "NORMAL"}
        elif state == "STRUCTURAL_BEAR_LOCK":
            if age >= self.FAILED_RECOVERY_CALLS and self._recovery_quality_ok(context, regime):
                episode["state"] = "RECOVERY_TEST"

        return episode


    def _start_episode(self, context: V42Context, sizing: V42Sizing) -> None:
        symbol = context.symbol
        setup = sizing.setup
        active = self._episodes_by_symbol.get(symbol)
        if active is not None:
            if self._is_protective_setup(str(active.get("setup", ""))) and self._is_protective_setup(setup):
                self._append_episode_sell_leg(context, active, sizing)
                return
            else:
                self._close_episode(symbol, "replaced")
        state = {
            "defense-sell": "DEFENSE_LOCK",
            "distribution-sell": "DISTRIBUTION_LOCK",
            "structural-exit-sell": "STRUCTURAL_BEAR_LOCK",
        }.get(setup, "DEFENSE_LOCK")
        sold_pct = sizing.quantity * context.price / context.total_value if context.total_value > 0.0 else 0.0
        post_sell_pct = max(0.0, context.current_pct - sold_pct)
        episode = {
            "episode_id": f"v4_2-{self._next_episode_id}",
            "symbol": symbol,
            "state": state,
            "setup": setup,
            "start_call": self._call_count,
            "cycle_start_call": self._call_count,
            "close_call": None,
            "sell_price": context.price,
            "lowest_price": context.price,
            "cycle_lowest_price": context.price,
            "sell_position_pct": context.current_pct,
            "sell_target_pct": float(sizing.get("target", 0.0) or 0.0),
            "sell_risk_score": context.risk_score,
            "sell_trend_risk": context.trend_risk,
            "sell_drawdown_risk": context.drawdown_risk,
            "first_sell_price": context.price,
            "last_sell_price": context.price,
            "avg_sell_price": context.price,
            "sell_count": 1,
            "sell_legs": [{
                "call": self._call_count,
                "setup": setup,
                "price": context.price,
                "sold_pct": sold_pct,
                "position_before_pct": context.current_pct,
                "position_after_pct": post_sell_pct,
                "target_pct": float(sizing.get("target", 0.0) or 0.0),
            }],
            "last_sell_call": self._call_count,
            "sold_pct": sold_pct,
            "cumulative_sold_pct": sold_pct,
            "post_sell_pct": post_sell_pct,
            "min_position_pct": post_sell_pct,
            "recovery_budget_pct": sold_pct if setup != "distribution-sell" else sold_pct * 0.50,
            "recovery_bought_pct": 0.0,
            "cumulative_recovered_pct": 0.0,
            "unrecovered_budget_pct": sold_pct if setup != "distribution-sell" else sold_pct * 0.50,
            "recovery_buy_count": 0,
            "recovery_legs": [],
            "recovery_buy_notional": 0.0,
            "last_recovery_buy_call": None,
            "last_recovery_buy_price": 0.0,
            "max_recovery_position_pct": post_sell_pct,
            "recovered_to_30_call": None,
            "recovered_to_50_call": None,
            "recovered_to_80_call": None,
            "had_value_recovery": False,
            "blocked_reason": "",
            "close_reason": "",
            "episode_contribution_notional": 0.0,
        }
        self._next_episode_id += 1
        self._episodes_by_symbol[symbol] = episode
        self._diag["v4_2_episode_started_count"] += 1

    def _append_episode_sell_leg(self, context: V42Context, episode: dict, sizing: V42Sizing) -> None:
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
        episode["start_call"] = self._call_count
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
        episode["last_sell_call"] = self._call_count
        episode["last_call"] = self._call_count
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
            "call": self._call_count,
            "setup": sizing.setup,
            "price": context.price,
            "sold_pct": sold_pct,
            "position_before_pct": context.current_pct,
            "position_after_pct": post_sell_pct,
            "target_pct": float(sizing.get("target", 0.0) or 0.0),
        })
        episode["sell_legs"] = legs

    @staticmethod
    def _is_protective_setup(setup: str) -> bool:
        return setup in {"defense-sell", "structural-exit-sell"}

    @staticmethod
    def _episode_continuity(active: dict) -> dict:
        budget = float(active.get("recovery_budget_pct", 0.0) or 0.0)
        bought = float(active.get("recovery_bought_pct", 0.0) or 0.0)
        return {
            "prior_episode_id": active.get("episode_id", ""),
            "prior_unrecovered_budget_pct": max(0.0, budget - bought),
            "prior_sold_pct": float(active.get("sold_pct", 0.0) or 0.0),
            "prior_recovery_bought_pct": bought,
        }

    def _close_episode(self, symbol: str, reason: str) -> None:
        episode = self._episodes_by_symbol.pop(symbol, None)
        if episode is None:
            return
        episode["close_reason"] = reason
        episode["close_call"] = self._call_count
        episode["state"] = "CLOSED"
        self._episode_log.append(dict(episode))
        self._diag["v4_2_episode_closed_count"] += 1

    def _episode_row(self, episode: dict) -> dict:
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        lowest_price = float(episode.get("cycle_lowest_price", episode.get("lowest_price", sell_price)) or sell_price)
        drawdown = 1.0 - lowest_price / sell_price if sell_price > 0.0 else 0.0
        return {
            "episode_id": episode.get("episode_id", ""),
            "symbol": episode.get("symbol", ""),
            "state": episode.get("state", ""),
            "setup": episode.get("setup", ""),
            "start_call": episode.get("cycle_start_call", episode.get("start_call")),
            "last_leg_start_call": episode.get("start_call"),
            "close_call": episode.get("close_call"),
            "sell_price": sell_price,
            "lowest_price": lowest_price,
            "drawdown_from_sell": drawdown,
            "sell_position_pct": float(episode.get("sell_position_pct", 0.0) or 0.0),
            "sell_target_pct": float(episode.get("sell_target_pct", 0.0) or 0.0),
            "first_sell_price": float(episode.get("first_sell_price", sell_price) or sell_price),
            "last_sell_price": float(episode.get("last_sell_price", sell_price) or sell_price),
            "avg_sell_price": float(episode.get("avg_sell_price", sell_price) or sell_price),
            "sell_count": int(episode.get("sell_count", 1) or 1),
            "sold_pct": float(episode.get("sold_pct", 0.0) or 0.0),
            "cumulative_sold_pct": float(episode.get("cumulative_sold_pct", episode.get("sold_pct", 0.0)) or 0.0),
            "post_sell_pct": float(episode.get("post_sell_pct", 0.0) or 0.0),
            "min_position_pct": float(episode.get("min_position_pct", episode.get("post_sell_pct", 0.0)) or 0.0),
            "recovery_budget_pct": float(episode.get("recovery_budget_pct", 0.0) or 0.0),
            "recovery_bought_pct": float(episode.get("recovery_bought_pct", 0.0) or 0.0),
            "cumulative_recovered_pct": float(episode.get("cumulative_recovered_pct", episode.get("recovery_bought_pct", 0.0)) or 0.0),
            "unrecovered_budget_pct": float(episode.get("unrecovered_budget_pct", max(0.0, float(episode.get("recovery_budget_pct", 0.0) or 0.0) - float(episode.get("recovery_bought_pct", 0.0) or 0.0))) or 0.0),
            "recovery_buy_count": int(episode.get("recovery_buy_count", 0) or 0),
            "recovery_buy_notional": float(episode.get("recovery_buy_notional", 0.0) or 0.0),
            "last_recovery_buy_call": episode.get("last_recovery_buy_call"),
            "last_recovery_buy_price": float(episode.get("last_recovery_buy_price", 0.0) or 0.0),
            "max_recovery_position_pct": float(episode.get("max_recovery_position_pct", episode.get("post_sell_pct", 0.0)) or 0.0),
            "recovered_to_30_call": episode.get("recovered_to_30_call"),
            "recovered_to_50_call": episode.get("recovered_to_50_call"),
            "recovered_to_80_call": episode.get("recovered_to_80_call"),
            "days_to_30_pct": self._episode_days_to_threshold(episode, "recovered_to_30_call"),
            "days_to_50_pct": self._episode_days_to_threshold(episode, "recovered_to_50_call"),
            "days_to_80_pct": self._episode_days_to_threshold(episode, "recovered_to_80_call"),
            "sell_leg_count": len(episode.get("sell_legs", []) or []),
            "recovery_leg_count": len(episode.get("recovery_legs", []) or []),
            "had_value_recovery": bool(episode.get("had_value_recovery", False)),
            "prior_episode_id": episode.get("prior_episode_id", ""),
            "prior_unrecovered_budget_pct": float(episode.get("prior_unrecovered_budget_pct", 0.0) or 0.0),
            "prior_sold_pct": float(episode.get("prior_sold_pct", 0.0) or 0.0),
            "prior_recovery_bought_pct": float(episode.get("prior_recovery_bought_pct", 0.0) or 0.0),
            "blocked_reason": episode.get("blocked_reason", ""),
            "close_reason": episode.get("close_reason", ""),
            "episode_contribution_notional": float(episode.get("episode_contribution_notional", 0.0) or 0.0),
        }

    @staticmethod
    def _episode_days_to_threshold(episode: dict, key: str):
        call = episode.get(key)
        if call is None:
            return None
        return int(call) - int(episode.get("cycle_start_call", episode.get("start_call", call)) or call)

    def _failed_recovery(self, context: V42Context, episode: dict) -> bool:
        recovery_start = episode.get("recovery_start_call")
        if recovery_start is None or self._call_count - int(recovery_start) < self.RECOVERY_TEST_CALLS:
            return False
        sell_price = float(episode.get("sell_price", 0.0) or 0.0)
        if sell_price <= 0.0:
            return context.trend_risk >= 3
        return bool(context.price < sell_price * 0.92 or context.trend_risk >= 3)

