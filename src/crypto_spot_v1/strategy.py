"""Clean V1 spot strategy.

The class in this module is the accepted V1 behavior flattened into one
maintainable implementation. It preserves the migrated V6.19 trading rules
without keeping historical experiment class names in the runtime code.
"""

from __future__ import annotations

import pandas as pd

from . import strategy_utils
from .strategy_rebalance import Action, PortfolioState, PortfolioStrategyBase, PositionState


class V1SpotStrategy(PortfolioStrategyBase):
    """Long-only spot portfolio strategy for one symbol at a time."""

    VERSION_LABEL = "v1"

    TARGET_TABLE = {
        "BULL": {0: 0.95, 1: 0.85, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.15},
        "MIXED": {0: 0.65, 1: 0.55, 2: 0.45, 3: 0.35, 4: 0.20, 5: 0.05},
        "BEAR": {0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08, 4: 0.00, 5: 0.00},
    }
    CONFIRM_BARS = {"BULL": 3, "MIXED": 5, "BEAR": 2}
    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 2},
        "MIXED": {"max_buy": 0.20, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }
    CORE_FLOOR = {
        "BTC/USDT": 0.35,
        "ETH/USDT": 0.25,
        "BNB/USDT": 0.20,
    }
    BTC_ADJUST = {
        "STRONG_BULL": 0.05,
        "BULL": 0.03,
        "RANGE": 0.00,
        "BEAR": -0.05,
    }

    MIN_ADJUST_THRESHOLD = 0.05
    TREND_CONTINUATION_BOOST = 0.06
    TREND_CONTINUATION_MAX_BUY_MULT = 1.25

    PARTIAL_CORE_K = 0.75
    POST_OVERRIDE_LOOKBACK_CALLS = 90
    POST_OVERRIDE_TARGET_GAP_MULT = 0.25
    BTC_BEAR_TARGET_GAP_MULT = 0.25

    def __init__(
        self,
        initial_capital: float = 100.0,
        reserve: float = 20.0,
        fee_rate: float = 0.001,
        confirm_bars_override: dict | None = None,
        state_config_override: dict | None = None,
    ):
        self.initial_capital = initial_capital
        self.reserve = reserve
        self.fee_rate = fee_rate
        self.min_notional = 10.0

        self._confirm_bars = dict(self.CONFIRM_BARS)
        if confirm_bars_override:
            self._confirm_bars.update(confirm_bars_override)

        self._state_config = {}
        for state, cfg in self.STATE_CONFIG.items():
            merged = dict(cfg)
            merged.update((state_config_override or {}).get(state, {}))
            self._state_config[state] = merged

        self._call_count = 0
        self._last_buy_call = -48
        self._peak_price = 0.0
        self._state_streak = 1
        self._current_state = "MIXED"
        self._pending_state: str | None = None
        self._pending_streak = 0
        self._prev_confirmed_state: str | None = None
        self._recovery_calls_remaining = 0
        self._last_core_override_sell_call = -10_000

    @property
    def name(self) -> str:
        return "v1_spot_btc_bear_target_gap_quarter_dca"

    @property
    def deployable_capital(self) -> float:
        return self.initial_capital

    compute_indicators = staticmethod(strategy_utils.compute_indicators)

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._call_count += 1

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return []

        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return []

        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return []

        pos = portfolio.positions.get(symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        current_pct = position_value / total_value if total_value > 0 else 0.0

        if current_pct < 0.20:
            self._peak_price = price
        elif pos.quantity > 1e-12:
            self._peak_price = max(self._peak_price, price)

        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        sell_target = self._lookup_target(raw_state, risk_score)
        buy_target = self._lookup_target(confirmed_state, risk_score)

        vol_multiplier = self._get_directional_vol_multiplier(latest, price)
        sell_target = max(0.0, min(1.0, sell_target * vol_multiplier))
        buy_target = max(0.0, min(1.0, buy_target * vol_multiplier))

        btc_adjust = self._get_btc_adjust(latest, symbol)
        sell_target = max(0.0, min(1.0, sell_target + btc_adjust))
        buy_target = max(0.0, min(1.0, buy_target + btc_adjust))

        sell_target = self._compose_target(
            symbol=symbol,
            tactical_target=sell_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side="sell",
        )
        buy_target = self._compose_target(
            symbol=symbol,
            tactical_target=buy_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side="buy",
        )

        trend_continuation = self._is_trend_continuation_setup(
            confirmed_state, latest, price, trend_risk
        )
        if trend_continuation:
            buy_target = min(self._target_cap(), buy_target + self.TREND_CONTINUATION_BOOST)

        sell_target = max(0.0, min(self._target_cap(), sell_target))
        buy_target = max(0.0, min(self._target_cap(), buy_target))

        self._track_recovery(confirmed_state)

        pullback_buy = self._is_safe_pullback_buy(
            confirmed_state, latest, price, trend_risk
        )
        safe_recovery = self._is_safe_recovery_buy(latest, price, trend_risk)

        sell_setup = self._classify_sell_setup(
            trend_risk=trend_risk,
            risk_score=risk_score,
            latest=latest,
            price=price,
            raw_state=raw_state,
            drawdown_risk=drawdown_risk,
        )
        sell_threshold, max_sell, sell_guard = self._adjust_sell_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
            sell_threshold=self.MIN_ADJUST_THRESHOLD,
            max_sell=self._base_max_sell(trend_risk, risk_score),
        )

        if current_pct > sell_target + sell_threshold:
            gap = current_pct - sell_target
            sell_pct = min(gap, max_sell)
            sell_qty = min(total_value * sell_pct / price, pos.quantity)
            if sell_qty > 1e-12:
                self._record_executed_action(side="sell", setup=sell_setup)
                return [
                    Action(
                        symbol=symbol,
                        side="sell",
                        quantity=sell_qty,
                        price=price,
                        reason=self._build_action_reason(
                            side="sell",
                            setup=sell_setup,
                            risk_score=risk_score,
                            trend_risk=trend_risk,
                            drawdown_risk=drawdown_risk,
                            raw_state=raw_state,
                            confirmed_state=confirmed_state,
                            target=sell_target,
                            guard=sell_guard,
                        ),
                    )
                ]

        if current_pct >= buy_target - self.MIN_ADJUST_THRESHOLD:
            return []

        cfg = self._state_config[confirmed_state]
        buy_setup = self._classify_buy_setup(
            trend_continuation,
            safe_recovery,
            pullback_buy,
        )

        if self._recovery_calls_remaining <= 0:
            effective_cooldown = self._compute_buy_cooldown(confirmed_state, cfg, risk_score)
            effective_cooldown, cooldown_guard = self._adjust_buy_cooldown(
                buy_setup=buy_setup,
                effective_cooldown=effective_cooldown,
            )
            if self._call_count - self._last_buy_call < effective_cooldown:
                return []
        else:
            cooldown_guard = ""

        gap = buy_target - current_pct
        if trend_continuation:
            max_buy = min(
                cfg.get("max_buy", 0.25) * self.TREND_CONTINUATION_MAX_BUY_MULT,
                gap,
            )
        elif safe_recovery:
            max_buy = min(cfg.get("max_buy", 0.25) * 2.0, gap)
        elif pullback_buy:
            max_buy = min(cfg.get("max_buy", 0.25) * 1.5, gap)
        else:
            max_buy = cfg.get("max_buy", 0.25)

        max_buy, buy_guard = self._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )

        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty * price < self.min_notional:
            return []

        self._last_buy_call = self._call_count
        self._recovery_calls_remaining = max(0, self._recovery_calls_remaining - 1)
        return [
            Action(
                symbol=symbol,
                side="buy",
                quantity=buy_qty,
                price=price,
                reason=self._build_action_reason(
                    side="buy",
                    setup=buy_setup,
                    risk_score=risk_score,
                    trend_risk=trend_risk,
                    drawdown_risk=drawdown_risk,
                    raw_state=raw_state,
                    confirmed_state=confirmed_state,
                    target=buy_target,
                    guard="-".join(part for part in (cooldown_guard, buy_guard) if part),
                ),
            )
        ]

    def _classify_buy_setup(
        self,
        trend_continuation: bool,
        safe_recovery: bool,
        pullback_buy: bool,
    ) -> str:
        if trend_continuation:
            return "trend-cont"
        if safe_recovery:
            return "safe-recovery"
        if pullback_buy:
            return "pullback"
        return "target-gap"

    def _classify_sell_setup(
        self,
        trend_risk: int,
        risk_score: int,
        latest: pd.Series,
        price: float,
        raw_state: str,
        drawdown_risk: int,
    ) -> str:
        override_setup = self._defensive_core_override_setup(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        )
        if override_setup:
            return f"core-override_{override_setup}"
        if trend_risk >= 3:
            return "trend-break"
        if risk_score >= 4:
            return "risk-reduce"
        return "target-reduce"

    def _base_max_sell(self, trend_risk: int, risk_score: int) -> float:
        if trend_risk >= 3:
            return 0.50
        if risk_score >= 4:
            return 0.40
        return 0.25

    def _adjust_sell_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
        sell_threshold: float,
        max_sell: float,
    ) -> tuple[float, float, str]:
        if sell_setup == "core-override_trend-break":
            return sell_threshold, 0.50, ""
        if sell_setup == "core-override_profit-giveback":
            return sell_threshold, 0.40, ""

        healthy_bull = self._is_healthy_bull_target_reduce_sell(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        )
        if not healthy_bull:
            return sell_threshold, max_sell, ""

        sell_threshold = max(sell_threshold, 0.10 if drawdown_risk == 0 else 0.08)
        return sell_threshold, max_sell, f"bull-bull-guard-thr-dd{drawdown_risk}"

    def _adjust_buy_cooldown(
        self,
        buy_setup: str,
        effective_cooldown: int,
    ) -> tuple[int, str]:
        if not self._is_post_override_target_gap(buy_setup):
            return effective_cooldown, ""
        return effective_cooldown, ""

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        guard = ""
        if self._is_post_override_target_gap(buy_setup):
            max_buy *= self.POST_OVERRIDE_TARGET_GAP_MULT
            guard = self._join_guard(
                guard,
                f"post-override-tgap-x{self.POST_OVERRIDE_TARGET_GAP_MULT:.2f}",
            )

        if buy_setup != "target-gap":
            return max_buy, guard

        btc_regime = str(latest.get("btc_regime", ""))
        if btc_regime == "BEAR":
            max_buy *= self.BTC_BEAR_TARGET_GAP_MULT
            guard = self._join_guard(
                guard,
                f"btc-bear-tgap-x{self.BTC_BEAR_TARGET_GAP_MULT:.2f}",
            )
        return max_buy, guard

    def _record_executed_action(self, side: str, setup: str) -> None:
        if side == "sell" and setup == "core-override_trend-break":
            self._last_core_override_sell_call = self._call_count

    def _compose_target(
        self,
        symbol: str,
        tactical_target: float,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        latest: pd.Series,
        price: float,
        side: str,
    ) -> float:
        if side == "sell" and self._defensive_core_override_setup(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        ):
            core = self.CORE_FLOOR.get(symbol, 0.0)
            return max(tactical_target, core * self.PARTIAL_CORE_K)
        return self._apply_core_floor(symbol, tactical_target)

    def _defensive_core_override_setup(
        self,
        latest: pd.Series | None,
        price: float,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
    ) -> str:
        if latest is None or price <= 0:
            return ""
        if self._is_structural_trend_break(latest, price, raw_state, trend_risk):
            return "trend-break"
        return ""

    def _is_structural_trend_break(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        trend_risk: int,
    ) -> bool:
        if raw_state != "BEAR" or trend_risk < 3:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        return bool(price < ema168 and ema24 < ema72 < ema168)

    def _is_healthy_bull_target_reduce_sell(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        if trend_risk != 0 or drawdown_risk > 1:
            return False

        ema72 = latest.get("ema72")
        slope = latest.get("ema168_slope")
        atr_rank = latest.get("atr_pct_rank")
        if pd.isna(ema72) or pd.isna(slope) or pd.isna(atr_rank):
            return False
        return bool(price > ema72 and slope > 0 and atr_rank < 0.90)

    def _is_post_override_target_gap(self, buy_setup: str) -> bool:
        if buy_setup != "target-gap":
            return False
        days_since = self._call_count - self._last_core_override_sell_call
        return 0 <= days_since <= self.POST_OVERRIDE_LOOKBACK_CALLS

    @staticmethod
    def _join_guard(existing: str, addition: str) -> str:
        return "-".join(part for part in (existing, addition) if part)

    def _target_cap(self) -> float:
        return 0.98

    def _build_action_reason(
        self,
        side: str,
        setup: str,
        risk_score: int,
        trend_risk: int,
        drawdown_risk: int,
        raw_state: str,
        confirmed_state: str,
        target: float,
        guard: str = "",
    ) -> str:
        reason = (
            f"{self.VERSION_LABEL}_{side}_{setup}"
            f"_r{risk_score}_tr{trend_risk}_dd{drawdown_risk}"
            f"_raw{raw_state}_conf{confirmed_state}_t{target:.0%}"
        )
        if guard:
            reason = f"{reason}_{guard}"
        return reason

    def _apply_state_confirmation(self, raw_state: str) -> str:
        if raw_state == self._current_state:
            self._state_streak += 1
            self._pending_state = None
            self._pending_streak = 0
            return self._current_state
        if raw_state == self._pending_state:
            self._pending_streak += 1
        else:
            self._pending_state = raw_state
            self._pending_streak = 1
        need = self._confirm_bars.get(raw_state, 3)
        if self._pending_streak >= need:
            self._current_state = raw_state
            self._state_streak = self._pending_streak
            self._pending_state = None
            self._pending_streak = 0
        return self._current_state

    def _track_recovery(self, confirmed_state: str) -> None:
        if (
            self._prev_confirmed_state is not None
            and self._prev_confirmed_state != "BULL"
            and confirmed_state == "BULL"
        ):
            self._recovery_calls_remaining = 2
        self._prev_confirmed_state = confirmed_state

    def _get_btc_adjust(self, latest: pd.Series, symbol: str) -> float:
        if symbol == "BTC/USDT":
            return 0.0
        btc_regime = latest.get("btc_regime")
        if pd.isna(btc_regime):
            return 0.0
        return self.BTC_ADJUST.get(btc_regime, 0.0)

    def _get_directional_vol_multiplier(self, latest: pd.Series, price: float) -> float:
        rank = latest.get("atr_pct_rank")
        if pd.isna(rank) or rank <= 0.80:
            return 1.0

        excess = rank - 0.80
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")

        strong_uptrend = (
            not pd.isna(ema24)
            and not pd.isna(ema72)
            and not pd.isna(ema168)
            and price > ema24 > ema72 > ema168
            and not pd.isna(slope)
            and slope > 0
        )
        weak_trend = (
            (not pd.isna(ema72) and price < ema72)
            or (not pd.isna(ema24) and not pd.isna(ema72) and ema24 < ema72)
        )

        if weak_trend:
            mult = 1.0 - excess * 1.8
        elif strong_uptrend:
            mult = 1.0 - excess * 0.5
        else:
            mult = 1.0 - excess * 1.2
        return max(0.70, min(1.00, mult))

    def _apply_core_floor(self, symbol: str, tactical_target: float) -> float:
        core = self.CORE_FLOOR.get(symbol, 0.25)
        return core + tactical_target * (1.0 - core)

    def _is_trend_continuation_setup(
        self,
        confirmed_state: str,
        latest: pd.Series,
        price: float,
        trend_risk: int,
    ) -> bool:
        if confirmed_state != "BULL" or trend_risk != 0:
            return False

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price > ema24 > ema72 > ema168):
            return False
        if pd.isna(slope) or slope <= 0:
            return False
        if price / ema24 >= 1.04:
            return False

        donchian_pos = latest.get("donchian_pos", 0.5)
        if not pd.isna(donchian_pos) and donchian_pos >= 0.92:
            return False

        atr_rank = latest.get("atr_pct_rank", 0.5)
        if not pd.isna(atr_rank) and atr_rank >= 0.90:
            return False
        return True

    def _is_safe_pullback_buy(
        self,
        confirmed_state: str,
        latest: pd.Series,
        price: float,
        trend_risk: int,
    ) -> bool:
        if confirmed_state != "BULL" or trend_risk >= 2:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        if pd.isna(ema24) or pd.isna(ema72):
            return False

        donchian_pos = latest.get("donchian_pos", 0.5)
        if not pd.isna(donchian_pos) and donchian_pos >= 0.80:
            return False
        return bool(ema24 > ema72 and ema72 < price < ema24)

    def _is_safe_recovery_buy(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
    ) -> bool:
        if self._recovery_calls_remaining <= 0 or trend_risk >= 2:
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price / ema24 >= 1.05:
            return False

        donchian_pos = latest.get("donchian_pos", 1.0)
        if not pd.isna(donchian_pos) and donchian_pos >= 0.85:
            return False

        atr_rank = latest.get("atr_pct_rank", 1.0)
        if not pd.isna(atr_rank) and atr_rank >= 0.95:
            return False
        return True

    def _compute_buy_cooldown(self, state: str, cfg: dict, risk_score: int) -> int:
        base = cfg.get("base_cooldown", 6)
        if state == "BEAR":
            return base
        return base + risk_score * 2

    def _calculate_trend_risk(self, latest: pd.Series, price: float) -> int:
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")

        ok24 = not pd.isna(ema24)
        ok72 = not pd.isna(ema72)
        ok168 = not pd.isna(ema168)

        if ok24 and ok72 and ok168 and price < ema168 and ema24 < ema72 < ema168:
            return 3
        if ok168 and price < ema168 and ok24 and ok72 and ema24 < ema72:
            return 2
        if ok72 and price < ema72 and ok24 and ema24 < ema72:
            return 1
        return 0

    def _calculate_drawdown_risk(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or self._peak_price <= 0:
            return 0

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        profit_pct = price / pos.avg_cost - 1
        dd_from_peak = 1 - price / self._peak_price

        ok24 = not pd.isna(ema24)
        ok72 = not pd.isna(ema72)
        if profit_pct > 0.30 and dd_from_peak > 0.18 and ok72 and price < ema72:
            return 2
        if profit_pct > 0.20 and dd_from_peak > 0.10 and ok24 and price < ema24:
            return 1
        return 0

    def _lookup_target(self, state: str, risk_score: int) -> float:
        table = self.TARGET_TABLE.get(state, self.TARGET_TABLE["MIXED"])
        return table.get(risk_score, 0.0)


SingleCoinSpotV1 = V1SpotStrategy
