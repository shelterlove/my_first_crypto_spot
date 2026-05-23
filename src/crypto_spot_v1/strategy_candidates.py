"""Validated strategy candidates kept separate from the accepted V1 baseline."""

from __future__ import annotations

import pandas as pd

from . import strategy_utils
from .strategy import V1SpotStrategy
from .strategy_rebalance import Action, PortfolioState, PositionState


class V1LessChurnStrategy(V1SpotStrategy):
    """V1 with a 6% rebalance band to reduce marginal target-gap churn."""

    VERSION_LABEL = "v1_less_churn"
    MIN_ADJUST_THRESHOLD = 0.06

    @property
    def name(self) -> str:
        return "v1_less_churn"


class V11Strategy(V1LessChurnStrategy):
    """V1.1: restrained bull participation upgrade over v1_less_churn."""

    VERSION_LABEL = "v1_1"

    STRONG_BULL_TARGET_REDUCE_BAND = 0.10
    STRONG_BULL_TARGET_REDUCE_SIZE_MULT = 0.50
    BULL_GUARD_MIN_POSITION_PCT = 0.70
    BULL_GUARD_TARGET_GAP_THRESHOLD = 0.04
    BULL_GUARD_MAX_ATR_PCT_RANK = 0.90
    BULL_GUARD_MAX_DONCHIAN_POS = 0.95
    RECOVERY_REBOUND_LOOKBACK = 20
    RECOVERY_REBOUND_THRESHOLD = 0.05
    RECOVERY_DONCHIAN_MIN = 0.35
    RECOVERY_DONCHIAN_MAX = 0.85
    RECOVERY_BUY_SIZE_MULT = 0.30
    RECOVERY_RISK_SCORE_REDUCTION = 1

    @property
    def name(self) -> str:
        return "v1_1"

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

        recovery_override = self._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        )
        effective_risk_score = (
            max(risk_score - self.RECOVERY_RISK_SCORE_REDUCTION, 0)
            if recovery_override
            else risk_score
        )

        sell_target = self._lookup_target(raw_state, risk_score)
        buy_target = self._lookup_target(confirmed_state, effective_risk_score)

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

        bull_guard = self._is_bull_guard_setup(
            latest=latest,
            price=price,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        )
        bull_guard_guard = ""
        if bull_guard:
            buy_target = max(buy_target, self.BULL_GUARD_MIN_POSITION_PCT)
            bull_guard_guard = f"{self.VERSION_LABEL}_bull_guard_floor"
        elif self._is_bull_guard_overheat_or_risk_skip(latest, price, confirmed_state, trend_risk, risk_score):
            bull_guard_guard = self._bull_guard_skip_reason(latest, trend_risk, risk_score)

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

        buy_threshold = (
            self.BULL_GUARD_TARGET_GAP_THRESHOLD if bull_guard else self.MIN_ADJUST_THRESHOLD
        )
        if current_pct >= buy_target - buy_threshold:
            return []

        cfg = self._state_config[confirmed_state]
        buy_setup = self._classify_buy_setup(
            trend_continuation,
            safe_recovery or recovery_override,
            pullback_buy,
        )

        if self._recovery_calls_remaining <= 0 and not recovery_override:
            effective_cooldown = self._compute_buy_cooldown(confirmed_state, cfg, effective_risk_score)
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
        elif recovery_override:
            max_buy = min(cfg.get("max_buy", 0.25) * self.RECOVERY_BUY_SIZE_MULT, gap)
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
        )

        guard = self._join_guard(cooldown_guard, buy_guard)
        guard = self._join_guard(guard, bull_guard_guard)
        if bull_guard:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_bull_guard_target_gap_buy")
        if recovery_override:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_risk_score_reduced")
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_small_buy")

        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty * price < 10.0:
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
                    guard=guard,
                ),
            )
        ]

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
        if sell_setup != "target-reduce":
            return sell_threshold, max_sell, ""

        if self._is_strong_bull_target_reduce_suppression(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            guard = self._join_guard(
                f"{self.VERSION_LABEL}_target_reduce_strong_bull_suppressed",
                f"{self.VERSION_LABEL}_target_reduce_strong_bull_half_size",
            )
            return (
                max(sell_threshold, self.STRONG_BULL_TARGET_REDUCE_BAND),
                max_sell * self.STRONG_BULL_TARGET_REDUCE_SIZE_MULT,
                guard,
            )

        threshold, adjusted_max_sell, base_guard = super()._adjust_sell_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
            sell_threshold=sell_threshold,
            max_sell=max_sell,
        )
        return threshold, adjusted_max_sell, self._join_guard(
            base_guard,
            f"{self.VERSION_LABEL}_target_reduce_normal",
        )

    def _is_strong_bull_target_reduce_suppression(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if confirmed_state != "BULL" or risk_score > 2 or trend_risk > 1:
            return False
        if not self._is_healthy_bull_target_reduce_sell(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        ):
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")
        atr_rank = latest.get("atr_pct_rank")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(slope):
            return False
        if not (ema24 > ema72 > ema168 and latest.get("close", price) > ema72 and slope > 0):
            return False
        return bool(pd.isna(atr_rank) or atr_rank <= self.BULL_GUARD_MAX_ATR_PCT_RANK)

    def _is_bull_guard_setup(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
    ) -> bool:
        if confirmed_state != "BULL" or risk_score > 2 or trend_risk > 1:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")
        atr_rank = latest.get("atr_pct_rank")
        donchian_pos = latest.get("donchian_pos")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(slope):
            return False
        if not (latest.get("close", price) > ema168 and ema24 > ema72 > ema168 and slope > 0):
            return False
        if not pd.isna(atr_rank) and atr_rank > self.BULL_GUARD_MAX_ATR_PCT_RANK:
            return False
        if not pd.isna(donchian_pos) and donchian_pos >= self.BULL_GUARD_MAX_DONCHIAN_POS:
            return False
        return True

    def _is_bull_guard_overheat_or_risk_skip(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
    ) -> bool:
        if confirmed_state != "BULL":
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(slope):
            return False
        return bool(latest.get("close", price) > ema168 and ema24 > ema72 > ema168 and slope > 0 and (risk_score >= 3 or trend_risk > 1 or self._is_overheated(latest)))

    def _bull_guard_skip_reason(self, latest: pd.Series, trend_risk: int, risk_score: int) -> str:
        if risk_score >= 3 or trend_risk > 1:
            return f"{self.VERSION_LABEL}_bull_guard_skipped_risk"
        if self._is_overheated(latest):
            return f"{self.VERSION_LABEL}_bull_guard_skipped_overheat"
        return ""

    def _is_overheated(self, latest: pd.Series) -> bool:
        atr_rank = latest.get("atr_pct_rank")
        donchian_pos = latest.get("donchian_pos")
        return bool(
            (not pd.isna(atr_rank) and atr_rank > self.BULL_GUARD_MAX_ATR_PCT_RANK)
            or (not pd.isna(donchian_pos) and donchian_pos >= self.BULL_GUARD_MAX_DONCHIAN_POS)
        )

    def _is_recovery_override_setup(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
    ) -> bool:
        if risk_score < 2:
            return False
        close = latest.get("close", price)
        if trend_risk >= 3 or close < latest.get("ema168", float("inf")):
            return False
        if confirmed_state == "BEAR" and raw_state == "BEAR":
            return False
        ema24 = latest.get("ema24")
        if pd.isna(ema24) or close <= ema24:
            return False
        if len(df) < max(self.RECOVERY_REBOUND_LOOKBACK, 4):
            return False
        ema24_prev = df["ema24"].iloc[-4]
        if pd.isna(ema24_prev) or ema24 <= ema24_prev:
            return False
        recent_low = df["low"].tail(self.RECOVERY_REBOUND_LOOKBACK).min()
        if pd.isna(recent_low) or recent_low <= 0:
            return False
        if close / recent_low - 1 < self.RECOVERY_REBOUND_THRESHOLD:
            return False
        donchian_pos = latest.get("donchian_pos")
        if pd.isna(donchian_pos):
            return False
        return bool(self.RECOVERY_DONCHIAN_MIN <= donchian_pos <= self.RECOVERY_DONCHIAN_MAX)


class V12Strategy(V11Strategy):
    """V1.2: faster bull confirmation with a restrained bull risk target curve.

    The research pass showed that recovery overrides and broad sell suppression
    added churn without improving bull median performance. V1.2 keeps the V1.1
    buy/sell mechanics intact and only changes the state confirmation speed and
    BULL target curve.
    """

    VERSION_LABEL = "v1_2"
    CONFIRM_BARS = {"BULL": 1, "MIXED": 5, "BEAR": 2}
    TARGET_TABLE = {
        "BULL": {0: 0.98, 1: 0.95, 2: 0.90, 3: 0.75, 4: 0.50, 5: 0.25},
        "MIXED": {0: 0.65, 1: 0.55, 2: 0.45, 3: 0.35, 4: 0.20, 5: 0.05},
        "BEAR": {0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08, 4: 0.00, 5: 0.00},
    }

    @property
    def name(self) -> str:
        return "v1_2"


class V13Strategy(V12Strategy):
    """V1.3: confirmed-BULL sell target override for raw MIXED flickers.

    V1.2 has an asymmetry: sell_target uses raw_state while buy_target uses
    confirmed_state.  When raw_state momentarily flicks to MIXED during an
    otherwise healthy bull, the sell target plunges from 98% to 65%, forcing
    a large premature sell (~44% of target-reduce sells are "too early" with
    avg +1.84% 20d forward return).

    V1.3 keeps the sell logic 100% intact except: when raw_state == MIXED
    and confirmed_state == BULL, the sell target uses the BULL table with a
    +1 risk_score penalty instead of the MIXED table.  This prevents the
    massive target gap on MIXED flickers while still applying a penalty for
    the MIXED signal.  All other cases (incl. fully confirmed transitions)
    are unchanged.
    """

    VERSION_LABEL = "v1_3"

    @property
    def name(self) -> str:
        return "v1_3"

    def _get_sell_target_state(self, raw_state: str, confirmed_state: str) -> tuple[str, int]:
        """Override state for sell target lookup.

        Returns (lookup_state, risk_penalty).
        When raw_state flicks to MIXED but confirmed still BULL, use BULL
        targets with +1 risk penalty instead of MIXED targets.
        """
        if raw_state == "MIXED" and confirmed_state == "BULL":
            return "BULL", 1
        return raw_state, 0

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

        recovery_override = self._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        )
        effective_risk_score = (
            max(risk_score - self.RECOVERY_RISK_SCORE_REDUCTION, 0)
            if recovery_override
            else risk_score
        )

        # V1.3: override sell lookup state for raw MIXED + confirmed BULL
        sell_lookup_state, sell_risk_penalty = self._get_sell_target_state(
            raw_state, confirmed_state,
        )
        sell_target = self._lookup_target(
            sell_lookup_state, min(risk_score + sell_risk_penalty, 5),
        )
        buy_target = self._lookup_target(confirmed_state, effective_risk_score)

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
            confirmed_state, latest, price, trend_risk,
        )
        if trend_continuation:
            buy_target = min(self._target_cap(), buy_target + self.TREND_CONTINUATION_BOOST)

        bull_guard = self._is_bull_guard_setup(
            latest=latest,
            price=price,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        )
        bull_guard_guard = ""
        if bull_guard:
            buy_target = max(buy_target, self.BULL_GUARD_MIN_POSITION_PCT)
            bull_guard_guard = f"{self.VERSION_LABEL}_bull_guard_floor"
        elif self._is_bull_guard_overheat_or_risk_skip(latest, price, confirmed_state, trend_risk, risk_score):
            bull_guard_guard = self._bull_guard_skip_reason(latest, trend_risk, risk_score)

        sell_target = max(0.0, min(self._target_cap(), sell_target))
        buy_target = max(0.0, min(self._target_cap(), buy_target))

        self._track_recovery(confirmed_state)

        pullback_buy = self._is_safe_pullback_buy(
            confirmed_state, latest, price, trend_risk,
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

        buy_threshold = (
            self.BULL_GUARD_TARGET_GAP_THRESHOLD if bull_guard else self.MIN_ADJUST_THRESHOLD
        )
        if current_pct >= buy_target - buy_threshold:
            return []

        cfg = self._state_config[confirmed_state]
        buy_setup = self._classify_buy_setup(
            trend_continuation,
            safe_recovery or recovery_override,
            pullback_buy,
        )

        if self._recovery_calls_remaining <= 0 and not recovery_override:
            effective_cooldown = self._compute_buy_cooldown(confirmed_state, cfg, effective_risk_score)
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
        elif recovery_override:
            max_buy = min(cfg.get("max_buy", 0.25) * self.RECOVERY_BUY_SIZE_MULT, gap)
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
        )

        guard = self._join_guard(cooldown_guard, buy_guard)
        guard = self._join_guard(guard, bull_guard_guard)
        if bull_guard:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_bull_guard_target_gap_buy")
        if recovery_override:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_risk_score_reduced")
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_small_buy")

        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty * price < 10.0:
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
                    guard=guard,
                ),
            )
        ]
