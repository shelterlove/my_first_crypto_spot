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
    """V1.3: BULL sell-target override for raw MIXED flickers + skip sells during bull pullbacks.

    V1.2 sells into raw-state transitions rather than confirmed-state ones.
    When raw_state flicks to MIXED during a healthy bull, the sell target
    plunges from 98% to 65%, forcing a premature sell.  Separately, even
    with the V1.2 less-churn band, the strategy sells during temporary bull
    pullbacks and then misses the recovery because trend_risk / btc_bear
    blocks prevent timely reentry for weeks or months.

    V1.3 changes:
    1. When raw_state == MIXED and confirmed_state == BULL, the sell target
       uses the BULL table with +1 risk penalty instead of the MIXED table.
    2. During bull pullbacks (confirmed=BULL, ema24>ema72, price between
       ema72 and ema24), target-reduce and risk-reduce sells are skipped.
    3. Wider sell threshold (0.10 vs 0.06) in BULL state prevents tiny
       trims (gap < 10%) that don't materially protect but do create a
       drag on bull-market participation.
    """

    VERSION_LABEL = "v1_3"

    @property
    def name(self) -> str:
        return "v1_3"

    def _get_sell_target_state(self, raw_state: str, confirmed_state: str) -> tuple[str, int]:
        """When raw_state flicks to MIXED but confirmed is still BULL,
        use the BULL table with +1 risk penalty.  The penalty ensures we
        still trim modestly (BULL[1]=0.95) even at low risk scores instead
        of holding at full 98% through a mixed signal.
        """
        if raw_state == "MIXED" and confirmed_state == "BULL":
            return "BULL", 1
        return raw_state, 0

    def _is_bull_pullback(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
    ) -> bool:
        """True when price has dipped within an otherwise healthy uptrend.

        A bull pullback means: confirmed regime is BULL, short-term trend
        is still rising (ema24 > ema72), but price has retraced below
        ema24 while staying above ema72.  Selling here crystallises a
        loss and the subsequent recovery is typically missed because
        trend_risk / btc_bear blocks buy reentry for weeks.
        """
        if confirmed_state != "BULL":
            return False
        if trend_risk >= 2:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        if pd.isna(ema24) or pd.isna(ema72):
            return False
        return bool(ema24 > ema72 and price > ema72 and price <= ema24)

    def _calculate_drawdown_risk(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        """Delegate to base class — no V1.3-specific overrides."""
        return super()._calculate_drawdown_risk(latest, pos, price)

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

        # V1.3+: skip target-reduce/risk-reduce sells during bull pullbacks
        #        and block target-reduce sells in healthy BULL (hook).
        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_pullback(latest, price, confirmed_state, trend_risk):
                sell_target = max(sell_target, current_pct)
        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_sell_blocked(confirmed_state, raw_state, trend_risk, risk_score, sell_setup):
                sell_target = max(sell_target, current_pct)

        # V1.3+: wider sell band in BULL — prevents tiny trims (gap < threshold)
        # that reduce position on normal volatility without real protection.
        bull_sell_threshold = self._get_bull_sell_threshold() if confirmed_state == "BULL" else self.MIN_ADJUST_THRESHOLD

        sell_threshold, max_sell, sell_guard = self._adjust_sell_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
            sell_threshold=bull_sell_threshold,
            max_sell=self._base_max_sell(trend_risk, risk_score),
        )
        max_sell = self._apply_sell_size_limit(max_sell, current_pct, pos, price, latest)

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

        max_buy = self._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)

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

    # ── Hook methods for V1.4 subclass overrides ──

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        """V1.3: faster reentry in BULL — rebuild position quicker after sells."""
        if confirmed_state == "BULL" and current_pct < 0.85:
            return max(max_buy, 0.35)
        return max_buy

    def _apply_sell_size_limit(
        self, max_sell: float, current_pct: float, pos: PositionState, price: float, latest: pd.Series,
    ) -> float:
        """Cap sell size under specific conditions. V1.3 default: no cap."""
        return max_sell

    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        """Override to block sells in healthy BULL. V1.3 default: no blocking."""
        return False

    def _get_bull_sell_threshold(self) -> float:
        """Wider gap threshold in BULL. V1.3 default: 0.10."""
        return 0.10


class V14AStrategy(V13Strategy):
    """V1.4A: Fast re-entry after sells in BULL regime.

    After a sell in BULL, if price re-crosses above EMA24 and risk is low,
    bypass the normal cooldown to buy back quickly (capped at 25%).
    """

    VERSION_LABEL = "v1_4A"
    FAST_REENTRY_WINDOW = 10
    FAST_REENTRY_RISK_CAP = 2
    FAST_REENTRY_MAX_BUY = 0.25

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sell_call = -100
        self._fast_reentry_state = None

    @property
    def name(self) -> str:
        return "v1_4A"

    def _record_executed_action(self, side: str, setup: str) -> None:
        super()._record_executed_action(side, setup)
        if side == "sell":
            self._last_sell_call = self._call_count

    def _is_fast_reentry_active(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        bars_since_sell = self._call_count - self._last_sell_call
        if not (0 < bars_since_sell <= self.FAST_REENTRY_WINDOW):
            return False

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        pos = portfolio.positions.get(symbol, PositionState())
        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        if confirmed_state != "BULL":
            return False
        if risk_score > self.FAST_REENTRY_RISK_CAP:
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False
        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._fast_reentry_state = self._is_fast_reentry_active(
            candles_by_symbol, portfolio, current_prices,
        )
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._fast_reentry_state = None

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._fast_reentry_state:
            return 0, "v1_4A_fast_reentry"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        if self._fast_reentry_state:
            return min(max_buy, self.FAST_REENTRY_MAX_BUY)
        return super()._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)


class V14BStrategy(V13Strategy):
    """V1.4B: Enhanced bull pullback recognition.

    When price < EMA24 but > EMA72 and the long-term trend is still rising,
    treat as a healthy pullback and skip target-reduce/risk-reduce sells.
    When price < EMA72 with EMA24 declining, treat as trend deterioration
    (already handled by existing trend_risk mechanics).
    """

    VERSION_LABEL = "v1_4B"

    @property
    def name(self) -> str:
        return "v1_4B"

    def _is_bull_pullback(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
    ) -> bool:
        if not super()._is_bull_pullback(latest, price, confirmed_state, trend_risk):
            return False
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema168_slope) or ema168_slope <= 0:
            return False
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema72) or pd.isna(ema168):
            return False
        return bool(ema72 > ema168)


class V14CStrategy(V13Strategy):
    """V1.4C: Lightweight drawdown protection.

    When holding a high position (>70%) with little profit buffer (<10%),
    drawdown >8%, and price below a weakening EMA24, cap any sell at
    10-15% to avoid panic-selling minor pullbacks.
    """

    VERSION_LABEL = "v1_4C"
    LIGHT_PROTECTION_MIN_POSITION = 0.70
    LIGHT_PROTECTION_MAX_PROFIT = 0.10
    LIGHT_PROTECTION_MIN_DD = 0.08
    LIGHT_PROTECTION_MAX_SELL = 0.15

    @property
    def name(self) -> str:
        return "v1_4C"

    def _apply_sell_size_limit(
        self,
        max_sell: float,
        current_pct: float,
        pos: PositionState,
        price: float,
        latest: pd.Series,
    ) -> float:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or self._peak_price <= 0:
            return max_sell
        if current_pct < self.LIGHT_PROTECTION_MIN_POSITION:
            return max_sell

        profit_pct = price / pos.avg_cost - 1
        if profit_pct >= self.LIGHT_PROTECTION_MAX_PROFIT:
            return max_sell

        dd_from_peak = 1 - price / self._peak_price
        if dd_from_peak <= self.LIGHT_PROTECTION_MIN_DD:
            return max_sell

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price >= ema24:
            return max_sell

        return min(max_sell, self.LIGHT_PROTECTION_MAX_SELL)


class V14DStrategy(V13Strategy):
    """V1.4D: Combination of A + B + C.

    - Fast re-entry after sells in BULL (A)
    - Enhanced bull pullback recognition (B)
    - Lightweight drawdown protection (C)
    """

    VERSION_LABEL = "v1_4D"

    # ── A: Fast re-entry ──
    FAST_REENTRY_WINDOW = 10
    FAST_REENTRY_RISK_CAP = 2
    FAST_REENTRY_MAX_BUY = 0.25

    # ── C: Lightweight protection ──
    LIGHT_PROTECTION_MIN_POSITION = 0.70
    LIGHT_PROTECTION_MAX_PROFIT = 0.10
    LIGHT_PROTECTION_MIN_DD = 0.08
    LIGHT_PROTECTION_MAX_SELL = 0.15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sell_call = -100
        self._fast_reentry_state = None

    @property
    def name(self) -> str:
        return "v1_4D"

    # ── A: Fast re-entry ──

    def _record_executed_action(self, side: str, setup: str) -> None:
        super()._record_executed_action(side, setup)
        if side == "sell":
            self._last_sell_call = self._call_count

    def _is_fast_reentry_active(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        bars_since_sell = self._call_count - self._last_sell_call
        if not (0 < bars_since_sell <= self.FAST_REENTRY_WINDOW):
            return False

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        pos = portfolio.positions.get(symbol, PositionState())
        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        if confirmed_state != "BULL":
            return False
        if risk_score > self.FAST_REENTRY_RISK_CAP:
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False
        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._fast_reentry_state = self._is_fast_reentry_active(
            candles_by_symbol, portfolio, current_prices,
        )
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._fast_reentry_state = None

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._fast_reentry_state:
            return 0, "v1_4D_fast_reentry"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        if self._fast_reentry_state:
            return min(max_buy, self.FAST_REENTRY_MAX_BUY)
        return super()._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)

    # ── B: Enhanced bull pullback ──

    def _is_bull_pullback(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
    ) -> bool:
        if not super()._is_bull_pullback(latest, price, confirmed_state, trend_risk):
            return False
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema168_slope) or ema168_slope <= 0:
            return False
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema72) or pd.isna(ema168):
            return False
        return bool(ema72 > ema168)

    # ── C: Lightweight protection ──

    def _apply_sell_size_limit(
        self,
        max_sell: float,
        current_pct: float,
        pos: PositionState,
        price: float,
        latest: pd.Series,
    ) -> float:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or self._peak_price <= 0:
            return max_sell
        if current_pct < self.LIGHT_PROTECTION_MIN_POSITION:
            return max_sell

        profit_pct = price / pos.avg_cost - 1
        if profit_pct >= self.LIGHT_PROTECTION_MAX_PROFIT:
            return max_sell

        dd_from_peak = 1 - price / self._peak_price
        if dd_from_peak <= self.LIGHT_PROTECTION_MIN_DD:
            return max_sell

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price >= ema24:
            return max_sell

        return min(max_sell, self.LIGHT_PROTECTION_MAX_SELL)


class V14EStrategy(V14CStrategy):
    """V1.4E: BULL retention — block target-reduce sells in healthy BULL + moderate fast re-entry.

    Inherits V1.4C's sell-size cap.  Adds two changes:
    1. Block target-reduce sells when confirmed BULL with low risk (risk_score <= 2).
       Prevents routine position trimming during healthy uptrends.
    2. After a sell in low-risk BULL, reduce cooldown to 1 instead of base 2.
       Gets back in faster without the V1.4A problem of buying every single bar.
    """

    VERSION_LABEL = "v1_4E"
    REENTRY_WINDOW = 10
    REENTRY_RISK_CAP = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sell_call = -100
        self._reentry_active = False

    @property
    def name(self) -> str:
        return "v1_4E"

    def _record_executed_action(self, side: str, setup: str) -> None:
        super()._record_executed_action(side, setup)
        if side == "sell":
            self._last_sell_call = self._call_count

    def _is_reentry_active(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        bars_since_sell = self._call_count - self._last_sell_call
        if not (0 < bars_since_sell <= self.REENTRY_WINDOW):
            return False

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        pos = portfolio.positions.get(symbol, PositionState())
        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        if confirmed_state != "BULL":
            return False
        if risk_score > self.REENTRY_RISK_CAP:
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False
        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._reentry_active = self._is_reentry_active(
            candles_by_symbol, portfolio, current_prices,
        )
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._reentry_active = False

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._reentry_active:
            return max(effective_cooldown - 1, 1), "v1_4E_reentry"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if confirmed_state != "BULL":
            return False
        if risk_score > 2:
            return False
        return True


class V14FStrategy(V14CStrategy):
    """V1.4F: Aggressive BULL retention — block all sells in very-low-risk BULL + zero cooldown.

    Inherits V1.4C's sell-size cap.  Three changes from V1.4E:
    1. Block BOTH target-reduce AND risk-reduce sells when confirmed BULL with risk_score <= 1.
    2. After a sell in low-risk BULL, zero cooldown (price > EMA24 + risk_score <= 1).
    3. Higher BULL buy floor: max_buy >= 0.50 (was 0.35).
    """

    VERSION_LABEL = "v1_4F"
    REENTRY_WINDOW = 10
    REENTRY_RISK_CAP = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sell_call = -100
        self._reentry_active = False

    @property
    def name(self) -> str:
        return "v1_4F"

    def _record_executed_action(self, side: str, setup: str) -> None:
        super()._record_executed_action(side, setup)
        if side == "sell":
            self._last_sell_call = self._call_count

    def _is_reentry_active(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        bars_since_sell = self._call_count - self._last_sell_call
        if not (0 < bars_since_sell <= self.REENTRY_WINDOW):
            return False

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        pos = portfolio.positions.get(symbol, PositionState())
        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)

        if confirmed_state != "BULL":
            return False
        if risk_score > self.REENTRY_RISK_CAP:
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False
        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._reentry_active = self._is_reentry_active(
            candles_by_symbol, portfolio, current_prices,
        )
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._reentry_active = False

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._reentry_active:
            return 0, "v1_4F_fast_reentry"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        if confirmed_state == "BULL" and current_pct < 0.85:
            return max(max_buy, 0.50)
        return max_buy

    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if confirmed_state != "BULL":
            return False
        if risk_score > 1:
            return False
        return True


class V14GStrategy(V14EStrategy):
    """V1.4G: V1.4E + wider BULL sell threshold (0.15 vs 0.10).

    Prevents even more small trims in BULL by requiring a wider gap (15%)
    between current position and target before selling.  This reduces the
    remaining target-reduce sells that V1.4E allows when risk_score >= 3.
    """

    VERSION_LABEL = "v1_4G"

    @property
    def name(self) -> str:
        return "v1_4G"

    def _get_bull_sell_threshold(self) -> float:
        return 0.15


class V14HStrategy(V14GStrategy):
    """V1.4H: V1.4G + BULL target floor at 0.75.

    When confirmed BULL, cap the sell target lookup so the target never
    drops below BULL[3]=0.75 — even when risk_score is 4+ (which would
    normally target 0.50 or 0.25).  Combined with V1.4G's 0.15 threshold
    and V1.4C's sell-size cap, deep sells in high-risk BULL are eliminated.
    """

    VERSION_LABEL = "v1_4H"

    @property
    def name(self) -> str:
        return "v1_4H"

    def _lookup_target(self, state: str, risk_score: int) -> float:
        target = super()._lookup_target(state, risk_score)
        if state == "BULL":
            target = max(target, self.TARGET_TABLE["BULL"][3])
        return target


class V15BStrategy(V14GStrategy):
    """V1.5B: Strong BULL minimum position 85% / normal BULL 75%.

    Uses risk_score inside _lookup_target to apply position floor:
    - risk_score <= 1 (strong BULL): never target below 85%
    - risk_score in (2, 3) (normal BULL): never target below 75%
    - risk_score >= 4 (weak BULL / turning): existing logic (no floor)

    Builds on V1.4G (target-reduce block + moderate reentry + 0.15 threshold + sell cap).
    """

    VERSION_LABEL = "v1_5B"
    STRONG_BULL_FLOOR = 0.85
    NORMAL_BULL_FLOOR = 0.75

    @property
    def name(self) -> str:
        return "v1_5B"

    def _lookup_target(self, state: str, risk_score: int) -> float:
        target = super()._lookup_target(state, risk_score)
        if state == "BULL":
            if risk_score <= 1:
                target = max(target, self.STRONG_BULL_FLOOR)
            elif risk_score <= 3:
                target = max(target, self.NORMAL_BULL_FLOOR)
        return target


class V15CStrategy(V14GStrategy):
    """V1.5C: Raw BULL early entry.

    When raw_state is BULL but confirmed_state is not yet BULL (still in
    MIXED confirmation window), allow reduced-position entry to capture
    early BULL upside instead of waiting for full confirmation.

    Two changes:
    1. Cooldown reduced to 1 bar when raw BULL is active.
    2. Max buy raised to at least 20% when raw BULL is active (vs normal
       MIXED max_buy which can be as low as ~5-15% depending on config).

    Only activates when risk_score <= 2 and price > EMA24 (healthy BULL
    conditions).  Inherits all V1.4G protections (target-reduce blocking,
    sell-size cap, 15% BULL sell threshold).
    """

    VERSION_LABEL = "v1_5C"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_bull_active = False

    @property
    def name(self) -> str:
        return "v1_5C"

    def _is_raw_bull_active(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        """Detect when raw state is BULL but confirmed is not yet BULL.

        Returns True when:
        - raw_state == "BULL"
        - confirmed_state != "BULL" (confirmation has not completed)
        - risk_score <= 2 (low risk environment)
        - price > EMA24 (healthy uptrend)
        """
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        raw_state = strategy_utils.detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)

        # Only activate when raw BULL but not yet confirmed BULL
        if raw_state != "BULL" or confirmed_state == "BULL":
            return False

        # Risk filter
        pos = portfolio.positions.get(symbol, PositionState())
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)
        if risk_score > 2:
            return False

        # Price must be above EMA24
        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False

        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._raw_bull_active = self._is_raw_bull_active(
            candles_by_symbol, portfolio, current_prices,
        )
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._raw_bull_active = False

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._raw_bull_active:
            return 1, "v1_5C_raw_bull"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        if self._raw_bull_active:
            return max(max_buy, 0.20)
        return super()._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)


class V15DStrategy(V14GStrategy):
    """V1.5D: Dynamic cash reserve.

    In confirmed BULL with low risk (risk_score <= 2), release the 20% cash
    reserve by targeting 100% position.  This deploys idle cash during
    favorable conditions instead of keeping it on the sidelines.

    Mechanically: overrides _compose_target to boost buy_target to 1.0 when
    confirmed BULL with risk_score <= 2, and raises _target_cap to 1.0 so
    the boost is not clamped.  Inherits all V1.4G protections.
    """

    VERSION_LABEL = "v1_5D"

    @property
    def name(self) -> str:
        return "v1_5D"

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
        if side == "buy":
            confirmed_state = self._current_state
            if confirmed_state == "BULL":
                risk_score = min(trend_risk + drawdown_risk, 5)
                if risk_score <= 2:
                    tactical_target = max(tactical_target, 1.0)
        return super()._compose_target(
            symbol, tactical_target, raw_state, trend_risk,
            drawdown_risk, latest, price, side,
        )

    def _target_cap(self) -> float:
        return 1.0


class V16AStrategy(V14GStrategy):
    """V1.6A: Buy/sell risk decoupling + post-sell fast reentry.

    Decouples buy logic from sell logic so risk_score controls sells and
    buy_risk controls buys.  buy_risk starts at trend_risk only (no
    drawdown_risk) with two recovery corrections:

      1. raw_state == BULL and price > EMA24 → buy_risk = max(0, buy_risk - 2)
      2. raw_state in (BULL, MIXED), price > EMA72, ema168_slope > 0
         → buy_risk = max(0, buy_risk - 1)

    Also adds post-sell aggressive reentry: after a sell in BULL/MIXED with
    uptrend structure (ema24 > ema72, price > EMA24), allows cooldown=0
    with max_buy capped at 20% to prevent full-rush reentry while getting
    back in quickly.
    """

    VERSION_LABEL = "v1_6A"
    POST_SELL_REENTRY_WINDOW = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buy_risk_override = None
        self._post_sell_reentry_active = False
        self._last_sell_call = -100

    @property
    def name(self) -> str:
        return "v1_6A"

    def _record_executed_action(self, side: str, setup: str) -> None:
        super()._record_executed_action(side, setup)
        if side == "sell":
            self._last_sell_call = self._call_count

    def _calc_buy_risk(self, trend_risk: int, raw_state: str, latest, price) -> int:
        buy_risk = trend_risk
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")

        if raw_state == "BULL" and not pd.isna(ema24) and price > ema24:
            buy_risk = max(0, buy_risk - 2)
        elif raw_state in ("BULL", "MIXED") and not pd.isna(ema72) and price > ema72:
            ema168_slope = latest.get("ema168_slope")
            if not pd.isna(ema168_slope) and ema168_slope > 0:
                buy_risk = max(0, buy_risk - 1)
        return buy_risk

    def _is_post_sell_aggressive_reentry(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> bool:
        bars_since_sell = self._call_count - self._last_sell_call
        if not (0 < bars_since_sell <= self.POST_SELL_REENTRY_WINDOW):
            return False

        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return False
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return False

        raw_state = strategy_utils.detect_market_state(latest)
        if raw_state not in ("BULL", "MIXED"):
            return False

        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price <= ema24:
            return False

        ema72 = latest.get("ema72")
        if pd.isna(ema72) or ema24 <= ema72:
            return False
        return True

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        self._buy_risk_override = None
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol:
            df = candles_by_symbol.get(symbol)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                price = current_prices.get(symbol, 0.0)
                if price > 0:
                    trend_risk = self._calculate_trend_risk(latest, price)
                    raw_state = strategy_utils.detect_market_state(latest)
                    self._buy_risk_override = self._calc_buy_risk(
                        trend_risk, raw_state, latest, price,
                    )

        self._post_sell_reentry_active = self._is_post_sell_aggressive_reentry(
            candles_by_symbol, portfolio, current_prices,
        )

        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._buy_risk_override = None
            self._post_sell_reentry_active = False

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
        if side == "buy" and self._buy_risk_override is not None:
            confirmed_state = self._current_state
            buy_risk = min(self._buy_risk_override, 5)
            new_target = self._lookup_target(confirmed_state, buy_risk)
            tactical_target = max(tactical_target, new_target)
        return super()._compose_target(
            symbol, tactical_target, raw_state, trend_risk,
            drawdown_risk, latest, price, side,
        )

    def _compute_buy_cooldown(self, state: str, cfg: dict, risk_score: int) -> int:
        if self._buy_risk_override is not None:
            return super()._compute_buy_cooldown(state, cfg, self._buy_risk_override)
        return super()._compute_buy_cooldown(state, cfg, risk_score)

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._post_sell_reentry_active:
            return 0, "v1_6A_post_sell_reentry"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        if self._post_sell_reentry_active:
            return min(max_buy, 0.20)
        return super()._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)


class V16BStrategy(V16AStrategy):
    """V1.6B: V1.6A + BULL dip buying.

    When confirmed BULL, price is between EMA24 and EMA72, EMA72 is above
    EMA168 (rising mid-term structure), and BTC is not in systemic crash:

      buy_target = max(buy_target, 0.80)
      max_buy    = max(max_buy, 0.10)

    This allows small dip-buys during healthy BULL pullbacks instead of
    treating every price drop below EMA24 as a sell signal.
    """

    VERSION_LABEL = "v1_6B"

    @property
    def name(self) -> str:
        return "v1_6B"

    def _is_bull_dip_setup(
        self,
        confirmed_state: str,
        latest: pd.Series,
        price: float,
    ) -> bool:
        if confirmed_state != "BULL":
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (ema72 < price < ema24):
            return False
        if ema72 <= ema168:
            return False
        btc_regime = str(latest.get("btc_regime", ""))
        if btc_regime == "BEAR":
            return False
        return True

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
        if side == "buy":
            confirmed_state = self._current_state
            if self._is_bull_dip_setup(confirmed_state, latest, price):
                tactical_target = max(tactical_target, 0.80)
        return super()._compose_target(
            symbol, tactical_target, raw_state, trend_risk,
            drawdown_risk, latest, price, side,
        )

    def _adjust_bull_buy_max_buy(self, max_buy: float, confirmed_state: str, current_pct: float) -> float:
        # Post-sell reentry cap must take precedence (from V1.6A)
        if self._post_sell_reentry_active:
            return min(max_buy, 0.20)
        # BULL dip buying — ensure at least 10% buy capacity
        if confirmed_state == "BULL":
            latest = getattr(self, "_latest_for_hooks", None)
            if latest is not None:
                price = latest.get("close", 0)
                if self._is_bull_dip_setup(confirmed_state, latest, price):
                    return max(max_buy, 0.10)
        return super(V16AStrategy, self)._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol:
            df = candles_by_symbol.get(symbol)
            if df is not None and not df.empty:
                self._latest_for_hooks = df.iloc[-1]
        try:
            return super().compute_actions(candles_by_symbol, portfolio, current_prices)
        finally:
            self._latest_for_hooks = None


class V16CStrategy(V14GStrategy):
    """V1.6C: BTC_BEAR layered buy restriction (standalone).

    Replaces the single BTC_BEAR_TARGET_GAP_MULT (0.25 for all assets)
    with a layered multiplier based on the asset's own raw state:

      - asset raw_state == BULL  → max_buy *= 0.60
      - asset raw_state == MIXED → max_buy *= 0.40
      - asset raw_state == BEAR  → max_buy *= 0.25

    This prevents BTC_BEAR from indiscriminately slashing altcoin buys
    when the altcoin itself is structurally strong.
    """

    VERSION_LABEL = "v1_6C"
    BTC_BEAR_LAYERED = {"BULL": 0.60, "MIXED": 0.40, "BEAR": 0.25}

    @property
    def name(self) -> str:
        return "v1_6C"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
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
            layer_mult = self.BTC_BEAR_LAYERED.get(raw_state, 0.25)
            max_buy *= layer_mult
            guard = self._join_guard(
                guard, f"btc-bear-{raw_state.lower()}-x{layer_mult:.2f}",
            )
        return max_buy, guard


class V16EStrategy(V16BStrategy):
    """V1.6E: V1.6A + BULL dip buying + BTC_BEAR layered.

    Combines all V1.6 features:
    - buy_risk / sell_risk decoupling
    - post-sell aggressive reentry
    - BULL dip buying
    - BTC_BEAR layered by asset raw_state
    """

    VERSION_LABEL = "v1_6E"
    BTC_BEAR_LAYERED = {"BULL": 0.60, "MIXED": 0.40, "BEAR": 0.25}

    @property
    def name(self) -> str:
        return "v1_6E"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
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
            layer_mult = self.BTC_BEAR_LAYERED.get(raw_state, 0.25)
            max_buy *= layer_mult
            guard = self._join_guard(
                guard, f"btc-bear-{raw_state.lower()}-x{layer_mult:.2f}",
            )
        return max_buy, guard


class V17Strategy(V13Strategy):
    """V1.7: V1.2 framework + targeted protections from V1.4/V1.5.

    Core = V1.2 (fast entry/exit, high BULL targets, 0.06 threshold,
    recovery override, strong BULL sell suppression, bull guard).

    Adds from V1.3:   MIXED flicker protection (raw=MIXED+conf=BULL
                       use BULL table).
    Adds from V1.4B:  Stricter bull pullback (requires ema168_slope>0
                       and ema72>ema168).
    Adds from V1.4C:  Sell-size cap (15% max in vulnerable positions).
    <!-- no P3 cash reserve -- incompatible with V1.2 fast framework -->

    Removes from V1.3:  aggressive BULL buying (_adjust_bull_buy_max_buy)
                         and wider BULL threshold (0.10 back to 0.06).
    Removes from V1.4E: BULL retention  covered by MIXED flicker
                         protection with less drawdown risk.

    Key design principle: only block target-reduce (routine trims),
    never risk-reduce or trend-break.  Keep sell decisions fast.
    """

    VERSION_LABEL = "v1_7"

    #  Sell-size cap (V1.4C)
    LIGHT_PROTECTION_MIN_POSITION = 0.70
    LIGHT_PROTECTION_MAX_PROFIT = 0.10
    LIGHT_PROTECTION_MIN_DD = 0.08
    LIGHT_PROTECTION_MAX_SELL = 0.15

    @property
    def name(self) -> str:
        return "v1_7"

    #  Restore V1.2s narrow BULL sell threshold
    def _get_bull_sell_threshold(self) -> float:
        return 0.06

    #  Disable V1.3s aggressive BULL buying floor
    def _adjust_bull_buy_max_buy(
        self, max_buy: float, confirmed_state: str, current_pct: float,
    ) -> float:
        return max_buy

    #  Disable BULL retention  MIXED flicker protection is sufficient
    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        return False

    #  V1.4B: stricter bull pullback (requires structure confirmation)
    def _is_bull_pullback(
        self,
        latest: pd.Series,
        price: float,
        confirmed_state: str,
        trend_risk: int,
    ) -> bool:
        if confirmed_state != "BULL":
            return False
        if trend_risk >= 2:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (ema24 > ema72 > ema168 and price > ema72 and price <= ema24):
            return False
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema168_slope) or ema168_slope <= 0:
            return False
        return True

    #  V1.4C: sell-size cap
    def _apply_sell_size_limit(
        self,
        max_sell: float,
        current_pct: float,
        pos: PositionState,
        price: float,
        latest: pd.Series,
    ) -> float:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or self._peak_price <= 0:
            return max_sell
        if current_pct < self.LIGHT_PROTECTION_MIN_POSITION:
            return max_sell
        profit_pct = price / pos.avg_cost - 1
        if profit_pct >= self.LIGHT_PROTECTION_MAX_PROFIT:
            return max_sell
        dd_from_peak = 1 - price / self._peak_price
        if dd_from_peak <= self.LIGHT_PROTECTION_MIN_DD:
            return max_sell
        ema24 = latest.get("ema24")
        if pd.isna(ema24) or price >= ema24:
            return max_sell
        return min(max_sell, self.LIGHT_PROTECTION_MAX_SELL)

    #  P3 removed (cash reserve): incompatible with V1.2 fast framework.
    #  V1.5D's cash reserve causes buy-high-sell-low on V1.2's 0.06 threshold.
    #  Rely on V1.2's native target table (max 0.98 via _target_cap default).


class V18Strategy(V17Strategy):
    """V1.8: V1.7 + instant confirmation + raw-state sell target.

    V1.7 analysis showed MIXED flicker protection is net negative in crypto
    — holding through MIXED signals in BULL gains small upside from fake
    signals but gets crushed when the signal is a real violent reversal.
    V1.2's "sell fast, buy cautious" asymmetry is correct for sells but
    the cautious buy side (5-bar MIXED confirm) also misses recovery entries.

    V1.8 changes from V1.7:
    1. Instant confirmation: CONFIRM_BARS=1 for ALL states.  Buys react
       as fast as sells to state changes.
    2. Remove MIXED flicker: sell target always uses raw_state directly.
       (With CONFIRM_BARS=1 this is largely redundant, but explicit.)
    """

    VERSION_LABEL = "v1_8"
    CONFIRM_BARS = {"BULL": 1, "MIXED": 1, "BEAR": 1}

    @property
    def name(self) -> str:
        return "v1_8"

    def _get_sell_target_state(
        self, raw_state: str, confirmed_state: str,
    ) -> tuple[str, int]:
        """Remove MIXED flicker — always use raw_state directly."""
        return raw_state, 0


class V19Strategy(V17Strategy):
    """V1.9: V1.2 + trend-quality-based BULL retention.

    All previous BULL retention approaches (MIXED flicker, blanket
    retention, cash reserve) failed because they applied to ALL BULL
    conditions equally.  When the trend reversed, they held too long.

    V1.9 is selective: only block target-reduce sells when the BULL
    trend is OBJECTIVELY STRONG — confirmed by multiple confluent
    conditions (trend_risk=0, risk_score<=1, EMA alignment, positive
    slope, price well above EMA24).  In weak or ambiguous BULL,
    behavior is identical to V1.2 (sell on raw_state, 6% threshold).

    Changes from V1.7:
    1. Remove MIXED flicker (proven harmful in V1.7 analysis).
    2. Block target-reduce sells only in objectively strong BULL.
    3. Wider sell threshold (0.10) in strong BULL to reduce marginal trims.
    """

    VERSION_LABEL = "v1_9"
    STRONG_BULL_MIN_PRICE_ABOVE_EMA24 = 0.03  # 3% above EMA24

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest_bar: pd.Series | None = None
        self._current_price: float = 0.0

    @property
    def name(self) -> str:
        return "v1_9"

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        """Save bar state for hook methods, then delegate."""
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol:
            df = candles_by_symbol.get(symbol)
            if df is not None and not df.empty:
                self._latest_bar = df.iloc[-1]
                self._current_price = current_prices.get(symbol, 0.0)
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _get_sell_target_state(
        self, raw_state: str, confirmed_state: str,
    ) -> tuple[str, int]:
        """Remove MIXED flicker — always use raw_state directly."""
        return raw_state, 0

    def _get_bull_sell_threshold(self) -> float:
        """Wider threshold in strong BULL lets position run before trimming."""
        if self._is_strong_bull_market():
            return 0.10
        return 0.06

    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        """Block target-reduce sells only in objectively strong BULL."""
        if sell_setup != "target-reduce":
            return False
        if confirmed_state != "BULL":
            return False
        if trend_risk != 0:
            return False
        if risk_score > 1:
            return False
        return self._is_strong_bull_market()

    def _is_strong_bull_market(self) -> bool:
        """True when multiple confluent conditions confirm a healthy BULL."""
        latest = self._latest_bar
        price = self._current_price
        if latest is None or price <= 0:
            return False

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False

        # Full EMA alignment
        if not (ema24 > ema72 > ema168):
            return False

        # Price must show clear momentum, not hugging support
        if price < ema24 * (1 + self.STRONG_BULL_MIN_PRICE_ABOVE_EMA24):
            return False

        # Long-term trend must be positive
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema168_slope) or ema168_slope <= 0:
            return False

        return True


class V19AStrategy(V19Strategy):
    """V1.9A: Block ALL target-reduce in BULL, not just strong BULL.

    V1.9 proved selective retention improves BULL win rate (0.278 -> 0.309)
    but the strong BULL filter is too strict to reach 0.35 threshold.

    Changes from V1.9:
    1. Block ALL target-reduce sells when confirmed_state == "BULL".
       Risk-reduce and trend-break sells still fire normally.
    2. Wider sell threshold (0.10) in ALL BULL, not just strong BULL.
    3. No MIXED flicker (inherited from V1.9).
    """

    VERSION_LABEL = "v1_9A"

    @property
    def name(self) -> str:
        return "v1_9A"

    def _get_bull_sell_threshold(self) -> float:
        """Wider threshold in ALL BULL -- let positions run longer."""
        return 0.10

    def _is_bull_sell_blocked(
        self,
        confirmed_state: str,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        """Block ALL target-reduce in BULL. Risk-reduce/trend-break still fire."""
        if sell_setup != "target-reduce":
            return False
        return confirmed_state == "BULL"


class V19BStrategy(V19AStrategy):
    """V1.9B: Block ALL sells in BULL — maximum possible retention.

    V1.9A blocks target-reduce in BULL but risk-reduce and trend-break
    still fire. V1.9B goes further: downgrade all sells to target-reduce
    in BULL so they all get blocked. This means the strategy holds 100%
    through BULL and only exits when BULL state ends.

    The risk is higher drawdown if BULL ends with a crash. CONFIRM_BARS
    (MIXED=5, BEAR=2) provide the exit signal delay.

    Changes from V1.9A:
    1. Downgrade ALL sell types to target-reduce in BULL via
       _classify_sell_setup override. All get blocked by parent's
       _is_bull_sell_blocked.
    2. Everything else identical to V1.9A.
    """

    VERSION_LABEL = "v1_9B"

    @property
    def name(self) -> str:
        return "v1_9B"

    def _classify_sell_setup(
        self,
        trend_risk: int,
        risk_score: int,
        latest: pd.Series,
        price: float,
        raw_state: str,
        drawdown_risk: int,
    ) -> str:
        """In BULL, all sells become target-reduce (blocked by parent hook)."""
        if raw_state == "BULL":
            return "target-reduce"
        return super()._classify_sell_setup(trend_risk, risk_score, latest, price, raw_state, drawdown_risk)


class V19CStrategy(V19AStrategy):
    """V1.9C: V1.9A + aggressive BULL buying + wider sell threshold.

    V1.9A blocks target-reduce in BULL (score 0.652, BULL win rate 0.3196)
    but still underperforms BH in 68% of BULL windows. The two remaining
    issues are slow re-entry after pullbacks and marginal trims that still
    leak through.

    Changes from V1.9A:
    1. Wider sell threshold in BULL (0.12 vs 0.10) — fewer marginal trims.
    2. Aggressive BULL buying — boost max_buy to 0.50 in confirmed BULL
       when position is below 85% (re-enable V1.3's floor).
    3. Block target-reduce in BULL (inherited).
    4. Wider sell threshold in all BULL (inherited).
    """

    VERSION_LABEL = "v1_9C"

    @property
    def name(self) -> str:
        return "v1_9C"

    def _get_bull_sell_threshold(self) -> float:
        """Wider threshold in ALL BULL — fewer marginal trims."""
        return 0.12

    def _adjust_bull_buy_max_buy(
        self, max_buy: float, confirmed_state: str, current_pct: float,
    ) -> float:
        """Aggressive BULL buying: boost max_buy to 0.50 when below 85%."""
        if confirmed_state == "BULL" and current_pct < 0.85:
            return max(max_buy, 0.50)
        return max_buy


class V19DStrategy(V19AStrategy):
    """V1.9D: V1.9A + faster MIXED position building.

    V1.9A (score 0.652, BULL win rate 0.3196) blocks target-reduce in BULL
    effectively but still enters BULL with low position because MIXED
    state has conservative buying (cooldown=4, max_buy=0.20).

    V1.9D accelerates MIXED buying so the strategy enters BULL markets
    with higher initial position:
    1. MIXED cooldown 4 -> 2 (buy twice as often)
    2. MIXED max_buy 0.20 -> 0.35 (buy almost 2x each time)
    3. Same BULL retention as V1.9A (block target-reduce in BULL).

    Combined effect: from a 5% BEAR position, reaching 85% takes ~5 bars
    in MIXED (vs ~9 bars with original params).
    """

    VERSION_LABEL = "v1_9D"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 2},
        "MIXED": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 2},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9D"


class V19EStrategy(V19AStrategy):
    """V1.9E: Use confirmed_state for sell decisions.

    V1.9A uses raw_state for sell lookups, which means when raw_state
    flickers to MIXED during a BULL pullback, the sell target drops to
    MIXED table levels. Even though target-reduce is blocked, the MIXED
    target cascades into future decisions.

    V1.9E always uses confirmed_state for sell_lookup_state (with 0 risk
    penalty).  This aligns sell decisions with the same state used for
    buy decisions, preventing premature target adjustment during brief
    pullbacks.

    Unlike V13Strategy's MIXED flicker protection (which used BULL table
    +1 risk), V1.9E uses confirmed_state directly with no penalty — the
    same state as buy lookups.  Risk-reduce (risk_score>=4) and trend-break
    (trend_risk>=3) still fire independently of the lookup state.

    Changes from V1.9A:
    1. _get_sell_target_state returns (confirmed_state, 0) always.
    2. Everything else identical to V1.9A.
    """

    VERSION_LABEL = "v1_9E"

    @property
    def name(self) -> str:
        return "v1_9E"

    def _get_sell_target_state(
        self, raw_state: str, confirmed_state: str,
    ) -> tuple[str, int]:
        """Always use confirmed_state for sell decisions."""
        return confirmed_state, 0


class V19FStrategy(V19AStrategy):
    """V1.9F: V1.9A + BULL cooldown 1 (buy every bar).

    V1.9A's BULL cooldown=2 means buying every 2 bars. After a sell or
    state transition, it takes ~4 days to reach full position. In a fast
    BULL rally, BH is at 100% from day 1.

    V1.9F reduces BULL base_cooldown from 2 to 1, allowing daily buying.
    Combined with max_buy=0.35, this fills from 30% to 98% in 2 days
    instead of 4.

    Result: BULL win rate 0.3299 (vs V1.9A 0.3196) — first measurable
    improvement, proving cooldown reduction works.

    Changes from V1.9A:
    1. BULL base_cooldown 2 -> 1.
    """

    VERSION_LABEL = "v1_9F"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.20, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9F"


class V19GStrategy(V19FStrategy):
    """V1.9G: V1.9F + BULL max_buy 0.50 (faster fill).

    V1.9F proved cooldown reduction works (BULL win 0.3299). V1.9G
    also increases BULL max_buy from 0.35 to 0.50, so the first buy
    after a BULL start fills to 80% instead of 65%, further increasing
    early BULL exposure.

    Combined cooldown=1 + max_buy=0.50:
    - Day 1: fill to 80% (vs 65% with 0.35)
    - Day 2: fill to 98% (full position)

    Result: BULL win 0.3299 (same as V1.9F). Max_buy increase didn't
    help further — cooldown is the binding constraint.

    Changes from V1.9F:
    1. BULL max_buy 0.35 -> 0.50.
    """

    VERSION_LABEL = "v1_9G"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.50, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.20, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9G"


class V19HStrategy(V19FStrategy):
    """V1.9H: V1.9F + MIXED max_buy 0.30 (better pre-BULL position).

    V1.9F (BULL cooldown=1) got BULL win to 0.3299. The remaining
    bottleneck is MIXED position building before BULL confirmation.

    When emerging from BEAR, confirmed_state must pass through MIXED
    (5 bars to confirm). With max_buy=0.20 and cooldown=4 in MIXED,
    reaching 65% takes 9 bars. Increasing MIXED max_buy to 0.30
    shortens this to 5 bars, so the strategy enters BULL with ~65%
    instead of ~45%.

    Result: BULL win 0.3402 — gap to 0.35 is now only 0.0098!

    Changes from V1.9F:
    1. MIXED max_buy 0.20 -> 0.30.
    2. BULL cooldown=1 (inherited).
    """

    VERSION_LABEL = "v1_9H"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.30, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9H"


class V19IStrategy(V19HStrategy):
    """V1.9I: V1.9H + MIXED max_buy 0.35 (even faster pre-BULL fill).

    V1.9H (MIXED max_buy=0.30) reached BULL win 0.3402. V1.9I
    pushes MIXED max_buy to 0.35 to test if more helps. It doesn't
    — still 0.3402. Max_buy plateau is at 0.30 for MIXED.

    Changes from V1.9H:
    1. MIXED max_buy 0.30 -> 0.35.
    """

    VERSION_LABEL = "v1_9I"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9I"


class V19JStrategy(V19HStrategy):
    """V1.9J: V1.9H + MIXED cooldown 3 (buy more often in MIXED).

    V1.9H (MIXED max_buy=0.30, cooldown=4) reached BULL win 0.3402.
    V1.9J reduces MIXED cooldown to 3 — still 0.3402. No further
    improvement.

    Changes from V1.9H:
    1. MIXED base_cooldown 4 -> 3.
    """

    VERSION_LABEL = "v1_9J"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.35, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.30, "max_sell": 0.25, "base_cooldown": 3},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9J"


class V19KStrategy(V19HStrategy):
    """V1.9K: V1.9H + BULL max_buy 0.50 (fill faster on BULL confirmation).

    V1.9H (MIXED max_buy=0.30, BULL cooldown=1, BULL max_buy=0.35)
    reached BULL win 0.3402. V1.9K adds BULL max_buy=0.50 so the
    first few BULL bars fill the remaining gap faster.

    With MIXED max_buy=0.30 at entry + BULL max_buy=0.50:
    - Enter BULL at ~65% position
    - Day 1 BULL: fill to 98% (vs 88% with max_buy=0.35)

    Changes from V1.9H:
    1. BULL max_buy 0.35 -> 0.50.
    """

    VERSION_LABEL = "v1_9K"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.50, "max_sell": 0.25, "base_cooldown": 1},
        "MIXED": {"max_buy": 0.30, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9"


class V19LStrategy(V19KStrategy):
    """V1.9L: V1.9K + BULL cooldown 0 (buy every bar, no wait).

    V1.9K's BULL cooldown=1 means buying every bar — full position
    in 2 bars. V1.9L tries cooldown=0. Result: identical to V1.9K.
    max_buy=0.50 is the binding constraint, not cooldown.

    Changes from V1.9K:
    1. BULL base_cooldown 1 -> 0.
    """

    VERSION_LABEL = "v1_9L"

    STATE_CONFIG = {
        "BULL": {"max_buy": 0.50, "max_sell": 0.25, "base_cooldown": 0},
        "MIXED": {"max_buy": 0.30, "max_sell": 0.25, "base_cooldown": 4},
        "BEAR": {"max_buy": 0.05, "max_sell": 0.25, "base_cooldown": 48},
    }

    @property
    def name(self) -> str:
        return "v1_9L"


class V19MStrategy(V19KStrategy):
    """V1.9M: V1.9K + MIXED confirm bars 5->3 (faster state transition).

    Result: score 0.6458, BULL win 0.3299. Reducing MIXED confirm HURT
    performance — the strategy exits BULL too early on 3-bar flickers.
    5-bar confirm is a feature, not a bug.

    Changes from V1.9K:
    1. CONFIRM_BARS MIXED 5 -> 3.
    """

    VERSION_LABEL = "v1_9M"
    CONFIRM_BARS = {"BULL": 1, "MIXED": 3, "BEAR": 2}

    @property
    def name(self) -> str:
        return "v1_9M"


class V19NStrategy(V19KStrategy):
    """V1.9N: V1.9K + trend-quality adaptive sell threshold.

    Dynamic threshold based on EMA alignment. Result: identical to V1.9K
    (0.6533, BULL win 0.3505). Threshold only affects risk-reduce
    and trend-break sells which don't fire in BULL — no impact.

    Changes from V1.9K:
    1. _get_bull_sell_threshold returns dynamic value based on EMAs.
    """

    VERSION_LABEL = "v1_9N"

    @property
    def name(self) -> str:
        return "v1_9N"

    def _get_bull_sell_threshold(self) -> float:
        latest = getattr(self, "_latest_bar", None)
        if latest is None:
            return 0.10
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if any(pd.isna(x) for x in (ema24, ema72, ema168)):
            return 0.10
        if ema24 > ema72 > ema168:
            return 0.15
        if ema24 > ema72:
            return 0.10
        return 0.06


class V19OStrategy(V19KStrategy):
    """V1.9O: V1.9K + post-pullback buy boost.

    After a MIXED->BULL state transition (the strategy survived a pullback
    and BULL is re-confirmed), temporarily boost buying for the next bars.
    This accelerates re-entry after selling during the pullback.

    Boost parameters:
    - COOLDOWN: 0 for POST_PULLBACK_BOOST_BARS after transition
    - MAX_BUY: 0.75 for POST_PULLBACK_BOOST_BARS after transition
    - Boost is consumed on each buy call, limited by POST_PULLBACK_BOOST_BARS

    Changes from V1.9K:
    1. Track MIXED->BULL transitions, activate boost.
    2. _adjust_buy_cooldown: return 0 during boost.
    3. _adjust_bull_buy_max_buy: return 0.75 during boost.
    """

    VERSION_LABEL = "v1_9O"
    POST_PULLBACK_BOOST_BARS = 3
    POST_PULLBACK_MAX_BUY = 0.75

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_pullback_boost = 0

    @property
    def name(self) -> str:
        return "v1_9O"

    def compute_actions(
        self,
        candles_by_symbol: dict[str, pd.DataFrame],
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> list[Action]:
        prev_state = getattr(self, "_current_state", None)
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        curr_state = getattr(self, "_current_state", None)
        if prev_state == "MIXED" and curr_state == "BULL":
            self._post_pullback_boost = self.POST_PULLBACK_BOOST_BARS
        return actions

    def _adjust_buy_cooldown(self, buy_setup: str, effective_cooldown: int) -> tuple[int, str]:
        if self._post_pullback_boost > 0:
            return 0, "post-pullback-boost"
        return super()._adjust_buy_cooldown(buy_setup, effective_cooldown)

    def _adjust_bull_buy_max_buy(
        self, max_buy: float, confirmed_state: str, current_pct: float,
    ) -> float:
        if self._post_pullback_boost > 0:
            self._post_pullback_boost -= 1
            return max(max_buy, self.POST_PULLBACK_MAX_BUY)
        return super()._adjust_bull_buy_max_buy(max_buy, confirmed_state, current_pct)
