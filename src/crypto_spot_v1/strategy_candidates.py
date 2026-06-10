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

        raw_state = self._detect_market_state(latest)
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
            confirmed_state=confirmed_state,
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
        if buy_qty <= 1e-12:
            return []
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

    def _detect_market_state(self, latest: pd.Series) -> str:
        """Override point for fast BULL detection. Base: standard EMA alignment."""
        return strategy_utils.detect_market_state(latest)

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

        raw_state = self._detect_market_state(latest)
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
            confirmed_state=confirmed_state,
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
        if buy_qty <= 1e-12:
            return []
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


class V110Strategy(V19KStrategy):
    """V1.10: V1.9K + volatility scaling.

    The ablation study showed vol scaling (B) is the only clearly effective
    improvement: it reduced buying at high-vol market tops while leaving
    the core V1.9K BULL retention and fast re-entry logic untouched.

    Scaling rules:
    - atr_pct_rank > 0.95: target-gap ×0.60, pullback/recovery ×0.75
    - atr_pct_rank 0.85–0.95: target-gap ×0.80
    """

    VERSION_LABEL = "v1_10"
    VOL_SCALE_HIGH = 0.80
    VOL_SCALE_EXTREME = 0.60
    VOL_SCALE_PULLBACK_RECOVERY = 0.75

    @property
    def name(self) -> str:
        return "v1_10"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        """Reduce max_buy when volatility is elevated."""
        atr_rank = latest.get("atr_pct_rank")
        if not pd.isna(atr_rank):
            if atr_rank > 0.95:
                if buy_setup in ("pullback", "safe-recovery"):
                    max_buy *= self.VOL_SCALE_PULLBACK_RECOVERY
                elif buy_setup == "target-gap":
                    max_buy *= self.VOL_SCALE_EXTREME
            elif atr_rank > 0.85 and buy_setup == "target-gap":
                max_buy *= self.VOL_SCALE_HIGH
        return super()._adjust_buy_execution(latest, price, raw_state, buy_setup, max_buy, confirmed_state=confirmed_state)


class V24Strategy(V110Strategy):
    """V2.4: fast BULL detection plus faster accumulation.

    Main changes over V1.10:
    - Fast BULL detection when ema72 > ema168, price > ema24,
      ema24_slope > 0, and roc_10 > 0.
    - Conditional MIXED cooldown acceleration with roc_5 > 0.
    - MIXED subtype buy sizing.
    - Extended volatility scaling.
    - Graduated cost-aware buy reduction.
    """

    VERSION_LABEL = "v2_4"

    # ── MIXED subtype buy multipliers ──
    MIXED_ACCUMULATION_BUY_MULT = 1.0
    MIXED_NEUTRAL_BUY_MULT = 0.60
    MIXED_DISTRIBUTION_BUY_MULT = 0.50

    # ── Extended vol scaling ──
    VOL_SCALE_HIGH = 0.75
    VOL_SCALE_EXTREME = 0.55
    VOL_SCALE_PULLBACK_RECOVERY_HIGH = 0.80
    VOL_SCALE_PULLBACK_RECOVERY_EXTREME = 0.65

    # ── Recovery override acceleration ──
    RECOVERY_BUY_SIZE_MULT = 1.5

    # ── Cost-aware buy protection ──
    COST_AWARE_BUY_PROTECTION_MULT = 0.50  # ratio 0.95-1.0 (near sell price)
    COST_AWARE_BUY_MODERATE_MULT = 0.75    # ratio 1.0-1.05 (moderate follow-through)

    # ── Compound reduction floor ──
    BUY_REDUCTION_FLOOR = 0.25

    # ── Faster BEAR exit (1 bar instead of 2) ──
    CONFIRM_BARS = {"BULL": 1, "MIXED": 5, "BEAR": 1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sell_price = 0.0

    @property
    def name(self) -> str:
        return "v2_4"

    # ── Direction A: fast BULL detection ──

    def _detect_market_state(self, latest: pd.Series) -> str:
        """Fast BULL detection: enter earlier when medium-term structure supports it.

        When standard EMA alignment returns MIXED, check if the medium-term
        trend is already bullish (ema72 > ema168) and short-term price action
        confirms (price > ema24, ema24_slope > 0, roc_10 > 0).

        The critical filter is ema72 > ema168 — this requires multi-month
        uptrend context.  Without it (as in the failed PRE_BULL experiment),
        fast detection fires on bear-market rallies and degrades results.
        """
        standard = strategy_utils.detect_market_state(latest)
        if standard != "MIXED":
            return standard

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        price = latest.get("close", 0.0)
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or price <= 0:
            return standard

        if not (ema72 > ema168 and price > ema24):
            return standard

        ema24_slope = latest.get("ema24_slope")
        roc_10 = latest.get("roc_10")
        if pd.isna(ema24_slope) or pd.isna(roc_10):
            return standard

        if ema24_slope > 0 and roc_10 > 0:
            return "BULL"

        return standard

    # ── Wrapper: save bar/position state, update trade tracking ──

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol:
            df = candles_by_symbol.get(symbol)
            if df is not None and not df.empty:
                self._latest_bar = df.iloc[-1]
                self._current_price = current_prices.get(symbol, 0.0)

        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)

        if actions:
            action = actions[0]
            if action.side == "sell":
                self._last_sell_price = action.price

        return actions

    def _compute_buy_cooldown(self, state: str, cfg: dict, risk_score: int) -> int:
        """Accelerate MIXED buying when price is above EMA24 with momentum.

        In MIXED with low risk and positive short-term price action, reduce
        the cooldown to build position faster before BULL confirmation.
        roc_5 > 0 filter prevents false breakouts — only accelerates when
        short-term momentum genuinely supports the move above EMA24.
        """
        base = super()._compute_buy_cooldown(state, cfg, risk_score)
        if state == "MIXED" and risk_score <= 1:
            latest = self._latest_bar
            if latest is not None:
                price = latest.get("close", 0.0)
                ema24 = latest.get("ema24")
                roc_5 = latest.get("roc_5")
                if (
                    not pd.isna(ema24) and not pd.isna(roc_5)
                    and price > ema24 and roc_5 > 0
                ):
                    return max(1, base // 3)
        return base

    @staticmethod
    def _classify_mixed_subtype(latest: pd.Series, price: float) -> str:
        """Classify MIXED environment as 'accumulation', 'neutral', or 'distribution'."""
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        if pd.isna(ema24) or pd.isna(ema72):
            return "neutral"
        if price > ema24:
            return "accumulation"
        if price < ema72:
            return "distribution"
        return "neutral"

    # ── Change 2+4: Combined buy execution ──

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        """MIXED subtype multiplier + extended vol scaling + cost-aware reduction."""
        guard = ""
        original_max_buy = max_buy  # for compound floor

        # 1 ── MIXED subtype multiplier (only when confirmed_state is also MIXED) ──
        if buy_setup == "target-gap" and raw_state == "MIXED" and confirmed_state == "MIXED":
            subtype = self._classify_mixed_subtype(latest, price)
            mult = {
                "accumulation": self.MIXED_ACCUMULATION_BUY_MULT,
                "neutral": self.MIXED_NEUTRAL_BUY_MULT,
                "distribution": self.MIXED_DISTRIBUTION_BUY_MULT,
            }.get(subtype, 1.0)
            max_buy *= mult
            guard = self._join_guard(guard, f"mixed_{subtype}_x{mult:.2f}")

        # 2 ── Extended vol scaling (replaces V110 logic entirely) ──
        atr_rank = latest.get("atr_pct_rank")
        if not pd.isna(atr_rank):
            if atr_rank > 0.95:
                if buy_setup in ("pullback", "safe-recovery"):
                    max_buy *= self.VOL_SCALE_PULLBACK_RECOVERY_EXTREME
                    guard = self._join_guard(
                        guard, f"v2_4_vol_ext_pb_x{self.VOL_SCALE_PULLBACK_RECOVERY_EXTREME:.2f}"
                    )
                elif buy_setup == "target-gap":
                    max_buy *= self.VOL_SCALE_EXTREME
                    guard = self._join_guard(
                        guard, f"v2_4_vol_ext_tgap_x{self.VOL_SCALE_EXTREME:.2f}"
                    )
            elif atr_rank > 0.85:
                if buy_setup == "target-gap":
                    max_buy *= self.VOL_SCALE_HIGH
                    guard = self._join_guard(
                        guard, f"v2_4_vol_high_tgap_x{self.VOL_SCALE_HIGH:.2f}"
                    )
                elif buy_setup in ("pullback", "safe-recovery"):
                    max_buy *= self.VOL_SCALE_PULLBACK_RECOVERY_HIGH
                    guard = self._join_guard(
                        guard, f"v2_4_vol_high_pb_x{self.VOL_SCALE_PULLBACK_RECOVERY_HIGH:.2f}"
                    )

        # 3 ── Graduated cost-aware buy reduction (skipped near 365d low) ──
        if buy_setup == "target-gap" and self._last_sell_price > 0:
            ratio = price / self._last_sell_price
            rolling_pos = latest.get("rolling_365d_pos", 0.5)
            if ratio > 0.95 and rolling_pos > 0.25:
                if ratio <= 1.0:
                    # Near sell price: full round-trip protection
                    mult = self.COST_AWARE_BUY_PROTECTION_MULT
                    label = "roundtrip"
                elif ratio <= 1.05:
                    # Moderate follow-through: partial protection
                    mult = self.COST_AWARE_BUY_MODERATE_MULT
                    label = "moderate"
                else:
                    # Clear trend continuation: no reduction
                    mult = 1.0
                    label = "trend"
                max_buy *= mult
                guard = self._join_guard(guard, f"v2_4_cost_{label}_x{mult:.2f}")

        # 4 ── Compound floor: V3 reductions alone can't cut below 15% of original ──
        min_allowed = original_max_buy * self.BUY_REDUCTION_FLOOR
        if max_buy < min_allowed:
            guard = self._join_guard(guard, f"v2_4_floor_x{max_buy/min_allowed:.2f}")
            max_buy = min_allowed

        # Call V1Spot base directly (skip V110 to avoid double vol scaling)
        base_max_buy, base_guard = V1SpotStrategy._adjust_buy_execution(
            self, latest, price, raw_state, buy_setup, max_buy,
            confirmed_state=confirmed_state,
        )
        guard = self._join_guard(guard, base_guard)

        return base_max_buy, guard


class V25Strategy(V24Strategy):
    """V2.5: long-term core-position variant of v2_4.

    The strategy keeps v2_4's fast BULL entry, but makes selling slower in
    constructive uptrends. Routine target-reduce sells are treated as small
    trims, while larger risk reduction is reserved for structural trend breaks.
    """

    VERSION_LABEL = "v2_5"

    TARGET_TABLE = V24Strategy.TARGET_TABLE

    LONG_TRIM_MAX_SELL = 0.06
    LONG_SELL_THRESHOLD = 0.10

    @property
    def name(self) -> str:
        return "v2_5"

    def _get_sell_target_state(
        self, raw_state: str, confirmed_state: str,
    ) -> tuple[str, int]:
        if raw_state == "MIXED" and confirmed_state == "BULL":
            return "BULL", 1
        return raw_state, 0

    def _is_constructive_mixed(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if (
            pd.isna(ema24)
            or pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
        ):
            return False
        if not (ema72 > ema168 and ema168_slope > 0 and price > ema168):
            return False
        roc_20 = latest.get("roc_20")
        if not pd.isna(roc_20) and roc_20 < -0.08:
            return False
        return True

    def _has_structural_break(self, latest: pd.Series, price: float) -> bool:
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(price < ema168 or ema72 < ema168 or ema168_slope <= 0)

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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

        if self._is_constructive_mixed(latest, price, raw_state, confirmed_state):
            if sell_setup == "target-reduce":
                threshold = max(threshold, self.LONG_SELL_THRESHOLD)
                adjusted_max_sell = min(adjusted_max_sell, self.LONG_TRIM_MAX_SELL)
                guard = self._join_guard(guard, "v2_5_constructive_mixed_small_trim")

        return threshold, adjusted_max_sell, guard

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        skip_cost = confirmed_state == "BULL"
        if not skip_cost:
            return super()._adjust_buy_execution(
                latest, price, raw_state, buy_setup, max_buy,
                confirmed_state=confirmed_state,
            )

        last_sell_price = self._last_sell_price
        self._last_sell_price = 0.0
        try:
            adjusted_max_buy, guard = super()._adjust_buy_execution(
                latest, price, raw_state, buy_setup, max_buy,
                confirmed_state=confirmed_state,
            )
        finally:
            self._last_sell_price = last_sell_price

        skip_cost_guard = "v2_5_skip_cost_in_uptrend"
        return adjusted_max_buy, self._join_guard(guard, skip_cost_guard)


class V26Strategy(V24Strategy):
    """V2.6: conservative EMA168 core overlay for constructive MIXED.

    V2.5's small-trim cap increased churn and hurt median results. V2.6 does
    not cap sell size or change buys. It only raises the sell target modestly
    when confirmed MIXED still has a healthy EMA168 structure.
    """

    VERSION_LABEL = "v2_6"

    @property
    def name(self) -> str:
        return "v2_6"

    def _get_sell_target_state(
        self, raw_state: str, confirmed_state: str,
    ) -> tuple[str, int]:
        latest = self._latest_bar
        if latest is not None and self._is_constructive_mixed(latest, self._current_price, raw_state, confirmed_state):
            return "BULL", 3
        return raw_state, 0

    @staticmethod
    def _is_constructive_mixed(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_20 = latest.get("roc_20")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema168 and ema72 > ema168 and ema168_slope > 0):
            return False
        return bool(pd.isna(roc_20) or roc_20 > -0.08)


class V27Strategy(V26Strategy):
    """V2.7: constructive-MIXED re-entry override.

    Diagnostics show the main BULL drag is slow re-entry after sells, not a
    simple lack of gross exposure. This variant only accelerates buy recovery
    when MIXED still has constructive EMA168 structure and BTC regime is not
    BEAR, leaving BTC-BEAR protection and sell rules unchanged.
    """

    VERSION_LABEL = "v2_7"

    @property
    def name(self) -> str:
        return "v2_7"

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
        if super()._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return True

        if trend_risk > 2 or risk_score < 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        if not self._is_constructive_mixed(latest, price, raw_state, confirmed_state):
            return False

        ema24 = latest.get("ema24")
        roc_5 = latest.get("roc_5")
        if pd.isna(ema24) or pd.isna(roc_5):
            return False
        return bool(price > ema24 and roc_5 > 0)


class V28Strategy(V27Strategy):
    """V2.8: broader alt constructive-MIXED recovery override."""

    VERSION_LABEL = "v2_8"

    @property
    def name(self) -> str:
        return "v2_8"

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
        if super()._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return True

        if self._is_btc_symbol() or trend_risk > 2 or risk_score < 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        if not self._is_constructive_mixed(latest, price, raw_state, confirmed_state):
            return False

        donchian_pos = latest.get("donchian_pos")
        if pd.isna(donchian_pos):
            return False
        return bool(donchian_pos >= 0.30)

    def _is_btc_symbol(self) -> bool:
        return "BTC/USDT" in getattr(self, "TARGET_ALLOC", {})


class V29Strategy(V28Strategy):
    """V2.9: wider alt recovery momentum tolerance."""

    VERSION_LABEL = "v2_9"
    ALT_RECOVERY_DONCHIAN_MIN = 0.15
    ALT_RECOVERY_MIN_ROC20 = -0.15

    @property
    def name(self) -> str:
        return "v2_9"

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
        if V27Strategy._is_recovery_override_setup(
            self,
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return True

        if self._is_btc_symbol() or trend_risk > 2 or risk_score < 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        if not self._is_alt_recovery_structure(latest, price, raw_state, confirmed_state):
            return False

        donchian_pos = latest.get("donchian_pos")
        if pd.isna(donchian_pos):
            return False
        return bool(donchian_pos >= self.ALT_RECOVERY_DONCHIAN_MIN)

    def _is_alt_recovery_structure(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_20 = latest.get("roc_20")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema168 and ema72 > ema168 and ema168_slope > 0):
            return False
        return bool(pd.isna(roc_20) or roc_20 > self.ALT_RECOVERY_MIN_ROC20)


class V210Strategy(V29Strategy):
    """V2.10: stronger alt recovery risk-score relief."""

    VERSION_LABEL = "v2_10"
    RECOVERY_RISK_SCORE_REDUCTION = 2

    @property
    def name(self) -> str:
        return "v2_10"


class V211AStrategy(V210Strategy):
    """V2.11A: structural-MIXED long-core floor for target-reduce sells.

    This candidate only limits routine target-reduce sells when long-term
    structure remains constructive. It leaves risk-reduce, trend-break, buy
    logic, cooldowns, and BTC-BEAR filters unchanged for clean attribution.
    """

    VERSION_LABEL = "v2_11A"
    BTC_STRUCTURAL_CORE_FLOOR = 0.80
    ALT_STRUCTURAL_CORE_FLOOR = 0.70

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_symbol = ""
        self._structural_core_floor = 0.0

    @property
    def name(self) -> str:
        return "v2_11A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._current_symbol = strategy_utils.resolve_symbol(candles_by_symbol) or ""
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

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
        self._structural_core_floor = 0.0
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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

        if sell_setup == "target-reduce" and self._is_structural_uptrend(latest, price, raw_state, confirmed_state):
            self._structural_core_floor = self._structural_core_floor_for_symbol()
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_structural_core_floor")

        return threshold, adjusted_max_sell, guard

    def _apply_sell_size_limit(
        self,
        max_sell: float,
        current_pct: float,
        pos: PositionState,
        price: float,
        latest: pd.Series,
    ) -> float:
        limited = super()._apply_sell_size_limit(max_sell, current_pct, pos, price, latest)
        if self._structural_core_floor <= 0:
            return limited
        allowed_sell = max(0.0, current_pct - self._structural_core_floor)
        return min(limited, allowed_sell)

    def _is_structural_uptrend(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(price > ema168 and ema72 > ema168 and ema168_slope > 0)

    def _structural_core_floor_for_symbol(self) -> float:
        if self._current_symbol == "BTC/USDT":
            return self.BTC_STRUCTURAL_CORE_FLOOR
        return self.ALT_STRUCTURAL_CORE_FLOOR


class V211BStrategy(V211AStrategy):
    """V2.11B: require structural damage before risk-reduce in uptrends."""

    VERSION_LABEL = "v2_11B"

    @property
    def name(self) -> str:
        return "v2_11B"

    def _classify_sell_setup(
        self,
        trend_risk: int,
        risk_score: int,
        latest: pd.Series,
        price: float,
        raw_state: str,
        drawdown_risk: int,
    ) -> str:
        setup = super()._classify_sell_setup(
            trend_risk=trend_risk,
            risk_score=risk_score,
            latest=latest,
            price=price,
            raw_state=raw_state,
            drawdown_risk=drawdown_risk,
        )
        if setup != "risk-reduce":
            return setup
        if not self._is_structural_uptrend(latest, price, raw_state, self._current_state):
            return setup
        if self._has_structural_sell_break(latest, price, trend_risk):
            return setup
        return "target-reduce"

    @staticmethod
    def _has_structural_sell_break(
        latest: pd.Series,
        price: float,
        trend_risk: int,
    ) -> bool:
        if trend_risk >= 3:
            return True
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        if pd.isna(ema72) or pd.isna(ema168):
            return True
        return bool(price < ema168 or ema72 < ema168)


class V211CStrategy(V211BStrategy):
    """V2.11C: add structural recovery buys after A/B sell protections."""

    VERSION_LABEL = "v2_11C"
    BTC_BEAR_ALT_RECOVERY_MIN_ROC5 = 0.0

    @property
    def name(self) -> str:
        return "v2_11C"

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
        if super()._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return True

        if risk_score < 2 or trend_risk > 2:
            return False
        if not self._is_structural_recovery_candidate(latest, price, raw_state, confirmed_state):
            return False

        ema24 = latest.get("ema24")
        roc_5 = latest.get("roc_5")
        if pd.isna(ema24) or pd.isna(roc_5):
            return False
        return bool(price > ema24 and roc_5 > self.BTC_BEAR_ALT_RECOVERY_MIN_ROC5)

    def _is_structural_recovery_candidate(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False

        btc_regime = str(latest.get("btc_regime", ""))
        if btc_regime == "BEAR" and self._current_symbol == "BTC/USDT":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(price > ema168 and ema72 > ema168 and ema168_slope > 0)


class V212AStrategy(V210Strategy):
    """V2.12A: delay low-quality target-reduce exits in noisy uptrends."""

    VERSION_LABEL = "v2_12A"
    TARGET_REDUCE_DELAY_BARS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_reduce_delay_streak = 0

    @property
    def name(self) -> str:
        return "v2_12A"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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

        if sell_setup != "target-reduce" or not self._is_noisy_uptrend_exit(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            self._target_reduce_delay_streak = 0
            return threshold, adjusted_max_sell, guard

        self._target_reduce_delay_streak += 1
        if self._target_reduce_delay_streak < self.TARGET_REDUCE_DELAY_BARS:
            return (
                threshold,
                0.0,
                self._join_guard(guard, f"{self.VERSION_LABEL}_target_reduce_delayed"),
            )
        return threshold, adjusted_max_sell, self._join_guard(
            guard,
            f"{self.VERSION_LABEL}_target_reduce_delay_confirmed",
        )

    @staticmethod
    def _is_noisy_uptrend_exit(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state not in {"BULL", "MIXED"}:
            return False
        if price <= 0 or trend_risk > 2 or risk_score > 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(price > ema72 and ema72 > ema168 and ema168_slope > 0)


class V212BStrategy(V210Strategy):
    """V2.12B: soften risk-reduce during raw-MIXED/confirmed-BULL pullbacks."""

    VERSION_LABEL = "v2_12B"
    RISK_REDUCE_PULLBACK_MAX_SELL_MULT = 0.50
    RISK_REDUCE_PULLBACK_MIN_THRESHOLD = 0.10

    @property
    def name(self) -> str:
        return "v2_12B"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if sell_setup == "risk-reduce" and self._is_bull_pullback_risk_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        ):
            return (
                max(threshold, self.RISK_REDUCE_PULLBACK_MIN_THRESHOLD),
                adjusted_max_sell * self.RISK_REDUCE_PULLBACK_MAX_SELL_MULT,
                self._join_guard(guard, f"{self.VERSION_LABEL}_risk_reduce_pullback_softened"),
            )
        return threshold, adjusted_max_sell, guard

    @staticmethod
    def _is_bull_pullback_risk_reduce(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "BULL":
            return False
        if trend_risk > 2 or drawdown_risk < 2 or price <= 0:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(price > ema168 and ema72 > ema168 and ema168_slope > 0)


class V212CStrategy(V212AStrategy):
    """V2.12C: combine target-reduce delay with pullback risk-reduce softening."""

    VERSION_LABEL = "v2_12C"
    RISK_REDUCE_PULLBACK_MAX_SELL_MULT = V212BStrategy.RISK_REDUCE_PULLBACK_MAX_SELL_MULT
    RISK_REDUCE_PULLBACK_MIN_THRESHOLD = V212BStrategy.RISK_REDUCE_PULLBACK_MIN_THRESHOLD

    @property
    def name(self) -> str:
        return "v2_12C"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if sell_setup == "risk-reduce" and V212BStrategy._is_bull_pullback_risk_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        ):
            return (
                max(threshold, self.RISK_REDUCE_PULLBACK_MIN_THRESHOLD),
                adjusted_max_sell * self.RISK_REDUCE_PULLBACK_MAX_SELL_MULT,
                self._join_guard(guard, f"{self.VERSION_LABEL}_risk_reduce_pullback_softened"),
            )
        return threshold, adjusted_max_sell, guard


class V219BStrategy(V212AStrategy):
    """V2.19B: skip tiny buys as an explicit percentage rule."""

    VERSION_LABEL = "v2_19B"
    MIN_BUY_PCT = 0.08

    @property
    def name(self) -> str:
        return "v2_19B"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest, price, raw_state, buy_setup, max_buy,
            confirmed_state=confirmed_state,
        )
        if adjusted < self.MIN_BUY_PCT:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_tiny_buy_skipped")
        return adjusted, guard


class V220AStrategy(V219BStrategy):
    """V2.20A: soften target-reduce trims while the long structure is intact."""

    VERSION_LABEL = "v2_20A"
    CONSTRUCTIVE_MIXED_MAX_SELL = 0.10
    CONSTRUCTIVE_MIXED_MIN_THRESHOLD = 0.10

    @property
    def name(self) -> str:
        return "v2_20A"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if sell_setup == "target-reduce" and self._is_constructive_mixed_trim(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return (
                max(threshold, self.CONSTRUCTIVE_MIXED_MIN_THRESHOLD),
                min(adjusted_max_sell, self.CONSTRUCTIVE_MIXED_MAX_SELL),
                self._join_guard(guard, f"{self.VERSION_LABEL}_constructive_mixed_trim_softened"),
            )
        return threshold, adjusted_max_sell, guard

    @staticmethod
    def _is_constructive_mixed_trim(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if price <= 0 or trend_risk > 2 or risk_score > 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        return bool(ema72 > ema168 and ema168_slope > 0)


class V220DStrategy(V219BStrategy):
    """V2.20D: delay only constructive low-risk target-reduce sells."""

    VERSION_LABEL = "v2_20D"
    TARGET_REDUCE_CONFIRM_CALLS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constructive_target_reduce_streak = 0

    @property
    def name(self) -> str:
        return "v2_20D"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_constructive_target_reduce_delay(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            self._constructive_target_reduce_streak = 0
            return threshold, adjusted_max_sell, guard

        self._constructive_target_reduce_streak += 1
        if self._constructive_target_reduce_streak <= self.TARGET_REDUCE_CONFIRM_CALLS:
            return (
                threshold,
                0.0,
                self._join_guard(guard, f"{self.VERSION_LABEL}_constructive_target_reduce_delay"),
            )
        return threshold, adjusted_max_sell, guard

    @staticmethod
    def _is_constructive_target_reduce_delay(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk > 1 or risk_score > 2:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_5 = latest.get("roc_5")
        if (
            price <= 0
            or pd.isna(ema24)
            or pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(roc_5)
        ):
            return False
        return bool(price > ema24 and roc_5 > 0 and ema72 > ema168 and ema168_slope >= 0)


class V221EStrategy(V220DStrategy):
    """V2.21E: short target-reduce grace period after structural recovery buys."""

    VERSION_LABEL = "v2_21E"
    STRUCTURAL_RECOVERY_MAX_RISK = 3
    RECOVERY_BUY_GRACE_CALLS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_recovery_buy_call = -10_000

    @property
    def name(self) -> str:
        return "v2_21E"

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
        if super()._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        ):
            return True
        if risk_score > self.STRUCTURAL_RECOVERY_MAX_RISK or trend_risk > 2:
            return False
        return self._is_structural_recovery_override(latest, price, raw_state, confirmed_state)

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                self._last_recovery_buy_call = self._call_count
        return actions

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_recovery_buy_grace_target_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            threshold,
            0.0,
            self._join_guard(guard, f"{self.VERSION_LABEL}_post_recovery_target_reduce_grace"),
        )

    def _is_recovery_buy_grace_target_reduce(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if self._call_count - self._last_recovery_buy_call > self.RECOVERY_BUY_GRACE_CALLS:
            return False
        if risk_score > 3 or trend_risk > 2:
            return False
        return self._is_structural_recovery_override(latest, price, raw_state, confirmed_state)

    @staticmethod
    def _is_structural_recovery_override(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_5 = latest.get("roc_5")
        if (
            pd.isna(ema24)
            or pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(roc_5)
        ):
            return False
        return bool(price > ema24 and price > ema168 and roc_5 > 0 and ema72 > ema168 and ema168_slope > 0)


class V222AStrategy(V221EStrategy):
    """V2.22A: skip target-reduce while MIXED remains above EMA72."""

    VERSION_LABEL = "v2_22A"

    @property
    def name(self) -> str:
        return "v2_22A"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if self._is_above_ema72_mixed_noise(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            sell_setup=sell_setup,
        ):
            return (
                threshold,
                0.0,
                self._join_guard(guard, f"{self.VERSION_LABEL}_above_ema72_mixed_noise"),
            )
        return threshold, adjusted_max_sell, guard

    @staticmethod
    def _is_above_ema72_mixed_noise(
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if price <= 0 or trend_risk > 1:
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_20 = latest.get("roc_20")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema72 and ema72 > ema168 and ema168_slope > 0):
            return False
        return bool(pd.isna(roc_20) or roc_20 >= 0)


class V223AStrategy(V221EStrategy):
    """V2.23A: modest buy-target floor for constructive MIXED regimes."""

    VERSION_LABEL = "v2_23A"
    CONSTRUCTIVE_MIXED_BUY_FLOOR = {
        "BTC/USDT": 0.74,
        "ETH/USDT": 0.70,
        "BNB/USDT": 0.68,
    }

    @property
    def name(self) -> str:
        return "v2_23A"

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
        target = super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        if side == "buy" and self._is_constructive_mixed_buy_target(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
        ):
            target = max(target, self.CONSTRUCTIVE_MIXED_BUY_FLOOR.get(symbol, 0.68))
        return target

    @staticmethod
    def _is_constructive_mixed_buy_target(
        latest: pd.Series,
        price: float,
        raw_state: str,
        trend_risk: int,
    ) -> bool:
        if raw_state != "MIXED" or price <= 0 or trend_risk > 1:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        ema168_slope = latest.get("ema168_slope")
        roc_20 = latest.get("roc_20")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema168 and ema72 > ema168 and ema168_slope > 0):
            return False
        return bool(pd.isna(roc_20) or roc_20 >= -0.03)


class V223BStrategy(V223AStrategy):
    """V2.23B: conservative constructive MIXED floor with distribution filters."""

    VERSION_LABEL = "v2_23B"
    CONSTRUCTIVE_MIXED_BUY_FLOOR = {
        "BTC/USDT": 0.72,
        "ETH/USDT": 0.68,
        "BNB/USDT": 0.66,
    }

    @property
    def name(self) -> str:
        return "v2_23B"

    @staticmethod
    def _is_constructive_mixed_buy_target(
        latest: pd.Series,
        price: float,
        raw_state: str,
        trend_risk: int,
    ) -> bool:
        if not V223AStrategy._is_constructive_mixed_buy_target(latest, price, raw_state, trend_risk):
            return False

        ema24 = latest.get("ema24")
        roc_5 = latest.get("roc_5")
        roc_10 = latest.get("roc_10")
        volume_strength = latest.get("volume_strength")
        donchian_pos = latest.get("donchian_pos")
        atr_rank = latest.get("atr_pct_rank")

        short_weak = (
            (not pd.isna(ema24) and price < ema24)
            or (not pd.isna(roc_5) and roc_5 < -0.03)
            or (not pd.isna(roc_10) and roc_10 < -0.08)
        )
        if short_weak and not pd.isna(volume_strength) and volume_strength >= 1.15:
            return False
        if (
            not pd.isna(donchian_pos)
            and donchian_pos >= 0.90
            and not pd.isna(atr_rank)
            and atr_rank >= 0.85
        ):
            return False
        return True


class V225FStrategy(V221EStrategy):
    """V2.25F: immediate safe-recovery target-reduce deadband."""

    VERSION_LABEL = "v2_25F"
    RECENT_BUY_DEADBAND_CALLS = 2
    RECENT_BUY_TARGET_REDUCE_BAND = 0.12
    RECENT_BUY_TARGET_REDUCE_MAX_SELL = 0.08

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_any_buy_call = -10_000

    @property
    def name(self) -> str:
        return "v2_25F"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                self._last_any_buy_call = self._call_count
        return actions

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_recent_buy_target_reduce_churn(
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard

        return (
            max(threshold, self.RECENT_BUY_TARGET_REDUCE_BAND),
            min(adjusted_max_sell, self.RECENT_BUY_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_recent_buy_target_reduce_deadband"),
        )

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if self._call_count - self._last_any_buy_call > self.RECENT_BUY_DEADBAND_CALLS:
            return False
        if raw_state == "BEAR" or trend_risk > 1 or risk_score > 3:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        return True


class V228AStrategy(V225FStrategy):
    """V2.28A: structure-gated safe-recovery target-reduce deadband."""

    VERSION_LABEL = "v2_28A"

    @property
    def name(self) -> str:
        return "v2_28A"

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if not super()._is_recent_buy_target_reduce_churn(
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return False

        close = float(latest.get("close", 0.0) or 0.0)
        ema24 = float(latest.get("ema24", 0.0) or 0.0)
        ema72 = float(latest.get("ema72", 0.0) or 0.0)
        ema168 = float(latest.get("ema168", 0.0) or 0.0)
        ema24_slope = float(latest.get("ema24_slope", 0.0) or 0.0)
        ema72_slope = float(latest.get("ema72_slope", 0.0) or 0.0)
        ema168_slope = float(latest.get("ema168_slope", 0.0) or 0.0)
        roc5 = float(latest.get("roc_5", 0.0) or 0.0)
        roc10 = float(latest.get("roc_10", 0.0) or 0.0)
        donchian_pos = float(latest.get("donchian_pos", 0.0) or 0.0)
        dd_from_120d_high = float(latest.get("dd_from_120d_high", 1.0) or 1.0)
        btc_vs_ema72 = float(latest.get("btc_price_vs_ema72", 0.0) or 0.0)

        local_structure_ok = (
            close > ema24
            and close > ema72
            and ema24 >= ema72
            and ema72 >= ema168 * 0.98
        )
        slope_ok = ema24_slope >= 0.0 and ema72_slope >= 0.0 and ema168_slope >= -0.005
        momentum_ok = roc5 >= 0.0 and roc10 >= 0.0 and donchian_pos >= 0.55
        drawdown_ok = dd_from_120d_high <= 0.30
        btc_ok = btc_vs_ema72 >= -0.02

        return local_structure_ok and slope_ok and momentum_ok and drawdown_ok and btc_ok


class V228BStrategy(V225FStrategy):
    """V2.28B: low-risk volume-confirmed safe-recovery target-reduce deadband."""

    VERSION_LABEL = "v2_28B"

    @property
    def name(self) -> str:
        return "v2_28B"

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if not super()._is_recent_buy_target_reduce_churn(
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return False
        if risk_score > 1:
            return False

        volume_strength = float(latest.get("volume_strength", 0.0) or 0.0)
        dd_from_120d_high = float(latest.get("dd_from_120d_high", 1.0) or 1.0)
        close = float(latest.get("close", 0.0) or 0.0)
        ema168 = float(latest.get("ema168", 0.0) or 0.0)
        price_vs_ema168 = close / ema168 - 1.0 if close > 0 and ema168 > 0 else -1.0
        ema168_slope = float(latest.get("ema168_slope", 0.0) or 0.0)

        return (
            drawdown_risk == 0
            and volume_strength >= 0.75
            and dd_from_120d_high <= 0.30
            and price_vs_ema168 >= 0.0
            and ema168_slope >= 0.0
        )


class V228CStrategy(V228BStrategy):
    """V2.28C: V2.28B with one extra call of post-recovery churn tolerance."""

    VERSION_LABEL = "v2_28C"
    RECENT_BUY_DEADBAND_CALLS = 3

    @property
    def name(self) -> str:
        return "v2_28C"


class V229AStrategy(V221EStrategy):
    """V2.29A: external recovery quality bands for MIXED buy targets and trims."""

    VERSION_LABEL = "v2_29A"
    LOW_QUALITY_BUY_CAP = 0.45
    MID_QUALITY_BUY_CAP = 0.65
    HIGH_QUALITY_BUY_FLOOR = 0.76
    HIGH_QUALITY_TARGET_REDUCE_BAND = 0.12
    HIGH_QUALITY_TARGET_REDUCE_MAX_SELL = 0.08

    @property
    def name(self) -> str:
        return "v2_29A"

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
        target = super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        if side != "buy" or raw_state != "MIXED" or price <= 0:
            return target

        grade = self._recovery_quality_grade(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        )
        if grade == "LOW":
            return min(target, self.LOW_QUALITY_BUY_CAP)
        if grade == "MID_WEAK":
            return min(target, self.MID_QUALITY_BUY_CAP)
        if grade == "HIGH":
            return max(target, self.HIGH_QUALITY_BUY_FLOOR)
        return target

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if sell_setup != "target-reduce" or raw_state != "MIXED" or confirmed_state != "MIXED":
            return threshold, adjusted_max_sell, guard
        if risk_score > 1 or trend_risk > 1 or drawdown_risk > 0:
            return threshold, adjusted_max_sell, guard
        if self._recovery_quality_grade(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        ) != "HIGH":
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.HIGH_QUALITY_TARGET_REDUCE_BAND),
            min(adjusted_max_sell, self.HIGH_QUALITY_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_high_quality_recovery_trim_softened"),
        )

    def _recovery_quality_grade(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
    ) -> str:
        if raw_state != "MIXED" or price <= 0:
            return "MID"
        if str(latest.get("btc_regime", "")) == "BEAR":
            return "LOW"

        score = self._recovery_quality_score(latest, price, trend_risk, drawdown_risk)
        if self._low_quality_recovery_context(latest, price, trend_risk, drawdown_risk, score):
            return "LOW"
        if self._high_quality_recovery_context(latest, price, trend_risk, drawdown_risk, score):
            return "HIGH"
        if score <= 3:
            return "MID_WEAK"
        return "MID"

    def _high_quality_recovery_context(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
        score: int,
    ) -> bool:
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        return bool(
            score >= 8
            and trend_risk <= 1
            and drawdown_risk == 0
            and price_vs_ema168 >= 0.0
            and self._value(latest, "ema168_slope") > 0.0
            and self._value(latest, "dd_from_120d_high", default=1.0) <= 0.30
            and self._value(latest, "donchian_pos") >= 0.45
            and 0.20 <= rolling_pos <= 0.75
            and not self._is_recovery_extension_risk(latest, price)
        )

    def _low_quality_recovery_context(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
        score: int,
    ) -> bool:
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        ema168_slope = self._value(latest, "ema168_slope")
        donchian_pos = self._value(latest, "donchian_pos")
        dd_120 = self._value(latest, "dd_from_120d_high", default=1.0)
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        roc_10 = self._value(latest, "roc_10")
        weak_structure = price_vs_ema168 < -0.03 or ema168_slope < 0
        weak_range = donchian_pos < 0.42 or dd_120 > 0.35
        weak_flow = volume_strength < 0.70 and roc_10 < 0
        high_position_failure = rolling_pos >= 0.62 and donchian_pos < 0.55 and roc_10 < 0
        risk_with_weak_path = (trend_risk >= 2 or drawdown_risk >= 2) and (weak_structure or weak_range or roc_10 < 0)
        return bool(risk_with_weak_path or (score <= 3 and (weak_structure or weak_range)) or weak_flow or high_position_failure)

    def _is_recovery_extension_risk(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        donchian_pos = self._value(latest, "donchian_pos")
        roc_20 = self._value(latest, "roc_20")
        return bool(
            (price_vs_ema168 >= 0.15 and rolling_pos >= 0.68)
            or (donchian_pos >= 0.80 and rolling_pos >= 0.64 and roc_20 >= 0.15)
        )

    def _recovery_quality_score(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
    ) -> int:
        score = 0
        if trend_risk == 0:
            score += 1
        if drawdown_risk == 0:
            score += 2
        if self._price_vs(latest, price, "ema168") >= 0:
            score += 1
        if self._value(latest, "ema168_slope") > 0:
            score += 1
        if self._value(latest, "roc_10") >= 0:
            score += 1
        if self._value(latest, "volume_strength", default=1.0) >= 0.75:
            score += 1
        if self._value(latest, "dd_from_120d_high", default=1.0) <= 0.30:
            score += 1
        if self._value(latest, "donchian_pos") >= 0.45:
            score += 1
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        if 0.20 <= rolling_pos <= 0.70:
            score += 1
        if self._value(latest, "btc_price_vs_ema72", default=0.0) >= -0.02:
            score += 1
        return score

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V229BStrategy(V229AStrategy):
    """V2.29B: stricter low-quality cap and no high-quality target floor."""

    VERSION_LABEL = "v2_29B"
    LOW_QUALITY_BUY_CAP = 0.35
    MID_QUALITY_BUY_CAP = 0.58
    HIGH_QUALITY_BUY_FLOOR = 0.0

    @property
    def name(self) -> str:
        return "v2_29B"

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
        target = super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        if side == "buy" and raw_state == "MIXED" and self._recovery_quality_grade(
            latest=latest,
            price=price,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
        ) == "HIGH":
            return target
        return target


class V229CStrategy(V229AStrategy):
    """V2.29C: high-quality target-reduce smoothing only, no buy target changes."""

    VERSION_LABEL = "v2_29C"

    @property
    def name(self) -> str:
        return "v2_29C"

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
        return V221EStrategy._compose_target(
            self,
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )


class V230AStrategy(V228CStrategy):
    """V2.30A: stateful recovery-path target-reduce smoothing."""

    VERSION_LABEL = "v2_30A"
    RECENT_BUY_DEADBAND_CALLS = 6
    MIN_RECOVERY_PATH_SCORE = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_recovery_buy_price = 0.0
        self._last_recovery_buy_path_score = 0

    @property
    def name(self) -> str:
        return "v2_30A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                symbol = strategy_utils.resolve_symbol(candles_by_symbol)
                df = candles_by_symbol.get(symbol) if symbol is not None else None
                latest = df.iloc[-1] if df is not None and not df.empty else pd.Series(dtype=float)
                self._last_recovery_buy_price = float(getattr(action, "price", 0.0) or 0.0)
                self._last_recovery_buy_path_score = self._recovery_path_score(latest, self._last_recovery_buy_price)
        return actions

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if not V225FStrategy._is_recent_buy_target_reduce_churn(
            self,
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return False
        if risk_score > 1 or trend_risk > 1 or drawdown_risk > 0:
            return False
        if self._last_recovery_buy_price <= 0:
            return False

        close = float(latest.get("close", 0.0) or 0.0)
        if close <= 0 or close < self._last_recovery_buy_price * 0.98:
            return False

        current_score = self._recovery_path_score(latest, close)
        if current_score < self.MIN_RECOVERY_PATH_SCORE:
            return False
        if current_score + 1 < self._last_recovery_buy_path_score:
            return False
        if self._recovery_path_extension_risk(latest, close):
            return False
        return True

    def _recovery_path_score(self, latest: pd.Series, price: float) -> int:
        score = 0
        if self._price_vs(latest, price, "ema168") >= 0:
            score += 1
        if self._value(latest, "ema168_slope") > 0:
            score += 1
        if self._value(latest, "ema72_slope") >= -0.005:
            score += 1
        if self._value(latest, "roc_10") >= -0.02:
            score += 1
        if self._value(latest, "donchian_pos") >= 0.45:
            score += 1
        if self._value(latest, "dd_from_120d_high", default=1.0) <= 0.30:
            score += 1
        if self._value(latest, "volume_strength", default=1.0) >= 0.75:
            score += 1
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        if 0.20 <= rolling_pos <= 0.75:
            score += 1
        if str(latest.get("btc_regime", "")) != "BEAR":
            score += 1
        return score

    def _recovery_path_extension_risk(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        donchian_pos = self._value(latest, "donchian_pos")
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        roc_20 = self._value(latest, "roc_20")
        return bool(
            (price_vs_ema168 >= 0.15 and rolling_pos >= 0.68)
            or (donchian_pos >= 0.82 and rolling_pos >= 0.64 and roc_20 >= 0.15)
        )

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V231AStrategy(V228CStrategy):
    """V2.31A: conservative low-quality recovery veto layer over V2.28C."""

    VERSION_LABEL = "v2_31A"
    RECOVERY_TARGET_GAP_MAX_CURRENT = 0.35
    LOW_QUALITY_RECOVERY_MAX_BUY = 0.04
    LOW_QUALITY_SAFE_RECOVERY_FLAGS = 5
    LOW_QUALITY_TARGET_GAP_FLAGS = 4
    SEVERE_LOW_QUALITY_FLAGS = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_current_pct = 0.0

    @property
    def name(self) -> str:
        return "v2_31A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if symbol is not None:
            price = current_prices.get(symbol, 0.0)
            pos = portfolio.positions.get(symbol, PositionState())
            position_value = pos.quantity * price if price > 0 else 0.0
            total_value = portfolio.cash + position_value
            if total_value > 0:
                self._overlay_current_pct = position_value / total_value
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._should_slow_low_quality_recovery(latest, price, raw_state, confirmed_state, buy_setup):
            return max_buy, guard

        score = self._low_quality_recovery_score(latest, price)
        if score >= self.SEVERE_LOW_QUALITY_FLAGS:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_low_quality_recovery_veto_s{score}")

        adjusted = min(max_buy, self.LOW_QUALITY_RECOVERY_MAX_BUY)
        return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_low_quality_recovery_slow_s{score}")

    def _should_slow_low_quality_recovery(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        if buy_setup == "safe-recovery":
            threshold = self.LOW_QUALITY_SAFE_RECOVERY_FLAGS
        elif buy_setup == "target-gap" and self._overlay_current_pct < self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            threshold = self.LOW_QUALITY_TARGET_GAP_FLAGS
        else:
            return False

        score = self._low_quality_recovery_score(latest, price)
        if score < threshold:
            return False
        return self._has_core_low_quality_recovery_failure(latest, price)

    def _low_quality_recovery_score(self, latest: pd.Series, price: float) -> int:
        score = 0
        btc_vs_ema72 = self._value(latest, "btc_price_vs_ema72", default=0.0)
        btc_vs_ema168 = self._value(latest, "btc_price_vs_ema168", default=0.0)
        btc_ema24_slope = self._value(latest, "btc_ema24_slope", default=0.0)
        btc_ema168_slope = self._value(latest, "btc_ema168_slope", default=0.0)
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        ema168_slope = self._value(latest, "ema168_slope")
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        dd_120 = self._value(latest, "dd_from_120d_high", default=0.0)
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        roc_20 = self._value(latest, "roc_20", default=0.0)
        atr_rank = self._value(latest, "atr_pct_rank", default=0.5)
        volume_strength = self._value(latest, "volume_strength", default=1.0)

        if str(latest.get("btc_regime", "")) == "BEAR":
            score += 2
        elif btc_vs_ema72 < -0.03 and btc_ema24_slope < 0:
            score += 1
        if btc_vs_ema168 < -0.03 and btc_ema168_slope < 0:
            score += 1
        if price_vs_ema168 < -0.03 or ema72_vs_ema168 < -0.04:
            score += 1
        if ema168_slope < 0:
            score += 1
        if donchian_pos < 0.38 and dd_120 > 0.35:
            score += 1
        if roc_10 < -0.03 and volume_strength < 0.85:
            score += 1
        if roc_20 < 0 and volume_strength < 0.75 and donchian_pos < 0.55:
            score += 1
        if rolling_pos >= 0.62 and donchian_pos < 0.55 and roc_10 < 0:
            score += 1
        if rolling_pos <= 0.20 and price_vs_ema168 < 0 and ema168_slope < 0:
            score += 1
        if atr_rank >= 0.85 and roc_10 < 0 and donchian_pos < 0.55:
            score += 1
        return score

    def _has_core_low_quality_recovery_failure(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        ema168_slope = self._value(latest, "ema168_slope")
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        dd_120 = self._value(latest, "dd_from_120d_high", default=0.0)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        structural_failure = (price_vs_ema168 < -0.03 or ema72_vs_ema168 < -0.04) and ema168_slope < 0
        range_failure = donchian_pos < 0.38 and dd_120 > 0.35
        flow_failure = roc_10 < -0.03 and volume_strength < 0.85
        return bool(structural_failure or range_failure or flow_failure)

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _ratio(cls, latest: pd.Series, numerator: str, denominator: str) -> float:
        num = cls._value(latest, numerator)
        den = cls._value(latest, denominator)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return num / den - 1.0

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V231BStrategy(V231AStrategy):
    """V2.31B: severe low-quality target-gap veto only."""

    VERSION_LABEL = "v2_31B"
    SEVERE_LOW_QUALITY_FLAGS = 6

    @property
    def name(self) -> str:
        return "v2_31B"

    def _should_slow_low_quality_recovery(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            return False
        if not self._macro_breakdown_context(latest):
            return False
        return self._low_quality_recovery_score(latest, price) >= self.SEVERE_LOW_QUALITY_FLAGS

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = V228CStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._should_slow_low_quality_recovery(latest, price, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        score = self._low_quality_recovery_score(latest, price)
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_severe_low_quality_target_gap_veto_s{score}")

    def _macro_breakdown_context(self, latest: pd.Series) -> bool:
        btc_vs_ema72 = self._value(latest, "btc_price_vs_ema72", default=0.0)
        btc_vs_ema168 = self._value(latest, "btc_price_vs_ema168", default=0.0)
        btc_ema24_slope = self._value(latest, "btc_ema24_slope", default=0.0)
        btc_ema168_slope = self._value(latest, "btc_ema168_slope", default=0.0)
        return bool(
            str(latest.get("btc_regime", "")) == "BEAR"
            or (btc_vs_ema72 < -0.04 and btc_ema24_slope < 0)
            or (btc_vs_ema168 < -0.03 and btc_ema168_slope < 0)
        )


class V231CStrategy(V231BStrategy):
    """V2.31C: strict low-quality target-gap veto with bull-repair deny gate."""

    VERSION_LABEL = "v2_31C"
    STRICT_LOW_QUALITY_FLAGS = 5

    @property
    def name(self) -> str:
        return "v2_31C"

    def _should_slow_low_quality_recovery(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED" or price <= 0:
            return False
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            return False
        if self._bull_repair_context(latest):
            return False
        if self._low_quality_recovery_score(latest, price) < self.STRICT_LOW_QUALITY_FLAGS:
            return False
        return self._has_core_low_quality_recovery_failure(latest, price)

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = V228CStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._should_slow_low_quality_recovery(latest, price, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        score = self._low_quality_recovery_score(latest, price)
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_strict_low_quality_target_gap_veto_s{score}")

    def _bull_repair_context(self, latest: pd.Series) -> bool:
        btc_regime = str(latest.get("btc_regime", ""))
        btc_vs_ema72 = self._value(latest, "btc_price_vs_ema72", default=0.0)
        btc_ema24_slope = self._value(latest, "btc_ema24_slope", default=0.0)
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        ema72_slope = self._value(latest, "ema72_slope", default=0.0)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        return bool(
            btc_regime in {"BULL", "STRONG_BULL"}
            or (btc_vs_ema72 > 0 and btc_ema24_slope >= 0 and roc_10 >= 0)
            or (ema72_vs_ema168 >= -0.02 and ema72_slope >= 0 and donchian_pos >= 0.45)
        )


class V232AStrategy(V231AStrategy):
    """V2.32A: veto BNB safe-recovery buys in high-rank failed-recovery setups."""

    VERSION_LABEL = "v2_32A"

    @property
    def name(self) -> str:
        return "v2_32A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._overlay_symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = V228CStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_failed_safe_recovery_context(latest, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_failed_safe_recovery_veto")

    def _is_failed_safe_recovery_context(
        self,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if getattr(self, "_overlay_symbol", None) != "BNB/USDT":
            return False
        if buy_setup != "safe-recovery":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False

        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        roc_20 = self._value(latest, "roc_20", default=0.0)
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        dd_120 = self._value(latest, "dd_from_120d_high", default=0.0)

        if rolling_pos < 0.62:
            return False
        weakening_path = roc_10 < 0 or roc_20 < 0 or volume_strength < 0.85
        not_fully_repaired = donchian_pos < 0.60 or dd_120 > 0.24
        return bool(weakening_path and not_fully_repaired)


class V232BStrategy(V228CStrategy):
    """V2.32B: veto high-ATR BULL target-gap chase buys."""

    VERSION_LABEL = "v2_32B"

    @property
    def name(self) -> str:
        return "v2_32B"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_high_atr_target_gap_chase(latest, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_high_atr_target_gap_veto")

    def _is_high_atr_target_gap_chase(
        self,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup != "target-gap":
            return False
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        atr_rank = self._value(latest, "atr_pct_rank", default=0.0)
        if atr_rank < 0.75:
            return False
        roc_20 = self._value(latest, "roc_20", default=0.0)
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        return bool(roc_20 >= 0.15 or volume_strength >= 1.15 or donchian_pos >= 0.80)

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)


class V233AStrategy(V231AStrategy):
    """V2.33A: one-shot BNB bear-market reclaim probe below the tiny-buy floor."""

    VERSION_LABEL = "v2_33A"
    BEAR_RECLAIM_PROBE_BUY = 0.05
    BEAR_RECLAIM_PROBE_MAX_CURRENT = 0.15
    BEAR_RECLAIM_PROBE_MIN_GAP = 0.20
    BEAR_RECLAIM_PROBE_COOLDOWN_CALLS = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_bear_reclaim_probe_call = -10_000

    @property
    def name(self) -> str:
        return "v2_33A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._overlay_symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            reason = str(getattr(actions[0], "reason", ""))
            if f"{self.VERSION_LABEL}_bear_reclaim_probe" in reason:
                self._last_bear_reclaim_probe_call = self._call_count
        return actions

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = V228CStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_bear_reclaim_probe_context(latest, price, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        adjusted = max(max_buy, self.BEAR_RECLAIM_PROBE_BUY)
        return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_bear_reclaim_probe")

    def _is_bear_reclaim_probe_context(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if getattr(self, "_overlay_symbol", None) != "BNB/USDT":
            return False
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.BEAR_RECLAIM_PROBE_MAX_CURRENT:
            return False
        if self._call_count - self._last_bear_reclaim_probe_call < self.BEAR_RECLAIM_PROBE_COOLDOWN_CALLS:
            return False
        if str(latest.get("btc_regime", "")) != "BEAR":
            return False
        if raw_state not in {"BEAR", "MIXED"} or confirmed_state not in {"BEAR", "MIXED"}:
            return False

        target_gap = max(0.0, self._value(latest, "buy_target", default=0.0) - self._overlay_current_pct)
        # The diagnostic hook does not expose buy_target here. Fall back to the
        # caller's observed current/gap pattern through structural filters.
        if target_gap and target_gap < self.BEAR_RECLAIM_PROBE_MIN_GAP:
            return False

        volume_strength = self._value(latest, "volume_strength", default=1.0)
        roc_20 = self._value(latest, "roc_20", default=0.0)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        donchian_pos = self._value(latest, "donchian_pos", default=0.5)
        rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
        price_vs_ema168 = self._price_vs(latest, price, "ema168")

        strong_reclaim = volume_strength >= 1.15 and roc_20 > 0 and roc_10 > -0.02
        bear_structure = ema72_vs_ema168 < -0.03 and rolling_pos < 0.55
        local_reclaim = donchian_pos >= 0.50 or price_vs_ema168 >= 0.0
        return bool(strong_reclaim and bear_structure and local_reclaim)


class V233BStrategy(V233AStrategy):
    """V2.33B: BNB bear reclaim probe only after MIXED confirmation."""

    VERSION_LABEL = "v2_33B"

    @property
    def name(self) -> str:
        return "v2_33B"

    def _is_bear_reclaim_probe_context(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        return super()._is_bear_reclaim_probe_context(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        )


class V234AStrategy(V228CStrategy):
    """V2.34A: BNB post-recovery external cap after short-path fade confirmation."""

    VERSION_LABEL = "v2_34A"
    FADE_SYMBOL = "BNB/USDT"
    FADE_CAP = 0.30
    FADE_MIN_AGE_CALLS = 5
    FADE_MAX_AGE_CALLS = 30
    FADE_ROLLING_DROP = 0.02
    FADE_BTC_EMA168_SLOPE_MIN = 0.03
    FADE_MAX_SELL = 0.12
    FADE_SELL_THRESHOLD = 0.03

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_symbol: str | None = None
        self._overlay_current_pct = 0.0
        self._fade_entry_call = -10_000
        self._fade_entry_rolling_pos = float("nan")

    @property
    def name(self) -> str:
        return "v2_34A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._track_overlay_position(candles_by_symbol, portfolio, current_prices)
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)

        latest = self._latest_overlay_bar(candles_by_symbol)
        if latest is None:
            return actions

        if actions:
            reason = str(getattr(actions[0], "reason", ""))
            if actions[0].side == "buy" and ("safe-recovery" in reason or "target-gap" in reason):
                self._fade_entry_call = self._call_count
                self._fade_entry_rolling_pos = self._value(latest, "rolling_365d_pos", default=0.5)
            if actions[0].side == "sell":
                return actions
            if (
                actions[0].side == "buy"
                and self._is_post_recovery_fade_active(latest)
                and self._overlay_current_pct >= self.FADE_CAP
            ):
                return []
            return actions

        if not self._is_post_recovery_fade_active(latest):
            return []
        if self._overlay_current_pct <= self.FADE_CAP + self.FADE_SELL_THRESHOLD:
            return []

        price = current_prices.get(self._overlay_symbol, 0.0) if self._overlay_symbol else 0.0
        if price <= 0:
            return []
        pos = portfolio.positions.get(self._overlay_symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        sell_pct = min(self._overlay_current_pct - self.FADE_CAP, self.FADE_MAX_SELL)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []

        return [
            Action(
                symbol=self._overlay_symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=(
                    f"{self.VERSION_LABEL}_sell_external-fade-cap"
                    f"_t{self.FADE_CAP:.0%}_drop{self.FADE_ROLLING_DROP:.0%}"
                ),
            )
        ]

    def _track_overlay_position(self, candles_by_symbol, portfolio, current_prices) -> None:
        self._overlay_symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if self._overlay_symbol is None:
            return
        price = current_prices.get(self._overlay_symbol, 0.0)
        pos = portfolio.positions.get(self._overlay_symbol, PositionState())
        position_value = pos.quantity * price if price > 0 else 0.0
        total_value = portfolio.cash + position_value
        if total_value > 0:
            self._overlay_current_pct = position_value / total_value

    def _latest_overlay_bar(self, candles_by_symbol) -> pd.Series | None:
        if self._overlay_symbol is None:
            return None
        frame = candles_by_symbol.get(self._overlay_symbol)
        if frame is None or frame.empty:
            return None
        return frame.iloc[-1]

    def _is_post_recovery_fade_active(self, latest: pd.Series) -> bool:
        if self._overlay_symbol != self.FADE_SYMBOL:
            return False
        if self._value(latest, "btc_ema168_slope", default=0.0) < self.FADE_BTC_EMA168_SLOPE_MIN:
            return False
        age = self._call_count - self._fade_entry_call
        if age < self.FADE_MIN_AGE_CALLS or age > self.FADE_MAX_AGE_CALLS:
            return False
        if pd.isna(self._fade_entry_rolling_pos):
            return False

        rolling_delta = self._value(latest, "rolling_365d_pos", default=0.5) - self._fade_entry_rolling_pos
        if rolling_delta > -self.FADE_ROLLING_DROP:
            return False

        weak_momentum = (
            self._value(latest, "roc_10", default=0.0) <= -0.03
            or self._value(latest, "roc_20", default=0.0) <= -0.05
        )
        weak_structure = (
            self._value(latest, "price_vs_ema72", default=self._close_vs(latest, "ema72")) < 0
            or self._value(latest, "donchian_pos", default=0.5) < 0.45
        )
        return bool(weak_momentum and weak_structure)

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _close_vs(cls, latest: pd.Series, column: str) -> float:
        close = cls._value(latest, "close")
        den = cls._value(latest, column)
        if pd.isna(close) or pd.isna(den) or den <= 0:
            return float("nan")
        return close / den - 1.0


class V234BStrategy(V234AStrategy):
    """V2.34B: V2.34A, only when BTC has also lost EMA72 support."""

    VERSION_LABEL = "v2_34B"
    FADE_BTC_PRICE_VS_EMA72_MAX = 0.0

    @property
    def name(self) -> str:
        return "v2_34B"

    def _is_post_recovery_fade_active(self, latest: pd.Series) -> bool:
        if self._value(latest, "btc_price_vs_ema72", default=0.0) > self.FADE_BTC_PRICE_VS_EMA72_MAX:
            return False
        return super()._is_post_recovery_fade_active(latest)


class V234CStrategy(V234BStrategy):
    """V2.34C: V2.34B with a stricter BTC EMA72 loss threshold."""

    VERSION_LABEL = "v2_34C"
    FADE_BTC_PRICE_VS_EMA72_MAX = -0.02

    @property
    def name(self) -> str:
        return "v2_34C"


class V234DStrategy(V234AStrategy):
    """V2.34D: V2.34A only after BNB is already in a low historical position."""

    VERSION_LABEL = "v2_34D"
    FADE_ROLLING_POS_MAX = 0.35

    @property
    def name(self) -> str:
        return "v2_34D"

    def _is_post_recovery_fade_active(self, latest: pd.Series) -> bool:
        if self._value(latest, "rolling_365d_pos", default=0.5) > self.FADE_ROLLING_POS_MAX:
            return False
        return super()._is_post_recovery_fade_active(latest)


class V235AStrategy(V221EStrategy):
    """V2.35A: MIXED external risk cap throttles buys over V2.21E."""

    VERSION_LABEL = "v2_35A"
    CAP_CRITICAL = 0.25
    CAP_CAUTION = 0.40
    CAP_DEFAULT = 0.60
    CAP_FULL = 1.00
    CAP_EPSILON = 0.001

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_current_pct = 0.0

    @property
    def name(self) -> str:
        return "v2_35A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if symbol is not None:
            price = current_prices.get(symbol, 0.0)
            pos = portfolio.positions.get(symbol, PositionState())
            position_value = pos.quantity * price if price > 0 else 0.0
            total_value = portfolio.cash + position_value
            if total_value > 0:
                self._overlay_current_pct = position_value / total_value
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._should_apply_mixed_external_cap(raw_state, confirmed_state, buy_setup):
            return max_buy, guard

        cap, label = self._mixed_external_cap(latest, price)
        allowed_buy = max(0.0, cap - self._overlay_current_pct)
        if allowed_buy + self.CAP_EPSILON >= max_buy:
            return max_buy, guard
        max_buy = max(0.0, allowed_buy)
        return max_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_{label}_t{cap:.0%}")

    @staticmethod
    def _should_apply_mixed_external_cap(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        return bool(
            raw_state == "MIXED"
            and confirmed_state == "MIXED"
            and buy_setup in {"safe-recovery", "target-gap"}
        )

    def _mixed_external_cap(self, latest: pd.Series, price: float) -> tuple[float, str]:
        profile = self._mixed_profile(latest, price)
        btc_signal = self._btc_signal(latest)
        if profile == "RECLAIM_EMA72_LONG_DOWN" and btc_signal == "BTC_SUPPORTIVE":
            return self.CAP_FULL, "external_cap_full_reclaim"
        if self._is_critical_mixed_risk(latest, price, profile, btc_signal):
            return self.CAP_CRITICAL, "external_cap_critical"
        if self._is_caution_mixed_risk(latest, price, profile, btc_signal):
            return self.CAP_CAUTION, "external_cap_caution"
        return self.CAP_DEFAULT, "external_cap_default"

    def _mixed_profile(self, latest: pd.Series, price: float) -> str:
        ema72 = self._value(latest, "ema72")
        ema168 = self._value(latest, "ema168")
        if price <= 0 or pd.isna(ema72) or pd.isna(ema168):
            return "OTHER_MIXED"
        if price > ema72 and ema72 <= ema168:
            return "RECLAIM_EMA72_LONG_DOWN"
        if price < ema168 and ema72 > ema168:
            return "LOST_EMA168_LONG_UP"
        if price > ema72 and ema72 > ema168:
            return "ABOVE_EMA72_LONG_UP"
        if ema168 < price <= ema72 and ema72 > ema168:
            return "PULLBACK_ABOVE_EMA168"
        if price < ema168 and ema72 <= ema168:
            return "BELOW_EMA168_LONG_DOWN"
        return "OTHER_MIXED"

    def _btc_signal(self, latest: pd.Series) -> str:
        regime = str(latest.get("btc_regime", "RANGE"))
        btc_price_vs_ema72 = self._value(latest, "btc_price_vs_ema72", default=0.0)
        btc_ema168_slope = self._value(latest, "btc_ema168_slope", default=0.0)
        btc_roc_20 = self._value(latest, "btc_roc_20", default=0.0)
        if regime in {"STRONG_BULL", "BULL"}:
            if btc_price_vs_ema72 >= 0 and btc_ema168_slope >= 0:
                return "BTC_SUPPORTIVE"
            return "BTC_MIXED_UP"
        if regime == "BEAR":
            return "BTC_BEAR"
        if btc_price_vs_ema72 >= 0 and btc_roc_20 >= 0:
            return "BTC_RANGE_IMPROVING"
        return "BTC_RANGE_WEAK"

    def _is_critical_mixed_risk(
        self,
        latest: pd.Series,
        price: float,
        profile: str,
        btc_signal: str,
    ) -> bool:
        vol_high = self._value(latest, "volume_strength", default=1.0) >= 1.15
        sharp_neg = self._value(latest, "roc_20") < -0.08
        weak_range = btc_signal == "BTC_RANGE_WEAK"
        btc_bear = btc_signal == "BTC_BEAR"
        failed_structure = (
            self._price_vs(latest, price, "ema72") < 0
            and self._value(latest, "roc_20") < 0
            and self._value(latest, "ema24_slope") < 0
        )
        bear_rally_trap = (
            btc_bear
            and self._value(latest, "rolling_365d_pos", default=0.5) <= 0.55
            and self._price_vs(latest, price, "ema72") >= 0
        )
        return bool(
            (profile == "PULLBACK_ABOVE_EMA168" and (vol_high or sharp_neg or weak_range))
            or (profile == "BELOW_EMA168_LONG_DOWN" and (btc_bear or weak_range))
            or (weak_range and failed_structure)
            or bear_rally_trap
        )

    def _is_caution_mixed_risk(
        self,
        latest: pd.Series,
        price: float,
        profile: str,
        btc_signal: str,
    ) -> bool:
        if btc_signal == "BTC_RANGE_WEAK" or profile == "PULLBACK_ABOVE_EMA168":
            return True
        if profile == "ABOVE_EMA72_LONG_UP":
            return bool(
                self._value(latest, "rolling_365d_pos", default=0.5) >= 0.75
                or self._value(latest, "donchian_pos", default=0.5) < 0.35
                or btc_signal == "BTC_BEAR"
            )
        if profile == "LOST_EMA168_LONG_UP" and btc_signal == "BTC_SUPPORTIVE":
            return bool(
                self._price_vs(latest, price, "ema72") < 0
                or self._value(latest, "volume_strength", default=1.0) >= 1.15
                or self._value(latest, "atr_pct_rank", default=0.0) >= 0.85
            )
        return False

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V235BStrategy(V221EStrategy):
    """V2.35B: replace position-peak drawdown risk with market drawdown anchor."""

    VERSION_LABEL = "v2_35B"

    @property
    def name(self) -> str:
        return "v2_35B"

    def _calculate_drawdown_risk(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or price <= 0:
            return 0

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        profit_pct = price / pos.avg_cost - 1.0
        market_dd = max(
            self._value(latest, "dd_from_120d_high", default=0.0),
            self._value(latest, "dd_from_180d_high", default=0.0),
        )

        ok24 = not pd.isna(ema24)
        ok72 = not pd.isna(ema72)
        if profit_pct > 0.30 and market_dd > 0.25 and ok72 and price < ema72:
            return 2
        if profit_pct > 0.20 and market_dd > 0.15 and ok24 and price < ema24:
            return 1
        return 0

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)


class V235CStrategy(V235BStrategy):
    """V2.35C: V2.35B plus narrow safe-recovery target-reduce protection."""

    VERSION_LABEL = "v2_35C"
    RECENT_BUY_DEADBAND_CALLS = 3
    RECENT_BUY_TARGET_REDUCE_BAND = 0.12
    RECENT_BUY_TARGET_REDUCE_MAX_SELL = 0.08

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_any_buy_call = -10_000

    @property
    def name(self) -> str:
        return "v2_35C"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                self._last_any_buy_call = self._call_count
        return actions

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_recent_buy_target_reduce_churn(
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.RECENT_BUY_TARGET_REDUCE_BAND),
            min(adjusted_max_sell, self.RECENT_BUY_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_recent_buy_target_reduce_deadband"),
        )

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if self._call_count - self._last_any_buy_call > self.RECENT_BUY_DEADBAND_CALLS:
            return False
        if raw_state == "BEAR" or trend_risk > 1 or risk_score > 3:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        close = float(latest.get("close", 0.0) or 0.0)
        ema24 = self._value(latest, "ema24", default=0.0)
        ema72 = self._value(latest, "ema72", default=0.0)
        ema168 = self._value(latest, "ema168", default=0.0)
        ema24_slope = self._value(latest, "ema24_slope", default=0.0)
        ema72_slope = self._value(latest, "ema72_slope", default=0.0)
        ema168_slope = self._value(latest, "ema168_slope", default=0.0)
        roc_5 = self._value(latest, "roc_5", default=0.0)
        roc_10 = self._value(latest, "roc_10", default=0.0)
        donchian_pos = self._value(latest, "donchian_pos", default=0.0)
        dd_from_120d_high = self._value(latest, "dd_from_120d_high", default=1.0)
        btc_vs_ema72 = self._value(latest, "btc_price_vs_ema72", default=0.0)

        local_structure_ok = (
            close > ema24
            and close > ema72
            and ema24 >= ema72
            and ema72 >= ema168 * 0.98
        )
        slope_ok = ema24_slope >= 0.0 and ema72_slope >= 0.0 and ema168_slope >= -0.005
        momentum_ok = roc_5 >= 0.0 and roc_10 >= 0.0 and donchian_pos >= 0.55
        drawdown_ok = dd_from_120d_high <= 0.30
        btc_ok = btc_vs_ema72 >= -0.02
        return bool(local_structure_ok and slope_ok and momentum_ok and drawdown_ok and btc_ok)


class V235DStrategy(V235BStrategy):
    """V2.35D: V2.35B plus broad V2.25F-style recent-buy deadband."""

    VERSION_LABEL = "v2_35D"
    RECENT_BUY_DEADBAND_CALLS = 2
    RECENT_BUY_TARGET_REDUCE_BAND = 0.12
    RECENT_BUY_TARGET_REDUCE_MAX_SELL = 0.08

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_any_buy_call = -10_000

    @property
    def name(self) -> str:
        return "v2_35D"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                self._last_any_buy_call = self._call_count
        return actions

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_recent_buy_target_reduce_churn(
            latest=latest,
            raw_state=raw_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.RECENT_BUY_TARGET_REDUCE_BAND),
            min(adjusted_max_sell, self.RECENT_BUY_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_recent_buy_target_reduce_deadband"),
        )

    def _is_recent_buy_target_reduce_churn(
        self,
        latest: pd.Series,
        raw_state: str,
        trend_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if self._call_count - self._last_any_buy_call > self.RECENT_BUY_DEADBAND_CALLS:
            return False
        if raw_state == "BEAR" or trend_risk > 1 or risk_score > 3:
            return False
        return str(latest.get("btc_regime", "")) != "BEAR"


class V236AStrategy(V221EStrategy):
    """V2.36A: split profit-giveback drawdown from core risk into a cap layer."""

    VERSION_LABEL = "v2_36A"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layered_pos = PositionState()

    @property
    def name(self) -> str:
        return "v2_36A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._layered_pos = PositionState()
        if symbol is not None:
            self._layered_pos = portfolio.positions.get(symbol, PositionState())
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _calculate_drawdown_risk(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        # Profit giveback is handled as an explicit target cap in _compose_target.
        return 0

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
        target = super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        giveback_risk = self._profit_giveback_risk(latest, self._layered_pos, price)
        if giveback_risk <= 0:
            return target
        cap = self._lookup_target(raw_state, min(trend_risk + giveback_risk, 5))
        return min(target, cap)

    def _profit_giveback_risk(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> int:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0 or self._peak_price <= 0:
            return 0

        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        profit_pct = price / pos.avg_cost - 1.0
        dd_from_peak = 1.0 - price / self._peak_price

        ok24 = not pd.isna(ema24)
        ok72 = not pd.isna(ema72)
        if profit_pct > 0.30 and dd_from_peak > 0.18 and ok72 and price < ema72:
            return 2
        if profit_pct > 0.20 and dd_from_peak > 0.10 and ok24 and price < ema24:
            return 1
        return 0


class V236BStrategy(V221EStrategy):
    """V2.36B: behavior-equivalent target layering scaffold over V2.21E."""

    VERSION_LABEL = "v2_36B"

    @property
    def name(self) -> str:
        return "v2_36B"

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
        core_target = self._core_layer_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        risk_cap = self._risk_cap_layer(
            symbol=symbol,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        return min(core_target, risk_cap)

    def _core_layer_target(
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
        return super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )

    def _risk_cap_layer(
        self,
        symbol: str,
        raw_state: str,
        trend_risk: int,
        drawdown_risk: int,
        latest: pd.Series,
        price: float,
        side: str,
    ) -> float:
        return self._target_cap()


class V236CStrategy(V236BStrategy):
    """V2.36C: behavior-equivalent target-band pipeline over V2.21E."""

    VERSION_LABEL = "v2_36C"

    @property
    def name(self) -> str:
        return "v2_36C"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._call_count += 1

        position = self._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None:
            return []
        symbol = position["symbol"]
        df = position["df"]
        latest = position["latest"]
        price = position["price"]
        pos = position["pos"]
        current_pct = position["current_pct"]
        total_value = position["total_value"]

        market = self._build_market_context(df, latest, pos, price)
        signals = self._build_signal_context(df, latest, price, market)
        band = self._build_target_band(symbol, latest, price, market, signals)
        sell_target = band["sell_boundary"]
        buy_target = band["buy_boundary"]

        self._track_recovery(market["confirmed_state"])
        pullback_buy = self._is_safe_pullback_buy(
            market["confirmed_state"],
            latest,
            price,
            market["trend_risk"],
        )
        safe_recovery = self._is_safe_recovery_buy(latest, price, market["trend_risk"])

        sell_action = self._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if sell_action:
            self._last_sell_price = sell_action[0].price
            return sell_action

        buy_action = self._maybe_buy(
            symbol=symbol,
            latest=latest,
            price=price,
            current_pct=current_pct,
            total_value=total_value,
            buy_target=buy_target,
            market=market,
            signals=signals,
            trend_continuation=signals["trend_continuation"],
            safe_recovery=safe_recovery,
            recovery_override=signals["recovery_override"],
            pullback_buy=pullback_buy,
        )
        if buy_action:
            action = buy_action[0]
            reason = str(getattr(action, "reason", ""))
            if action.side == "buy" and "safe-recovery" in reason:
                self._last_recovery_buy_call = self._call_count
        return buy_action

    def _prepare_position_context(self, candles_by_symbol, portfolio, current_prices) -> dict | None:
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return None
        df = candles_by_symbol.get(symbol)
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return None

        self._latest_bar = latest
        self._current_price = price

        pos = portfolio.positions.get(symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        current_pct = position_value / total_value if total_value > 0 else 0.0

        if current_pct < 0.20:
            self._peak_price = price
        elif pos.quantity > 1e-12:
            self._peak_price = max(self._peak_price, price)

        return {
            "symbol": symbol,
            "df": df,
            "latest": latest,
            "price": price,
            "pos": pos,
            "current_pct": current_pct,
            "total_value": total_value,
        }

    def _build_market_context(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> dict:
        raw_state = self._detect_market_state(latest)
        confirmed_state = self._apply_state_confirmation(raw_state)
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, pos, price)
        risk_score = min(trend_risk + drawdown_risk, 5)
        return {
            "raw_state": raw_state,
            "confirmed_state": confirmed_state,
            "trend_risk": trend_risk,
            "drawdown_risk": drawdown_risk,
            "risk_score": risk_score,
        }

    def _build_signal_context(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> dict:
        recovery_override = self._is_recovery_override_setup(
            df=df,
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            confirmed_state=market["confirmed_state"],
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
        )
        effective_risk_score = (
            max(market["risk_score"] - self.RECOVERY_RISK_SCORE_REDUCTION, 0)
            if recovery_override
            else market["risk_score"]
        )
        trend_continuation = self._is_trend_continuation_setup(
            market["confirmed_state"],
            latest,
            price,
            market["trend_risk"],
        )
        bull_guard = self._is_bull_guard_setup(
            latest=latest,
            price=price,
            confirmed_state=market["confirmed_state"],
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
        )
        bull_guard_guard = ""
        if bull_guard:
            bull_guard_guard = f"{self.VERSION_LABEL}_bull_guard_floor"
        elif self._is_bull_guard_overheat_or_risk_skip(
            latest,
            price,
            market["confirmed_state"],
            market["trend_risk"],
            market["risk_score"],
        ):
            bull_guard_guard = self._bull_guard_skip_reason(
                latest,
                market["trend_risk"],
                market["risk_score"],
            )
        return {
            "recovery_override": recovery_override,
            "effective_risk_score": effective_risk_score,
            "trend_continuation": trend_continuation,
            "bull_guard": bull_guard,
            "bull_guard_guard": bull_guard_guard,
        }

    def _build_target_band(
        self,
        symbol: str,
        latest: pd.Series,
        price: float,
        market: dict,
        signals: dict,
    ) -> dict:
        sell_lookup_state, sell_risk_penalty = self._get_sell_target_state(
            market["raw_state"],
            market["confirmed_state"],
        )
        sell_boundary = self._lookup_target(
            sell_lookup_state,
            min(market["risk_score"] + sell_risk_penalty, 5),
        )
        buy_boundary = self._lookup_target(
            market["confirmed_state"],
            signals["effective_risk_score"],
        )

        vol_multiplier = self._get_directional_vol_multiplier(latest, price)
        sell_boundary = max(0.0, min(1.0, sell_boundary * vol_multiplier))
        buy_boundary = max(0.0, min(1.0, buy_boundary * vol_multiplier))

        btc_adjust = self._get_btc_adjust(latest, symbol)
        sell_boundary = max(0.0, min(1.0, sell_boundary + btc_adjust))
        buy_boundary = max(0.0, min(1.0, buy_boundary + btc_adjust))

        sell_boundary = self._compose_target(
            symbol=symbol,
            tactical_target=sell_boundary,
            raw_state=market["raw_state"],
            trend_risk=market["trend_risk"],
            drawdown_risk=market["drawdown_risk"],
            latest=latest,
            price=price,
            side="sell",
        )
        buy_boundary = self._compose_target(
            symbol=symbol,
            tactical_target=buy_boundary,
            raw_state=market["raw_state"],
            trend_risk=market["trend_risk"],
            drawdown_risk=market["drawdown_risk"],
            latest=latest,
            price=price,
            side="buy",
        )

        if signals["trend_continuation"]:
            buy_boundary = min(self._target_cap(), buy_boundary + self.TREND_CONTINUATION_BOOST)
        if signals["bull_guard"]:
            buy_boundary = max(buy_boundary, self.BULL_GUARD_MIN_POSITION_PCT)

        return {
            "sell_boundary": max(0.0, min(self._target_cap(), sell_boundary)),
            "buy_boundary": max(0.0, min(self._target_cap(), buy_boundary)),
        }

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        sell_setup = self._classify_sell_setup(
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            drawdown_risk=market["drawdown_risk"],
        )

        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_pullback(latest, price, market["confirmed_state"], market["trend_risk"]):
                sell_target = max(sell_target, current_pct)
        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_sell_blocked(
                market["confirmed_state"],
                market["raw_state"],
                market["trend_risk"],
                market["risk_score"],
                sell_setup,
            ):
                sell_target = max(sell_target, current_pct)

        bull_sell_threshold = (
            self._get_bull_sell_threshold()
            if market["confirmed_state"] == "BULL"
            else self.MIN_ADJUST_THRESHOLD
        )
        sell_threshold, max_sell, sell_guard = self._adjust_sell_execution(
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            confirmed_state=market["confirmed_state"],
            trend_risk=market["trend_risk"],
            drawdown_risk=market["drawdown_risk"],
            risk_score=market["risk_score"],
            sell_setup=sell_setup,
            sell_threshold=bull_sell_threshold,
            max_sell=self._base_max_sell(market["trend_risk"], market["risk_score"]),
        )
        max_sell = self._apply_sell_size_limit(max_sell, current_pct, pos, price, latest)

        if current_pct <= sell_target + sell_threshold:
            return []
        gap = current_pct - sell_target
        sell_pct = min(gap, max_sell)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
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
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=sell_target,
                    guard=sell_guard,
                ),
            )
        ]

    def _maybe_buy(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        current_pct: float,
        total_value: float,
        buy_target: float,
        market: dict,
        signals: dict,
        trend_continuation: bool,
        safe_recovery: bool,
        recovery_override: bool,
        pullback_buy: bool,
    ) -> list[Action]:
        buy_threshold = (
            self.BULL_GUARD_TARGET_GAP_THRESHOLD
            if signals["bull_guard"]
            else self.MIN_ADJUST_THRESHOLD
        )
        if current_pct >= buy_target - buy_threshold:
            return []

        cfg = self._state_config[market["confirmed_state"]]
        buy_setup = self._classify_buy_setup(
            trend_continuation,
            safe_recovery or recovery_override,
            pullback_buy,
        )

        if self._recovery_calls_remaining <= 0 and not recovery_override:
            effective_cooldown = self._compute_buy_cooldown(
                market["confirmed_state"],
                cfg,
                signals["effective_risk_score"],
            )
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

        max_buy = self._adjust_bull_buy_max_buy(max_buy, market["confirmed_state"], current_pct)
        max_buy, buy_guard = self._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=market["confirmed_state"],
        )

        guard = self._join_guard(cooldown_guard, buy_guard)
        guard = self._join_guard(guard, signals["bull_guard_guard"])
        if signals["bull_guard"]:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_bull_guard_target_gap_buy")
        if recovery_override:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_risk_score_reduced")
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_small_buy")

        buy_pct = min(gap, max_buy)
        buy_qty = total_value * buy_pct / price
        if buy_qty <= 1e-12:
            return []
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
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=buy_target,
                    guard=guard,
                ),
            )
        ]


class RecentRecoveryTargetReduceGuardMixin:
    """Shared post safe-recovery target-reduce guard."""

    RECOVERY_BUY_TARGET_REDUCE_BLOCK_CALLS = 3
    RECOVERY_TARGET_REDUCE_MIN_VOLUME_STRENGTH: float | None = None

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        sell_setup = self._classify_sell_setup(
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            drawdown_risk=market["drawdown_risk"],
        )
        if self._should_block_recent_recovery_target_reduce(latest, market, sell_setup):
            return []

        return super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )

    def _should_block_recent_recovery_target_reduce(
        self,
        latest: pd.Series,
        market: dict,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if self._call_count - self._last_recovery_buy_call > self.RECOVERY_BUY_TARGET_REDUCE_BLOCK_CALLS:
            return False
        if market["raw_state"] == "BEAR" or str(latest.get("btc_regime", "")) == "BEAR":
            return False
        if market["drawdown_risk"] > 0 or market["risk_score"] >= 3:
            return False
        if market["trend_risk"] > 1:
            return False
        min_volume = self.RECOVERY_TARGET_REDUCE_MIN_VOLUME_STRENGTH
        if min_volume is not None:
            volume_strength = latest.get("volume_strength")
            if pd.isna(volume_strength):
                return False
            if float(volume_strength) < min_volume:
                return False
        return True


class V236DStrategy(RecentRecoveryTargetReduceGuardMixin, V236CStrategy):
    """V2.36D: narrow post safe-recovery target-reduce block."""

    VERSION_LABEL = "v2_36D"

    @property
    def name(self) -> str:
        return "v2_36D"


class V236EStrategy(V236DStrategy):
    """V2.36E: V2.36D gated by minimum recovery volume strength."""

    VERSION_LABEL = "v2_36E"
    RECOVERY_TARGET_REDUCE_MIN_VOLUME_STRENGTH = 0.75

    @property
    def name(self) -> str:
        return "v2_36E"


class V236FStrategy(V236CStrategy):
    """V2.36F: soften early-recovery MIXED target-reduce sells."""

    VERSION_LABEL = "v2_36F"
    MIXED_REPAIR_TARGET_REDUCE_BAND = 0.12
    MIXED_REPAIR_TARGET_REDUCE_MAX_SELL = 0.08

    @property
    def name(self) -> str:
        return "v2_36F"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if not self._is_mixed_early_repair_target_reduce(
            latest=latest,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.MIXED_REPAIR_TARGET_REDUCE_BAND),
            min(adjusted_max_sell, self.MIXED_REPAIR_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_early_repair_trim_softened"),
        )

    def _is_mixed_early_repair_target_reduce(
        self,
        *,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str,
        drawdown_risk: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if drawdown_risk != 0:
            return False
        ema168_slope = latest.get("ema168_slope")
        if pd.isna(ema168_slope):
            return False
        return float(ema168_slope) < 0.0


class V236GStrategy(V236CStrategy):
    """V2.36G: scored MIXED target-reduce sell-fly softening."""

    VERSION_LABEL = "v2_36G"
    SELLFLY_STRICT_EMA168_SLOPE = -0.0019
    SELLFLY_LIGHT_MAX_SELL_MULT = 0.75
    SELLFLY_MEDIUM_TARGET_REDUCE_BAND = 0.08
    SELLFLY_MEDIUM_TARGET_REDUCE_MAX_SELL = 0.10
    SELLFLY_STRONG_TARGET_REDUCE_BAND = 0.12
    SELLFLY_STRONG_TARGET_REDUCE_MAX_SELL = 0.08

    @property
    def name(self) -> str:
        return "v2_36G"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        self._sellfly_current_sell_target = sell_target
        try:
            return super()._maybe_sell(
                symbol=symbol,
                latest=latest,
                price=price,
                pos=pos,
                current_pct=current_pct,
                total_value=total_value,
                sell_target=sell_target,
                market=market,
            )
        finally:
            self._sellfly_current_sell_target = None

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        score = self._sellfly_softening_score(
            latest=latest,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        )
        if score >= 9:
            return (
                max(threshold, self.SELLFLY_STRONG_TARGET_REDUCE_BAND),
                min(adjusted_max_sell, self.SELLFLY_STRONG_TARGET_REDUCE_MAX_SELL),
                self._join_guard(guard, f"{self.VERSION_LABEL}_sellfly_score{score}_strong"),
            )
        if score >= 7:
            return (
                max(threshold, self.SELLFLY_MEDIUM_TARGET_REDUCE_BAND),
                min(adjusted_max_sell, self.SELLFLY_MEDIUM_TARGET_REDUCE_MAX_SELL),
                self._join_guard(guard, f"{self.VERSION_LABEL}_sellfly_score{score}_medium"),
            )
        if score >= 5:
            return (
                threshold,
                adjusted_max_sell * self.SELLFLY_LIGHT_MAX_SELL_MULT,
                self._join_guard(guard, f"{self.VERSION_LABEL}_sellfly_score{score}_light"),
            )
        return threshold, adjusted_max_sell, guard

    def _sellfly_softening_score(
        self,
        *,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str,
        drawdown_risk: int,
        sell_setup: str,
    ) -> int:
        if not self._is_sellfly_target_reduce_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        ):
            return 0
        if self._is_sellfly_panic_setup(latest):
            return 9

        ema168_slope = self._sellfly_value(latest, "ema168_slope")
        roc20 = self._sellfly_value(latest, "roc_20")
        roc5 = self._sellfly_value(latest, "roc_5")
        donchian_pos = self._sellfly_value(latest, "donchian_pos")
        volume_strength = self._sellfly_value(latest, "volume_strength", default=1.0)
        price_vs_ema72 = self._sellfly_price_vs(latest, "ema72")
        btc_signal = self._sellfly_btc_signal(latest)
        sell_target = getattr(self, "_sellfly_current_sell_target", None)

        if pd.isna(ema168_slope) or pd.isna(roc20) or pd.isna(donchian_pos):
            return 0
        low_donchian = donchian_pos < 0.25
        deep_breakdown = price_vs_ema72 < -0.18 and donchian_pos < 0.35
        weak_momentum = roc20 < -0.12 and roc5 < 0 and volume_strength < 1.15
        veto = low_donchian or deep_breakdown or weak_momentum

        strict_repair = ema168_slope <= self.SELLFLY_STRICT_EMA168_SLOPE and not veto
        supportive = (
            btc_signal == "BTC_SUPPORTIVE"
            and volume_strength >= 1.0
            and roc20 >= 0
            and 0.45 <= donchian_pos <= 0.70
            and sell_target is not None
            and sell_target >= 0.60
        )
        if strict_repair:
            return 7
        if supportive:
            return 5
        return 0

    def _is_sellfly_target_reduce_context(
        self,
        *,
        raw_state: str,
        confirmed_state: str,
        drawdown_risk: int,
        sell_setup: str,
    ) -> bool:
        return (
            sell_setup == "target-reduce"
            and raw_state == "MIXED"
            and confirmed_state == "MIXED"
            and drawdown_risk == 0
        )

    def _is_sellfly_panic_setup(self, latest: pd.Series) -> bool:
        ema168_slope = self._sellfly_value(latest, "ema168_slope")
        roc20 = self._sellfly_value(latest, "roc_20")
        donchian_pos = self._sellfly_value(latest, "donchian_pos")
        volume_strength = self._sellfly_value(latest, "volume_strength", default=1.0)
        if pd.isna(ema168_slope) or pd.isna(roc20) or pd.isna(donchian_pos):
            return False
        return (
            ema168_slope < 0
            and volume_strength >= 1.15
            and roc20 <= -0.20
            and donchian_pos >= 0.45
        )

    @staticmethod
    def _sellfly_value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _sellfly_price_vs(cls, latest: pd.Series, column: str) -> float:
        close = cls._sellfly_value(latest, "close")
        base = cls._sellfly_value(latest, column)
        if pd.isna(close) or pd.isna(base) or base <= 0:
            return float("nan")
        return close / base - 1.0

    @classmethod
    def _sellfly_btc_signal(cls, latest: pd.Series) -> str:
        regime = str(latest.get("btc_regime", ""))
        btc_vs_ema72 = cls._sellfly_value(latest, "btc_price_vs_ema72")
        btc_roc20 = cls._sellfly_value(latest, "btc_roc_20")
        if regime == "BEAR":
            return "BTC_BEAR"
        if btc_vs_ema72 >= 0.02 and btc_roc20 >= 0:
            return "BTC_SUPPORTIVE"
        if btc_vs_ema72 >= -0.02 and btc_roc20 >= 0:
            return "BTC_RANGE_IMPROVING"
        if btc_vs_ema72 < -0.02 and btc_roc20 < 0:
            return "BTC_RANGE_WEAK"
        return "BTC_NEUTRAL"


class V236HStrategy(V236GStrategy):
    """V2.36H: panic-repair sell-fly softening only."""

    VERSION_LABEL = "v2_36H"

    @property
    def name(self) -> str:
        return "v2_36H"

    def _sellfly_softening_score(
        self,
        *,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str,
        drawdown_risk: int,
        sell_setup: str,
    ) -> int:
        if not self._is_sellfly_target_reduce_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        ):
            return 0
        return 9 if self._is_sellfly_panic_setup(latest) else 0


class V236IStrategy(V236HStrategy):
    """V2.36I: panic sell-fly plus light supportive-recovery softening."""

    VERSION_LABEL = "v2_36I"

    @property
    def name(self) -> str:
        return "v2_36I"

    def _sellfly_softening_score(
        self,
        *,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str,
        drawdown_risk: int,
        sell_setup: str,
    ) -> int:
        panic_score = super()._sellfly_softening_score(
            latest=latest,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        )
        if panic_score:
            return panic_score
        if not self._is_sellfly_target_reduce_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            drawdown_risk=drawdown_risk,
            sell_setup=sell_setup,
        ):
            return 0

        roc20 = self._sellfly_value(latest, "roc_20")
        donchian_pos = self._sellfly_value(latest, "donchian_pos")
        volume_strength = self._sellfly_value(latest, "volume_strength", default=1.0)
        btc_signal = self._sellfly_btc_signal(latest)
        sell_target = getattr(self, "_sellfly_current_sell_target", None)
        if (
            btc_signal == "BTC_SUPPORTIVE"
            and volume_strength >= 1.0
            and roc20 >= 0
            and 0.45 <= donchian_pos <= 0.70
            and sell_target is not None
            and sell_target >= 0.60
        ):
            return 5
        return 0


class V3Strategy(RecentRecoveryTargetReduceGuardMixin, V236HStrategy):
    """V3: conservative live-run repair over the V2.36C refactor.

    Keeps the V2.36C target-band pipeline and only adds two narrow guards:
    1. protect volume-confirmed safe-recovery buys from immediate routine
       target-reduce trims;
    2. soften target-reduce only in panic-repair MIXED/MIXED sell-fly setups.
    """

    VERSION_LABEL = "v3"
    RECOVERY_TARGET_REDUCE_MIN_VOLUME_STRENGTH = 0.75

    @property
    def name(self) -> str:
        return "v3"


class V31AStrategy(V3Strategy):
    """V3.1A: ordinary MIXED/MIXED target-gap and target-reduce denoise."""

    VERSION_LABEL = "v3_1A"
    MIXED_TARGET_GAP_MAX_BUY = 0.08
    MIXED_TARGET_GAP_MAX_BUY_HIGH_QUALITY = 0.14
    MIXED_TARGET_REDUCE_BAND = 0.10
    MIXED_TARGET_REDUCE_MAX_SELL = 0.08

    @property
    def name(self) -> str:
        return "v3_1A"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if raw_state != "MIXED" or confirmed_state != "MIXED" or buy_setup != "target-gap":
            return adjusted, guard
        cap = (
            self.MIXED_TARGET_GAP_MAX_BUY_HIGH_QUALITY
            if self._v3_is_quality_mixed_reclaim(latest, price)
            else self.MIXED_TARGET_GAP_MAX_BUY
        )
        if adjusted <= cap:
            return adjusted, guard
        return min(adjusted, cap), self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_target_gap_capped")

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        if (
            sell_setup == "target-reduce"
            and raw_state == "MIXED"
            and confirmed_state == "MIXED"
            and trend_risk <= 1
            and drawdown_risk == 0
            and risk_score <= 2
        ):
            return (
                max(threshold, self.MIXED_TARGET_REDUCE_BAND),
                min(adjusted_max_sell, self.MIXED_TARGET_REDUCE_MAX_SELL),
                self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_target_reduce_small_trim"),
            )
        return threshold, adjusted_max_sell, guard

    def _v3_is_quality_mixed_reclaim(self, latest: pd.Series, price: float) -> bool:
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        return bool(
            price > self._v3_value(latest, "ema72")
            and self._v3_value(latest, "ema24_slope") > 0
            and self._v3_value(latest, "donchian_pos") >= 0.54
            and self._v3_ratio(latest, "ema72", "ema168") >= -0.08
        )

    @staticmethod
    def _v3_value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _v3_ratio(cls, latest: pd.Series, numerator: str, denominator: str) -> float:
        num = cls._v3_value(latest, numerator)
        den = cls._v3_value(latest, denominator)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return num / den - 1.0

    @classmethod
    def _v3_price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._v3_value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V32AStrategy(V3Strategy):
    """V3.2A: small BEAR/MIXED structural reclaim probe."""

    VERSION_LABEL = "v3_2A"
    BEAR_RECLAIM_MIN_TARGET = 0.45
    BEAR_RECLAIM_MAX_BUY = 0.12

    @property
    def name(self) -> str:
        return "v3_2A"

    def _build_target_band(
        self,
        symbol: str,
        latest: pd.Series,
        price: float,
        market: dict,
        signals: dict,
    ) -> dict:
        band = super()._build_target_band(
            symbol=symbol,
            latest=latest,
            price=price,
            market=market,
            signals=signals,
        )
        if self._v3_is_bear_reclaim(latest, price, market):
            band["buy_boundary"] = max(band["buy_boundary"], self.BEAR_RECLAIM_MIN_TARGET)
        return band

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        market = {"raw_state": raw_state, "confirmed_state": confirmed_state}
        if self._v3_is_bear_reclaim(latest, price, market):
            adjusted = min(max(adjusted, self.BEAR_RECLAIM_MAX_BUY), self.BEAR_RECLAIM_MAX_BUY)
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_bear_reclaim_probe")
        return adjusted, guard

    def _v3_is_bear_reclaim(self, latest: pd.Series, price: float, market: dict) -> bool:
        raw_state = market.get("raw_state")
        confirmed_state = market.get("confirmed_state")
        if raw_state not in {"BEAR", "MIXED"} or confirmed_state not in {"BEAR", "MIXED"}:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        return bool(
            price > self._v3_value(latest, "ema72")
            and price > self._v3_value(latest, "ema168")
            and self._v3_value(latest, "ema24_slope") > 0
            and self._v3_value(latest, "donchian_pos") >= 0.55
            and self._v3_ratio(latest, "ema24", "ema72") >= -0.02
        )

    @staticmethod
    def _v3_value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _v3_ratio(cls, latest: pd.Series, numerator: str, denominator: str) -> float:
        num = cls._v3_value(latest, numerator)
        den = cls._v3_value(latest, denominator)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return num / den - 1.0


class V33AStrategy(V3Strategy):
    """V3.3A: light multi-factor profit taking in confirmed bull regimes."""

    VERSION_LABEL = "v3_3A"
    PROFIT_TAKE_MAX_SELL = 0.10
    PROFIT_TAKE_MIN_POSITION = 0.72

    @property
    def name(self) -> str:
        return "v3_3A"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        baseline_action = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if baseline_action:
            return baseline_action
        if not self._v3_should_light_profit_take(latest, price, current_pct, market):
            return []
        sell_pct = min(self.PROFIT_TAKE_MAX_SELL, current_pct - self.PROFIT_TAKE_MIN_POSITION)
        if sell_pct <= 0:
            return []
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="light-profit-take",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=current_pct - sell_pct,
                    guard=f"{self.VERSION_LABEL}_multi_factor_light_profit_take",
                ),
            )
        ]

    def _v3_should_light_profit_take(
        self,
        latest: pd.Series,
        price: float,
        current_pct: float,
        market: dict,
    ) -> bool:
        if market["confirmed_state"] != "BULL" or market["raw_state"] not in {"BULL", "MIXED"}:
            return False
        if market["trend_risk"] > 0 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.PROFIT_TAKE_MIN_POSITION + 0.05:
            return False
        score = 0
        if self._v3_value(latest, "rolling_365d_pos") >= 0.82:
            score += 1
        if self._v3_price_vs(latest, price, "ema168") >= 0.22:
            score += 1
        if self._v3_value(latest, "donchian_pos") >= 0.88:
            score += 1
        if self._v3_value(latest, "roc_20") >= 0.18:
            score += 1
        if self._v3_value(latest, "atr_pct_rank", default=0.0) >= 0.70:
            score += 1
        return score >= 4

    @staticmethod
    def _v3_value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _v3_price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._v3_value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V31BStrategy(V3Strategy):
    """V3.1B: stateful MIXED/MIXED cooldown to suppress flip-flop churn."""

    VERSION_LABEL = "v3_1B"
    MIXED_BUY_AFTER_SELL_COOLDOWN_CALLS = 6
    MIXED_SELL_AFTER_BUY_COOLDOWN_CALLS = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_mixed_target_gap_buy_call = -10_000
        self._last_mixed_target_reduce_sell_call = -10_000

    @property
    def name(self) -> str:
        return "v3_1B"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if not actions:
            return actions
        action = actions[0]
        reason = str(getattr(action, "reason", ""))
        if action.side == "buy" and "_buy_target-gap_" in reason and "_rawMIXED_confMIXED_" in reason:
            self._last_mixed_target_gap_buy_call = self._call_count
        elif action.side == "sell" and "_sell_target-reduce_" in reason and "_rawMIXED_confMIXED_" in reason:
            self._last_mixed_target_reduce_sell_call = self._call_count
        return actions

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_mixed_target_gap_context(raw_state, confirmed_state, buy_setup):
            return adjusted, guard
        if self._call_count - self._last_mixed_target_reduce_sell_call <= self.MIXED_BUY_AFTER_SELL_COOLDOWN_CALLS:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_buy_cooldown")
        return adjusted, guard

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        sell_setup = self._classify_sell_setup(
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            drawdown_risk=market["drawdown_risk"],
        )
        if (
            sell_setup == "target-reduce"
            and market["raw_state"] == "MIXED"
            and market["confirmed_state"] == "MIXED"
            and market["trend_risk"] <= 1
            and market["drawdown_risk"] == 0
            and market["risk_score"] <= 2
            and self._call_count - self._last_mixed_target_gap_buy_call <= self.MIXED_SELL_AFTER_BUY_COOLDOWN_CALLS
        ):
            return []
        return super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )

    @staticmethod
    def _is_mixed_target_gap_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        return raw_state == "MIXED" and confirmed_state == "MIXED" and buy_setup == "target-gap"


class V33BStrategy(V3Strategy):
    """V3.3B: stateful high-profit trailing trim with rebuy cooldown."""

    VERSION_LABEL = "v3_3B"
    PROFIT_TAKE_MIN_PROFIT = 0.85
    PROFIT_TAKE_MIN_PEAK_PROFIT = 1.00
    PROFIT_TAKE_PEAK_PULLBACK = 0.14
    PROFIT_TAKE_MIN_POSITION = 0.82
    PROFIT_TAKE_SELL_PCT = 0.08
    PROFIT_TAKE_MIN_GAP_CALLS = 30
    PROFIT_TAKE_REBUY_COOLDOWN_CALLS = 24
    PROFIT_TAKE_NEW_PEAK_BUFFER = 0.08

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_profit_take_call = -10_000
        self._last_profit_take_peak_price = 0.0

    @property
    def name(self) -> str:
        return "v3_3B"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        baseline_action = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if baseline_action:
            return baseline_action
        if not self._should_trailing_profit_take(latest, price, pos, current_pct, market):
            return []
        sell_qty = min(total_value * self.PROFIT_TAKE_SELL_PCT / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        self._last_profit_take_call = self._call_count
        self._last_profit_take_peak_price = self._peak_price
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="trailing-profit-take",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=max(0.0, current_pct - self.PROFIT_TAKE_SELL_PCT),
                    guard=f"{self.VERSION_LABEL}_high_profit_trailing_trim",
                ),
            )
        ]

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if buy_setup not in {"target-gap", "pullback"}:
            return adjusted, guard
        if self._call_count - self._last_profit_take_call <= self.PROFIT_TAKE_REBUY_COOLDOWN_CALLS:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_post_profit_take_buy_cooldown")
        return adjusted, guard

    def _should_trailing_profit_take(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["confirmed_state"] != "BULL" or market["raw_state"] not in {"BULL", "MIXED"}:
            return False
        if market["trend_risk"] > 1:
            return False
        if current_pct < self.PROFIT_TAKE_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.PROFIT_TAKE_MIN_GAP_CALLS:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.PROFIT_TAKE_MIN_PROFIT:
            return False
        if peak_profit_pct < self.PROFIT_TAKE_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.PROFIT_TAKE_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.PROFIT_TAKE_NEW_PEAK_BUFFER):
            return False

        if self._v3_value(latest, "rolling_365d_pos") < 0.78:
            return False
        if self._v3_price_vs(latest, price, "ema168") < 0.20:
            return False
        if self._v3_value(latest, "donchian_pos") < 0.80:
            return False
        ema72 = self._v3_value(latest, "ema72")
        if pd.isna(ema72) or price <= ema72:
            return False
        return True

    @staticmethod
    def _v3_value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _v3_price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._v3_value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V33CStrategy(V33BStrategy):
    """V3.3C: trailing trim only after explicit giveback risk appears."""

    VERSION_LABEL = "v3_3C"

    @property
    def name(self) -> str:
        return "v3_3C"

    def _should_trailing_profit_take(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if market["drawdown_risk"] < 1:
            return False
        return super()._should_trailing_profit_take(
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            market=market,
        )


class V33DStrategy(V33BStrategy):
    """V3.3D: allow trims on giveback risk or BTC-weak range overextension."""

    VERSION_LABEL = "v3_3D"
    RANGE_TRIM_MIN_ROC20 = 0.12

    @property
    def name(self) -> str:
        return "v3_3D"

    def _should_trailing_profit_take(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if not super()._should_trailing_profit_take(
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            market=market,
        ):
            return False
        if market["drawdown_risk"] >= 1 and str(latest.get("btc_regime", "")) != "STRONG_BULL":
            return True
        return bool(
            market["drawdown_risk"] == 0
            and str(latest.get("btc_regime", "")) == "RANGE"
            and self._v3_value(latest, "btc_price_vs_ema72") < 0
            and self._v3_value(latest, "roc_20") >= self.RANGE_TRIM_MIN_ROC20
        )


class V33EStrategy(V33CStrategy):
    """V3.3E: trailing trim plus discounted rebuy ceiling, trend-cont exempt."""

    VERSION_LABEL = "v3_3E"
    PROFIT_TAKE_REBUY_MAX_PRICE_MULT = 0.96
    PROFIT_TAKE_PRICE_CEILING_CALLS = 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_profit_take_sell_price = 0.0

    @property
    def name(self) -> str:
        return "v3_3E"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if actions and actions[0].side == "sell" and "trailing-profit-take" in str(actions[0].reason):
            self._last_profit_take_sell_price = float(actions[0].price)
        return actions

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if buy_setup == "trend-cont":
            return adjusted, guard
        if self._last_profit_take_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_profit_take_call > self.PROFIT_TAKE_PRICE_CEILING_CALLS:
            return adjusted, guard
        rebuy_max_price = self._last_profit_take_sell_price * self.PROFIT_TAKE_REBUY_MAX_PRICE_MULT
        if price > rebuy_max_price:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_post_profit_take_price_ceiling")
        return adjusted, guard


class V34AStrategy(V33EStrategy):
    """V3.4A: earlier bull giveback trims plus constructive-MIXED trim suppression."""

    VERSION_LABEL = "v3_4A"
    EARLY_TRIM_MIN_PROFIT = 0.55
    EARLY_TRIM_MIN_PEAK_PROFIT = 0.72
    EARLY_TRIM_PEAK_PULLBACK = 0.09
    EARLY_TRIM_MIN_POSITION = 0.88
    EARLY_TRIM_SELL_PCT = 0.06
    EARLY_TRIM_MIN_GAP_CALLS = 18
    EARLY_TRIM_NEW_PEAK_BUFFER = 0.05
    CONSTRUCTIVE_MIXED_MIN_ROLLING_POS = 0.55
    CONSTRUCTIVE_MIXED_MIN_DONCHIAN = 0.45

    @property
    def name(self) -> str:
        return "v3_4A"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if actions:
            action = actions[0]
            if (
                action.side == "sell"
                and self._is_constructive_mixed_target_reduce(action, latest, price, market)
            ):
                return []
            return actions

        if not self._should_early_giveback_trim(latest, price, pos, current_pct, market):
            return []
        sell_qty = min(total_value * self.EARLY_TRIM_SELL_PCT / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        self._last_profit_take_call = self._call_count
        self._last_profit_take_peak_price = self._peak_price
        self._last_profit_take_sell_price = price
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="early-profit-giveback",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=max(0.0, current_pct - self.EARLY_TRIM_SELL_PCT),
                    guard=f"{self.VERSION_LABEL}_bull_giveback_trim",
                ),
            )
        ]

    def _should_early_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["confirmed_state"] != "BULL" or market["raw_state"] != "BULL":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.EARLY_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.EARLY_TRIM_MIN_GAP_CALLS:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price < ema24 and price > ema72 and ema72 > ema168):
            return False
        if self._v3_value(latest, "ema168_slope") <= 0:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.EARLY_TRIM_MIN_PROFIT:
            return False
        if peak_profit_pct < self.EARLY_TRIM_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.EARLY_TRIM_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.EARLY_TRIM_NEW_PEAK_BUFFER):
            return False

        if self._v3_value(latest, "rolling_365d_pos") < 0.82:
            return False
        if self._v3_value(latest, "donchian_pos") < 0.84:
            return False
        if self._v3_price_vs(latest, price, "ema168") < 0.16:
            return False
        return True

    def _is_constructive_mixed_target_reduce(
        self,
        action: Action,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> bool:
        reason = str(getattr(action, "reason", ""))
        if "_sell_target-reduce_" not in reason:
            return False
        if market["raw_state"] != "MIXED" or market["confirmed_state"] != "MIXED":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] != 0 or market["risk_score"] > 2:
            return False
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema72 and ema72 > ema168 and ema168_slope > 0):
            return False
        if self._v3_value(latest, "rolling_365d_pos") < self.CONSTRUCTIVE_MIXED_MIN_ROLLING_POS:
            return False
        if self._v3_value(latest, "donchian_pos") < self.CONSTRUCTIVE_MIXED_MIN_DONCHIAN:
            return False
        return True


class V34BStrategy(V34AStrategy):
    """V3.4B: looser bull giveback trims and EMA168-based constructive MIXED suppression."""

    VERSION_LABEL = "v3_4B"
    EARLY_TRIM_MIN_PROFIT = 0.45
    EARLY_TRIM_MIN_PEAK_PROFIT = 0.60
    EARLY_TRIM_PEAK_PULLBACK = 0.06
    EARLY_TRIM_MIN_POSITION = 0.82
    EARLY_TRIM_MIN_GAP_CALLS = 14
    CONSTRUCTIVE_MIXED_MIN_ROLLING_POS = 0.62
    CONSTRUCTIVE_MIXED_MIN_DONCHIAN = 0.30

    @property
    def name(self) -> str:
        return "v3_4B"

    def _should_early_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if not super(V34AStrategy, self)._should_trailing_profit_take(
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            market=market,
        ):
            return False
        if market["confirmed_state"] != "BULL" or market["raw_state"] != "BULL":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.EARLY_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.EARLY_TRIM_MIN_GAP_CALLS:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price < ema24 and price > ema168 and ema72 > ema168):
            return False
        if self._v3_value(latest, "ema168_slope") <= 0:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.EARLY_TRIM_MIN_PROFIT:
            return False
        if peak_profit_pct < self.EARLY_TRIM_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.EARLY_TRIM_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.EARLY_TRIM_NEW_PEAK_BUFFER):
            return False

        if self._v3_value(latest, "rolling_365d_pos") < 0.72:
            return False
        if self._v3_value(latest, "donchian_pos") < 0.68:
            return False
        if self._v3_price_vs(latest, price, "ema168") < 0.12:
            return False
        return True

    def _is_constructive_mixed_target_reduce(
        self,
        action: Action,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> bool:
        reason = str(getattr(action, "reason", ""))
        if "_sell_target-reduce_" not in reason:
            return False
        if market["raw_state"] != "MIXED" or market["confirmed_state"] != "MIXED":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] != 0 or market["risk_score"] > 2:
            return False
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        if pd.isna(ema72) or pd.isna(ema168) or pd.isna(ema168_slope):
            return False
        if not (price > ema168 * 0.995 and ema72 > ema168 and ema168_slope > 0):
            return False
        if self._v3_value(latest, "rolling_365d_pos") < self.CONSTRUCTIVE_MIXED_MIN_ROLLING_POS:
            return False
        if self._v3_value(latest, "donchian_pos") < self.CONSTRUCTIVE_MIXED_MIN_DONCHIAN:
            return False
        return True


class V34CStrategy(V34BStrategy):
    """V3.4C: BTC-only bull giveback trim + constructive-MIXED suppression."""

    VERSION_LABEL = "v3_4C"

    @property
    def name(self) -> str:
        return "v3_4C"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = super(V34BStrategy, self)._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if symbol != "BTC/USDT":
            return actions
        if actions:
            action = actions[0]
            if (
                action.side == "sell"
                and self._is_constructive_mixed_target_reduce(action, latest, price, market)
            ):
                return []
            return actions
        if not self._should_early_giveback_trim(latest, price, pos, current_pct, market):
            return []
        sell_qty = min(total_value * self.EARLY_TRIM_SELL_PCT / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        self._last_profit_take_call = self._call_count
        self._last_profit_take_peak_price = self._peak_price
        self._last_profit_take_sell_price = price
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="early-profit-giveback",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=max(0.0, current_pct - self.EARLY_TRIM_SELL_PCT),
                    guard=f"{self.VERSION_LABEL}_btc_bull_giveback_trim",
                ),
            )
        ]

    def _should_early_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["confirmed_state"] != "BULL" or market["raw_state"] != "BULL":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.EARLY_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.EARLY_TRIM_MIN_GAP_CALLS:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price < ema24 and price > ema168 and ema72 > ema168):
            return False
        if self._v3_value(latest, "ema168_slope") <= 0:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.EARLY_TRIM_MIN_PROFIT:
            return False
        if peak_profit_pct < self.EARLY_TRIM_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.EARLY_TRIM_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.EARLY_TRIM_NEW_PEAK_BUFFER):
            return False

        if self._v3_value(latest, "rolling_365d_pos") < 0.72:
            return False
        if self._v3_value(latest, "donchian_pos") < 0.68:
            return False
        if self._v3_price_vs(latest, price, "ema168") < 0.12:
            return False
        return True


class V34DStrategy(V34CStrategy):
    """V3.4D: BTC-only, stricter giveback trim with longer no-higher rebuy ceiling."""

    VERSION_LABEL = "v3_4D"
    PROFIT_TAKE_REBUY_MAX_PRICE_MULT = 1.00
    PROFIT_TAKE_PRICE_CEILING_CALLS = 120

    @property
    def name(self) -> str:
        return "v3_4D"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        if symbol != "BTC/USDT":
            return V33EStrategy._maybe_sell(
                self,
                symbol=symbol,
                latest=latest,
                price=price,
                pos=pos,
                current_pct=current_pct,
                total_value=total_value,
                sell_target=sell_target,
                market=market,
            )
        return super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )

    def _should_early_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if not super()._should_early_giveback_trim(
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            market=market,
        ):
            return False
        if self._v3_value(latest, "roc_20") > -0.06:
            return False
        if self._v3_value(latest, "donchian_pos") > 0.82:
            return False
        if self._v3_value(latest, "atr_pct_rank") < 0.65:
            return False
        if self._v3_value(latest, "rolling_365d_pos") < 0.78:
            return False
        return True


class V34EStrategy(V34DStrategy):
    """V3.4E: same BTC-only trims, but medium rebuy ceiling horizon."""

    VERSION_LABEL = "v3_4E"
    PROFIT_TAKE_PRICE_CEILING_CALLS = 75

    @property
    def name(self) -> str:
        return "v3_4E"


class V34FStrategy(V33EStrategy):
    """V3.4F: BTC-only constructive-MIXED deadband plus mature-bull giveback trim."""

    VERSION_LABEL = "v3_4F"
    CONSTRUCTIVE_MIXED_DELAY_CALLS = 2
    EARLY_TRIM_SELL_PCT = 0.06
    EARLY_TRIM_MIN_POSITION = 0.82
    EARLY_TRIM_MIN_PROFIT = 0.45
    EARLY_TRIM_MIN_PEAK_PROFIT = 0.60
    EARLY_TRIM_PEAK_PULLBACK = 0.06
    EARLY_TRIM_MIN_GAP_CALLS = 18
    EARLY_TRIM_NEW_PEAK_BUFFER = 0.05
    PROFIT_TAKE_REBUY_MAX_PRICE_MULT = 1.00
    PROFIT_TAKE_PRICE_CEILING_CALLS = 90

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constructive_mixed_streaks: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "v3_4F"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = V33EStrategy._maybe_sell(
            self,
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if symbol != "BTC/USDT":
            return actions

        if actions:
            action = actions[0]
            if action.side == "sell" and self._is_constructive_mixed_target_reduce(action, latest, price, market):
                streak = self._constructive_mixed_streaks.get(symbol, 0) + 1
                self._constructive_mixed_streaks[symbol] = streak
                if streak <= self.CONSTRUCTIVE_MIXED_DELAY_CALLS:
                    return []
            else:
                self._constructive_mixed_streaks[symbol] = 0
            return actions

        self._constructive_mixed_streaks[symbol] = 0
        if not self._should_mature_bull_giveback_trim(latest, price, pos, current_pct, market):
            return []
        sell_qty = min(total_value * self.EARLY_TRIM_SELL_PCT / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        self._last_profit_take_call = self._call_count
        self._last_profit_take_peak_price = self._peak_price
        self._last_profit_take_sell_price = price
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="early-profit-giveback",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=max(0.0, current_pct - self.EARLY_TRIM_SELL_PCT),
                    guard=f"{self.VERSION_LABEL}_mature_bull_giveback_trim",
                ),
            )
        ]

    def _should_mature_bull_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["raw_state"] != "BULL" or market["confirmed_state"] != "BULL":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.EARLY_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.EARLY_TRIM_MIN_GAP_CALLS:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price < ema24 and price > ema72 and ema72 > ema168):
            return False
        if self._v3_value(latest, "ema168_slope") <= 0:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.EARLY_TRIM_MIN_PROFIT:
            return False
        if peak_profit_pct < self.EARLY_TRIM_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.EARLY_TRIM_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.EARLY_TRIM_NEW_PEAK_BUFFER):
            return False

        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        atr_rank = self._v3_value(latest, "atr_pct_rank")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(rolling_pos) or pd.isna(donchian_pos) or pd.isna(atr_rank) or pd.isna(roc_20):
            return False
        return bool(
            0.78 <= rolling_pos <= 0.90
            and 0.70 <= donchian_pos <= 0.83
            and atr_rank >= 0.65
            and roc_20 <= -0.06
        )

    def _is_constructive_mixed_target_reduce(
        self,
        action: Action,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> bool:
        reason = str(getattr(action, "reason", ""))
        if "_sell_target-reduce_" not in reason:
            return False
        if market["raw_state"] != "MIXED" or market["confirmed_state"] != "MIXED":
            return False
        if market["risk_score"] > 2 or market["trend_risk"] > 2 or market["drawdown_risk"] != 0:
            return False
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        atr_rank = self._v3_value(latest, "atr_pct_rank")
        if (
            pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(atr_rank)
        ):
            return False
        return bool(
            ema72 > ema168
            and ema168_slope > 0
            and price > ema168 * 0.995
            and 0.62 <= rolling_pos <= 0.80
            and 0.30 <= donchian_pos <= 0.72
            and atr_rank < 0.80
        )


class V34GStrategy(V34FStrategy):
    """V3.4G: same split-context logic, but only delay the first constructive MIXED trim."""

    VERSION_LABEL = "v3_4G"
    CONSTRUCTIVE_MIXED_DELAY_CALLS = 1

    @property
    def name(self) -> str:
        return "v3_4G"


class V34HStrategy(V34FStrategy):
    """V3.4H: only suppress higher-quality constructive MIXED trims."""

    VERSION_LABEL = "v3_4H"
    CONSTRUCTIVE_MIXED_DELAY_CALLS = 1

    @property
    def name(self) -> str:
        return "v3_4H"

    def _is_constructive_mixed_target_reduce(
        self,
        action: Action,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> bool:
        if not super()._is_constructive_mixed_target_reduce(action, latest, price, market):
            return False
        ema168_slope = self._v3_value(latest, "ema168_slope")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        atr_rank = self._v3_value(latest, "atr_pct_rank")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        btc_regime = str(latest.get("btc_regime", ""))
        if pd.isna(ema168_slope) or pd.isna(donchian_pos) or pd.isna(atr_rank) or pd.isna(rolling_pos):
            return False
        return bool(
            ema168_slope >= 0.005
            and donchian_pos >= 0.50
            and atr_rank <= 0.42
            and rolling_pos <= 0.74
            and btc_regime in {"BULL", "RANGE"}
        )


class V34IStrategy(V34HStrategy):
    """V3.4I: keep mature giveback trim, but only soften high-quality constructive MIXED trims."""

    VERSION_LABEL = "v3_4I"
    CONSTRUCTIVE_MIXED_DELAY_CALLS = 0
    CONSTRUCTIVE_MIXED_MIN_THRESHOLD = 0.10
    CONSTRUCTIVE_MIXED_MAX_SELL = 0.10

    @property
    def name(self) -> str:
        return "v3_4I"

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
        threshold, adjusted_max_sell, guard = super()._adjust_sell_execution(
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
        dummy_action = Action(symbol="BTC/USDT", side="sell", quantity=0.0, price=price, reason=f"{self.VERSION_LABEL}_sell_target-reduce")
        market = {
            "raw_state": raw_state,
            "confirmed_state": confirmed_state,
            "risk_score": risk_score,
            "trend_risk": trend_risk,
            "drawdown_risk": drawdown_risk,
        }
        if not self._is_constructive_mixed_target_reduce(dummy_action, latest, price, market):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.CONSTRUCTIVE_MIXED_MIN_THRESHOLD),
            min(adjusted_max_sell, self.CONSTRUCTIVE_MIXED_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_constructive_mixed_soft_trim"),
        )


class V35AStrategy(V33EStrategy):
    """V3.5A: all-coin mature giveback trim + MIXED rebuy price ceiling with breakout exception."""

    VERSION_LABEL = "v3_5A"
    EARLY_TRIM_SELL_PCT = 0.06
    EARLY_TRIM_MIN_POSITION = 0.80
    EARLY_TRIM_MIN_PROFIT = 0.40
    EARLY_TRIM_MIN_PEAK_PROFIT = 0.55
    EARLY_TRIM_PEAK_PULLBACK = 0.06
    EARLY_TRIM_MIN_GAP_CALLS = 18
    EARLY_TRIM_NEW_PEAK_BUFFER = 0.05
    MIXED_REBUY_MAX_PRICE_MULT = 0.99
    MIXED_REBUY_PRICE_CEILING_CALLS = 45

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_mixed_trim_sell_price = 0.0
        self._last_mixed_trim_sell_call = -10_000

    @property
    def name(self) -> str:
        return "v3_5A"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if actions:
            action = actions[0]
            reason = str(getattr(action, "reason", ""))
            if (
                action.side == "sell"
                and "_sell_target-reduce_" in reason
                and market["raw_state"] == "MIXED"
                and market["confirmed_state"] == "MIXED"
            ):
                self._last_mixed_trim_sell_price = float(action.price)
                self._last_mixed_trim_sell_call = self._call_count
            if action.side == "sell" and "early-profit-giveback" in reason:
                self._last_profit_take_sell_price = float(action.price)
            return actions

        if not self._should_mature_giveback_trim(latest, price, pos, current_pct, market):
            return []
        sell_qty = min(total_value * self.EARLY_TRIM_SELL_PCT / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        self._last_profit_take_call = self._call_count
        self._last_profit_take_peak_price = self._peak_price
        self._last_profit_take_sell_price = price
        return [
            Action(
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=self._build_action_reason(
                    side="sell",
                    setup="early-profit-giveback",
                    risk_score=market["risk_score"],
                    trend_risk=market["trend_risk"],
                    drawdown_risk=market["drawdown_risk"],
                    raw_state=market["raw_state"],
                    confirmed_state=market["confirmed_state"],
                    target=max(0.0, current_pct - self.EARLY_TRIM_SELL_PCT),
                    guard=f"{self.VERSION_LABEL}_mature_giveback_trim",
                ),
            )
        ]

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_mixed_rebuy_context(raw_state, confirmed_state, buy_setup):
            return adjusted, guard
        if self._is_clear_breakout_rebuy_exception(latest, price, raw_state, confirmed_state, buy_setup):
            return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_exception")
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_PRICE_CEILING_CALLS:
            return adjusted, guard
        rebuy_max_price = self._last_mixed_trim_sell_price * self.MIXED_REBUY_MAX_PRICE_MULT
        if price > rebuy_max_price:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_price_ceiling")
        return adjusted, guard

    def _should_mature_giveback_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["raw_state"] != "BULL" or market["confirmed_state"] != "BULL":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.EARLY_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_profit_take_call <= self.EARLY_TRIM_MIN_GAP_CALLS:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            return False
        if not (price < ema24 and price > ema72 and ema72 > ema168):
            return False
        if self._v3_value(latest, "ema168_slope") <= 0:
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = self._peak_price / pos.avg_cost - 1.0 if self._peak_price > 0 else profit_pct
        peak_pullback = 1.0 - price / self._peak_price if self._peak_price > 0 else 0.0
        if profit_pct < self.EARLY_TRIM_MIN_PROFIT:
            return False
        if peak_profit_pct < self.EARLY_TRIM_MIN_PEAK_PROFIT:
            return False
        if peak_pullback < self.EARLY_TRIM_PEAK_PULLBACK:
            return False
        if self._last_profit_take_peak_price > 0 and self._peak_price < self._last_profit_take_peak_price * (1 + self.EARLY_TRIM_NEW_PEAK_BUFFER):
            return False

        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        atr_rank = self._v3_value(latest, "atr_pct_rank")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(rolling_pos) or pd.isna(donchian_pos) or pd.isna(atr_rank) or pd.isna(roc_20):
            return False
        if rolling_pos < 0.76 or donchian_pos < 0.68 or donchian_pos > 0.88:
            return False
        if atr_rank < 0.55 or roc_20 > -0.05:
            return False
        if self._v3_price_vs(latest, price, "ema168") < 0.12:
            return False
        return True

    @staticmethod
    def _is_mixed_rebuy_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup not in {"target-gap", "safe-recovery", "pullback"}:
            return False
        return raw_state == "MIXED" or confirmed_state == "MIXED"

    def _is_clear_breakout_rebuy_exception(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup == "trend-cont":
            return True
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            price > ema72 * 1.01
            and ema72 > ema168
            and ema168_slope > 0.01
            and donchian_pos >= 0.78
            and roc_20 >= 0.08
            and str(latest.get("btc_regime", "")) != "BEAR"
        )


class V35BStrategy(V35AStrategy):
    """V3.5B: keep all-coin giveback trim, but only price-cap ordinary MIXED chase buys."""

    VERSION_LABEL = "v3_5B"

    @property
    def name(self) -> str:
        return "v3_5B"

    @staticmethod
    def _is_mixed_rebuy_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup not in {"target-gap", "pullback"}:
            return False
        return raw_state == "MIXED" or confirmed_state == "MIXED"


class V35CStrategy(V34IStrategy):
    """V3.5C: keep accepted sell logic, but defer MIXED chase buys into pullback-first rebuy intents."""

    VERSION_LABEL = "v3_5C"
    MIXED_REBUY_TRACK_CALLS = 45
    MIXED_REBUY_INTENT_WAIT_CALLS = 12
    MIXED_REBUY_INTENT_MAX_AGE = 28
    MIXED_REBUY_PULLBACK_FROM_INTENT = 0.975
    MIXED_REBUY_PULLBACK_FROM_SELL = 0.992
    MIXED_REBUY_BREAKOUT_FROM_INTENT = 1.04
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_mixed_trim_sell_price = 0.0
        self._last_mixed_trim_sell_call = -10_000
        self._pending_mixed_rebuy_call = -10_000
        self._pending_mixed_rebuy_price = 0.0
        self._pending_mixed_rebuy_sell_price = 0.0
        self._pending_mixed_rebuy_setup = ""
        self._pending_mixed_rebuy_budget = 0.0

    @property
    def name(self) -> str:
        return "v3_5C"

    def _maybe_sell(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        sell_target: float,
        market: dict,
    ) -> list[Action]:
        actions = super()._maybe_sell(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if not actions:
            return actions
        action = actions[0]
        reason = str(getattr(action, "reason", ""))
        if (
            action.side == "sell"
            and "_sell_target-reduce_" in reason
            and market["raw_state"] == "MIXED"
            and market["confirmed_state"] == "MIXED"
        ):
            self._last_mixed_trim_sell_price = float(action.price)
            self._last_mixed_trim_sell_call = self._call_count
            self._pending_mixed_rebuy_call = -10_000
            self._pending_mixed_rebuy_price = 0.0
            self._pending_mixed_rebuy_sell_price = float(action.price)
            self._pending_mixed_rebuy_setup = ""
            self._pending_mixed_rebuy_budget = 0.0
        return actions

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        adjusted, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_mixed_intent_context(raw_state, confirmed_state, buy_setup):
            self._clear_mixed_rebuy_intent_if_stale()
            return adjusted, guard
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_TRACK_CALLS:
            self._clear_mixed_rebuy_intent()
            return adjusted, guard

        if not self._mixed_rebuy_intent_active():
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            self._pending_mixed_rebuy_sell_price = self._last_mixed_trim_sell_price
            self._pending_mixed_rebuy_setup = buy_setup
            self._pending_mixed_rebuy_budget = min(adjusted, self.MIXED_REBUY_BREAKOUT_MAX_BUY)
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_open")

        if price < self._pending_mixed_rebuy_price:
            self._pending_mixed_rebuy_price = price

        age = self._call_count - self._pending_mixed_rebuy_call
        if self._is_mixed_pullback_fill(price):
            self._clear_mixed_rebuy_intent()
            return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_fill")

        if age >= self.MIXED_REBUY_INTENT_WAIT_CALLS and self._is_mixed_breakout_fill(
            latest, price, raw_state, confirmed_state, buy_setup
        ):
            self._clear_mixed_rebuy_intent()
            return (
                min(adjusted, self.MIXED_REBUY_BREAKOUT_MAX_BUY),
                self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_fill"),
            )

        if age > self.MIXED_REBUY_INTENT_MAX_AGE:
            self._clear_mixed_rebuy_intent()
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_expired")
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_wait")

    @staticmethod
    def _is_mixed_intent_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup not in {"target-gap", "pullback"}:
            return False
        return raw_state == "MIXED" or confirmed_state == "MIXED"

    def _mixed_rebuy_intent_active(self) -> bool:
        if self._pending_mixed_rebuy_call < 0 or self._pending_mixed_rebuy_sell_price <= 0:
            return False
        return self._call_count - self._pending_mixed_rebuy_call <= self.MIXED_REBUY_INTENT_MAX_AGE

    def _clear_mixed_rebuy_intent_if_stale(self) -> None:
        if self._mixed_rebuy_intent_active():
            return
        self._clear_mixed_rebuy_intent()

    def _clear_mixed_rebuy_intent(self) -> None:
        self._pending_mixed_rebuy_call = -10_000
        self._pending_mixed_rebuy_price = 0.0
        self._pending_mixed_rebuy_sell_price = 0.0
        self._pending_mixed_rebuy_setup = ""
        self._pending_mixed_rebuy_budget = 0.0

    def _is_mixed_pullback_fill(self, price: float) -> bool:
        if self._pending_mixed_rebuy_price <= 0 or self._pending_mixed_rebuy_sell_price <= 0:
            return False
        return bool(
            price <= self._pending_mixed_rebuy_price * self.MIXED_REBUY_PULLBACK_FROM_INTENT
            or price <= self._pending_mixed_rebuy_sell_price * self.MIXED_REBUY_PULLBACK_FROM_SELL
        )

    def _is_mixed_breakout_fill(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup == "trend-cont":
            return True
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or self._pending_mixed_rebuy_price <= 0
        ):
            return False
        return bool(
            price >= self._pending_mixed_rebuy_price * self.MIXED_REBUY_BREAKOUT_FROM_INTENT
            and price > ema72 * 1.01
            and ema72 > ema168
            and ema168_slope > 0.01
            and donchian_pos >= 0.82
            and roc_20 >= 0.10
            and str(latest.get("btc_regime", "")) != "BEAR"
        )


class V35DStrategy(V35CStrategy):
    """V3.5D: BTC-only, target-gap-only MIXED pullback-first rebuy intent."""

    VERSION_LABEL = "v3_5D"
    MIXED_REBUY_TRACK_CALLS = 28
    MIXED_REBUY_INTENT_WAIT_CALLS = 8
    MIXED_REBUY_INTENT_MAX_AGE = 18
    MIXED_REBUY_PULLBACK_FROM_INTENT = 0.985
    MIXED_REBUY_PULLBACK_FROM_SELL = 0.997
    MIXED_REBUY_BREAKOUT_FROM_INTENT = 1.05
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.10

    @property
    def name(self) -> str:
        return "v3_5D"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._active_symbol_name() != "BTC/USDT":
            return V34IStrategy._adjust_buy_execution(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )
        return super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )

    @staticmethod
    def _is_mixed_intent_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        return raw_state == "MIXED" and confirmed_state == "MIXED" and buy_setup == "target-gap"

    def _active_symbol_name(self) -> str:
        target_alloc = getattr(self, "TARGET_ALLOC", {})
        if isinstance(target_alloc, dict) and len(target_alloc) == 1:
            return next(iter(target_alloc.keys()))
        return ""


class V35EStrategy(V35DStrategy):
    """V3.5E: BTC-only pullback-first rebuy with deferred budget carry-forward."""

    VERSION_LABEL = "v3_5E"
    MIXED_REBUY_PULLBACK_FROM_INTENT = 0.99
    MIXED_REBUY_PULLBACK_FROM_SELL = 0.998
    MIXED_REBUY_BREAKOUT_FROM_INTENT = 1.045
    MIXED_REBUY_PULLBACK_MAX_BUY = 0.16
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.14
    MIXED_REBUY_PULLBACK_BUDGET_SHARE = 0.80
    MIXED_REBUY_BREAKOUT_BUDGET_SHARE = 0.60

    @property
    def name(self) -> str:
        return "v3_5E"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._active_symbol_name() != "BTC/USDT":
            return V34IStrategy._adjust_buy_execution(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )

        adjusted, guard = V34IStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_mixed_intent_context(raw_state, confirmed_state, buy_setup):
            self._clear_mixed_rebuy_intent_if_stale()
            return adjusted, guard
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_TRACK_CALLS:
            self._clear_mixed_rebuy_intent()
            return adjusted, guard

        if not self._mixed_rebuy_intent_active():
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            self._pending_mixed_rebuy_sell_price = self._last_mixed_trim_sell_price
            self._pending_mixed_rebuy_setup = buy_setup
            self._pending_mixed_rebuy_budget = min(adjusted, self.MIXED_REBUY_PULLBACK_MAX_BUY)
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_open")

        if price < self._pending_mixed_rebuy_price:
            self._pending_mixed_rebuy_price = price

        age = self._call_count - self._pending_mixed_rebuy_call
        if self._is_mixed_pullback_fill(price):
            deferred = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_PULLBACK_BUDGET_SHARE
            self._clear_mixed_rebuy_intent()
            return (
                min(adjusted + deferred, self.MIXED_REBUY_PULLBACK_MAX_BUY),
                self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_fill"),
            )

        if age >= self.MIXED_REBUY_INTENT_WAIT_CALLS and self._is_mixed_breakout_fill(
            latest, price, raw_state, confirmed_state, buy_setup
        ):
            deferred = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_BREAKOUT_BUDGET_SHARE
            self._clear_mixed_rebuy_intent()
            return (
                min(adjusted + deferred, self.MIXED_REBUY_BREAKOUT_MAX_BUY),
                self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_fill"),
            )

        if age > self.MIXED_REBUY_INTENT_MAX_AGE:
            self._clear_mixed_rebuy_intent()
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_expired")
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_wait")


class V35FStrategy(V35DStrategy):
    """V3.5F: BTC-only staged MIXED rebuy intent with pullback-first partial fills."""

    VERSION_LABEL = "v3_5F"
    MIXED_REBUY_TRACK_CALLS = 28
    MIXED_REBUY_INTENT_WAIT_CALLS = 6
    MIXED_REBUY_INTENT_MAX_AGE = 20
    MIXED_REBUY_PULLBACK_FROM_INTENT = 0.992
    MIXED_REBUY_PULLBACK_FROM_SELL = 0.998
    MIXED_REBUY_BREAKOUT_FROM_INTENT = 1.04
    MIXED_REBUY_STAGE1_MAX_BUY = 0.08
    MIXED_REBUY_STAGE2_MAX_BUY = 0.08
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.06
    MIXED_REBUY_STAGE1_BUDGET_SHARE = 0.55
    MIXED_REBUY_STAGE2_BUDGET_SHARE = 0.45
    MIXED_REBUY_BREAKOUT_BUDGET_SHARE = 0.40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_mixed_rebuy_stage = 0

    @property
    def name(self) -> str:
        return "v3_5F"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._active_symbol_name() != "BTC/USDT":
            return V34IStrategy._adjust_buy_execution(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )

        adjusted, guard = V34IStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_staged_mixed_intent_context(raw_state, confirmed_state, buy_setup):
            self._clear_mixed_rebuy_intent_if_stale()
            return adjusted, guard
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_TRACK_CALLS:
            self._clear_mixed_rebuy_intent()
            return adjusted, guard

        if not self._mixed_rebuy_intent_active():
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            self._pending_mixed_rebuy_sell_price = self._last_mixed_trim_sell_price
            self._pending_mixed_rebuy_setup = buy_setup
            self._pending_mixed_rebuy_budget = min(adjusted, self.MIXED_REBUY_STAGE1_MAX_BUY + self.MIXED_REBUY_STAGE2_MAX_BUY)
            self._pending_mixed_rebuy_stage = 0
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_open")

        if price < self._pending_mixed_rebuy_price:
            self._pending_mixed_rebuy_price = price

        age = self._call_count - self._pending_mixed_rebuy_call
        if self._is_stage1_pullback_fill(latest, price):
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_STAGE1_BUDGET_SHARE
            fill_buy = min(adjusted + fill_budget, self.MIXED_REBUY_STAGE1_MAX_BUY)
            self._pending_mixed_rebuy_budget = max(0.0, self._pending_mixed_rebuy_budget - fill_buy)
            self._pending_mixed_rebuy_stage = 1
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_stage1")

        if self._pending_mixed_rebuy_stage >= 1 and self._is_stage2_pullback_fill(latest, price):
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_STAGE2_BUDGET_SHARE
            fill_buy = min(adjusted + fill_budget, self.MIXED_REBUY_STAGE2_MAX_BUY)
            self._clear_mixed_rebuy_intent()
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_stage2")

        if age >= self.MIXED_REBUY_INTENT_WAIT_CALLS and self._is_mixed_breakout_fill(
            latest, price, raw_state, confirmed_state, buy_setup
        ):
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_BREAKOUT_BUDGET_SHARE
            fill_buy = min(adjusted + fill_budget, self.MIXED_REBUY_BREAKOUT_MAX_BUY)
            self._clear_mixed_rebuy_intent()
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_fill")

        if age > self.MIXED_REBUY_INTENT_MAX_AGE:
            self._clear_mixed_rebuy_intent()
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_expired")
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_wait")

    @staticmethod
    def _is_staged_mixed_intent_context(
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        return raw_state == "MIXED" and confirmed_state == "MIXED" and buy_setup == "target-gap"

    def _is_stage1_pullback_fill(self, latest: pd.Series, price: float) -> bool:
        if self._pending_mixed_rebuy_stage != 0:
            return False
        ema24 = self._v3_value(latest, "ema24")
        if pd.isna(ema24):
            return False
        return bool(
            price <= self._pending_mixed_rebuy_price * self.MIXED_REBUY_PULLBACK_FROM_INTENT
            and price <= self._pending_mixed_rebuy_sell_price * self.MIXED_REBUY_PULLBACK_FROM_SELL
            and price <= ema24 * 1.005
        )

    def _is_stage2_pullback_fill(self, latest: pd.Series, price: float) -> bool:
        ema72 = self._v3_value(latest, "ema72")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if pd.isna(ema72) or pd.isna(donchian_pos):
            return False
        return bool(
            price <= self._pending_mixed_rebuy_price * 0.996
            and price <= ema72 * 1.01
            and donchian_pos <= 0.60
        )

    def _clear_mixed_rebuy_intent(self) -> None:
        super()._clear_mixed_rebuy_intent()
        self._pending_mixed_rebuy_stage = 0


class V35GStrategy(V35FStrategy):
    """V3.5G: delay only weak BTC MIXED repairs, keep a small starter fill, then optimize the rest."""

    VERSION_LABEL = "v3_5G"
    MIXED_REBUY_STARTER_MAX_BUY = 0.05
    MIXED_REBUY_TOTAL_BUDGET_MAX = 0.14
    MIXED_REBUY_STAGE1_MAX_BUY = 0.06
    MIXED_REBUY_STAGE2_MAX_BUY = 0.05
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.04
    MIXED_REBUY_INTENT_WAIT_CALLS = 5
    MIXED_REBUY_INTENT_MAX_AGE = 18

    @property
    def name(self) -> str:
        return "v3_5G"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._active_symbol_name() != "BTC/USDT":
            return V34IStrategy._adjust_buy_execution(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )

        adjusted, guard = V34IStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._is_staged_mixed_intent_context(raw_state, confirmed_state, buy_setup):
            self._clear_mixed_rebuy_intent_if_stale()
            return adjusted, guard
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_TRACK_CALLS:
            self._clear_mixed_rebuy_intent()
            return adjusted, guard

        # High-quality early repair should still buy normally.
        if not self._should_delay_mixed_rebuy(latest):
            self._clear_mixed_rebuy_intent()
            return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_immediate_repair")

        total_budget = min(adjusted, self.MIXED_REBUY_TOTAL_BUDGET_MAX)
        if not self._mixed_rebuy_intent_active():
            starter_buy = min(total_budget, self.MIXED_REBUY_STARTER_MAX_BUY)
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            self._pending_mixed_rebuy_sell_price = self._last_mixed_trim_sell_price
            self._pending_mixed_rebuy_setup = buy_setup
            self._pending_mixed_rebuy_budget = max(0.0, total_budget - starter_buy)
            self._pending_mixed_rebuy_stage = 0
            return starter_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_starter")

        if price < self._pending_mixed_rebuy_price:
            self._pending_mixed_rebuy_price = price

        age = self._call_count - self._pending_mixed_rebuy_call
        if self._is_stage1_pullback_fill(latest, price) and self._pending_mixed_rebuy_budget > 0:
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_STAGE1_BUDGET_SHARE
            fill_buy = min(fill_budget, self.MIXED_REBUY_STAGE1_MAX_BUY)
            self._pending_mixed_rebuy_budget = max(0.0, self._pending_mixed_rebuy_budget - fill_buy)
            self._pending_mixed_rebuy_stage = 1
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_stage1")

        if (
            self._pending_mixed_rebuy_stage >= 1
            and self._is_stage2_pullback_fill(latest, price)
            and self._pending_mixed_rebuy_budget > 0
        ):
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_STAGE2_BUDGET_SHARE
            fill_buy = min(fill_budget, self.MIXED_REBUY_STAGE2_MAX_BUY)
            self._clear_mixed_rebuy_intent()
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_pullback_stage2")

        if (
            age >= self.MIXED_REBUY_INTENT_WAIT_CALLS
            and self._pending_mixed_rebuy_budget > 0
            and self._is_mixed_breakout_fill(latest, price, raw_state, confirmed_state, buy_setup)
        ):
            fill_budget = self._pending_mixed_rebuy_budget * self.MIXED_REBUY_BREAKOUT_BUDGET_SHARE
            fill_buy = min(fill_budget, self.MIXED_REBUY_BREAKOUT_MAX_BUY)
            self._clear_mixed_rebuy_intent()
            return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_fill")

        if age > self.MIXED_REBUY_INTENT_MAX_AGE:
            self._clear_mixed_rebuy_intent()
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_expired")
        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_wait")

    def _should_delay_mixed_rebuy(self, latest: pd.Series) -> bool:
        ema24_slope = self._v3_value(latest, "ema24_slope")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        roc_20 = self._v3_value(latest, "roc_20")
        atr_rank = self._v3_value(latest, "atr_pct_rank")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        if (
            pd.isna(ema24_slope)
            or pd.isna(ema168_slope)
            or pd.isna(roc_20)
            or pd.isna(atr_rank)
            or pd.isna(donchian_pos)
            or pd.isna(rolling_pos)
        ):
            return False
        weakness_score = 0
        if ema24_slope <= 0:
            weakness_score += 1
        if ema168_slope <= 0.005:
            weakness_score += 1
        if roc_20 <= 0.02:
            weakness_score += 1
        if atr_rank >= 0.65:
            weakness_score += 1
        if donchian_pos <= 0.45:
            weakness_score += 1
        if rolling_pos >= 0.60:
            weakness_score += 1
        return weakness_score >= 3


class V35HStrategy(V35GStrategy):
    """V3.5H: weak BTC MIXED repair keeps a larger starter and refills deferred budget on later better recovery prices."""

    VERSION_LABEL = "v3_5H"
    MIXED_REBUY_TRACK_CALLS = 45
    MIXED_REBUY_INTENT_WAIT_CALLS = 4
    MIXED_REBUY_INTENT_MAX_AGE = 40
    MIXED_REBUY_STARTER_MAX_BUY = 0.10
    MIXED_REBUY_TOTAL_BUDGET_MAX = 0.16
    MIXED_REBUY_STAGE1_MAX_BUY = 0.06
    MIXED_REBUY_BREAKOUT_MAX_BUY = 0.04
    MIXED_REBUY_REFILL_PRICE_MULT = 1.002

    @property
    def name(self) -> str:
        return "v3_5H"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._active_symbol_name() != "BTC/USDT":
            return V34IStrategy._adjust_buy_execution(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )

        adjusted, guard = V34IStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )

        if self._mixed_rebuy_intent_active():
            age = self._call_count - self._pending_mixed_rebuy_call
            if buy_setup in {"safe-recovery", "target-gap"} and self._is_deferred_refill_price(price):
                fill_buy = min(self._pending_mixed_rebuy_budget, self.MIXED_REBUY_STAGE1_MAX_BUY)
                self._pending_mixed_rebuy_budget = max(0.0, self._pending_mixed_rebuy_budget - fill_buy)
                if self._pending_mixed_rebuy_budget <= 1e-12:
                    self._clear_mixed_rebuy_intent()
                else:
                    self._pending_mixed_rebuy_stage = max(self._pending_mixed_rebuy_stage, 1)
                    self._pending_mixed_rebuy_call = self._call_count
                    self._pending_mixed_rebuy_price = min(self._pending_mixed_rebuy_price, price)
                return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_refill_fill")
            if (
                age >= self.MIXED_REBUY_INTENT_WAIT_CALLS
                and self._pending_mixed_rebuy_budget > 0
                and self._is_mixed_breakout_fill(latest, price, raw_state, confirmed_state, buy_setup)
            ):
                fill_buy = min(self._pending_mixed_rebuy_budget, self.MIXED_REBUY_BREAKOUT_MAX_BUY)
                self._clear_mixed_rebuy_intent()
                return fill_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_breakout_fill")
            if age > self.MIXED_REBUY_INTENT_MAX_AGE:
                self._clear_mixed_rebuy_intent()

        if not self._is_staged_mixed_intent_context(raw_state, confirmed_state, buy_setup):
            self._clear_mixed_rebuy_intent_if_stale()
            return adjusted, guard
        if self._last_mixed_trim_sell_price <= 0:
            return adjusted, guard
        if self._call_count - self._last_mixed_trim_sell_call > self.MIXED_REBUY_TRACK_CALLS:
            self._clear_mixed_rebuy_intent()
            return adjusted, guard
        if not self._should_delay_mixed_rebuy(latest):
            self._clear_mixed_rebuy_intent()
            return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_immediate_repair")

        total_budget = min(adjusted, self.MIXED_REBUY_TOTAL_BUDGET_MAX)
        if not self._mixed_rebuy_intent_active():
            starter_buy = min(total_budget, self.MIXED_REBUY_STARTER_MAX_BUY)
            self._pending_mixed_rebuy_call = self._call_count
            self._pending_mixed_rebuy_price = price
            self._pending_mixed_rebuy_sell_price = self._last_mixed_trim_sell_price
            self._pending_mixed_rebuy_setup = buy_setup
            self._pending_mixed_rebuy_budget = max(0.0, total_budget - starter_buy)
            self._pending_mixed_rebuy_stage = 0
            return starter_buy, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_starter")

        return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_mixed_rebuy_intent_wait")

    def _is_deferred_refill_price(self, price: float) -> bool:
        if self._pending_mixed_rebuy_price <= 0:
            return False
        return bool(
            price <= self._pending_mixed_rebuy_price * self.MIXED_REBUY_REFILL_PRICE_MULT
            or (
                self._pending_mixed_rebuy_sell_price > 0
                and price <= self._pending_mixed_rebuy_sell_price * self.MIXED_REBUY_REFILL_PRICE_MULT
            )
        )


class V226AStrategy(V221EStrategy):
    """V2.26A: external recovery score cap layered over v2.21E buy targets."""

    VERSION_LABEL = "v2_26A"
    RECOVERY_EXTERNAL_BASE_CAP = 0.40
    RECOVERY_EXTERNAL_MID_CAP = 0.50
    RECOVERY_EXTERNAL_HIGH_CAP = 0.65

    @property
    def name(self) -> str:
        return "v2_26A"

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
        target = super()._compose_target(
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )
        if side != "buy" or raw_state != "MIXED" or price <= 0:
            return target
        if target <= self.RECOVERY_EXTERNAL_BASE_CAP:
            return target
        cap = self._external_recovery_cap(latest, price, trend_risk, drawdown_risk)
        return min(target, cap)

    def _external_recovery_cap(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
    ) -> float:
        cap = self.RECOVERY_EXTERNAL_BASE_CAP
        if trend_risk > 1 or drawdown_risk > 1:
            return cap
        if str(latest.get("btc_regime", "")) == "BEAR":
            return cap

        macro_repair = self._macro_repair(latest)
        donchian_pos = self._value(latest, "donchian_pos")
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        if (
            macro_repair
            and donchian_pos >= 0.54
            and volume_strength <= 1.00
            and not self._is_severe_extension(latest, price)
        ):
            cap = self.RECOVERY_EXTERNAL_MID_CAP

        if (
            trend_risk == 0
            and drawdown_risk == 0
            and self._recovery_overlay_score(latest, price) >= 7
            and self._ratio(latest, "ema72", "ema168") >= -0.03
            and donchian_pos >= 0.54
            and not self._is_extension_risk(latest, price)
        ):
            cap = self.RECOVERY_EXTERNAL_HIGH_CAP
        return cap

    def _recovery_overlay_score(self, latest: pd.Series, price: float) -> int:
        score = 0
        if self._ratio(latest, "ema72", "ema168") >= -0.03:
            score += 2
        if self._ratio(latest, "ema24", "ema168") >= -0.02:
            score += 1
        if self._macro_repair(latest):
            score += 1
        if self._value(latest, "donchian_pos") >= 0.54:
            score += 2
        if self._value(latest, "volume_strength", default=1.0) <= 0.75:
            score += 1
        rolling_pos = self._value(latest, "rolling_365d_pos")
        if 0.20 <= rolling_pos <= 0.75:
            score += 1
        if self._value(latest, "roc_20") < 0.20:
            score += 1
        if self._value(latest, "price_vs_ema168", default=self._price_vs(latest, price, "ema168")) >= -0.03:
            score += 1
        return score

    def _macro_repair(self, latest: pd.Series) -> bool:
        return (
            self._value(latest, "btc_price_vs_ema72") > 0
            or self._value(latest, "btc_ema24_slope") > 0
        )

    def _is_severe_extension(self, latest: pd.Series, price: float) -> bool:
        roc_20 = self._value(latest, "roc_20")
        price_vs_ema168 = self._value(
            latest,
            "price_vs_ema168",
            default=self._price_vs(latest, price, "ema168"),
        )
        donchian_pos = self._value(latest, "donchian_pos")
        rolling_pos = self._value(latest, "rolling_365d_pos")
        return bool(
            (roc_20 >= 0.30 and price_vs_ema168 >= 0.18 and donchian_pos >= 0.85)
            or (rolling_pos >= 0.80 and donchian_pos >= 0.90 and price_vs_ema168 >= 0.18)
        )

    def _is_extension_risk(self, latest: pd.Series, price: float) -> bool:
        roc_20 = self._value(latest, "roc_20")
        price_vs_ema168 = self._value(
            latest,
            "price_vs_ema168",
            default=self._price_vs(latest, price, "ema168"),
        )
        donchian_pos = self._value(latest, "donchian_pos")
        rolling_pos = self._value(latest, "rolling_365d_pos")
        return bool(
            (roc_20 >= 0.20 and price_vs_ema168 >= 0.12 and donchian_pos >= 0.75)
            or (rolling_pos >= 0.64 and donchian_pos >= 0.80 and price_vs_ema168 >= 0.12)
        )

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _ratio(cls, latest: pd.Series, numerator: str, denominator: str) -> float:
        num = cls._value(latest, numerator)
        den = cls._value(latest, denominator)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return num / den - 1.0

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class V226BStrategy(V226AStrategy):
    """V2.26B: external recovery overlay caps buy size, not strategy target."""

    VERSION_LABEL = "v2_26B"
    RECOVERY_EXTERNAL_BASE_CAP = 0.25
    RECOVERY_EXTERNAL_MID_CAP = 0.40
    RECOVERY_EXTERNAL_HIGH_CAP = 0.65
    RECOVERY_TARGET_GAP_MAX_CURRENT = 0.35
    RECOVERY_CAP_EPSILON = 0.001

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_current_pct = 0.0

    @property
    def name(self) -> str:
        return "v2_26B"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if symbol is not None:
            price = current_prices.get(symbol, 0.0)
            pos = portfolio.positions.get(symbol, PositionState())
            position_value = pos.quantity * price if price > 0 else 0.0
            total_value = portfolio.cash + position_value
            if total_value > 0:
                self._overlay_current_pct = position_value / total_value
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

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
        return V221EStrategy._compose_target(
            self,
            symbol=symbol,
            tactical_target=tactical_target,
            raw_state=raw_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            latest=latest,
            price=price,
            side=side,
        )

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        max_buy, guard = super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if not self._should_apply_external_recovery_overlay(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            return max_buy, guard

        current_pct = self._overlay_current_pct
        # Risk scores are not available at this hook; the overlay is deliberately
        # structural here and only caps buy size.
        cap = self._external_recovery_cap(
            latest=latest,
            price=price,
            trend_risk=0,
            drawdown_risk=0,
        )
        allowed_buy = max(0.0, cap - current_pct)
        if allowed_buy + self.RECOVERY_CAP_EPSILON < max_buy:
            max_buy = max(0.0, allowed_buy)
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_external_cap_t{cap:.0%}")
        return max_buy, guard

    def _external_recovery_cap(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
    ) -> float:
        cap = self.RECOVERY_EXTERNAL_BASE_CAP
        if str(latest.get("btc_regime", "")) == "BEAR":
            return cap

        macro_repair = self._macro_repair(latest)
        donchian_pos = self._value(latest, "donchian_pos")
        volume_strength = self._value(latest, "volume_strength", default=1.0)
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        if (
            macro_repair
            and donchian_pos >= 0.54
            and ema72_vs_ema168 >= -0.03
            and volume_strength <= 1.00
            and not self._is_severe_extension(latest, price)
        ):
            cap = self.RECOVERY_EXTERNAL_MID_CAP

        if (
            self._recovery_overlay_score(latest, price) >= 7
            and ema72_vs_ema168 >= -0.03
            and donchian_pos >= 0.54
            and not self._is_extension_risk(latest, price)
        ):
            cap = self.RECOVERY_EXTERNAL_HIGH_CAP
        return cap

    def _should_apply_external_recovery_overlay(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if buy_setup == "safe-recovery":
            return True
        if buy_setup == "target-gap" and self._overlay_current_pct < self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            return True
        return False


class V226CStrategy(V226BStrategy):
    """V2.26C: cap safe-recovery, but only cap weak low-position target-gap buys."""

    VERSION_LABEL = "v2_26C"
    WEAK_TARGET_GAP_MIN_FLAGS = 3

    @property
    def name(self) -> str:
        return "v2_26C"

    def _should_apply_external_recovery_overlay(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if buy_setup == "safe-recovery":
            return True
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            return False
        return self._weak_recovery_structure_score(latest, price) >= self.WEAK_TARGET_GAP_MIN_FLAGS

    def _weak_recovery_structure_score(self, latest: pd.Series, price: float) -> int:
        score = 0
        if self._ratio(latest, "ema72", "ema168") < -0.03:
            score += 1
        if self._value(latest, "ema168_slope") < 0:
            score += 1
        if self._value(latest, "donchian_pos") < 0.54:
            score += 1
        if self._value(latest, "roc_20") < 0.08:
            score += 1
        if self._value(latest, "dd_from_120d_high") > 0.35:
            score += 1
        return score


class V226DStrategy(V226CStrategy):
    """V2.26D: only cap structurally weak safe-recovery buys."""

    VERSION_LABEL = "v2_26D"

    @property
    def name(self) -> str:
        return "v2_26D"

    def _should_apply_external_recovery_overlay(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if buy_setup != "safe-recovery":
            return False
        return self._weak_recovery_structure_score(latest, price) >= self.WEAK_TARGET_GAP_MIN_FLAGS


class V226EStrategy(V226BStrategy):
    """V2.26E: cap only low-position target-gap weak-range recovery buys."""

    VERSION_LABEL = "v2_26E"
    RECOVERY_EXTERNAL_BASE_CAP = 0.30

    @property
    def name(self) -> str:
        return "v2_26E"

    def _external_recovery_cap(
        self,
        latest: pd.Series,
        price: float,
        trend_risk: int,
        drawdown_risk: int,
    ) -> float:
        return self.RECOVERY_EXTERNAL_BASE_CAP

    def _should_apply_external_recovery_overlay(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.RECOVERY_TARGET_GAP_MAX_CURRENT:
            return False
        return self._is_low_quality_target_gap_context(latest)

    def _is_low_quality_target_gap_context(self, latest: pd.Series) -> bool:
        mid_history_weak_range = (
            self._value(latest, "rolling_365d_pos") >= 0.3198
            and self._value(latest, "donchian_pos") < 0.54
        )
        deep_ema_weak_low_volume = (
            self._ratio(latest, "ema72", "ema168") <= -0.05
            and self._value(latest, "volume_strength", default=1.0) <= 0.8709
        )
        return bool(mid_history_weak_range or deep_ema_weak_low_volume)


class V226FStrategy(V226EStrategy):
    """V2.26F: cap only BTC-hot, local-EMA-weak low-position target-gap buys."""

    VERSION_LABEL = "v2_26F"

    @property
    def name(self) -> str:
        return "v2_26F"

    def _is_low_quality_target_gap_context(self, latest: pd.Series) -> bool:
        return bool(
            self._value(latest, "btc_price_vs_ema72") >= 0.096
            and self._ratio(latest, "ema72", "ema168") <= -0.05
        )


class V227AStrategy(V221EStrategy):
    """V2.27A: relax tiny low-position MIXED target-gap buys."""

    VERSION_LABEL = "v2_27A"
    RELAXED_TINY_MAX_CURRENT = 0.35
    RELAXED_TINY_MIN_TARGET_GAP = 0.05
    RELAXED_TINY_MIN_BUY = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_current_pct = 0.0

    @property
    def name(self) -> str:
        return "v2_27A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if symbol is not None:
            price = current_prices.get(symbol, 0.0)
            pos = portfolio.positions.get(symbol, PositionState())
            position_value = pos.quantity * price if price > 0 else 0.0
            total_value = portfolio.cash + position_value
            if total_value > 0:
                self._overlay_current_pct = position_value / total_value
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if not self._is_relaxed_tiny_target_gap_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        ):
            return super()._adjust_buy_execution(
                latest=latest,
                price=price,
                raw_state=raw_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
                confirmed_state=confirmed_state,
            )

        adjusted, guard = V212AStrategy._adjust_buy_execution(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )
        if 0.0 < adjusted < self.RELAXED_TINY_MIN_BUY:
            return 0.0, self._join_guard(guard, f"{self.VERSION_LABEL}_relaxed_tiny_floor_skipped")
        return adjusted, self._join_guard(guard, f"{self.VERSION_LABEL}_relaxed_tiny_target_gap")

    def _is_relaxed_tiny_target_gap_context(
        self,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.RELAXED_TINY_MAX_CURRENT:
            return False
        return max_buy >= self.RELAXED_TINY_MIN_TARGET_GAP


class V227BStrategy(V227AStrategy):
    """V2.27B: relaxed tiny target-gap buys with a narrow external deny gate."""

    VERSION_LABEL = "v2_27B"

    @property
    def name(self) -> str:
        return "v2_27B"

    def _adjust_buy_execution(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        buy_setup: str,
        max_buy: float,
        confirmed_state: str | None = None,
    ) -> tuple[float, str]:
        if self._is_relaxed_tiny_target_gap_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        ) and self._external_deny_target_gap(latest):
            return 0.0, f"{self.VERSION_LABEL}_external_deny_f"
        return super()._adjust_buy_execution(
            latest=latest,
            price=price,
            raw_state=raw_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
            confirmed_state=confirmed_state,
        )

    def _external_deny_target_gap(self, latest: pd.Series) -> bool:
        return bool(
            V226FStrategy._value(latest, "btc_price_vs_ema72") >= 0.096
            and V226FStrategy._ratio(latest, "ema72", "ema168") <= -0.05
        )


class V227CStrategy(V227AStrategy):
    """V2.27C: relax tiny target-gap buys only in high-quality MIXED recovery."""

    VERSION_LABEL = "v2_27C"

    @property
    def name(self) -> str:
        return "v2_27C"

    def _is_relaxed_tiny_target_gap_context(
        self,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> bool:
        if not super()._is_relaxed_tiny_target_gap_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        ):
            return False
        latest = getattr(self, "_latest_bar", None)
        price = getattr(self, "_current_price", 0.0)
        if latest is None or price <= 0:
            return False
        return self._is_high_quality_target_gap_context(latest, price)

    def _is_high_quality_target_gap_context(self, latest: pd.Series, price: float) -> bool:
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        if V226FStrategy._value(latest, "btc_price_vs_ema72") < -0.02:
            return False
        if price <= V226FStrategy._value(latest, "ema24"):
            return False
        if V226FStrategy._ratio(latest, "ema24", "ema72") <= 0:
            return False
        if V226FStrategy._ratio(latest, "ema72", "ema168") < -0.10:
            return False
        if V226FStrategy._value(latest, "donchian_pos") < 0.54:
            return False
        if V226FStrategy._value(latest, "roc_20") < 0.08:
            return False
        if V226FStrategy._value(latest, "volume_strength", default=1.0) > 1.20:
            return False
        return True


class V227DStrategy(V227CStrategy):
    """V2.27D: ultra-conservative tiny relaxation after long trend turns up."""

    VERSION_LABEL = "v2_27D"

    @property
    def name(self) -> str:
        return "v2_27D"

    def _is_relaxed_tiny_target_gap_context(
        self,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> bool:
        if self._is_post_override_target_gap(buy_setup):
            return False
        return super()._is_relaxed_tiny_target_gap_context(
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )

    def _is_high_quality_target_gap_context(self, latest: pd.Series, price: float) -> bool:
        if not super()._is_high_quality_target_gap_context(latest, price):
            return False
        return V226FStrategy._value(latest, "ema168_slope") >= 0
