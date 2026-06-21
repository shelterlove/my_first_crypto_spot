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


class V36AStrategy(V34IStrategy):
    """V3.6A: BNB weak-MIXED safe-recovery starter with native deferred fills."""

    VERSION_LABEL = "v3_6A"
    DEFER_SYMBOL = "BNB/USDT"
    DEFER_STARTER_MAX_BUY = 0.06
    DEFER_TOTAL_BUDGET_MAX = 0.16
    DEFER_PULLBACK_MAX_BUY = 0.07
    DEFER_BREAKOUT_MAX_BUY = 0.04
    DEFER_MAX_AGE_CALLS = 45
    DEFER_WAIT_CALLS = 4
    DEFER_PULLBACK_FROM_INTENT = 0.985
    DEFER_PULLBACK_FROM_SELL = 0.995
    DEFER_BREAKOUT_FROM_INTENT = 1.055

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deferred_mixed_rebuy_call = -10_000
        self._deferred_mixed_rebuy_price = 0.0
        self._deferred_mixed_rebuy_sell_price = 0.0
        self._deferred_mixed_rebuy_budget = 0.0

    @property
    def name(self) -> str:
        return "v3_6A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        symbol = self._active_symbol_name()
        if symbol != self.DEFER_SYMBOL:
            return actions

        position = self._defer_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None:
            self._clear_deferred_mixed_rebuy()
            return actions

        if actions:
            action = actions[0]
            if action.side == "sell":
                self._deferred_mixed_rebuy_sell_price = float(action.price)
                if "_sell_trend-break_" in action.reason or "_sell_risk-reduce_" in action.reason:
                    self._clear_deferred_mixed_rebuy()
                return actions
            adjusted = self._maybe_convert_weak_mixed_recovery_to_starter(action, position)
            return [adjusted] if adjusted is not None else actions

        fill = self._maybe_deferred_mixed_rebuy_fill(position)
        return [fill] if fill is not None else []

    def _maybe_convert_weak_mixed_recovery_to_starter(
        self,
        action: Action,
        position: dict,
    ) -> Action | None:
        parsed = self._parse_action_context(action)
        if parsed["side"] != "buy" or parsed["setup"] != "safe-recovery":
            return None
        if parsed["raw_state"] != "MIXED" and parsed["confirmed_state"] != "MIXED":
            self._clear_deferred_mixed_rebuy_if_stale()
            return None
        latest = position["latest"]
        price = position["price"]
        if not self._is_weak_bnb_mixed_recovery(latest, price, parsed):
            self._clear_deferred_mixed_rebuy()
            return None

        total_value = position["total_value"]
        original_buy_pct = action.quantity * action.price / total_value if total_value > 0 else 0.0
        total_budget = min(original_buy_pct, self.DEFER_TOTAL_BUDGET_MAX)
        starter_buy = min(total_budget, self.DEFER_STARTER_MAX_BUY)
        if starter_buy <= 1e-12:
            return None

        self._deferred_mixed_rebuy_call = self._call_count
        self._deferred_mixed_rebuy_price = price
        self._deferred_mixed_rebuy_budget = max(0.0, total_budget - starter_buy)
        starter_qty = total_value * starter_buy / price
        return Action(
            symbol=action.symbol,
            side="buy",
            quantity=starter_qty,
            price=action.price,
            reason=self._join_guard(action.reason, f"{self.VERSION_LABEL}_weak_mixed_starter"),
            order_type=action.order_type,
        )

    def _maybe_deferred_mixed_rebuy_fill(self, position: dict) -> Action | None:
        if not self._deferred_mixed_rebuy_active():
            self._clear_deferred_mixed_rebuy()
            return None
        if self._deferred_mixed_rebuy_budget <= 1e-12:
            self._clear_deferred_mixed_rebuy()
            return None

        latest = position["latest"]
        price = position["price"]
        age = self._call_count - self._deferred_mixed_rebuy_call
        raw_state = self._detect_market_state(latest)
        confirmed_state = self._current_state
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = self._calculate_drawdown_risk(latest, position["pos"], price)
        risk_score = min(trend_risk + drawdown_risk, 5)
        if trend_risk >= 3 or raw_state == "BEAR":
            self._clear_deferred_mixed_rebuy()
            return None

        fill_setup = ""
        max_fill = 0.0
        if self._is_deferred_pullback_fill(latest, price):
            fill_setup = "safe-recovery"
            max_fill = self.DEFER_PULLBACK_MAX_BUY
        elif age >= self.DEFER_WAIT_CALLS and self._is_deferred_breakout_fill(latest, price, raw_state, confirmed_state):
            fill_setup = "safe-recovery"
            max_fill = self.DEFER_BREAKOUT_MAX_BUY
        elif age > self.DEFER_MAX_AGE_CALLS:
            self._clear_deferred_mixed_rebuy()
            return None

        if not fill_setup:
            return None

        fill_buy = min(self._deferred_mixed_rebuy_budget, max_fill)
        if fill_buy <= 1e-12:
            self._clear_deferred_mixed_rebuy()
            return None
        self._deferred_mixed_rebuy_budget = max(0.0, self._deferred_mixed_rebuy_budget - fill_buy)
        if self._deferred_mixed_rebuy_budget <= 1e-12:
            self._clear_deferred_mixed_rebuy()
        else:
            self._deferred_mixed_rebuy_call = self._call_count
            self._deferred_mixed_rebuy_price = min(self._deferred_mixed_rebuy_price, price)

        qty = position["total_value"] * fill_buy / price
        self._last_buy_call = self._call_count
        self._last_recovery_buy_call = self._call_count
        return Action(
            symbol=position["symbol"],
            side="buy",
            quantity=qty,
            price=price,
            reason=self._build_action_reason(
                side="buy",
                setup=fill_setup,
                risk_score=risk_score,
                trend_risk=trend_risk,
                drawdown_risk=drawdown_risk,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                target=position["current_pct"] + fill_buy,
                guard=f"{self.VERSION_LABEL}_deferred_mixed_fill",
            ),
        )

    def _is_weak_bnb_mixed_recovery(self, latest: pd.Series, price: float, parsed: dict) -> bool:
        if parsed["trend_risk"] >= 3:
            return False
        if parsed["drawdown_risk"] >= 2:
            return True
        weakness = 0
        if price < self._v3_value(latest, "ema72"):
            weakness += 1
        if self._v3_value(latest, "ema24_slope") <= 0:
            weakness += 1
        if self._v3_value(latest, "roc_20") <= 0:
            weakness += 1
        if self._v3_value(latest, "donchian_pos") <= 0.52:
            weakness += 1
        if str(latest.get("btc_regime", "")) == "BEAR":
            weakness += 1
        return weakness >= 3

    def _is_deferred_pullback_fill(self, latest: pd.Series, price: float) -> bool:
        if self._deferred_mixed_rebuy_price <= 0:
            return False
        ema72 = self._v3_value(latest, "ema72")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if pd.isna(ema72) or pd.isna(donchian_pos):
            return False
        return bool(
            (
                price <= self._deferred_mixed_rebuy_price * self.DEFER_PULLBACK_FROM_INTENT
                or (
                    self._deferred_mixed_rebuy_sell_price > 0
                    and price <= self._deferred_mixed_rebuy_sell_price * self.DEFER_PULLBACK_FROM_SELL
                )
            )
            and price <= ema72 * 1.02
            and donchian_pos <= 0.58
        )

    def _is_deferred_breakout_fill(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        if self._deferred_mixed_rebuy_price <= 0:
            return False
        if raw_state != "BULL" and confirmed_state != "BULL":
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False
        return bool(
            price >= self._deferred_mixed_rebuy_price * self.DEFER_BREAKOUT_FROM_INTENT
            and price > self._v3_value(latest, "ema72") * 1.01
            and self._v3_value(latest, "ema24_slope") > 0
            and self._v3_value(latest, "donchian_pos") >= 0.62
            and self._v3_value(latest, "roc_20") >= 0.04
        )

    def _defer_position_context(self, candles_by_symbol, portfolio, current_prices) -> dict | None:
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
        pos = portfolio.positions.get(symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        current_pct = position_value / total_value if total_value > 0 else 0.0
        return {
            "symbol": symbol,
            "latest": latest,
            "price": price,
            "pos": pos,
            "current_pct": current_pct,
            "total_value": total_value,
        }

    def _active_symbol_name(self) -> str:
        target_alloc = getattr(self, "TARGET_ALLOC", {})
        if isinstance(target_alloc, dict) and len(target_alloc) == 1:
            return next(iter(target_alloc.keys()))
        return ""

    def _deferred_mixed_rebuy_active(self) -> bool:
        return (
            self._deferred_mixed_rebuy_call > 0
            and self._call_count - self._deferred_mixed_rebuy_call <= self.DEFER_MAX_AGE_CALLS
        )

    def _clear_deferred_mixed_rebuy_if_stale(self) -> None:
        if not self._deferred_mixed_rebuy_active():
            self._clear_deferred_mixed_rebuy()

    def _clear_deferred_mixed_rebuy(self) -> None:
        self._deferred_mixed_rebuy_call = -10_000
        self._deferred_mixed_rebuy_price = 0.0
        self._deferred_mixed_rebuy_budget = 0.0

    @staticmethod
    def _parse_action_context(action: Action) -> dict:
        from .decision import parse_action_reason

        return parse_action_reason(getattr(action, "reason", ""))


class V36BStrategy(V36AStrategy):
    """V3.6B: let the BNB weak-MIXED pending state own later safe-recovery fills."""

    VERSION_LABEL = "v3_6B"
    DEFER_BREAKOUT_MAX_BUY = 0.0

    @property
    def name(self) -> str:
        return "v3_6B"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = V34IStrategy.compute_actions(self, candles_by_symbol, portfolio, current_prices)
        symbol = self._active_symbol_name()
        if symbol != self.DEFER_SYMBOL:
            return actions

        position = self._defer_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None:
            self._clear_deferred_mixed_rebuy()
            return actions

        if actions:
            action = actions[0]
            if action.side == "sell":
                self._deferred_mixed_rebuy_sell_price = float(action.price)
                if "_sell_trend-break_" in action.reason or "_sell_risk-reduce_" in action.reason:
                    self._clear_deferred_mixed_rebuy()
                return actions

            parsed = self._parse_action_context(action)
            if self._deferred_mixed_rebuy_active() and self._is_deferred_owned_recovery_buy(parsed):
                fill = self._maybe_deferred_mixed_rebuy_fill(position)
                return [fill] if fill is not None else []

            adjusted = self._maybe_convert_weak_mixed_recovery_to_starter(action, position)
            return [adjusted] if adjusted is not None else actions

        fill = self._maybe_deferred_mixed_rebuy_fill(position)
        return [fill] if fill is not None else []

    @staticmethod
    def _is_deferred_owned_recovery_buy(parsed: dict) -> bool:
        return (
            parsed["side"] == "buy"
            and parsed["setup"] == "safe-recovery"
            and (parsed["raw_state"] == "MIXED" or parsed["confirmed_state"] == "MIXED")
        )

    def _is_deferred_breakout_fill(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        return False


class V36CStrategy(V34IStrategy):
    """V3.6C: mean-reversion overlay that softens panic-end target-reduce sells."""

    VERSION_LABEL = "v3_6C"
    MR_TARGET_REDUCE_THRESHOLD = 0.12
    MR_TARGET_REDUCE_MAX_SELL = 0.08

    @property
    def name(self) -> str:
        return "v3_6C"

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
        if not self._is_panic_reversion_target_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.MR_TARGET_REDUCE_THRESHOLD),
            min(adjusted_max_sell, self.MR_TARGET_REDUCE_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_panic_reversion_soft_sell"),
        )

    def _is_panic_reversion_target_reduce(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk != 2 or drawdown_risk != 0 or risk_score != 2:
            return False

        ema24 = self._v3_value(latest, "ema24")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(ema24)
            or pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        if not (price < ema24 < ema72):
            return False
        if price > ema168 * 1.02:
            return False
        if price / ema72 - 1.0 > -0.12:
            return False
        if donchian_pos > 0.55:
            return False
        if roc_20 > -0.08:
            return False
        return True


class V37AStrategy(V34IStrategy):
    """V3.7A: behavior-equivalent timing-gate scaffold over V3.4I."""

    VERSION_LABEL = "v3_7A"

    @property
    def name(self) -> str:
        return "v3_7A"

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
        gate = self._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=adjusted,
        )
        return self._apply_timing_gate_to_buy(adjusted, guard, gate)

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
        gate = self._evaluate_sell_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
            sell_threshold=threshold,
            max_sell=adjusted_max_sell,
        )
        return self._apply_timing_gate_to_sell(threshold, adjusted_max_sell, guard, gate)

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        return {
            "decision": "allow",
            "max_pct_mult": 1.0,
            "max_pct_cap": None,
            "guard": "",
        }

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        return {
            "decision": "allow",
            "threshold": sell_threshold,
            "max_pct_mult": 1.0,
            "max_pct_cap": None,
            "guard": "",
        }

    def _apply_timing_gate_to_buy(
        self,
        max_buy: float,
        guard: str,
        gate: dict,
    ) -> tuple[float, str]:
        decision = str(gate.get("decision", "allow"))
        if decision in {"block", "defer"}:
            return 0.0, self._join_guard(guard, str(gate.get("guard", "")))
        adjusted = max_buy * float(gate.get("max_pct_mult", 1.0))
        cap = gate.get("max_pct_cap")
        if cap is not None:
            adjusted = min(adjusted, float(cap))
        return adjusted, self._join_guard(guard, str(gate.get("guard", "")))

    def _apply_timing_gate_to_sell(
        self,
        sell_threshold: float,
        max_sell: float,
        guard: str,
        gate: dict,
    ) -> tuple[float, float, str]:
        decision = str(gate.get("decision", "allow"))
        adjusted_threshold = float(gate.get("threshold", sell_threshold))
        if decision in {"block", "defer", "freeze"}:
            return 1.0, 0.0, self._join_guard(guard, str(gate.get("guard", "")))
        adjusted_max_sell = max_sell * float(gate.get("max_pct_mult", 1.0))
        cap = gate.get("max_pct_cap")
        if cap is not None:
            adjusted_max_sell = min(adjusted_max_sell, float(cap))
        return adjusted_threshold, adjusted_max_sell, self._join_guard(
            guard,
            str(gate.get("guard", "")),
        )


class V37BStrategy(V37AStrategy):
    """V3.7B: generic buy timing gate that reduces extended chase buys to starters."""

    VERSION_LABEL = "v3_7B"
    EXTENDED_CHASE_STARTER_CAP = 0.06

    @property
    def name(self) -> str:
        return "v3_7B"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return gate
        if not self._is_extended_chase_buy(latest, price, raw_state, confirmed_state):
            return gate
        return {
            "decision": "starter",
            "max_pct_mult": 1.0,
            "max_pct_cap": self.EXTENDED_CHASE_STARTER_CAP,
            "guard": f"{self.VERSION_LABEL}_extended_chase_starter",
        }

    def _is_extended_chase_buy(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state == "BULL" and confirmed_state == "BULL":
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        atr_rank = self._v3_value(latest, "atr_pct_rank", default=0.0)
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            price_vs_ema72 >= 0.14
            and price_vs_ema168 >= 0.07
            and donchian_pos >= 0.85
            and roc_20 >= 0.18
            and atr_rank >= 0.20
        )


class V37CStrategy(V37AStrategy):
    """V3.7C: generic sell timing gate that freezes low routine target-reduce sells."""

    VERSION_LABEL = "v3_7C"

    @property
    def name(self) -> str:
        return "v3_7C"

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if not self._is_low_reversion_freeze_sell(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return gate
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": f"{self.VERSION_LABEL}_low_reversion_freeze",
        }

    def _is_low_reversion_freeze_sell(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk != 2 or drawdown_risk != 0 or risk_score != 2:
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        roc_5 = self._v3_value(latest, "roc_5")
        roc_10 = self._v3_value(latest, "roc_10")
        roc_20 = self._v3_value(latest, "roc_20")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(roc_5)
            or pd.isna(roc_10)
            or pd.isna(roc_20)
            or pd.isna(donchian_pos)
        ):
            return False
        deeply_low = price_vs_ema72 <= -0.18 and price_vs_ema168 <= -0.14
        short_rebound = roc_5 >= 0.02 or roc_10 >= 0.04
        panic_context = roc_20 <= -0.18 and donchian_pos <= 0.55
        return bool(deeply_low and short_rebound and panic_context)


class V37DStrategy(V37BStrategy, V37CStrategy):
    """V3.7D: combine generic buy and sell timing gates."""

    VERSION_LABEL = "v3_7D"

    @property
    def name(self) -> str:
        return "v3_7D"


class V37EStrategy(V37BStrategy):
    """V3.7E: buy starter gate plus stateful low-sell freeze."""

    VERSION_LABEL = "v3_7E"
    LOW_SELL_FREEZE_MAX_CALLS = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._low_sell_freeze_call = -10_000
        self._low_sell_freeze_price = 0.0
        self._low_sell_freeze_low = 0.0

    @property
    def name(self) -> str:
        return "v3_7E"

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = V37AStrategy._evaluate_sell_timing_gate(
            self,
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
        if sell_setup != "target-reduce":
            if sell_setup in {"risk-reduce", "core-override_trend-break"}:
                self._clear_low_sell_freeze()
            return gate

        if self._low_sell_freeze_active():
            self._low_sell_freeze_low = min(self._low_sell_freeze_low or price, price)
            if self._should_cancel_low_sell_freeze(latest, price, raw_state, confirmed_state):
                self._clear_low_sell_freeze()
                return self._freeze_gate(sell_threshold, "low_sell_recovered_cancel")
            if self._should_expire_low_sell_freeze(latest, price):
                self._clear_low_sell_freeze()
                return gate
            return self._freeze_gate(sell_threshold, "low_sell_freeze_wait")

        if not self._is_low_sell_freeze_context(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
        ):
            return gate

        self._low_sell_freeze_call = self._call_count
        self._low_sell_freeze_price = price
        self._low_sell_freeze_low = price
        return self._freeze_gate(sell_threshold, "low_sell_freeze_open")

    def _is_low_sell_freeze_context(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk != 2 or drawdown_risk != 0 or risk_score != 2:
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        roc_20 = self._v3_value(latest, "roc_20")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(roc_20)
            or pd.isna(donchian_pos)
        ):
            return False
        return bool(
            price_vs_ema72 <= -0.18
            and price_vs_ema168 <= -0.14
            and roc_20 <= -0.18
            and donchian_pos <= 0.55
        )

    def _should_cancel_low_sell_freeze(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        roc_20 = self._v3_value(latest, "roc_20")
        if raw_state == "BULL" or confirmed_state == "BULL":
            return True
        if not pd.isna(price_vs_ema72) and price_vs_ema72 >= -0.06:
            return True
        if not pd.isna(roc_20) and roc_20 >= -0.04:
            return True
        if self._low_sell_freeze_price > 0 and price >= self._low_sell_freeze_price * 1.16:
            return True
        return False

    def _should_expire_low_sell_freeze(self, latest: pd.Series, price: float) -> bool:
        age = self._call_count - self._low_sell_freeze_call
        if age <= self.LOW_SELL_FREEZE_MAX_CALLS:
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        roc_20 = self._v3_value(latest, "roc_20")
        return bool(
            (pd.isna(price_vs_ema72) or price_vs_ema72 < -0.12)
            and (pd.isna(roc_20) or roc_20 < -0.10)
        )

    def _low_sell_freeze_active(self) -> bool:
        return self._low_sell_freeze_call > 0

    def _clear_low_sell_freeze(self) -> None:
        self._low_sell_freeze_call = -10_000
        self._low_sell_freeze_price = 0.0
        self._low_sell_freeze_low = 0.0

    def _freeze_gate(self, sell_threshold: float, label: str) -> dict:
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": f"{self.VERSION_LABEL}_{label}",
        }


class V37FStrategy(V37EStrategy):
    """V3.7F: stricter low-sell recovery plus staged extended-chase starters."""

    VERSION_LABEL = "v3_7F"
    EXTENDED_CHASE_PENDING_CALLS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extended_chase_call = -10_000
        self._extended_chase_price = 0.0

    @property
    def name(self) -> str:
        return "v3_7F"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if self._extended_chase_pending_active() and buy_setup in {"target-gap", "safe-recovery"}:
            if self._should_release_extended_chase_pending(latest, price, raw_state, confirmed_state):
                self._clear_extended_chase_pending()
            elif raw_state != "BULL" or confirmed_state != "BULL":
                return {
                    "decision": "starter",
                    "max_pct_mult": 1.0,
                    "max_pct_cap": self.EXTENDED_CHASE_STARTER_CAP,
                    "guard": f"{self.VERSION_LABEL}_extended_chase_pending_starter",
                }

        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
        return gate

    def _should_cancel_low_sell_freeze(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
    ) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        roc_20 = self._v3_value(latest, "roc_20")
        if raw_state == "BULL" or confirmed_state == "BULL":
            return True
        if (
            not pd.isna(price_vs_ema72)
            and not pd.isna(roc_20)
            and price_vs_ema72 >= 0.0
            and roc_20 >= 0.02
        ):
            return True
        if self._low_sell_freeze_price > 0 and price >= self._low_sell_freeze_price * 1.25:
            return True
        return False

    def _extended_chase_pending_active(self) -> bool:
        return (
            self._extended_chase_call > 0
            and self._call_count - self._extended_chase_call <= self.EXTENDED_CHASE_PENDING_CALLS
        )

    def _should_release_extended_chase_pending(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state == "BULL" and confirmed_state == "BULL":
            return True
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if (
            not pd.isna(price_vs_ema72)
            and not pd.isna(donchian_pos)
            and price_vs_ema72 <= 0.08
            and donchian_pos <= 0.68
        ):
            return True
        if self._extended_chase_price > 0 and price <= self._extended_chase_price * 0.96:
            return True
        return False

    def _clear_extended_chase_pending(self) -> None:
        self._extended_chase_call = -10_000
        self._extended_chase_price = 0.0


class V37GStrategy(V37FStrategy):
    """V3.7G: defer late mixed chase after the first starter."""

    VERSION_LABEL = "v3_7G"
    LATE_CHASE_PENDING_CALLS = 8

    @property
    def name(self) -> str:
        return "v3_7G"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        if self._extended_chase_pending_active():
            if self._should_release_extended_chase_pending(latest, price, raw_state, confirmed_state):
                self._clear_extended_chase_pending()
            elif self._is_late_mixed_chase(latest, price, raw_state, confirmed_state):
                return {
                    "decision": "defer",
                    "max_pct_mult": 0.0,
                    "max_pct_cap": 0.0,
                    "guard": f"{self.VERSION_LABEL}_late_mixed_chase_defer",
                }
            else:
                return {
                    "decision": "starter",
                    "max_pct_mult": 1.0,
                    "max_pct_cap": self.EXTENDED_CHASE_STARTER_CAP,
                    "guard": f"{self.VERSION_LABEL}_extended_chase_pending_starter",
                }

        if self._is_late_mixed_chase(latest, price, raw_state, confirmed_state):
            if self._extended_chase_call > 0:
                age = self._call_count - self._extended_chase_call
                if age <= self.LATE_CHASE_PENDING_CALLS:
                    return {
                        "decision": "defer",
                        "max_pct_mult": 0.0,
                        "max_pct_cap": 0.0,
                        "guard": f"{self.VERSION_LABEL}_late_mixed_chase_defer",
                    }

        gate = V37EStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
        return gate

    def _is_late_mixed_chase(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if str(latest.get("btc_regime", "")) != "STRONG_BULL":
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            price_vs_ema72 >= 0.07
            and price_vs_ema168 >= 0.02
            and donchian_pos >= 0.70
            and roc_20 >= 0.08
        )


class V38AStrategy(V37GStrategy):
    """V3.8A: buy-quality classifier for MIXED repair versus late chase."""

    VERSION_LABEL = "v3_8A"
    LATE_CHASE_PENDING_CALLS = 16

    @property
    def name(self) -> str:
        return "v3_8A"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )

        if self._extended_chase_pending_active():
            if self._should_release_quality_pending(latest, price, raw_state, confirmed_state):
                self._clear_extended_chase_pending()
            elif quality in {"late_mixed_chase", "bad_chase_mixed"}:
                return self._buy_quality_gate("defer", f"{quality}_defer", 0.0)
            elif quality == "mean_reversion_repair":
                self._clear_extended_chase_pending()

        if quality == "bad_chase_mixed":
            return self._buy_quality_gate("defer", "bad_chase_mixed_defer", 0.0)

        if quality == "late_mixed_chase":
            if self._extended_chase_call > 0:
                age = self._call_count - self._extended_chase_call
                if age <= self.LATE_CHASE_PENDING_CALLS:
                    return self._buy_quality_gate("defer", "late_mixed_chase_defer", 0.0)

        gate = V37EStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
            if quality in {"late_mixed_chase", "bad_chase_mixed"}:
                gate["guard"] = f"{self.VERSION_LABEL}_{quality}_starter"
        return gate

    def _classify_mixed_buy_quality(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> str:
        if raw_state == "BULL" and confirmed_state == "BULL":
            return "confirmed_bull"
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return "other"

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_5 = self._v3_value(latest, "roc_5")
        roc_10 = self._v3_value(latest, "roc_10")
        roc_20 = self._v3_value(latest, "roc_20")
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(rolling_365d_pos)
        ):
            return "other"

        btc_regime = str(latest.get("btc_regime", ""))
        if (
            price_vs_ema72 <= 0.03
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.68
            and (pd.isna(roc_5) or roc_5 >= -0.04)
            and (pd.isna(roc_10) or roc_10 >= -0.06)
        ):
            return "mean_reversion_repair"

        if (
            price_vs_ema72 <= 0.08
            and donchian_pos <= 0.82
            and -0.04 <= roc_20 <= 0.16
            and btc_regime != "BEAR"
        ):
            return "early_repair_mixed"

        if (
            btc_regime == "STRONG_BULL"
            and price_vs_ema72 >= 0.10
            and price_vs_ema168 >= 0.04
            and donchian_pos >= 0.82
            and roc_20 >= 0.14
            and rolling_365d_pos <= 0.45
        ):
            return "bad_chase_mixed"

        if self._is_late_mixed_chase(latest, price, raw_state, confirmed_state):
            return "late_mixed_chase"
        return "other"

    def _should_release_quality_pending(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state == "BULL" and confirmed_state == "BULL":
            return True
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            not pd.isna(price_vs_ema72)
            and not pd.isna(price_vs_ema168)
            and not pd.isna(donchian_pos)
            and not pd.isna(roc_20)
            and price_vs_ema72 <= 0.04
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.66
            and roc_20 <= 0.08
        ):
            return True
        if self._extended_chase_price > 0 and price <= self._extended_chase_price * 0.94:
            return True
        return False

    def _buy_quality_gate(self, decision: str, label: str, cap: float | None) -> dict:
        return {
            "decision": decision,
            "max_pct_mult": 1.0 if decision != "defer" else 0.0,
            "max_pct_cap": cap,
            "guard": f"{self.VERSION_LABEL}_{label}",
        }


class V38BStrategy(V38AStrategy):
    """V3.8B: force first late-MIXED chase fill to be a starter."""

    VERSION_LABEL = "v3_8B"

    @property
    def name(self) -> str:
        return "v3_8B"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )

        if self._extended_chase_pending_active():
            if self._should_release_quality_pending(latest, price, raw_state, confirmed_state):
                self._clear_extended_chase_pending()
            elif quality in {"late_mixed_chase", "bad_chase_mixed"}:
                return self._buy_quality_gate("defer", f"{quality}_defer", 0.0)
            elif quality == "mean_reversion_repair":
                self._clear_extended_chase_pending()

        if quality == "bad_chase_mixed":
            return self._buy_quality_gate("defer", "bad_chase_mixed_defer", 0.0)

        if quality == "late_mixed_chase":
            if self._extended_chase_call > 0:
                age = self._call_count - self._extended_chase_call
                if age <= self.LATE_CHASE_PENDING_CALLS:
                    return self._buy_quality_gate("defer", "late_mixed_chase_defer", 0.0)
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
            return self._buy_quality_gate(
                "starter",
                "late_mixed_chase_starter",
                self.EXTENDED_CHASE_STARTER_CAP,
            )

        gate = V37EStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
        return gate


class V38CStrategy(V38BStrategy):
    """V3.8C: keep late-MIXED chase pending until a true release condition."""

    VERSION_LABEL = "v3_8C"

    @property
    def name(self) -> str:
        return "v3_8C"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )

        if self._late_chase_quality_pending_active():
            if self._should_release_quality_pending(latest, price, raw_state, confirmed_state):
                self._clear_extended_chase_pending()
            else:
                return self._buy_quality_gate("defer", "late_mixed_pending_defer", 0.0)

        if quality == "bad_chase_mixed":
            return self._buy_quality_gate("defer", "bad_chase_mixed_defer", 0.0)

        if quality == "late_mixed_chase":
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
            return self._buy_quality_gate(
                "starter",
                "late_mixed_chase_starter",
                self.EXTENDED_CHASE_STARTER_CAP,
            )

        gate = V37EStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
        return gate

    def _late_chase_quality_pending_active(self) -> bool:
        return (
            self._extended_chase_call > 0
            and self._call_count - self._extended_chase_call <= self.LATE_CHASE_PENDING_CALLS
        )


class V38DStrategy(V38CStrategy):
    """V3.8D: strict pending only for low-ranked late-MIXED chase."""

    VERSION_LABEL = "v3_8D"

    @property
    def name(self) -> str:
        return "v3_8D"

    def _is_late_mixed_chase(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if not super()._is_late_mixed_chase(latest, price, raw_state, confirmed_state):
            return False
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        if pd.isna(rolling_365d_pos):
            return False
        return bool(rolling_365d_pos <= 0.45)


class V39AStrategy(V38DStrategy):
    """V3.9A: execution-layer pending fill using latest target gap."""

    VERSION_LABEL = "v3_9A"
    PENDING_FILL_MAX_BUY = 0.08
    PENDING_FILL_MIN_GAP = 0.04

    @property
    def name(self) -> str:
        return "v3_9A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            if action.side == "sell":
                self._clear_extended_chase_pending()
            return actions

        position = self._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None or not self._late_chase_quality_pending_active():
            return []

        fill = self._maybe_pending_target_gap_fill(position)
        return [fill] if fill is not None else []

    def _maybe_pending_target_gap_fill(self, position: dict) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        market = self._build_market_context(position["df"], latest, position["pos"], price)
        if market["trend_risk"] >= 3 or market["raw_state"] == "BEAR":
            self._clear_extended_chase_pending()
            return None
        if not self._should_pending_fill_on_mean_reversion(latest, price):
            return None

        signals = self._build_signal_context(position["df"], latest, price, market)
        band = self._build_target_band(position["symbol"], latest, price, market, signals)
        buy_target = band["buy_boundary"]
        gap = buy_target - position["current_pct"]
        if gap < self.PENDING_FILL_MIN_GAP:
            self._clear_extended_chase_pending()
            return None

        buy_pct = min(gap, self.PENDING_FILL_MAX_BUY)
        buy_qty = position["total_value"] * buy_pct / price
        if buy_qty <= 1e-12 or buy_qty * price < self.min_notional:
            return None

        self._last_buy_call = self._call_count
        self._clear_extended_chase_pending()
        return Action(
            symbol=position["symbol"],
            side="buy",
            quantity=buy_qty,
            price=price,
            reason=self._build_action_reason(
                side="buy",
                setup="target-gap",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=buy_target,
                guard=f"{self.VERSION_LABEL}_pending_mean_reversion_fill",
            ),
        )

    def _should_pending_fill_on_mean_reversion(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            price_vs_ema72 <= 0.04
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.66
            and roc_20 <= 0.08
        )


class V39BStrategy(V39AStrategy):
    """V3.9B: earlier execution-layer fill after a pending chase pullback."""

    VERSION_LABEL = "v3_9B"
    PENDING_FILL_MAX_BUY = 0.06

    @property
    def name(self) -> str:
        return "v3_9B"

    def _should_pending_fill_on_mean_reversion(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        deep_mean_reversion = (
            price_vs_ema72 <= 0.04
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.66
            and roc_20 <= 0.08
        )
        pullback_from_intent = (
            self._extended_chase_price > 0
            and price <= self._extended_chase_price * 0.96
            and price_vs_ema72 <= 0.08
            and price_vs_ema168 <= 0.04
            and donchian_pos <= 0.74
            and roc_20 <= 0.12
        )
        return bool(deep_mean_reversion or pullback_from_intent)


class V39CStrategy(V39BStrategy):
    """V3.9C: let execution layer own fills while a late-chase intent is pending."""

    VERSION_LABEL = "v3_9C"

    @property
    def name(self) -> str:
        return "v3_9C"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        if self._late_chase_quality_pending_active():
            if raw_state == "BEAR":
                self._clear_extended_chase_pending()
                return self._buy_quality_gate("defer", "late_mixed_pending_cancel", 0.0)
            return self._buy_quality_gate("defer", "late_mixed_pending_execution_owned", 0.0)

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )
        if quality == "bad_chase_mixed":
            return self._buy_quality_gate("defer", "bad_chase_mixed_defer", 0.0)
        if quality == "late_mixed_chase":
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
            return self._buy_quality_gate(
                "starter",
                "late_mixed_chase_starter",
                self.EXTENDED_CHASE_STARTER_CAP,
            )

        gate = V37EStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
        return gate


class V39DStrategy(V39CStrategy):
    """V3.9D: execution-owned pending fill is single-shot per intent."""

    VERSION_LABEL = "v3_9D"

    @property
    def name(self) -> str:
        return "v3_9D"

    def _maybe_pending_target_gap_fill(self, position: dict) -> Action | None:
        action = super()._maybe_pending_target_gap_fill(position)
        if action is not None:
            self._clear_extended_chase_pending()
        return action


class V39EStrategy(V39DStrategy):
    """V3.9E: execution-owned pending only for ranked late-MIXED chase."""

    VERSION_LABEL = "v3_9E"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_owned_late_chase_pending = False

    @property
    def name(self) -> str:
        return "v3_9E"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return V37EStrategy._evaluate_buy_timing_gate(
                self,
                latest=latest,
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                buy_setup=buy_setup,
                max_buy=max_buy,
            )

        if self._execution_owned_late_chase_pending and self._late_chase_quality_pending_active():
            if raw_state == "BEAR":
                self._clear_extended_chase_pending()
                return self._buy_quality_gate("defer", "late_mixed_pending_cancel", 0.0)
            return self._buy_quality_gate("defer", "late_mixed_pending_execution_owned", 0.0)

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )
        if quality == "bad_chase_mixed":
            return self._buy_quality_gate("defer", "bad_chase_mixed_defer", 0.0)
        if quality == "late_mixed_chase":
            self._extended_chase_call = self._call_count
            self._extended_chase_price = price
            self._execution_owned_late_chase_pending = True
            return self._buy_quality_gate(
                "starter",
                "late_mixed_chase_starter",
                self.EXTENDED_CHASE_STARTER_CAP,
            )

        gate = V38DStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("guard", "")).endswith("extended_chase_starter"):
            self._execution_owned_late_chase_pending = False
        return gate

    def _late_chase_quality_pending_active(self) -> bool:
        return (
            self._execution_owned_late_chase_pending
            and super()._late_chase_quality_pending_active()
        )

    def _clear_extended_chase_pending(self) -> None:
        super()._clear_extended_chase_pending()
        self._execution_owned_late_chase_pending = False


class V39FStrategy(V39EStrategy):
    """V3.9F: typed pending fill with a full target-gap-sized cap."""

    VERSION_LABEL = "v3_9F"
    PENDING_FILL_MAX_BUY = 0.08

    @property
    def name(self) -> str:
        return "v3_9F"


class V310AStrategy(V39FStrategy):
    """V3.10A: typed pending sell intent for low routine target-reduce."""

    VERSION_LABEL = "v3_10A"
    PENDING_SELL_MAX_PCT = 0.08
    PENDING_SELL_MIN_GAP = 0.04
    PENDING_SELL_MAX_CALLS = 24

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_low_sell_call = -10_000
        self._pending_low_sell_price = 0.0
        self._pending_low_sell_low = 0.0

    @property
    def name(self) -> str:
        return "v3_10A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            if action.side == "sell":
                self._clear_pending_low_sell()
            return actions

        position = self._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None or not self._pending_low_sell_active():
            return []

        fill = self._maybe_pending_low_sell_fill(position)
        return [fill] if fill is not None else []

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if sell_setup in {"risk-reduce", "trend-break", "core-override_trend-break"}:
            self._clear_pending_low_sell()
            return gate

        guard = str(gate.get("guard", ""))
        if "low_sell_freeze_open" in guard or "low_sell_freeze_wait" in guard:
            if not self._pending_low_sell_active():
                self._pending_low_sell_call = self._call_count
                self._pending_low_sell_price = price
                self._pending_low_sell_low = price
            else:
                self._pending_low_sell_low = min(self._pending_low_sell_low or price, price)
        return gate

    def _maybe_pending_low_sell_fill(self, position: dict) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        market = self._build_market_context(position["df"], latest, position["pos"], price)
        if market["trend_risk"] >= 3 or market["raw_state"] == "BEAR":
            self._clear_pending_low_sell()
            return None
        if market["raw_state"] == "BULL" and market["confirmed_state"] == "BULL":
            self._clear_pending_low_sell()
            return None
        if not self._should_pending_low_sell_execute(latest, price):
            if self._pending_low_sell_expired():
                self._clear_pending_low_sell()
            return None

        signals = self._build_signal_context(position["df"], latest, price, market)
        band = self._build_target_band(position["symbol"], latest, price, market, signals)
        sell_target = band["sell_boundary"]
        gap = position["current_pct"] - sell_target
        if gap < self.PENDING_SELL_MIN_GAP:
            self._clear_pending_low_sell()
            return None

        sell_pct = min(gap, self.PENDING_SELL_MAX_PCT)
        sell_qty = min(position["total_value"] * sell_pct / price, position["pos"].quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return None

        self._clear_pending_low_sell()
        self._record_executed_action(side="sell", setup="target-reduce")
        return Action(
            symbol=position["symbol"],
            side="sell",
            quantity=sell_qty,
            price=price,
            reason=self._build_action_reason(
                side="sell",
                setup="target-reduce",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=sell_target,
                guard=f"{self.VERSION_LABEL}_pending_low_sell_recovery_fill",
            ),
        )

    def _should_pending_low_sell_execute(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        recovered_to_mean = (
            price_vs_ema72 >= -0.03
            and price_vs_ema168 >= -0.06
            and donchian_pos >= 0.45
            and roc_20 >= -0.03
        )
        rebound_from_freeze = (
            self._pending_low_sell_price > 0
            and price >= self._pending_low_sell_price * 1.12
            and donchian_pos >= 0.38
            and roc_20 >= -0.06
        )
        return bool(recovered_to_mean or rebound_from_freeze)

    def _pending_low_sell_active(self) -> bool:
        return (
            self._pending_low_sell_call > 0
            and self._call_count - self._pending_low_sell_call <= self.PENDING_SELL_MAX_CALLS
        )

    def _pending_low_sell_expired(self) -> bool:
        return (
            self._pending_low_sell_call > 0
            and self._call_count - self._pending_low_sell_call > self.PENDING_SELL_MAX_CALLS
        )

    def _clear_pending_low_sell(self) -> None:
        self._pending_low_sell_call = -10_000
        self._pending_low_sell_price = 0.0
        self._pending_low_sell_low = 0.0


class V310BStrategy(V39FStrategy):
    """V3.10B: path-memory gate for high-price buybacks after routine trims."""

    VERSION_LABEL = "v3_10B"
    PATH_BUYBACK_MEMORY_CALLS = 30
    PATH_BUYBACK_STARTER_CAP = 0.04
    PATH_BUYBACK_HIGHER_PRICE = 1.02
    PATH_BUYBACK_DISCOUNT_PRICE = 0.98
    PATH_STARTER_SELL_FREEZE_CALLS = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_routine_sell_call = -10_000
        self._last_routine_sell_price = 0.0
        self._last_path_starter_buy_call = -10_000
        self._last_path_starter_buy_price = 0.0

    @property
    def name(self) -> str:
        return "v3_10B"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            self._record_path_memory_from_action(actions[0])
        return actions

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return gate
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._recent_routine_sell_active():
            return gate
        if price < self._last_routine_sell_price * self.PATH_BUYBACK_HIGHER_PRICE:
            return gate
        if self._is_path_buyback_quality_ok(latest, price, raw_state, confirmed_state):
            return gate
        cap = min(float(gate.get("max_pct_cap") or max_buy), self.PATH_BUYBACK_STARTER_CAP)
        return {
            "decision": "starter",
            "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
            "max_pct_cap": cap,
            "guard": self._join_guard(str(gate.get("guard", "")), "path_buyback_starter"),
        }

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if sell_setup != "target-reduce":
            return gate
        if not self._recent_path_starter_buy_active():
            return gate
        if trend_risk >= 3 or risk_score >= 4:
            return gate
        if price > self._last_path_starter_buy_price * 1.02:
            return gate
        if price < self._last_path_starter_buy_price * 0.97:
            return gate
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "path_starter_sell_freeze"),
        }

    def _is_path_buyback_quality_ok(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if self._last_routine_sell_price <= 0:
            return True
        if price <= self._last_routine_sell_price * self.PATH_BUYBACK_DISCOUNT_PRICE:
            return True

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(rolling_365d_pos)
        ):
            return False

        mean_reversion_buyback = (
            price_vs_ema72 <= 0.02
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.55
            and roc_20 <= 0.04
        )
        fresh_breakout = (
            raw_state == "BULL"
            and confirmed_state == "BULL"
            and price >= self._last_routine_sell_price * 1.08
            and roc_20 >= 0.18
            and donchian_pos >= 0.72
            and rolling_365d_pos <= 0.55
        )
        return bool(mean_reversion_buyback or fresh_breakout)

    def _recent_routine_sell_active(self) -> bool:
        return (
            self._last_routine_sell_call > 0
            and self._call_count - self._last_routine_sell_call <= self.PATH_BUYBACK_MEMORY_CALLS
            and self._last_routine_sell_price > 0
        )

    def _recent_path_starter_buy_active(self) -> bool:
        return (
            self._last_path_starter_buy_call > 0
            and self._call_count - self._last_path_starter_buy_call <= self.PATH_STARTER_SELL_FREEZE_CALLS
            and self._last_path_starter_buy_price > 0
        )

    def _record_path_memory_from_action(self, action: Action) -> None:
        parsed = self._parse_action_context(action)
        side = str(parsed.get("side", ""))
        setup = str(parsed.get("setup", ""))
        reason = str(getattr(action, "reason", ""))
        if side == "sell":
            if setup == "target-reduce":
                self._last_routine_sell_call = self._call_count
                self._last_routine_sell_price = float(action.price)
            elif setup in {"risk-reduce", "trend-break", "core-override_trend-break"}:
                self._last_routine_sell_call = -10_000
                self._last_routine_sell_price = 0.0
            return
        if side == "buy" and "path_buyback_starter" in reason:
            self._last_path_starter_buy_call = self._call_count
            self._last_path_starter_buy_price = float(action.price)

    @staticmethod
    def _parse_action_context(action: Action) -> dict:
        from .decision import parse_action_reason

        return parse_action_reason(getattr(action, "reason", ""))


class V310CStrategy(V310BStrategy):
    """V3.10C: defer only late-ranked high-price buybacks after routine trims."""

    VERSION_LABEL = "v3_10C"
    PATH_LATE_RANK_MIN = 0.70
    PATH_LATE_BREAKOUT_DONCHIAN_MIN = 0.70

    @property
    def name(self) -> str:
        return "v3_10C"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = V39FStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if buy_setup not in {"target-gap", "safe-recovery"}:
            return gate
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._recent_routine_sell_active():
            return gate
        if price < self._last_routine_sell_price * self.PATH_BUYBACK_HIGHER_PRICE:
            return gate
        if not self._is_late_rank_high_price_buyback(latest):
            return gate
        return {
            "decision": "defer",
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "late_rank_buyback_defer"),
        }

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        return V39FStrategy._evaluate_sell_timing_gate(
            self,
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

    def _is_late_rank_high_price_buyback(self, latest: pd.Series) -> bool:
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if pd.isna(rolling_365d_pos) or pd.isna(donchian_pos):
            return False
        return bool(
            rolling_365d_pos >= self.PATH_LATE_RANK_MIN
            and donchian_pos <= self.PATH_LATE_BREAKOUT_DONCHIAN_MIN
        )


class V310DStrategy(V310CStrategy):
    """V3.10D: avoid weak repair buys and boost low-rank mean-reversion fills."""

    VERSION_LABEL = "v3_10D"
    LOW_RANK_MR_BUY_MULT = 1.35
    LOW_RANK_MR_BUY_CAP = 0.12

    @property
    def name(self) -> str:
        return "v3_10D"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if self._pending_weak_mixed_buy_intent_active() and buy_setup in {"target-gap", "safe-recovery"}:
            release = self._classify_pending_buy_release(
                latest,
                price,
                {"raw_state": raw_state, "confirmed_state": confirmed_state},
            )
            if release == "":
                return {
                    "decision": "defer",
                    "max_pct_mult": 0.0,
                    "max_pct_cap": 0.0,
                    "guard": self._join_guard(str(gate.get("guard", "")), "weak_mixed_pending_owned"),
                }
            assert self._pending_intent is not None
            existing_cap = gate.get("max_pct_cap")
            cap = max_buy if existing_cap is None else min(max_buy, float(existing_cap))
            cap = min(cap, float(self._pending_intent.get("budget_pct", 0.0)), self.INTENT_MAX_MR_BUY)
            return {
                "decision": "allow",
                "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
                "max_pct_cap": cap,
                "guard": self._join_guard(str(gate.get("guard", "")), f"weak_mixed_pending_{release}_release"),
            }

        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate

        if self._is_weak_repair_buy(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            return {
                "decision": "defer",
                "max_pct_mult": 0.0,
                "max_pct_cap": 0.0,
                "guard": self._join_guard(str(gate.get("guard", "")), "weak_repair_buy_defer"),
            }

        if self._is_low_rank_mean_reversion_add(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            existing_cap = gate.get("max_pct_cap")
            cap = self.LOW_RANK_MR_BUY_CAP if existing_cap is None else max(
                float(existing_cap),
                self.LOW_RANK_MR_BUY_CAP,
            )
            return {
                "decision": "allow",
                "max_pct_mult": max(float(gate.get("max_pct_mult", 1.0)), self.LOW_RANK_MR_BUY_MULT),
                "max_pct_cap": cap,
                "guard": self._join_guard(str(gate.get("guard", "")), "low_rank_mr_add"),
            }

        return gate

    def _is_weak_repair_buy(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup == "safe-recovery":
            return self._is_weak_repair_safe_recovery(latest, price, raw_state, confirmed_state)
        if buy_setup == "target-gap":
            return self._is_bear_high_risk_target_gap(latest, raw_state, confirmed_state)
        return False

    def _is_weak_repair_safe_recovery(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state != "MIXED":
            return False
        trend_risk = self._calculate_trend_risk(latest, price)
        drawdown_risk = 0
        # In the timing gate we do not have the position object, so use market
        # shape only and keep the conditions narrow.
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        if (
            pd.isna(rolling_365d_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(price_vs_ema168)
        ):
            return False
        weak_down_repair = (
            trend_risk >= 1
            and rolling_365d_pos >= 0.60
            and donchian_pos <= 0.50
            and roc_20 <= -0.04
            and price_vs_ema168 >= -0.02
        )
        stale_mid_repair = (
            confirmed_state == "MIXED"
            and rolling_365d_pos >= 0.60
            and 0.50 <= donchian_pos <= 0.62
            and roc_20 <= 0.02
            and price_vs_ema168 >= 0.00
        )
        return bool(weak_down_repair or stale_mid_repair)

    def _is_bear_high_risk_target_gap(
        self,
        latest: pd.Series,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if str(latest.get("btc_regime", "")) != "BEAR":
            return False
        if raw_state == "BEAR":
            return False
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(rolling_365d_pos) or pd.isna(donchian_pos) or pd.isna(roc_20):
            return False
        return bool(
            rolling_365d_pos >= 0.55
            and donchian_pos <= 0.50
            and roc_20 <= 0.02
            and confirmed_state == "BULL"
        )

    def _is_low_rank_mean_reversion_add(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup != "target-gap":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        if (
            pd.isna(rolling_365d_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(price_vs_ema168)
        ):
            return False
        return bool(
            rolling_365d_pos <= 0.30
            and 0.30 <= donchian_pos <= 0.55
            and price_vs_ema168 <= 0.00
            and roc_20 <= 0.00
        )


class V310EStrategy(V310CStrategy):
    """V3.10E: lightly add to low-rank MIXED mean-reversion target-gap buys."""

    VERSION_LABEL = "v3_10E"
    LOW_RANK_MR_BUY_MULT = 1.25
    LOW_RANK_MR_BUY_CAP = 0.10

    @property
    def name(self) -> str:
        return "v3_10E"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._is_low_rank_mean_reversion_add(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            return gate

        existing_cap = gate.get("max_pct_cap")
        cap = self.LOW_RANK_MR_BUY_CAP if existing_cap is None else max(
            float(existing_cap),
            self.LOW_RANK_MR_BUY_CAP,
        )
        return {
            "decision": "allow",
            "max_pct_mult": max(float(gate.get("max_pct_mult", 1.0)), self.LOW_RANK_MR_BUY_MULT),
            "max_pct_cap": cap,
            "guard": self._join_guard(str(gate.get("guard", "")), "low_rank_mr_add"),
        }

    def _is_low_rank_mean_reversion_add(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup != "target-gap":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        if (
            pd.isna(rolling_365d_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(price_vs_ema168)
        ):
            return False
        return bool(
            rolling_365d_pos <= 0.30
            and 0.30 <= donchian_pos <= 0.55
            and price_vs_ema168 <= 0.00
            and roc_20 <= 0.00
        )


class V310FStrategy(V310CStrategy):
    """V3.10F: low-rank MIXED mean-reversion add without imposing a new cap."""

    VERSION_LABEL = "v3_10F"
    LOW_RANK_MR_BUY_MULT = 1.20

    @property
    def name(self) -> str:
        return "v3_10F"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not V310EStrategy._is_low_rank_mean_reversion_add(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            return gate
        return {
            "decision": "allow",
            "max_pct_mult": max(float(gate.get("max_pct_mult", 1.0)), self.LOW_RANK_MR_BUY_MULT),
            "max_pct_cap": gate.get("max_pct_cap"),
            "guard": self._join_guard(str(gate.get("guard", "")), "low_rank_mr_add"),
        }


class V310GStrategy(V310CStrategy):
    """V3.10G: execution intent for deferred high-rank buybacks."""

    VERSION_LABEL = "v3_10G"
    PATH_BUYBACK_PENDING_MAX_CALLS = 30
    PATH_BUYBACK_PENDING_MIN_GAP = 0.04
    PATH_BUYBACK_PENDING_MAX_BUY = 0.08
    PATH_BUYBACK_PULLBACK_FROM_INTENT = 0.96

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_path_buyback_call = -10_000
        self._pending_path_buyback_price = 0.0
        self._pending_path_buyback_sell_price = 0.0

    @property
    def name(self) -> str:
        return "v3_10G"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            action = actions[0]
            parsed = self._parse_action_context(action)
            if parsed.get("side") == "sell" and parsed.get("setup") in {
                "risk-reduce",
                "trend-break",
                "core-override_trend-break",
            }:
                self._clear_pending_path_buyback()
            return actions

        position = self._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None or not self._pending_path_buyback_active():
            return []

        fill = self._maybe_pending_path_buyback_fill(position)
        return [fill] if fill is not None else []

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        guard = str(gate.get("guard", ""))
        if "late_rank_buyback_defer" in guard:
            self._pending_path_buyback_call = self._call_count
            self._pending_path_buyback_price = price
            self._pending_path_buyback_sell_price = self._last_routine_sell_price
        elif buy_setup not in {"target-gap", "safe-recovery"}:
            self._clear_pending_path_buyback_if_stale()
        return gate

    def _maybe_pending_path_buyback_fill(self, position: dict) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        market = self._build_market_context(position["df"], latest, position["pos"], price)
        if market["trend_risk"] >= 3 or market["raw_state"] == "BEAR":
            self._clear_pending_path_buyback()
            return None
        if self._pending_path_buyback_expired():
            self._clear_pending_path_buyback()
            return None
        if not self._should_release_path_buyback_pending(latest, price):
            return None

        signals = self._build_signal_context(position["df"], latest, price, market)
        band = self._build_target_band(position["symbol"], latest, price, market, signals)
        buy_target = band["buy_boundary"]
        gap = buy_target - position["current_pct"]
        if gap < self.PATH_BUYBACK_PENDING_MIN_GAP:
            self._clear_pending_path_buyback()
            return None

        buy_pct = min(gap, self.PATH_BUYBACK_PENDING_MAX_BUY)
        buy_qty = position["total_value"] * buy_pct / price
        if buy_qty <= 1e-12 or buy_qty * price < self.min_notional:
            return None

        self._last_buy_call = self._call_count
        self._clear_pending_path_buyback()
        return Action(
            symbol=position["symbol"],
            side="buy",
            quantity=buy_qty,
            price=price,
            reason=self._build_action_reason(
                side="buy",
                setup="target-gap",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=buy_target,
                guard=f"{self.VERSION_LABEL}_path_buyback_pending_mr_fill",
            ),
        )

    def _should_release_path_buyback_pending(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False

        pullback_from_intent = (
            self._pending_path_buyback_price > 0
            and price <= self._pending_path_buyback_price * self.PATH_BUYBACK_PULLBACK_FROM_INTENT
        )
        back_near_last_sell = (
            self._pending_path_buyback_sell_price > 0
            and price <= self._pending_path_buyback_sell_price * 1.01
        )
        mean_reversion_quality = (
            price_vs_ema72 <= 0.04
            and price_vs_ema168 <= 0.03
            and donchian_pos <= 0.58
            and roc_20 <= 0.06
        )
        return bool((pullback_from_intent or back_near_last_sell) and mean_reversion_quality)

    def _pending_path_buyback_active(self) -> bool:
        return (
            self._pending_path_buyback_call > 0
            and self._call_count - self._pending_path_buyback_call <= self.PATH_BUYBACK_PENDING_MAX_CALLS
        )

    def _pending_path_buyback_expired(self) -> bool:
        return (
            self._pending_path_buyback_call > 0
            and self._call_count - self._pending_path_buyback_call > self.PATH_BUYBACK_PENDING_MAX_CALLS
        )

    def _clear_pending_path_buyback_if_stale(self) -> None:
        if not self._pending_path_buyback_active():
            self._clear_pending_path_buyback()

    def _clear_pending_path_buyback(self) -> None:
        self._pending_path_buyback_call = -10_000
        self._pending_path_buyback_price = 0.0
        self._pending_path_buyback_sell_price = 0.0


class V310HStrategy(V310GStrategy):
    """V3.10H: protect pending buyback fills from immediate routine trims."""

    VERSION_LABEL = "v3_10H"
    PATH_PENDING_FILL_SELL_FREEZE_CALLS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_path_pending_fill_call = -10_000
        self._last_path_pending_fill_price = 0.0

    @property
    def name(self) -> str:
        return "v3_10H"

    def _maybe_pending_path_buyback_fill(self, position: dict) -> Action | None:
        action = super()._maybe_pending_path_buyback_fill(position)
        if action is not None:
            self._last_path_pending_fill_call = self._call_count
            self._last_path_pending_fill_price = float(action.price)
        return action

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if sell_setup != "target-reduce":
            return gate
        if not self._recent_path_pending_fill_active():
            return gate
        if trend_risk >= 3 or risk_score >= 4:
            return gate
        if price > self._last_path_pending_fill_price * 1.06:
            return gate
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "path_pending_fill_sell_freeze"),
        }

    def _recent_path_pending_fill_active(self) -> bool:
        return (
            self._last_path_pending_fill_call > 0
            and self._call_count - self._last_path_pending_fill_call <= self.PATH_PENDING_FILL_SELL_FREEZE_CALLS
            and self._last_path_pending_fill_price > 0
        )


class V310IStrategy(V310GStrategy):
    """V3.10I: reserved intent budget for deferred path buybacks."""

    VERSION_LABEL = "v3_10I"
    PATH_BUYBACK_PENDING_MAX_BUY = 0.08
    PATH_RESERVED_SELL_PROTECT_CALLS = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_path_buyback_budget = 0.0
        self._reserved_path_budget = 0.0
        self._reserved_path_call = -10_000
        self._reserved_path_price = 0.0

    @property
    def name(self) -> str:
        return "v3_10I"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if not actions:
            self._clear_reserved_path_budget_if_stale()
            return actions

        action = actions[0]
        parsed = self._parse_action_context(action)
        side = str(parsed.get("side", ""))
        setup = str(parsed.get("setup", ""))
        if side == "sell" and setup in {"risk-reduce", "trend-break", "core-override_trend-break"}:
            self._clear_reserved_path_budget()
            return actions
        if side == "sell" and setup == "target-reduce":
            self._consume_reserved_path_budget(candles_by_symbol, portfolio, current_prices, action)
        return actions

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        guard = str(gate.get("guard", ""))
        if "late_rank_buyback_defer" in guard:
            self._pending_path_buyback_budget = min(
                max(float(max_buy), 0.0),
                self.PATH_BUYBACK_PENDING_MAX_BUY,
            )
        elif not self._pending_path_buyback_active():
            self._pending_path_buyback_budget = 0.0
        return gate

    def _maybe_pending_path_buyback_fill(self, position: dict) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        market = self._build_market_context(position["df"], latest, position["pos"], price)
        if market["trend_risk"] >= 3 or market["raw_state"] == "BEAR":
            self._clear_pending_path_buyback()
            return None
        if self._pending_path_buyback_expired():
            self._clear_pending_path_buyback()
            return None
        if not self._should_release_path_buyback_pending(latest, price):
            return None

        signals = self._build_signal_context(position["df"], latest, price, market)
        band = self._build_target_band(position["symbol"], latest, price, market, signals)
        buy_target = band["buy_boundary"]
        gap = buy_target - position["current_pct"]
        if gap < self.PATH_BUYBACK_PENDING_MIN_GAP:
            self._clear_pending_path_buyback()
            return None

        budget = min(self._pending_path_buyback_budget, self.PATH_BUYBACK_PENDING_MAX_BUY)
        if budget <= 1e-12:
            self._clear_pending_path_buyback()
            return None

        buy_pct = min(gap, budget)
        buy_qty = position["total_value"] * buy_pct / price
        if buy_qty <= 1e-12 or buy_qty * price < self.min_notional:
            return None

        self._last_buy_call = self._call_count
        self._reserved_path_budget = buy_pct
        self._reserved_path_call = self._call_count
        self._reserved_path_price = price
        self._clear_pending_path_buyback()
        return Action(
            symbol=position["symbol"],
            side="buy",
            quantity=buy_qty,
            price=price,
            reason=self._build_action_reason(
                side="buy",
                setup="target-gap",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=buy_target,
                guard=f"{self.VERSION_LABEL}_reserved_path_buyback_fill",
            ),
        )

    def _should_release_path_buyback_pending(self, latest: pd.Series, price: float) -> bool:
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False

        pullback_from_intent = (
            self._pending_path_buyback_price > 0
            and price <= self._pending_path_buyback_price * self.PATH_BUYBACK_PULLBACK_FROM_INTENT
        )
        back_near_last_sell = (
            self._pending_path_buyback_sell_price > 0
            and price <= self._pending_path_buyback_sell_price * 1.01
        )
        mean_reversion_quality = (
            price_vs_ema72 <= 0.03
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.55
            and roc_20 <= 0.04
        )
        return bool((pullback_from_intent or back_near_last_sell) and mean_reversion_quality)

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if sell_setup != "target-reduce":
            return gate
        if not self._reserved_path_budget_active():
            return gate
        if trend_risk >= 3 or risk_score >= 4:
            return gate
        existing_cap = gate.get("max_pct_cap")
        adjusted_cap = max(0.0, max_sell - self._reserved_path_budget)
        if existing_cap is not None:
            adjusted_cap = min(adjusted_cap, float(existing_cap))
        return {
            "decision": str(gate.get("decision", "allow")),
            "threshold": float(gate.get("threshold", sell_threshold)),
            "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
            "max_pct_cap": adjusted_cap,
            "guard": self._join_guard(str(gate.get("guard", "")), "reserved_path_budget_protect"),
        }

    def _consume_reserved_path_budget(self, candles_by_symbol, portfolio, current_prices, action: Action) -> None:
        if not self._reserved_path_budget_active():
            return
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            return
        pos = portfolio.positions.get(symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        if total_value <= 0:
            return
        sold_pct = action.quantity * action.price / total_value
        self._reserved_path_budget = max(0.0, self._reserved_path_budget - sold_pct)
        if self._reserved_path_budget <= 1e-12:
            self._clear_reserved_path_budget()

    def _reserved_path_budget_active(self) -> bool:
        return (
            self._reserved_path_call > 0
            and self._call_count - self._reserved_path_call <= self.PATH_RESERVED_SELL_PROTECT_CALLS
            and self._reserved_path_budget > 1e-12
        )

    def _clear_reserved_path_budget_if_stale(self) -> None:
        if not self._reserved_path_budget_active():
            self._clear_reserved_path_budget()

    def _clear_reserved_path_budget(self) -> None:
        self._reserved_path_budget = 0.0
        self._reserved_path_call = -10_000
        self._reserved_path_price = 0.0

    def _clear_pending_path_buyback(self) -> None:
        super()._clear_pending_path_buyback()
        self._pending_path_buyback_budget = 0.0


class V310JStrategy(V310IStrategy):
    """V3.10J: protect reserved intent budget by raising routine sell threshold."""

    VERSION_LABEL = "v3_10J"

    @property
    def name(self) -> str:
        return "v3_10J"

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = V310GStrategy._evaluate_sell_timing_gate(
            self,
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
        if sell_setup != "target-reduce":
            return gate
        if not self._reserved_path_budget_active():
            return gate
        if trend_risk >= 3 or risk_score >= 4:
            return gate
        return {
            "decision": str(gate.get("decision", "allow")),
            "threshold": float(gate.get("threshold", sell_threshold)) + self._reserved_path_budget,
            "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
            "max_pct_cap": gate.get("max_pct_cap"),
            "guard": self._join_guard(str(gate.get("guard", "")), "reserved_path_threshold_protect"),
        }


class V311AStrategy(V310CStrategy):
    """V3.11A: explicit single-intent execution arbiter over V3.10C."""

    VERSION_LABEL = "v3_11A"
    INTENT_MAX_CALLS = 34
    INTENT_MIN_GAP = 0.04
    INTENT_MAX_MR_BUY = 0.08
    INTENT_MAX_BREAKOUT_BUY = 0.04
    INTENT_PULLBACK_FROM_CREATED = 0.94
    INTENT_PULLBACK_TO_ANCHOR = 1.00

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_intent: dict | None = None

    @property
    def name(self) -> str:
        return "v3_11A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions:
            self._update_intent_after_action(actions[0])
            return actions

        if not self._pending_buy_intent_active():
            self._clear_pending_buy_intent_if_stale()
            return []

        context = self._prepare_pending_execution_context(candles_by_symbol, portfolio, current_prices)
        if context is None:
            return []
        fill = self._maybe_fill_pending_buy_intent(
            context["position"],
            context["market"],
            context["band"],
        )
        if fill is not None:
            self._record_path_memory_from_action(fill)
            self._update_intent_after_action(fill)
        return [fill] if fill is not None else []

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        guard = str(gate.get("guard", ""))
        if "late_rank_buyback_defer" in guard:
            self._open_or_update_buy_intent(
                source=buy_setup,
                price=price,
                budget=max_buy,
            )
        elif buy_setup not in {"target-gap", "safe-recovery"}:
            self._clear_pending_buy_intent_if_stale()
        return gate

    def _prepare_pending_execution_context(self, candles_by_symbol, portfolio, current_prices) -> dict | None:
        position = self._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None:
            return None
        market = self._build_market_context_without_confirmation(
            position["latest"],
            position["pos"],
            position["price"],
        )
        signals = self._build_signal_context(
            position["df"],
            position["latest"],
            position["price"],
            market,
        )
        band = self._build_target_band(
            position["symbol"],
            position["latest"],
            position["price"],
            market,
            signals,
        )
        return {
            "position": position,
            "market": market,
            "signals": signals,
            "band": band,
        }

    def _build_market_context_without_confirmation(
        self,
        latest: pd.Series,
        pos: PositionState,
        price: float,
    ) -> dict:
        raw_state = self._detect_market_state(latest)
        confirmed_state = self._current_state
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

    def _open_or_update_buy_intent(
        self,
        *,
        source: str,
        price: float,
        budget: float,
        anchor_price: float | None = None,
    ) -> None:
        anchor_price = float(anchor_price if anchor_price is not None else self._last_routine_sell_price or price)
        budget_pct = min(max(float(budget), 0.0), self.INTENT_MAX_MR_BUY)
        if budget_pct <= 0.0 or anchor_price <= 0.0:
            return

        if self._pending_buy_intent_active():
            assert self._pending_intent is not None
            self._pending_intent["created_price"] = min(
                float(self._pending_intent.get("created_price", price)),
                float(price),
            )
            self._pending_intent["anchor_price"] = anchor_price
            self._pending_intent["budget_pct"] = max(
                float(self._pending_intent.get("budget_pct", 0.0)),
                budget_pct,
            )
            self._pending_intent["source"] = source
            return

        self._pending_intent = {
            "side": "buy",
            "source": source,
            "created_call": self._call_count,
            "created_price": float(price),
            "anchor_price": anchor_price,
            "budget_pct": budget_pct,
        }

    def _maybe_fill_pending_buy_intent(
        self,
        position: dict,
        market: dict,
        band: dict,
    ) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        if self._should_cancel_pending_buy_intent(market):
            self._clear_pending_intent()
            return None
        if self._pending_buy_intent_expired():
            self._clear_pending_intent()
            return None

        release = self._classify_pending_buy_release(latest, price, market)
        if release == "":
            return None

        buy_target = band["buy_boundary"]
        gap = buy_target - position["current_pct"]
        if gap < self.INTENT_MIN_GAP:
            self._clear_pending_intent()
            return None

        assert self._pending_intent is not None
        release_cap = (
            self.INTENT_MAX_BREAKOUT_BUY
            if release == "breakout"
            else self.INTENT_MAX_MR_BUY
        )
        buy_pct = min(gap, float(self._pending_intent.get("budget_pct", 0.0)), release_cap)
        buy_qty = position["total_value"] * buy_pct / price
        if buy_qty <= 1e-12 or buy_qty * price < self.min_notional:
            return None

        self._last_buy_call = self._call_count
        self._clear_pending_intent()
        return Action(
            symbol=position["symbol"],
            side="buy",
            quantity=buy_qty,
            price=price,
            reason=self._build_action_reason(
                side="buy",
                setup="target-gap",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=buy_target,
                guard=f"{self.VERSION_LABEL}_intent_{release}_fill",
            ),
        )

    def _classify_pending_buy_release(
        self,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> str:
        if self._pending_intent is None:
            return ""
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(rolling_365d_pos)
        ):
            return ""

        created_price = float(self._pending_intent.get("created_price", 0.0))
        anchor_price = float(self._pending_intent.get("anchor_price", 0.0))
        pullback_from_created = (
            created_price > 0
            and price <= created_price * self.INTENT_PULLBACK_FROM_CREATED
        )
        back_to_anchor = (
            anchor_price > 0
            and price <= anchor_price * self.INTENT_PULLBACK_TO_ANCHOR
        )
        mean_reversion_quality = (
            price_vs_ema72 <= 0.02
            and price_vs_ema168 <= 0.02
            and donchian_pos <= 0.58
            and roc_20 <= 0.06
            and str(latest.get("btc_regime", "")) != "BEAR"
        )
        if (pullback_from_created or back_to_anchor) and mean_reversion_quality:
            return "mr"

        breakout_quality = (
            market["raw_state"] == "BULL"
            and market["confirmed_state"] == "BULL"
            and created_price > 0
            and price >= created_price * 1.10
            and price_vs_ema72 <= 0.12
            and donchian_pos >= 0.74
            and roc_20 >= 0.18
            and rolling_365d_pos <= 0.60
            and str(latest.get("btc_regime", "")) != "BEAR"
        )
        if breakout_quality:
            return "breakout"
        return ""

    def _should_cancel_pending_buy_intent(self, market: dict) -> bool:
        return bool(
            market["trend_risk"] >= 3
            or market["risk_score"] >= 4
            or market["raw_state"] == "BEAR"
        )

    def _update_intent_after_action(self, action: Action) -> None:
        parsed = self._parse_action_context(action)
        side = str(parsed.get("side", ""))
        if side in {"buy", "sell"}:
            self._clear_pending_intent()

    def _pending_buy_intent_active(self) -> bool:
        return (
            self._pending_intent is not None
            and self._pending_intent.get("side") == "buy"
            and not self._pending_buy_intent_expired()
        )

    def _pending_buy_intent_expired(self) -> bool:
        if self._pending_intent is None:
            return False
        if self._pending_intent.get("side") != "buy":
            return False
        return self._call_count - int(self._pending_intent.get("created_call", -10_000)) > self.INTENT_MAX_CALLS

    def _clear_pending_buy_intent_if_stale(self) -> None:
        if self._pending_intent is None or self._pending_intent.get("side") != "buy":
            return
        if self._pending_buy_intent_expired():
            self._clear_pending_intent()

    def _clear_pending_intent(self) -> None:
        self._pending_intent = None


class V311BStrategy(V311AStrategy):
    """V3.11B: add narrow low-price routine-sell intent to the 11A arbiter."""

    VERSION_LABEL = "v3_11B"
    SELL_INTENT_MAX_CALLS = 20
    SELL_INTENT_MIN_GAP = 0.04
    SELL_INTENT_MAX_SELL = 0.08
    SELL_INTENT_REBOUND_FROM_CREATED = 1.06

    @property
    def name(self) -> str:
        return "v3_11B"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = V310CStrategy.compute_actions(self, candles_by_symbol, portfolio, current_prices)
        if actions:
            self._update_intent_after_action(actions[0])
            return actions

        context = self._prepare_pending_execution_context(candles_by_symbol, portfolio, current_prices)
        if context is None:
            return []
        position = context["position"]
        market = context["market"]
        band = context["band"]

        sell_fill = self._maybe_fill_pending_sell_intent(position, market, band)
        if sell_fill is not None:
            self._record_path_memory_from_action(sell_fill)
            self._update_intent_after_action(sell_fill)
            return [sell_fill]

        if not self._pending_buy_intent_active():
            self._clear_pending_sell_intent_if_stale()
            self._clear_pending_buy_intent_if_stale()
            return []
        buy_fill = self._maybe_fill_pending_buy_intent(position, market, band)
        if buy_fill is not None:
            self._record_path_memory_from_action(buy_fill)
            self._update_intent_after_action(buy_fill)
        return [buy_fill] if buy_fill is not None else []

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if not self._should_defer_low_routine_sell(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return gate

        self._open_or_update_sell_intent(price=price, budget=max_sell)
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "low_sell_intent_defer"),
        }

    def _should_defer_low_routine_sell(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk != 0 or drawdown_risk != 0 or risk_score > 1:
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            price_vs_ema72 <= -0.04
            and price_vs_ema168 <= -0.04
            and donchian_pos <= 0.42
            and roc_20 <= 0.02
        )

    def _open_or_update_sell_intent(self, *, price: float, budget: float) -> None:
        budget_pct = min(max(float(budget), 0.0), self.SELL_INTENT_MAX_SELL)
        if budget_pct <= 0.0:
            return
        if self._pending_sell_intent_active():
            assert self._pending_intent is not None
            self._pending_intent["created_price"] = min(
                float(self._pending_intent.get("created_price", price)),
                float(price),
            )
            self._pending_intent["budget_pct"] = max(
                float(self._pending_intent.get("budget_pct", 0.0)),
                budget_pct,
            )
            return
        self._pending_intent = {
            "side": "sell",
            "source": "target-reduce",
            "created_call": self._call_count,
            "created_price": float(price),
            "anchor_price": float(price),
            "budget_pct": budget_pct,
        }

    def _maybe_fill_pending_sell_intent(
        self,
        position: dict,
        market: dict,
        band: dict,
    ) -> Action | None:
        if not self._pending_sell_intent_active():
            return None
        if market["trend_risk"] >= 2 or market["risk_score"] >= 3 or market["raw_state"] == "BEAR":
            self._clear_pending_intent()
            return None
        latest = position["latest"]
        price = position["price"]
        if not self._should_release_pending_sell_intent(latest, price):
            return None

        sell_target = band["sell_boundary"]
        gap = position["current_pct"] - sell_target
        if gap < self.SELL_INTENT_MIN_GAP:
            self._clear_pending_intent()
            return None

        assert self._pending_intent is not None
        sell_pct = min(gap, float(self._pending_intent.get("budget_pct", 0.0)), self.SELL_INTENT_MAX_SELL)
        sell_qty = min(position["total_value"] * sell_pct / price, position["pos"].quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return None

        self._clear_pending_intent()
        self._record_executed_action(side="sell", setup="target-reduce")
        return Action(
            symbol=position["symbol"],
            side="sell",
            quantity=sell_qty,
            price=price,
            reason=self._build_action_reason(
                side="sell",
                setup="target-reduce",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=sell_target,
                guard=f"{self.VERSION_LABEL}_intent_rebound_sell",
            ),
        )

    def _should_release_pending_sell_intent(self, latest: pd.Series, price: float) -> bool:
        if self._pending_intent is None:
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(price_vs_ema72) or pd.isna(donchian_pos) or pd.isna(roc_20):
            return False
        created_price = float(self._pending_intent.get("created_price", 0.0))
        rebound_from_created = (
            created_price > 0
            and price >= created_price * self.SELL_INTENT_REBOUND_FROM_CREATED
        )
        recovered_to_mean = (
            price_vs_ema72 >= -0.01
            and donchian_pos >= 0.50
            and roc_20 >= -0.02
        )
        return bool(rebound_from_created or recovered_to_mean)

    def _pending_sell_intent_active(self) -> bool:
        return (
            self._pending_intent is not None
            and self._pending_intent.get("side") == "sell"
            and not self._pending_sell_intent_expired()
        )

    def _pending_sell_intent_expired(self) -> bool:
        if self._pending_intent is None:
            return False
        if self._pending_intent.get("side") != "sell":
            return False
        return self._call_count - int(self._pending_intent.get("created_call", -10_000)) > self.SELL_INTENT_MAX_CALLS

    def _clear_pending_sell_intent_if_stale(self) -> None:
        if self._pending_intent is None or self._pending_intent.get("side") != "sell":
            return
        if self._pending_sell_intent_expired():
            self._clear_pending_intent()


class V311CStrategy(V311AStrategy):
    """V3.11C: weak MIXED target-gap starter with a unified buy intent."""

    VERSION_LABEL = "v3_11C"
    WEAK_MIXED_STARTER_CAP = 0.025
    WEAK_MIXED_PENDING_PULLBACK = 0.96

    @property
    def name(self) -> str:
        return "v3_11C"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._is_weak_mixed_target_gap(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        ):
            return gate

        existing_cap = gate.get("max_pct_cap")
        base_cap = max_buy if existing_cap is None else min(max_buy, float(existing_cap))
        starter_cap = min(base_cap, self.WEAK_MIXED_STARTER_CAP)
        deferred_budget = max(max_buy - starter_cap, self.INTENT_MIN_GAP)
        self._open_or_update_buy_intent(
            source="weak-mixed-target-gap",
            price=price,
            budget=deferred_budget,
            anchor_price=price * self.WEAK_MIXED_PENDING_PULLBACK,
        )
        return {
            "decision": "starter",
            "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
            "max_pct_cap": starter_cap,
            "guard": self._join_guard(str(gate.get("guard", "")), "weak_mixed_target_gap_starter"),
        }

    def _is_weak_mixed_target_gap(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> bool:
        if buy_setup != "target-gap":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if str(latest.get("btc_regime", "")) == "BEAR":
            return False

        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema24_slope = self._v3_value(latest, "ema24_slope")
        if (
            pd.isna(rolling_365d_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema24_slope)
        ):
            return False

        high_rank_weak_repair = (
            rolling_365d_pos >= 0.65
            and -0.06 <= price_vs_ema72 <= 0.03
            and price_vs_ema168 >= -0.02
            and donchian_pos <= 0.45
            and roc_20 <= 0.04
            and ema24_slope <= 0.0
        )
        if high_rank_weak_repair:
            return True

        quality = self._classify_mixed_buy_quality(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
        )
        if quality in {"mean_reversion_repair", "early_repair_mixed", "confirmed_bull"}:
            return False
        if quality in {"bad_chase_mixed", "late_mixed_chase"}:
            return False

        return bool(
            rolling_365d_pos >= 0.55
            and 0.00 <= price_vs_ema72 <= 0.10
            and price_vs_ema168 >= -0.02
            and 0.45 <= donchian_pos <= 0.68
            and roc_20 <= 0.06
            and ema24_slope <= 0.0
        )

    def _update_intent_after_action(self, action: Action) -> None:
        parsed = self._parse_action_context(action)
        side = str(parsed.get("side", ""))
        reason = str(getattr(action, "reason", ""))
        if side == "buy" and "weak_mixed_target_gap_starter" in reason:
            return
        if side in {"buy", "sell"}:
            self._clear_pending_intent()

    def _pending_weak_mixed_buy_intent_active(self) -> bool:
        return (
            self._pending_buy_intent_active()
            and self._pending_intent is not None
            and self._pending_intent.get("source") == "weak-mixed-target-gap"
        )


class V312AStrategy(V311AStrategy):
    """V3.12A: peak-memory sell timing over the V3.11A intent base."""

    VERSION_LABEL = "v3_12A"
    PEAK_TRIM_SELL_PCT = 0.06
    PEAK_TRIM_MIN_POSITION = 0.70
    PEAK_TRIM_MIN_PROFIT = 0.25
    PEAK_TRIM_MIN_PEAK_PROFIT = 0.45
    PEAK_TRIM_MIN_PULLBACK = 0.08
    PEAK_TRIM_MAX_PULLBACK = 0.20
    PEAK_TRIM_MIN_DAYS_FROM_PEAK = 3
    PEAK_TRIM_MAX_DAYS_FROM_PEAK = 35
    PEAK_TRIM_MIN_GAP_CALLS = 28
    PEAK_TRIM_NEW_PEAK_BUFFER = 0.08

    LATE_SELL_INTENT_MAX_CALLS = 24
    LATE_SELL_INTENT_MIN_GAP = 0.04
    LATE_SELL_INTENT_MAX_SELL = 0.08
    LATE_SELL_DEFER_PEAK_DD = 0.18
    LATE_SELL_DEFER_MIN_DAYS_FROM_PEAK = 10
    LATE_SELL_REBOUND_FROM_CREATED = 1.06

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v312_peak_price = 0.0
        self._v312_peak_call = -10_000
        self._last_v312_peak_trim_call = -10_000
        self._last_v312_peak_trim_sell_price = 0.0
        self._last_v312_peak_trim_peak_price = 0.0

    @property
    def name(self) -> str:
        return "v3_12A"

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        actions = V310CStrategy.compute_actions(self, candles_by_symbol, portfolio, current_prices)
        if actions:
            self._update_intent_after_action(actions[0])
            return actions

        context = self._prepare_pending_execution_context(candles_by_symbol, portfolio, current_prices)
        if context is None:
            return []

        sell_fill = self._maybe_fill_late_sell_intent(context["position"], context["market"], context["band"])
        if sell_fill is not None:
            self._record_path_memory_from_action(sell_fill)
            self._update_intent_after_action(sell_fill)
            return [sell_fill]

        if not self._pending_buy_intent_active():
            self._clear_late_sell_intent_if_stale()
            self._clear_pending_buy_intent_if_stale()
            return []

        buy_fill = self._maybe_fill_pending_buy_intent(context["position"], context["market"], context["band"])
        if buy_fill is not None:
            self._record_path_memory_from_action(buy_fill)
            self._update_intent_after_action(buy_fill)
        return [buy_fill] if buy_fill is not None else []

    def _prepare_position_context(self, candles_by_symbol, portfolio, current_prices) -> dict | None:
        position = super()._prepare_position_context(candles_by_symbol, portfolio, current_prices)
        if position is None:
            return None
        price = float(position["price"])
        pos = position["pos"]
        current_pct = float(position["current_pct"])
        if current_pct < 0.20 or pos.quantity <= 1e-12:
            self._v312_peak_price = price
            self._v312_peak_call = self._call_count
        elif price >= self._v312_peak_price:
            self._v312_peak_price = price
            self._v312_peak_call = self._call_count
        return position

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
        if actions or self._pending_late_sell_intent_active():
            return actions
        if not self._should_peak_memory_trim(latest, price, pos, current_pct, market):
            return []

        sell_pct = min(self.PEAK_TRIM_SELL_PCT, current_pct)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return []

        self._last_v312_peak_trim_call = self._call_count
        self._last_v312_peak_trim_sell_price = price
        self._last_v312_peak_trim_peak_price = self._v312_peak_price
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
                    target=max(0.0, current_pct - sell_pct),
                    guard=f"{self.VERSION_LABEL}_peak_memory_light_trim",
                ),
            )
        ]

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if str(gate.get("decision", "allow")) in {"block", "defer", "freeze"}:
            return gate
        if not self._should_defer_late_target_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return gate

        self._open_or_update_late_sell_intent(price=price, budget=max_sell)
        return {
            "decision": "freeze",
            "threshold": sell_threshold,
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "late_peak_target_reduce_defer"),
        }

    def _should_peak_memory_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if market["confirmed_state"] != "BULL":
            return False
        if market["raw_state"] not in {"BULL", "MIXED"}:
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False
        if current_pct < self.PEAK_TRIM_MIN_POSITION:
            return False
        if self._call_count - self._last_v312_peak_trim_call <= self.PEAK_TRIM_MIN_GAP_CALLS:
            return False
        if (
            self._last_v312_peak_trim_peak_price > 0
            and self._v312_peak_price < self._last_v312_peak_trim_peak_price * (1 + self.PEAK_TRIM_NEW_PEAK_BUFFER)
        ):
            return False

        peak_price = self._v312_peak_price
        if peak_price <= 0:
            return False
        days_from_peak = self._call_count - self._v312_peak_call
        peak_pullback = 1.0 - price / peak_price
        if not (self.PEAK_TRIM_MIN_DAYS_FROM_PEAK <= days_from_peak <= self.PEAK_TRIM_MAX_DAYS_FROM_PEAK):
            return False
        if not (self.PEAK_TRIM_MIN_PULLBACK <= peak_pullback <= self.PEAK_TRIM_MAX_PULLBACK):
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = peak_price / pos.avg_cost - 1.0
        if profit_pct < self.PEAK_TRIM_MIN_PROFIT or peak_profit_pct < self.PEAK_TRIM_MIN_PEAK_PROFIT:
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema72 = self._v3_value(latest, "ema72")
        ema168 = self._v3_value(latest, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema72)
            or pd.isna(ema168)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            ema72 > ema168
            and ema168_slope > 0
            and price_vs_ema72 >= -0.04
            and price_vs_ema168 >= 0.08
            and rolling_pos >= 0.70
            and 0.48 <= donchian_pos <= 0.86
            and roc_20 <= 0.04
            and str(latest.get("btc_regime", "")) != "BEAR"
        )

    def _should_defer_late_target_reduce(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if trend_risk > 2 or risk_score > 3:
            return False
        if self._v312_peak_price <= 0:
            return False
        peak_drawdown = 1.0 - price / self._v312_peak_price
        if peak_drawdown < self.LATE_SELL_DEFER_PEAK_DD:
            return False
        if self._call_count - self._v312_peak_call < self.LATE_SELL_DEFER_MIN_DAYS_FROM_PEAK:
            return False

        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(price_vs_ema168) or pd.isna(rolling_pos) or pd.isna(donchian_pos) or pd.isna(roc_20):
            return False
        return bool(
            rolling_pos <= 0.52
            and donchian_pos <= 0.42
            and price_vs_ema168 <= 0.04
            and roc_20 <= 0.08
        )

    def _open_or_update_late_sell_intent(self, *, price: float, budget: float) -> None:
        budget_pct = min(max(float(budget), 0.0), self.LATE_SELL_INTENT_MAX_SELL)
        if budget_pct <= 0.0:
            return
        if self._pending_late_sell_intent_active():
            assert self._pending_intent is not None
            self._pending_intent["created_price"] = min(
                float(self._pending_intent.get("created_price", price)),
                float(price),
            )
            self._pending_intent["budget_pct"] = max(
                float(self._pending_intent.get("budget_pct", 0.0)),
                budget_pct,
            )
            return
        self._pending_intent = {
            "side": "sell",
            "source": "late-target-reduce",
            "created_call": self._call_count,
            "created_price": float(price),
            "anchor_price": float(price),
            "budget_pct": budget_pct,
        }

    def _maybe_fill_late_sell_intent(
        self,
        position: dict,
        market: dict,
        band: dict,
    ) -> Action | None:
        if not self._pending_late_sell_intent_active():
            return None
        if market["trend_risk"] >= 3 or market["risk_score"] >= 4 or market["raw_state"] == "BEAR":
            self._clear_pending_intent()
            return None
        latest = position["latest"]
        price = position["price"]
        if not self._should_release_late_sell_intent(latest, price):
            return None

        sell_target = band["sell_boundary"]
        gap = position["current_pct"] - sell_target
        if gap < self.LATE_SELL_INTENT_MIN_GAP:
            self._clear_pending_intent()
            return None

        assert self._pending_intent is not None
        sell_pct = min(gap, float(self._pending_intent.get("budget_pct", 0.0)), self.LATE_SELL_INTENT_MAX_SELL)
        sell_qty = min(position["total_value"] * sell_pct / price, position["pos"].quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return None

        self._clear_pending_intent()
        self._record_executed_action(side="sell", setup="target-reduce")
        return Action(
            symbol=position["symbol"],
            side="sell",
            quantity=sell_qty,
            price=price,
            reason=self._build_action_reason(
                side="sell",
                setup="target-reduce",
                risk_score=market["risk_score"],
                trend_risk=market["trend_risk"],
                drawdown_risk=market["drawdown_risk"],
                raw_state=market["raw_state"],
                confirmed_state=market["confirmed_state"],
                target=sell_target,
                guard=f"{self.VERSION_LABEL}_late_sell_intent_rebound",
            ),
        )

    def _should_release_late_sell_intent(self, latest: pd.Series, price: float) -> bool:
        if self._pending_intent is None:
            return False
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(price_vs_ema72) or pd.isna(donchian_pos) or pd.isna(roc_20):
            return False
        created_price = float(self._pending_intent.get("created_price", 0.0))
        rebound_from_created = created_price > 0 and price >= created_price * self.LATE_SELL_REBOUND_FROM_CREATED
        recovered_to_mean = price_vs_ema72 >= -0.01 and donchian_pos >= 0.48 and roc_20 >= -0.02
        return bool(rebound_from_created or recovered_to_mean)

    def _pending_late_sell_intent_active(self) -> bool:
        return (
            self._pending_intent is not None
            and self._pending_intent.get("side") == "sell"
            and self._pending_intent.get("source") == "late-target-reduce"
            and not self._pending_late_sell_intent_expired()
        )

    def _pending_late_sell_intent_expired(self) -> bool:
        if self._pending_intent is None:
            return False
        if self._pending_intent.get("side") != "sell":
            return False
        return self._call_count - int(self._pending_intent.get("created_call", -10_000)) > self.LATE_SELL_INTENT_MAX_CALLS

    def _clear_late_sell_intent_if_stale(self) -> None:
        if self._pending_intent is None or self._pending_intent.get("side") != "sell":
            return
        if self._pending_late_sell_intent_expired():
            self._clear_pending_intent()


class V312BStrategy(V312AStrategy):
    """V3.12B: stricter peak trim and narrower low-repair sell defer."""

    VERSION_LABEL = "v3_12B"
    LATE_SELL_DEFER_PEAK_DD = 0.25

    @property
    def name(self) -> str:
        return "v3_12B"

    def _should_peak_memory_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if not super()._should_peak_memory_trim(latest, price, pos, current_pct, market):
            return False

        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema24_slope = self._v3_value(latest, "ema24_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(price_vs_ema168) or pd.isna(ema24_slope) or pd.isna(rolling_pos) or pd.isna(roc_20):
            return False
        return bool(
            0.18 <= price_vs_ema168 <= 0.55
            and rolling_pos >= 0.78
            and roc_20 <= -0.03
            and ema24_slope <= 0.02
        )

    def _should_defer_late_target_reduce(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if not super()._should_defer_late_target_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return False
        if trend_risk != 2 or drawdown_risk != 0 or risk_score != 2:
            return False

        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        if pd.isna(price_vs_ema168) or pd.isna(rolling_pos) or pd.isna(donchian_pos):
            return False
        return bool(
            rolling_pos <= 0.48
            and donchian_pos <= 0.25
            and price_vs_ema168 <= 0.00
        )


class V312CStrategy(V312BStrategy):
    """V3.12C: peak trim with post-trim high-price rebuy protection."""

    VERSION_LABEL = "v3_12C"
    PEAK_TRIM_REBUY_GUARD_CALLS = 24
    PEAK_TRIM_REBUY_MAX_PRICE_MULT = 0.99

    @property
    def name(self) -> str:
        return "v3_12C"

    def _should_peak_memory_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if not super()._should_peak_memory_trim(latest, price, pos, current_pct, market):
            return False
        ema24_slope = self._v3_value(latest, "ema24_slope")
        if pd.isna(ema24_slope):
            return False
        return bool(ema24_slope <= 0.0)

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._post_peak_trim_rebuy_guard_active():
            return gate
        if buy_setup not in {"target-gap", "safe-recovery", "trend-cont"}:
            return gate
        if price <= self._last_v312_peak_trim_sell_price * self.PEAK_TRIM_REBUY_MAX_PRICE_MULT:
            return gate
        if self._is_post_peak_fresh_breakout(latest, price, raw_state, confirmed_state):
            return gate
        return {
            "decision": "defer",
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "post_peak_trim_high_rebuy_defer"),
        }

    def _post_peak_trim_rebuy_guard_active(self) -> bool:
        return (
            self._last_v312_peak_trim_call > 0
            and self._last_v312_peak_trim_sell_price > 0
            and self._call_count - self._last_v312_peak_trim_call <= self.PEAK_TRIM_REBUY_GUARD_CALLS
        )

    def _is_post_peak_fresh_breakout(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        if pd.isna(rolling_pos) or pd.isna(donchian_pos) or pd.isna(roc_20) or pd.isna(price_vs_ema72):
            return False
        return bool(
            rolling_pos <= 0.65
            and donchian_pos >= 0.82
            and roc_20 >= 0.16
            and price_vs_ema72 <= 0.12
        )


class V313AStrategy(V312CStrategy):
    """V3.13A: staged deep-drawdown risk sells over V3.12C."""

    VERSION_LABEL = "v3_13A"
    DEEP_RISK_STAGE_PEAK_DD = 0.35
    DEEP_RISK_STAGE_MAX_SELL = 0.14

    @property
    def name(self) -> str:
        return "v3_13A"

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        gate = super()._evaluate_sell_timing_gate(
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
        if str(gate.get("decision", "allow")) in {"block", "defer", "freeze"}:
            return gate
        if not self._should_stage_deep_risk_reduce(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return gate

        existing_cap = gate.get("max_pct_cap")
        staged_cap = self.DEEP_RISK_STAGE_MAX_SELL if existing_cap is None else min(
            float(existing_cap),
            self.DEEP_RISK_STAGE_MAX_SELL,
        )
        return {
            "decision": str(gate.get("decision", "allow")),
            "threshold": float(gate.get("threshold", sell_threshold)),
            "max_pct_mult": float(gate.get("max_pct_mult", 1.0)),
            "max_pct_cap": staged_cap,
            "guard": self._join_guard(str(gate.get("guard", "")), "deep_risk_reduce_staged"),
        }

    def _should_stage_deep_risk_reduce(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "risk-reduce":
            return False
        if raw_state == "BEAR" or confirmed_state == "BEAR":
            return False
        if trend_risk != 2 or drawdown_risk != 2 or risk_score != 4:
            return False
        if self._v312_peak_price <= 0:
            return False
        peak_drawdown = 1.0 - price / self._v312_peak_price
        if peak_drawdown < self.DEEP_RISK_STAGE_PEAK_DD:
            return False

        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        roc_20 = self._v3_value(latest, "roc_20")
        if pd.isna(rolling_pos) or pd.isna(donchian_pos) or pd.isna(price_vs_ema168) or pd.isna(roc_20):
            return False
        return bool(
            rolling_pos <= 0.50
            and donchian_pos <= 0.25
            and price_vs_ema168 <= 0.02
            and roc_20 <= 0.08
        )


class V313BStrategy(V313AStrategy):
    """V3.13B: staged deep risk sell with a short no-repeat guard."""

    VERSION_LABEL = "v3_13B"
    DEEP_RISK_STAGE_GUARD_CALLS = 10
    DEEP_RISK_STAGE_BREAKDOWN_MULT = 0.90

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_deep_risk_stage_call = -10_000
        self._last_deep_risk_stage_price = 0.0

    @property
    def name(self) -> str:
        return "v3_13B"

    def _evaluate_sell_timing_gate(
        self,
        *,
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
    ) -> dict:
        if (
            sell_setup == "risk-reduce"
            and self._deep_risk_stage_guard_active()
            and not self._deep_risk_stage_breakdown(
                price=price,
                raw_state=raw_state,
                confirmed_state=confirmed_state,
                trend_risk=trend_risk,
            )
        ):
            return {
                "decision": "freeze",
                "threshold": sell_threshold,
                "max_pct_mult": 0.0,
                "max_pct_cap": 0.0,
                "guard": f"{self.VERSION_LABEL}_deep_risk_stage_no_repeat",
            }

        gate = super()._evaluate_sell_timing_gate(
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
        if "deep_risk_reduce_staged" in str(gate.get("guard", "")):
            self._last_deep_risk_stage_call = self._call_count
            self._last_deep_risk_stage_price = float(price)
        return gate

    def _deep_risk_stage_guard_active(self) -> bool:
        return (
            self._last_deep_risk_stage_call > 0
            and self._last_deep_risk_stage_price > 0
            and self._call_count - self._last_deep_risk_stage_call <= self.DEEP_RISK_STAGE_GUARD_CALLS
        )

    def _deep_risk_stage_breakdown(
        self,
        *,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
    ) -> bool:
        return bool(
            raw_state == "BEAR"
            or confirmed_state == "BEAR"
            or trend_risk >= 3
            or price <= self._last_deep_risk_stage_price * self.DEEP_RISK_STAGE_BREAKDOWN_MULT
        )


class V314AStrategy(V312CStrategy):
    """V3.14A: two-stage profit-taking layer over V3.12C."""

    VERSION_LABEL = "v3_14A"
    PROFIT_STAGE2_MIN_WAIT_CALLS = 5
    PROFIT_STAGE2_MAX_CALLS = 34
    PROFIT_STAGE2_SELL_PCT = 0.08
    PROFIT_STAGE2_MIN_POSITION = 0.78
    PROFIT_STAGE2_NEW_PEAK_BUFFER = 0.04
    PROFIT_STAGE2_MIN_REBOUND = 0.03

    @property
    def name(self) -> str:
        return "v3_14A"

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
            return actions
        if not self._should_stage2_profit_trim(latest, price, pos, current_pct, market):
            return []

        sell_pct = min(self.PROFIT_STAGE2_SELL_PCT, current_pct)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return []

        self._last_v312_peak_trim_call = self._call_count
        self._last_v312_peak_trim_sell_price = price
        self._last_v312_peak_trim_peak_price = max(self._last_v312_peak_trim_peak_price, self._v312_peak_price)
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
                    target=max(0.0, current_pct - sell_pct),
                    guard=f"{self.VERSION_LABEL}_profit_stage2_failed_rebound_trim",
                ),
            )
        ]

    def _should_stage2_profit_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if self._last_v312_peak_trim_call <= 0 or self._last_v312_peak_trim_sell_price <= 0:
            return False
        age = self._call_count - self._last_v312_peak_trim_call
        if age < self.PROFIT_STAGE2_MIN_WAIT_CALLS or age > self.PROFIT_STAGE2_MAX_CALLS:
            return False
        if current_pct < self.PROFIT_STAGE2_MIN_POSITION:
            return False
        if market["raw_state"] == "BEAR" or market["confirmed_state"] == "BEAR":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False

        peak_price = self._v312_peak_price
        if peak_price <= 0:
            return False
        if (
            self._last_v312_peak_trim_peak_price > 0
            and peak_price >= self._last_v312_peak_trim_peak_price * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER)
        ):
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = peak_price / pos.avg_cost - 1.0
        if profit_pct < self.PEAK_TRIM_MIN_PROFIT or peak_profit_pct < self.PEAK_TRIM_MIN_PEAK_PROFIT:
            return False
        if price < self._last_v312_peak_trim_sell_price * (1 - self.PEAK_TRIM_MIN_PULLBACK):
            return False

        rebound_from_trim = price >= self._last_v312_peak_trim_sell_price * (1 + self.PROFIT_STAGE2_MIN_REBOUND)
        failed_near_trim = price >= self._last_v312_peak_trim_sell_price * 0.98
        if not (rebound_from_trim or failed_near_trim):
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema24_slope = self._v3_value(latest, "ema24_slope")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema24_slope)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            ema168_slope > 0
            and price_vs_ema168 >= 0.12
            and price_vs_ema72 <= 0.06
            and rolling_pos >= 0.70
            and 0.45 <= donchian_pos <= 0.78
            and ema24_slope <= 0.0
            and roc_20 <= 0.02
            and str(latest.get("btc_regime", "")) != "BEAR"
        )


class V314BStrategy(V314AStrategy):
    """V3.14B: second-stage trim on high failed-rebound retests after V3.12C peak trims."""

    VERSION_LABEL = "v3_14B"
    PROFIT_STAGE2_MAX_CALLS = 18
    PROFIT_STAGE2_MIN_REBOUND = 0.02
    PROFIT_STAGE2_MIN_POSITION = 0.82

    @property
    def name(self) -> str:
        return "v3_14B"

    def _should_stage2_profit_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if self._last_v312_peak_trim_call <= 0 or self._last_v312_peak_trim_sell_price <= 0:
            return False
        age = self._call_count - self._last_v312_peak_trim_call
        if age < self.PROFIT_STAGE2_MIN_WAIT_CALLS or age > self.PROFIT_STAGE2_MAX_CALLS:
            return False
        if current_pct < self.PROFIT_STAGE2_MIN_POSITION:
            return False
        if market["raw_state"] == "BEAR" or market["confirmed_state"] == "BEAR":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False

        peak_price = self._v312_peak_price
        if peak_price <= 0:
            return False
        if (
            self._last_v312_peak_trim_peak_price > 0
            and peak_price >= self._last_v312_peak_trim_peak_price * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER)
        ):
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = peak_price / pos.avg_cost - 1.0
        if profit_pct < self.PEAK_TRIM_MIN_PROFIT or peak_profit_pct < self.PEAK_TRIM_MIN_PEAK_PROFIT:
            return False
        if price < self._last_v312_peak_trim_sell_price * (1 + self.PROFIT_STAGE2_MIN_REBOUND):
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            ema168_slope > 0
            and 0.08 <= price_vs_ema72 <= 0.18
            and price_vs_ema168 >= 0.18
            and rolling_pos >= 0.88
            and 0.82 <= donchian_pos <= 0.94
            and roc_20 <= 0.12
            and str(latest.get("btc_regime", "")) != "BEAR"
        )


class V314CStrategy(V314BStrategy):
    """V3.14C: V3.14B with cycle-local peak anchors and full post-stage2 rebuy guard."""

    VERSION_LABEL = "v3_14C"
    PROFIT_STAGE2_REBUY_GUARD_CALLS = 24
    PROFIT_STAGE2_REBUY_DISCOUNT = 0.88

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v314_stage1_anchor_peak_price = 0.0
        self._last_v314_stage2_trim_call = -10_000
        self._last_v314_stage2_trim_sell_price = 0.0
        self._last_v314_stage2_anchor_peak_price = 0.0

    @property
    def name(self) -> str:
        return "v3_14C"

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

        reason = str(actions[0].reason)
        if "profit_stage2_failed_rebound_trim" in reason:
            self._last_v314_stage2_trim_call = self._call_count
            self._last_v314_stage2_trim_sell_price = float(price)
            self._last_v314_stage2_anchor_peak_price = self._v314_stage1_anchor_peak_price
        elif "peak_memory_light_trim" in reason or "mature_bull_giveback_trim" in reason:
            self._v314_stage1_anchor_peak_price = self._v312_peak_price
        return actions

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._post_stage2_trim_rebuy_guard_active():
            return gate
        if buy_setup not in {"target-gap", "safe-recovery", "trend-cont", "pullback"}:
            return gate
        if price <= self._last_v314_stage2_trim_sell_price * self.PROFIT_STAGE2_REBUY_DISCOUNT:
            return gate
        if self._is_stage2_true_breakout(latest, price, raw_state, confirmed_state):
            return gate
        return {
            "decision": "defer",
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "post_stage2_trim_near_rebuy_defer"),
        }

    def _should_stage2_profit_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if self._last_v312_peak_trim_call <= 0 or self._last_v312_peak_trim_sell_price <= 0:
            return False
        age = self._call_count - self._last_v312_peak_trim_call
        if age < self.PROFIT_STAGE2_MIN_WAIT_CALLS or age > self.PROFIT_STAGE2_MAX_CALLS:
            return False
        if current_pct < self.PROFIT_STAGE2_MIN_POSITION:
            return False
        if market["raw_state"] == "BEAR" or market["confirmed_state"] == "BEAR":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False

        peak_price = self._v312_peak_price
        anchor_peak = self._v314_stage1_anchor_peak_price or self._last_v312_peak_trim_peak_price
        if peak_price <= 0 or anchor_peak <= 0:
            return False
        if peak_price >= anchor_peak * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER):
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = peak_price / pos.avg_cost - 1.0
        if profit_pct < self.PEAK_TRIM_MIN_PROFIT or peak_profit_pct < self.PEAK_TRIM_MIN_PEAK_PROFIT:
            return False
        if price < self._last_v312_peak_trim_sell_price * (1 + self.PROFIT_STAGE2_MIN_REBOUND):
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            ema168_slope > 0
            and 0.08 <= price_vs_ema72 <= 0.18
            and price_vs_ema168 >= 0.18
            and rolling_pos >= 0.88
            and 0.82 <= donchian_pos <= 0.94
            and roc_20 <= 0.12
            and str(latest.get("btc_regime", "")) != "BEAR"
        )

    def _post_stage2_trim_rebuy_guard_active(self) -> bool:
        return (
            self._last_v314_stage2_trim_call > 0
            and self._last_v314_stage2_trim_sell_price > 0
            and self._call_count - self._last_v314_stage2_trim_call <= self.PROFIT_STAGE2_REBUY_GUARD_CALLS
        )

    def _is_stage2_true_breakout(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        anchor_peak = self._last_v314_stage2_anchor_peak_price
        if anchor_peak <= 0 or price < anchor_peak * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER):
            return False
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        if pd.isna(donchian_pos) or pd.isna(roc_20) or pd.isna(price_vs_ema72):
            return False
        return bool(donchian_pos >= 0.90 and roc_20 >= 0.12 and price_vs_ema72 <= 0.18)


class V314DStrategy(V314CStrategy):
    """V3.14D: stricter post-profit-trim rebuy discount guard over V3.14C."""

    VERSION_LABEL = "v3_14D"
    PROFIT_STAGE2_REBUY_GUARD_CALLS = 34

    @property
    def name(self) -> str:
        return "v3_14D"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        if not self._post_profit_trim_rebuy_guard_active():
            return gate
        if buy_setup not in {"target-gap", "safe-recovery", "trend-cont", "pullback"}:
            return gate
        if price <= self._last_v312_peak_trim_sell_price * self.PROFIT_STAGE2_REBUY_DISCOUNT:
            return gate
        if self._is_stage2_true_breakout(latest, price, raw_state, confirmed_state):
            return gate
        return {
            "decision": "defer",
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), "post_profit_trim_discount_rebuy_defer"),
        }

    def _post_profit_trim_rebuy_guard_active(self) -> bool:
        return (
            self._last_v312_peak_trim_call > 0
            and self._last_v312_peak_trim_sell_price > 0
            and self._call_count - self._last_v312_peak_trim_call <= self.PROFIT_STAGE2_REBUY_GUARD_CALLS
        )


class V315AStrategy(V314CStrategy):
    """V3.15A: V3.14D behavior through a unified post-sell buy permission layer."""

    VERSION_LABEL = "v3_15A"
    PROFIT_STAGE2_REBUY_GUARD_CALLS = 34

    @property
    def name(self) -> str:
        return "v3_15A"

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = super()._evaluate_buy_timing_gate(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        return self._evaluate_buy_permission(
            gate=gate,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        )

    def _maybe_fill_pending_buy_intent(
        self,
        position: dict,
        market: dict,
        band: dict,
    ) -> Action | None:
        fill = super()._maybe_fill_pending_buy_intent(position, market, band)
        if fill is None:
            return None
        parsed = self._parse_action_context(fill)
        gate = self._evaluate_buy_permission(
            gate={"decision": "allow", "max_pct_mult": 1.0, "max_pct_cap": None, "guard": ""},
            latest=position["latest"],
            price=position["price"],
            raw_state=market["raw_state"],
            confirmed_state=market["confirmed_state"],
            buy_setup=str(parsed.get("setup", "target-gap")),
        )
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return None
        return fill

    def _evaluate_buy_permission(
        self,
        *,
        gate: dict,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> dict:
        if str(gate.get("decision", "allow")) in {"block", "defer"}:
            return gate
        deny_guard = self._post_profit_trim_rebuy_deny_guard(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        )
        if not deny_guard:
            return gate
        return {
            "decision": "defer",
            "max_pct_mult": 0.0,
            "max_pct_cap": 0.0,
            "guard": self._join_guard(str(gate.get("guard", "")), deny_guard),
        }

    def _post_profit_trim_rebuy_deny_guard(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> str:
        if not self._post_profit_trim_rebuy_guard_active():
            return ""
        if buy_setup not in {"target-gap", "safe-recovery", "trend-cont", "pullback"}:
            return ""
        if price <= self._last_v312_peak_trim_sell_price * self.PROFIT_STAGE2_REBUY_DISCOUNT:
            return ""
        if self._is_stage2_true_breakout(latest, price, raw_state, confirmed_state):
            return ""
        return "post_profit_trim_discount_rebuy_defer"

    def _post_profit_trim_rebuy_guard_active(self) -> bool:
        return (
            self._last_v312_peak_trim_call > 0
            and self._last_v312_peak_trim_sell_price > 0
            and self._call_count - self._last_v312_peak_trim_call <= self.PROFIT_STAGE2_REBUY_GUARD_CALLS
        )


class V315BStrategy(V315AStrategy):
    """V3.15B: consolidate profit-taking path memory into a profit cycle state."""

    VERSION_LABEL = "v3_15B"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profit_cycle = self._empty_profit_cycle()

    @property
    def name(self) -> str:
        return "v3_15B"

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
            self._record_profit_cycle_from_action(actions[0], price=price)
        return actions

    def _evaluate_buy_timing_gate(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
        max_buy: float,
    ) -> dict:
        gate = V314BStrategy._evaluate_buy_timing_gate(
            self,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
            max_buy=max_buy,
        )
        return self._evaluate_buy_permission(
            gate=gate,
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            buy_setup=buy_setup,
        )

    def _should_stage2_profit_trim(
        self,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        market: dict,
    ) -> bool:
        cycle = self._profit_cycle
        if pos.quantity <= 1e-12 or pos.avg_cost <= 0:
            return False
        if not cycle.get("active") or int(cycle.get("stage", 0)) != 1:
            return False
        stage1_call = int(cycle.get("stage1_call", -10_000))
        stage1_sell_price = float(cycle.get("stage1_sell_price", 0.0))
        if stage1_call <= 0 or stage1_sell_price <= 0:
            return False
        age = self._call_count - stage1_call
        if age < self.PROFIT_STAGE2_MIN_WAIT_CALLS or age > self.PROFIT_STAGE2_MAX_CALLS:
            return False
        if current_pct < self.PROFIT_STAGE2_MIN_POSITION:
            return False
        if market["raw_state"] == "BEAR" or market["confirmed_state"] == "BEAR":
            return False
        if market["trend_risk"] > 1 or market["drawdown_risk"] > 0:
            return False

        peak_price = self._v312_peak_price
        anchor_peak = float(cycle.get("stage1_anchor_peak", 0.0))
        if peak_price <= 0 or anchor_peak <= 0:
            return False
        if peak_price >= anchor_peak * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER):
            return False

        profit_pct = price / pos.avg_cost - 1.0
        peak_profit_pct = peak_price / pos.avg_cost - 1.0
        if profit_pct < self.PEAK_TRIM_MIN_PROFIT or peak_profit_pct < self.PEAK_TRIM_MIN_PEAK_PROFIT:
            return False
        if price < stage1_sell_price * (1 + self.PROFIT_STAGE2_MIN_REBOUND):
            return False

        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        ema168_slope = self._v3_value(latest, "ema168_slope")
        rolling_pos = self._v3_value(latest, "rolling_365d_pos")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(price_vs_ema72)
            or pd.isna(price_vs_ema168)
            or pd.isna(ema168_slope)
            or pd.isna(rolling_pos)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
        ):
            return False
        return bool(
            ema168_slope > 0
            and 0.08 <= price_vs_ema72 <= 0.18
            and price_vs_ema168 >= 0.18
            and rolling_pos >= 0.88
            and 0.82 <= donchian_pos <= 0.94
            and roc_20 <= 0.12
            and str(latest.get("btc_regime", "")) != "BEAR"
        )

    def _post_profit_trim_rebuy_guard_active(self) -> bool:
        cycle = self._profit_cycle
        return (
            bool(cycle.get("active"))
            and int(cycle.get("last_trim_call", -10_000)) > 0
            and float(cycle.get("last_trim_sell_price", 0.0)) > 0
            and self._call_count - int(cycle.get("last_trim_call", -10_000)) <= self.PROFIT_STAGE2_REBUY_GUARD_CALLS
        )

    def _post_profit_trim_rebuy_deny_guard(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
        buy_setup: str,
    ) -> str:
        if not self._post_profit_trim_rebuy_guard_active():
            return ""
        if buy_setup not in {"target-gap", "safe-recovery", "trend-cont", "pullback"}:
            return ""
        sell_price = float(self._profit_cycle.get("last_trim_sell_price", 0.0))
        if sell_price <= 0 or price <= sell_price * self.PROFIT_STAGE2_REBUY_DISCOUNT:
            return ""
        if self._is_stage2_true_breakout(latest, price, raw_state, confirmed_state):
            return ""
        return "post_profit_trim_discount_rebuy_defer"

    def _is_stage2_true_breakout(
        self,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str | None,
    ) -> bool:
        if raw_state != "BULL" or confirmed_state != "BULL":
            return False
        anchor_peak = float(self._profit_cycle.get("stage2_anchor_peak", 0.0))
        if anchor_peak <= 0 or price < anchor_peak * (1 + self.PROFIT_STAGE2_NEW_PEAK_BUFFER):
            return False
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        if pd.isna(donchian_pos) or pd.isna(roc_20) or pd.isna(price_vs_ema72):
            return False
        return bool(donchian_pos >= 0.90 and roc_20 >= 0.12 and price_vs_ema72 <= 0.18)

    def _record_profit_cycle_from_action(self, action: Action, *, price: float) -> None:
        reason = str(getattr(action, "reason", ""))
        if "profit_stage2_failed_rebound_trim" in reason:
            stage1_anchor = float(self._profit_cycle.get("stage1_anchor_peak", 0.0))
            self._profit_cycle.update(
                {
                    "active": True,
                    "stage": 2,
                    "last_trim_call": self._call_count,
                    "last_trim_sell_price": float(price),
                    "stage2_call": self._call_count,
                    "stage2_sell_price": float(price),
                    "stage2_anchor_peak": stage1_anchor,
                }
            )
            return
        if "peak_memory_light_trim" in reason:
            self._profit_cycle.update(
                {
                    "active": True,
                    "stage": 1,
                    "last_trim_call": self._call_count,
                    "last_trim_sell_price": float(price),
                    "stage1_call": self._call_count,
                    "stage1_sell_price": float(price),
                    "stage1_anchor_peak": float(self._v312_peak_price),
                }
            )

    @staticmethod
    def _empty_profit_cycle() -> dict:
        return {
            "active": False,
            "stage": 0,
            "last_trim_call": -10_000,
            "last_trim_sell_price": 0.0,
            "stage1_call": -10_000,
            "stage1_sell_price": 0.0,
            "stage1_anchor_peak": 0.0,
            "stage2_call": -10_000,
            "stage2_sell_price": 0.0,
            "stage2_anchor_peak": 0.0,
        }


class V315CStrategy(V315BStrategy):
    """V3.15C: route pending buy fills through an explicit candidate/decision flow."""

    VERSION_LABEL = "v3_15C"

    @property
    def name(self) -> str:
        return "v3_15C"

    def _maybe_fill_pending_buy_intent(
        self,
        position: dict,
        market: dict,
        band: dict,
    ) -> Action | None:
        latest = position["latest"]
        price = position["price"]
        if self._should_cancel_pending_buy_intent(market):
            self._clear_pending_intent()
            return None
        if self._pending_buy_intent_expired():
            self._clear_pending_intent()
            return None

        release = self._classify_pending_buy_release(latest, price, market)
        if release == "":
            return None

        buy_target = band["buy_boundary"]
        gap = buy_target - position["current_pct"]
        if gap < self.INTENT_MIN_GAP:
            self._clear_pending_intent()
            return None

        candidate = self._build_pending_buy_candidate(
            position=position,
            market=market,
            buy_target=buy_target,
            gap=gap,
            release=release,
        )
        if candidate is None:
            return None

        decision = self._decide_buy_candidate(candidate, position=position, market=market)
        if decision["decision"] != "allow":
            return None

        self._last_buy_call = self._call_count
        self._clear_pending_intent()
        return self._action_from_buy_candidate(candidate, decision)

    def _build_pending_buy_candidate(
        self,
        *,
        position: dict,
        market: dict,
        buy_target: float,
        gap: float,
        release: str,
    ) -> dict | None:
        if self._pending_intent is None:
            return None
        price = float(position["price"])
        release_cap = self.INTENT_MAX_BREAKOUT_BUY if release == "breakout" else self.INTENT_MAX_MR_BUY
        buy_pct = min(gap, float(self._pending_intent.get("budget_pct", 0.0)), release_cap)
        buy_qty = position["total_value"] * buy_pct / price
        if buy_qty <= 1e-12 or buy_qty * price < self.min_notional:
            return None
        return {
            "symbol": position["symbol"],
            "side": "buy",
            "setup": "target-gap",
            "quantity": buy_qty,
            "price": price,
            "target": buy_target,
            "guard": f"{self.VERSION_LABEL}_intent_{release}_fill",
            "risk_score": market["risk_score"],
            "trend_risk": market["trend_risk"],
            "drawdown_risk": market["drawdown_risk"],
            "raw_state": market["raw_state"],
            "confirmed_state": market["confirmed_state"],
        }

    def _decide_buy_candidate(self, candidate: dict, *, position: dict, market: dict) -> dict:
        gate = self._evaluate_buy_permission(
            gate={
                "decision": "allow",
                "max_pct_mult": 1.0,
                "max_pct_cap": None,
                "guard": str(candidate.get("guard", "")),
            },
            latest=position["latest"],
            price=float(position["price"]),
            raw_state=market["raw_state"],
            confirmed_state=market["confirmed_state"],
            buy_setup=str(candidate.get("setup", "target-gap")),
        )
        decision = str(gate.get("decision", "allow"))
        if decision in {"block", "defer"}:
            return {"decision": "defer", "guard": str(gate.get("guard", ""))}
        return {"decision": "allow", "guard": str(gate.get("guard", candidate.get("guard", "")))}

    def _action_from_buy_candidate(self, candidate: dict, decision: dict) -> Action:
        return Action(
            symbol=str(candidate["symbol"]),
            side="buy",
            quantity=float(candidate["quantity"]),
            price=float(candidate["price"]),
            reason=self._build_action_reason(
                side="buy",
                setup=str(candidate["setup"]),
                risk_score=int(candidate["risk_score"]),
                trend_risk=int(candidate["trend_risk"]),
                drawdown_risk=int(candidate["drawdown_risk"]),
                raw_state=str(candidate["raw_state"]),
                confirmed_state=str(candidate["confirmed_state"]),
                target=float(candidate["target"]),
                guard=str(decision.get("guard", candidate.get("guard", ""))),
            ),
        )


class V315DStrategy(V315CStrategy):
    """V3.15D: route regular buys through the same candidate/decision/action shape."""

    VERSION_LABEL = "v3_15D"

    @property
    def name(self) -> str:
        return "v3_15D"

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
        candidate = self._build_regular_buy_candidate(
            symbol=symbol,
            latest=latest,
            price=price,
            current_pct=current_pct,
            total_value=total_value,
            buy_target=buy_target,
            market=market,
            signals=signals,
            trend_continuation=trend_continuation,
            safe_recovery=safe_recovery,
            recovery_override=recovery_override,
            pullback_buy=pullback_buy,
        )
        if candidate is None:
            return []
        decision = self._decide_regular_buy_candidate(candidate, latest=latest, market=market)
        if decision["decision"] != "allow":
            return []
        action = self._action_from_buy_candidate(candidate, decision)
        self._last_buy_call = self._call_count
        self._recovery_calls_remaining = max(0, self._recovery_calls_remaining - 1)
        return [action]

    def _build_regular_buy_candidate(
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
    ) -> dict | None:
        buy_threshold = (
            self.BULL_GUARD_TARGET_GAP_THRESHOLD
            if signals["bull_guard"]
            else self.MIN_ADJUST_THRESHOLD
        )
        if current_pct >= buy_target - buy_threshold:
            return None

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
                return None
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
        guard = str(cooldown_guard)
        guard = self._join_guard(guard, signals["bull_guard_guard"])
        if signals["bull_guard"]:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_bull_guard_target_gap_buy")
        if recovery_override:
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_risk_score_reduced")
            guard = self._join_guard(guard, f"{self.VERSION_LABEL}_recovery_override_small_buy")

        return {
            "symbol": symbol,
            "side": "buy",
            "setup": buy_setup,
            "price": float(price),
            "total_value": float(total_value),
            "gap": float(gap),
            "max_buy": float(max_buy),
            "target": float(buy_target),
            "guard": guard,
            "risk_score": market["risk_score"],
            "trend_risk": market["trend_risk"],
            "drawdown_risk": market["drawdown_risk"],
            "raw_state": market["raw_state"],
            "confirmed_state": market["confirmed_state"],
        }

    def _decide_regular_buy_candidate(self, candidate: dict, *, latest: pd.Series, market: dict) -> dict:
        adjusted_max_buy, buy_guard = self._adjust_buy_execution(
            latest=latest,
            price=float(candidate["price"]),
            raw_state=market["raw_state"],
            buy_setup=str(candidate["setup"]),
            max_buy=float(candidate["max_buy"]),
            confirmed_state=market["confirmed_state"],
        )
        guard = self._join_guard(str(candidate.get("guard", "")), buy_guard)
        buy_pct = min(float(candidate["gap"]), adjusted_max_buy)
        buy_qty = float(candidate["total_value"]) * buy_pct / float(candidate["price"])
        if buy_qty <= 1e-12 or buy_qty * float(candidate["price"]) < self.min_notional:
            return {"decision": "defer", "guard": guard}
        return {
            "decision": "allow",
            "guard": guard,
            "quantity": buy_qty,
        }

    def _action_from_buy_candidate(self, candidate: dict, decision: dict) -> Action:
        quantity = float(decision.get("quantity", candidate.get("quantity", 0.0)))
        return Action(
            symbol=str(candidate["symbol"]),
            side="buy",
            quantity=quantity,
            price=float(candidate["price"]),
            reason=self._build_action_reason(
                side="buy",
                setup=str(candidate["setup"]),
                risk_score=int(candidate["risk_score"]),
                trend_risk=int(candidate["trend_risk"]),
                drawdown_risk=int(candidate["drawdown_risk"]),
                raw_state=str(candidate["raw_state"]),
                confirmed_state=str(candidate["confirmed_state"]),
                target=float(candidate["target"]),
                guard=str(decision.get("guard", candidate.get("guard", ""))),
            ),
        )


class V315EStrategy(V315DStrategy):
    """V3.15E: route regular sells through candidate/decision/action shape."""

    VERSION_LABEL = "v3_15E"

    @property
    def name(self) -> str:
        return "v3_15E"

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
        candidate = self._candidate_from_sell_action(action, market=market)
        if candidate is None:
            return actions
        decision = {
            "decision": "allow",
            "guard": candidate.get("guard", ""),
            "quantity": float(action.quantity),
        }
        return [self._action_from_sell_candidate(candidate, decision)]

    def _build_regular_sell_candidate(
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
    ) -> dict | None:
        sell_setup = self._classify_sell_setup(
            trend_risk=market["trend_risk"],
            risk_score=market["risk_score"],
            latest=latest,
            price=price,
            raw_state=market["raw_state"],
            drawdown_risk=market["drawdown_risk"],
        )

        effective_sell_target = sell_target
        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_pullback(latest, price, market["confirmed_state"], market["trend_risk"]):
                effective_sell_target = max(effective_sell_target, current_pct)
        if sell_setup in ("target-reduce", "risk-reduce"):
            if self._is_bull_sell_blocked(
                market["confirmed_state"],
                market["raw_state"],
                market["trend_risk"],
                market["risk_score"],
                sell_setup,
            ):
                effective_sell_target = max(effective_sell_target, current_pct)

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
        return {
            "symbol": symbol,
            "side": "sell",
            "setup": sell_setup,
            "price": float(price),
            "pos": pos,
            "current_pct": float(current_pct),
            "total_value": float(total_value),
            "target": float(effective_sell_target),
            "threshold": float(sell_threshold),
            "max_sell": float(max_sell),
            "guard": sell_guard,
            "risk_score": market["risk_score"],
            "trend_risk": market["trend_risk"],
            "drawdown_risk": market["drawdown_risk"],
            "raw_state": market["raw_state"],
            "confirmed_state": market["confirmed_state"],
        }

    def _decide_regular_sell_candidate(self, candidate: dict) -> dict:
        if float(candidate["current_pct"]) <= float(candidate["target"]) + float(candidate["threshold"]):
            return {"decision": "defer", "guard": str(candidate.get("guard", ""))}
        gap = float(candidate["current_pct"]) - float(candidate["target"])
        sell_pct = min(gap, float(candidate["max_sell"]))
        sell_qty = min(
            float(candidate["total_value"]) * sell_pct / float(candidate["price"]),
            candidate["pos"].quantity,
        )
        if sell_qty <= 1e-12:
            return {"decision": "defer", "guard": str(candidate.get("guard", ""))}
        return {
            "decision": "allow",
            "guard": str(candidate.get("guard", "")),
            "quantity": sell_qty,
        }

    def _action_from_sell_candidate(self, candidate: dict, decision: dict) -> Action:
        return Action(
            symbol=str(candidate["symbol"]),
            side="sell",
            quantity=float(decision["quantity"]),
            price=float(candidate["price"]),
            reason=self._build_action_reason(
                side="sell",
                setup=str(candidate["setup"]),
                risk_score=int(candidate["risk_score"]),
                trend_risk=int(candidate["trend_risk"]),
                drawdown_risk=int(candidate["drawdown_risk"]),
                raw_state=str(candidate["raw_state"]),
                confirmed_state=str(candidate["confirmed_state"]),
                target=float(candidate["target"]),
                guard=str(decision.get("guard", candidate.get("guard", ""))),
            ),
        )

    def _candidate_from_sell_action(self, action: Action, *, market: dict) -> dict | None:
        parsed = self._parse_action_context(action)
        setup = str(parsed.get("setup", ""))
        if setup not in {"target-reduce", "risk-reduce", "trend-break", "core-override_trend-break"}:
            return None
        return {
            "symbol": action.symbol,
            "side": "sell",
            "setup": setup,
            "price": float(action.price),
            "target": float(parsed.get("target_pct", parsed.get("target", 0.0)) or 0.0),
            "guard": str(parsed.get("guard", "")),
            "risk_score": int(parsed.get("risk_score", market["risk_score"])),
            "trend_risk": int(parsed.get("trend_risk", market["trend_risk"])),
            "drawdown_risk": int(parsed.get("drawdown_risk", market["drawdown_risk"])),
            "raw_state": str(parsed.get("raw_state", market["raw_state"])),
            "confirmed_state": str(parsed.get("confirmed_state", market["confirmed_state"])),
        }

    def _maybe_peak_memory_trim_action(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        market: dict,
    ) -> Action | None:
        if not self._should_peak_memory_trim(latest, price, pos, current_pct, market):
            return None
        sell_pct = min(self.PEAK_TRIM_SELL_PCT, current_pct)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return None

        self._last_v312_peak_trim_call = self._call_count
        self._last_v312_peak_trim_sell_price = price
        self._last_v312_peak_trim_peak_price = self._v312_peak_price
        return Action(
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
                target=max(0.0, current_pct - sell_pct),
                guard=f"{self.VERSION_LABEL}_peak_memory_light_trim",
            ),
        )

    def _maybe_stage2_profit_trim_action(
        self,
        *,
        symbol: str,
        latest: pd.Series,
        price: float,
        pos: PositionState,
        current_pct: float,
        total_value: float,
        market: dict,
    ) -> Action | None:
        if not self._should_stage2_profit_trim(latest, price, pos, current_pct, market):
            return None
        sell_pct = min(self.PROFIT_STAGE2_SELL_PCT, current_pct)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12 or sell_qty * price < self.min_notional:
            return None

        self._last_v312_peak_trim_call = self._call_count
        self._last_v312_peak_trim_sell_price = price
        self._last_v312_peak_trim_peak_price = max(self._last_v312_peak_trim_peak_price, self._v312_peak_price)
        return Action(
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
                target=max(0.0, current_pct - sell_pct),
                guard=f"{self.VERSION_LABEL}_profit_stage2_failed_rebound_trim",
            ),
        )


class V315FStrategy(V315EStrategy):
    """V3.15F: separate position needs from trading-opportunity setup names."""

    VERSION_LABEL = "v3_15F"

    @property
    def name(self) -> str:
        return "v3_15F"

    @staticmethod
    def _buy_candidate_metadata(setup: str) -> dict:
        if setup == "target-gap":
            return {
                "candidate_type": "position_buy_need",
                "intent_type": "position_need",
                "source_setup": "position_buy_need",
                "public_setup": "target-gap",
                "priority": 100,
            }
        return {
            "candidate_type": "buy_opportunity",
            "intent_type": "trade_opportunity",
            "source_setup": setup,
            "public_setup": setup,
            "priority": 50,
        }

    @staticmethod
    def _sell_candidate_metadata(setup: str) -> dict:
        if setup == "target-reduce":
            return {
                "candidate_type": "position_sell_need",
                "intent_type": "position_need",
                "source_setup": "position_sell_need",
                "public_setup": "target-reduce",
                "priority": 100,
            }
        return {
            "candidate_type": "sell_opportunity",
            "intent_type": "trade_opportunity",
            "source_setup": setup,
            "public_setup": setup,
            "priority": 50,
        }

    def _build_regular_buy_candidate(
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
    ) -> dict | None:
        candidate = super()._build_regular_buy_candidate(
            symbol=symbol,
            latest=latest,
            price=price,
            current_pct=current_pct,
            total_value=total_value,
            buy_target=buy_target,
            market=market,
            signals=signals,
            trend_continuation=trend_continuation,
            safe_recovery=safe_recovery,
            recovery_override=recovery_override,
            pullback_buy=pullback_buy,
        )
        if candidate is None:
            return None
        setup = str(candidate["setup"])
        candidate.update(self._buy_candidate_metadata(setup))
        candidate["target_pct"] = float(buy_target)
        candidate["current_pct"] = float(current_pct)
        candidate["gap"] = float(buy_target - current_pct)
        return candidate

    def _build_regular_sell_candidate(
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
    ) -> dict | None:
        candidate = super()._build_regular_sell_candidate(
            symbol=symbol,
            latest=latest,
            price=price,
            pos=pos,
            current_pct=current_pct,
            total_value=total_value,
            sell_target=sell_target,
            market=market,
        )
        if candidate is None:
            return None
        setup = str(candidate["setup"])
        candidate.update(self._sell_candidate_metadata(setup))
        candidate["target_pct"] = float(candidate["target"])
        candidate["gap"] = float(current_pct - float(candidate["target"]))
        return candidate

    def _candidate_from_sell_action(self, action: Action, *, market: dict) -> dict | None:
        candidate = super()._candidate_from_sell_action(action, market=market)
        if candidate is None:
            return None
        setup = str(candidate["setup"])
        candidate.update(self._sell_candidate_metadata(setup))
        candidate["target_pct"] = float(candidate["target"])
        return candidate

    def _action_from_buy_candidate(self, candidate: dict, decision: dict) -> Action:
        candidate = dict(candidate)
        candidate["setup"] = str(candidate.get("public_setup", candidate.get("setup", "")))
        return super()._action_from_buy_candidate(candidate, decision)

    def _action_from_sell_candidate(self, candidate: dict, decision: dict) -> Action:
        candidate = dict(candidate)
        candidate["setup"] = str(candidate.get("public_setup", candidate.get("setup", "")))
        return super()._action_from_sell_candidate(candidate, decision)


class V316AStrategy(V315FStrategy):
    """V3.16A: soften low-risk recovery-path position_sell_need execution only."""

    VERSION_LABEL = "v3_16A"
    HEALTHY_POSITION_SELL_MIN_THRESHOLD = 0.12
    HEALTHY_POSITION_SELL_MAX_SELL = 0.06

    @property
    def name(self) -> str:
        return "v3_16A"

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
        if not self._is_healthy_position_sell_need(
            latest=latest,
            price=price,
            raw_state=raw_state,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            drawdown_risk=drawdown_risk,
            risk_score=risk_score,
            sell_setup=sell_setup,
        ):
            return threshold, adjusted_max_sell, guard
        return (
            max(threshold, self.HEALTHY_POSITION_SELL_MIN_THRESHOLD),
            min(adjusted_max_sell, self.HEALTHY_POSITION_SELL_MAX_SELL),
            self._join_guard(guard, f"{self.VERSION_LABEL}_healthy_position_sell_need_softened"),
        )

    def _is_healthy_position_sell_need(
        self,
        *,
        latest: pd.Series,
        price: float,
        raw_state: str,
        confirmed_state: str,
        trend_risk: int,
        drawdown_risk: int,
        risk_score: int,
        sell_setup: str,
    ) -> bool:
        if sell_setup != "target-reduce":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        if drawdown_risk != 0 or risk_score > 2:
            return False
        if trend_risk > 2:
            return False
        if str(latest.get("btc_regime", "")) != "BULL":
            return False

        ema24_slope = self._v3_value(latest, "ema24_slope")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        if (
            pd.isna(ema24_slope)
            or pd.isna(donchian_pos)
            or pd.isna(roc_20)
            or price <= 0
        ):
            return False
        return bool(
            ema24_slope >= -0.025
            and donchian_pos >= 0.55
            and roc_20 >= -0.10
        )


class V317AStrategy(V316AStrategy):
    """V3.17A: add transition-context diagnostics without changing execution."""

    VERSION_LABEL = "v3_17A"

    @property
    def name(self) -> str:
        return "v3_17A"

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
        actions = super()._maybe_buy(
            symbol=symbol,
            latest=latest,
            price=price,
            current_pct=current_pct,
            total_value=total_value,
            buy_target=buy_target,
            market=market,
            signals=signals,
            trend_continuation=trend_continuation,
            safe_recovery=safe_recovery,
            recovery_override=recovery_override,
            pullback_buy=pullback_buy,
        )
        return [self._with_transition_guard(action, latest=latest, price=price, market=market) for action in actions]

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
        return [self._with_transition_guard(action, latest=latest, price=price, market=market) for action in actions]

    def _with_transition_guard(
        self,
        action: Action,
        *,
        latest: pd.Series,
        price: float,
        market: dict,
    ) -> Action:
        guard = self._transition_guard(latest=latest, price=price, market=market)
        if not guard:
            return action
        return Action(
            symbol=action.symbol,
            side=action.side,
            quantity=action.quantity,
            price=action.price,
            reason=self._append_action_guard(action, guard),
            order_type=action.order_type,
        )

    def _append_action_guard(self, action: Action, guard: str) -> str:
        parsed = self._parse_action_context(action)
        separator = "-" if parsed.get("guards") else "_"
        return f"{action.reason}{separator}{guard}"

    def _transition_guard(self, *, latest: pd.Series, price: float, market: dict) -> str:
        context = self._transition_context(latest=latest, price=price, market=market)
        if context == "none":
            return ""
        score = self._transition_score(latest=latest, price=price, market=market)
        return f"tc_{context}_s{score}"

    def _transition_context(self, *, latest: pd.Series, price: float, market: dict) -> str:
        raw_state = str(market["raw_state"])
        confirmed_state = str(market["confirmed_state"])
        trend_risk = int(market["trend_risk"])
        drawdown_risk = int(market["drawdown_risk"])
        risk_score = int(market["risk_score"])
        btc_regime = str(latest.get("btc_regime", ""))

        price_vs_ema168 = self._v3_price_vs(latest, price, "ema168")
        price_vs_ema72 = self._v3_price_vs(latest, price, "ema72")
        ema24_slope = self._v3_value(latest, "ema24_slope")
        donchian_pos = self._v3_value(latest, "donchian_pos")
        roc_20 = self._v3_value(latest, "roc_20")
        rolling_365d_pos = self._v3_value(latest, "rolling_365d_pos")

        if any(pd.isna(value) for value in (price_vs_ema168, price_vs_ema72, ema24_slope, donchian_pos, roc_20)):
            return "none"

        if raw_state == "BEAR" or confirmed_state == "BEAR":
            if trend_risk >= 3 or risk_score >= 3 or btc_regime == "BEAR":
                return "failed_recovery"

        if (
            (raw_state == "BULL" or confirmed_state == "BULL")
            and drawdown_risk == 0
            and trend_risk <= 1
            and (price_vs_ema72 >= 0.0 or ema24_slope > 0.0)
        ):
            if (
                not pd.isna(rolling_365d_pos)
                and rolling_365d_pos >= 0.75
                and donchian_pos >= 0.75
                and roc_20 >= 0.08
            ):
                return "mature_uptrend"
            return "confirmed_recovery"

        if (
            (raw_state == "MIXED" or confirmed_state == "MIXED")
            and drawdown_risk <= 1
            and trend_risk <= 2
            and price_vs_ema168 >= -0.08
            and donchian_pos >= 0.45
            and roc_20 >= -0.08
        ):
            return "early_recovery"

        if (
            raw_state != "BEAR"
            and trend_risk <= 2
            and drawdown_risk <= 1
            and price_vs_ema168 >= -0.12
            and donchian_pos >= 0.35
            and roc_20 >= -0.12
        ):
            return "bear_repair"

        return "none"

    def _transition_score(self, *, latest: pd.Series, price: float, market: dict) -> int:
        checks = [
            self._v3_price_vs(latest, price, "ema168") >= -0.08,
            self._v3_price_vs(latest, price, "ema72") >= -0.04,
            self._v3_value(latest, "ema24_slope") >= -0.02,
            self._v3_value(latest, "donchian_pos") >= 0.45,
            self._v3_value(latest, "roc_20") >= -0.08,
            str(latest.get("btc_regime", "")) != "BEAR",
            int(market["drawdown_risk"]) <= 1,
            int(market["trend_risk"]) <= 2,
        ]
        return sum(1 for passed in checks if bool(passed))


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
