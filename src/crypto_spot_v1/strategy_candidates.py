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
