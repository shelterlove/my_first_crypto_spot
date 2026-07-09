"""Signal construction for V4.7."""

from __future__ import annotations

import pandas as pd

from ..v42_types import V42Context, V42Regime, V42Signals


class V47SignalEngine:
    def build_signals(self, owner, context: V42Context, regime: V42Regime, episode: dict) -> V42Signals:
        starter = self.starter_signal(context, regime)
        value_recovery = self.value_recovery(owner, context, regime)
        trend_continuation = self.trend_continuation(owner, context, regime)
        distribution_exhaustion = self.distribution_exhaustion(owner, context, regime)
        recovery_quality_ok = self.recovery_quality_ok(owner, context, regime)
        recovery_signal = self.recovery_signal_from_parts(
            owner,
            context=context,
            regime=regime,
            value_recovery=value_recovery,
            trend_continuation=trend_continuation,
        )
        strong_recovery_signal = bool(
            recovery_quality_ok
            or trend_continuation
            or (value_recovery and context.trend_risk <= 1)
        )
        return V42Signals(
            starter=starter,
            value_recovery=value_recovery,
            trend_continuation=trend_continuation,
            distribution_exhaustion=distribution_exhaustion,
            recovery_signal=recovery_signal,
            strong_recovery_signal=strong_recovery_signal,
            recovery_quality_ok=recovery_quality_ok,
        )

    @staticmethod
    def accumulation_signal(episode: dict, signals: V42Signals) -> bool:
        if str(episode.get("state", "NORMAL")) in {"DEFENSE_LOCK", "FAILED_RECOVERY_LOCK", "RECOVERY_TEST"}:
            return signals.recovery_signal
        return signals.accumulation

    @staticmethod
    def starter_signal(context: V42Context, regime: V42Regime) -> bool:
        return bool(
            context.current_pct <= 0.08
            and regime.regime != "BEAR"
            and context.trend_risk <= 2
        )

    @staticmethod
    def value_recovery(owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        if regime.regime == "BEAR" or context.trend_risk > 2 or regime.btc_regime == "BEAR":
            return False
        rolling_pos = owner._value(latest, "rolling_365d_pos")
        donchian_pos = owner._value(latest, "donchian_pos")
        roc_20 = owner._value(latest, "roc_20")
        price_vs_ema168 = regime.price_vs_ema168
        if not pd.isna(roc_20) and roc_20 < -0.15:
            return False
        return bool(
            (not pd.isna(rolling_pos) and rolling_pos <= 0.55)
            or (not pd.isna(donchian_pos) and donchian_pos <= 0.45)
            or (not pd.isna(price_vs_ema168) and price_vs_ema168 <= -0.04)
        )

    @staticmethod
    def trend_continuation(owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        price = context.price
        if regime.regime != "BULL" or context.trend_risk != 0:
            return False
        ema24 = latest.get("ema24")
        ema72 = latest.get("ema72")
        ema168 = latest.get("ema168")
        slope = latest.get("ema168_slope")
        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168) or pd.isna(slope):
            return False
        donchian_pos = owner._value(latest, "donchian_pos", 0.5)
        atr_rank = owner._value(latest, "atr_pct_rank", 0.5)
        return bool(
            price > float(ema24) > float(ema72) > float(ema168)
            and float(slope) > 0.0
            and price / float(ema24) < 1.04
            and (pd.isna(donchian_pos) or donchian_pos < 0.92)
            and (pd.isna(atr_rank) or atr_rank < 0.90)
        )

    def late_trend_continuation_risk(self, owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        rolling_pos = owner._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = owner._value(latest, "donchian_pos", 0.5)
        roc_20 = owner._value(latest, "roc_20", 0.0)
        price_vs_ema72 = regime.price_vs_ema72
        price_vs_ema168 = regime.price_vs_ema168
        hot_location = bool(
            (not pd.isna(rolling_pos) and rolling_pos >= 0.84)
            or (not pd.isna(donchian_pos) and donchian_pos >= 0.88)
            or (not pd.isna(price_vs_ema168) and price_vs_ema168 >= 0.34)
        )
        extended = bool(
            (not pd.isna(price_vs_ema72) and price_vs_ema72 >= 0.16)
            or (not pd.isna(price_vs_ema168) and price_vs_ema168 >= 0.42)
        )
        momentum_dulling = bool(not pd.isna(roc_20) and roc_20 <= 0.08)
        return bool(hot_location and (extended or momentum_dulling or self.distribution_exhaustion(owner, context, regime)))

    @staticmethod
    def distribution_exhaustion(owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        if context.trend_risk >= 2 or context.current_pct < 0.70:
            return False
        donchian_pos = owner._value(latest, "donchian_pos")
        rolling_pos = owner._value(latest, "rolling_365d_pos")
        roc_20 = owner._value(latest, "roc_20")
        atr_rank = owner._value(latest, "atr_pct_rank")
        price_vs_ema72 = regime.price_vs_ema72
        return bool(
            (not pd.isna(donchian_pos) and donchian_pos >= 0.94 and (pd.isna(roc_20) or roc_20 <= 0.10))
            or (not pd.isna(rolling_pos) and rolling_pos >= 0.92 and not pd.isna(roc_20) and roc_20 <= 0.08)
            or (not pd.isna(price_vs_ema72) and price_vs_ema72 >= 0.22 and not pd.isna(roc_20) and roc_20 <= 0.12)
            or (not pd.isna(atr_rank) and atr_rank >= 0.96 and not pd.isna(donchian_pos) and donchian_pos >= 0.86)
        )

    @staticmethod
    def recovery_signal_from_parts(
        owner,
        *,
        context: V42Context,
        regime: V42Regime,
        value_recovery: bool,
        trend_continuation: bool,
    ) -> bool:
        latest = context.latest
        price = context.price
        price_vs_ema72 = regime.price_vs_ema72
        roc_20 = owner._value(latest, "roc_20")
        ema24_slope = owner._value(latest, "ema24_slope")
        donchian_pos = owner._value(latest, "donchian_pos")
        return bool(
            regime.regime in {"BULL", "RANGE", "TRANSITION"}
            and context.trend_risk <= 2
            and regime.btc_regime != "BEAR"
            and (
                value_recovery
                or trend_continuation
                or (not pd.isna(price_vs_ema72) and price_vs_ema72 >= -0.03)
                or (not pd.isna(roc_20) and roc_20 >= -0.06 and not pd.isna(ema24_slope) and ema24_slope >= -0.015)
                or (not pd.isna(donchian_pos) and donchian_pos >= 0.45 and price > 0.0)
            )
        )

    def recovery_signal(self, owner, context: V42Context, regime: V42Regime) -> bool:
        return self.recovery_signal_from_parts(
            owner,
            context=context,
            regime=regime,
            value_recovery=self.value_recovery(owner, context, regime),
            trend_continuation=self.trend_continuation(owner, context, regime),
        )

    def strong_recovery_signal(self, owner, context: V42Context, regime: V42Regime) -> bool:
        return bool(
            self.recovery_quality_ok(owner, context, regime)
            or self.trend_continuation(owner, context, regime)
            or (self.value_recovery(owner, context, regime) and context.trend_risk <= 1)
        )

    @staticmethod
    def recovery_quality_ok(owner, context: V42Context, regime: V42Regime) -> bool:
        latest = context.latest
        atr_rank = owner._value(latest, "atr_pct_rank", 0.5)
        volume_strength = owner._value(latest, "volume_strength", 1.0)
        return bool(
            regime.regime in {"BULL", "RANGE"}
            and regime.btc_regime in {"BULL", "STRONG_BULL", "RANGE"}
            and context.trend_risk <= 1
            and (pd.isna(regime.price_vs_ema72) or regime.price_vs_ema72 >= -0.02)
            and (pd.isna(atr_rank) or atr_rank <= 0.90)
            and (pd.isna(volume_strength) or volume_strength <= 1.35)
        )
