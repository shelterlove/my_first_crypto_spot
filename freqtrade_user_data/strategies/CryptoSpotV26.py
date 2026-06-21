"""Base Freqtrade shell for native crypto_spot_v1 target-position strategies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from freqtrade.persistence import Trade
    from freqtrade.strategy import IStrategy
except ImportError:  # Allows local compile/test without Freqtrade installed.
    Trade = Any

    class IStrategy:  # type: ignore[no-redef]
        pass

from crypto_spot_v1 import strategy_utils
from crypto_spot_v1.freqtrade_adapter import build_native_signal_frame


class CryptoSpotV26(IStrategy):
    """Legacy-named base shell; subclasses select the native strategy version."""

    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = False
    position_adjustment_enable = True
    startup_candle_count = 220

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True

    strategy_name = "v2_6"
    decision_capital = 100.0
    fee_rate = 0.001
    reserve = 20.0
    min_notional = 0.0
    min_delta_pct = 0.02
    bootstrap_position_alignment_enable = True
    bootstrap_position_alignment_max_pct = 0.35
    bootstrap_position_alignment_tag = "bootstrap-position-align"
    pair_allocations = {
        "BTC/USDT": 0.333,
        "ETH/USDT": 0.333,
        "BNB/USDT": 0.334,
    }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        frame = dataframe.copy()
        if "timestamp" not in frame.columns:
            frame["timestamp"] = frame["date"] if "date" in frame.columns else frame.index
        frame = strategy_utils.compute_indicators(frame)
        return build_native_signal_frame(
            pair=metadata["pair"],
            dataframe=frame,
            strategy_name=self.strategy_name,
            capital=self.decision_capital,
            reserve=self.reserve,
            fee_rate=self.fee_rate,
            min_notional=self.min_notional,
            startup_candle_count=self.startup_candle_count,
        )

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        if dataframe.empty:
            return dataframe

        native_action = dataframe.get("native_action", "")
        native_delta = dataframe.get("native_delta_pct", 0.0).astype(float)
        native_current = dataframe.get("native_current_pct", 0.0).astype(float)
        mask = (
            (native_action == "buy")
            & (native_delta >= self.min_delta_pct)
        )
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "enter_tag"] = dataframe.loc[mask, "native_reason"].astype(str).str[:255]
        bootstrap_mask = (
            bool(self.bootstrap_position_alignment_enable)
            & (native_action == "hold")
            & (native_current >= self.min_delta_pct)
        )
        dataframe.loc[bootstrap_mask, "enter_long"] = 1
        dataframe.loc[bootstrap_mask, "enter_tag"] = self.bootstrap_position_alignment_tag
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        if dataframe.empty:
            return dataframe

        native_delta = dataframe.get("native_delta_pct", 0.0).astype(float).abs()
        native_current = dataframe.get("native_current_pct", 0.0).astype(float)
        full_exit_buffer = max(self.min_delta_pct * 0.5, 0.005)
        mask = (
            (dataframe.get("native_action", "") == "sell")
            & (native_delta >= self.min_delta_pct)
            & (native_delta >= (native_current - full_exit_buffer))
        )
        dataframe.loc[mask, "exit_long"] = 1
        dataframe.loc[mask, "exit_tag"] = dataframe.loc[mask, "native_reason"].astype(str).str[:255]
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        dataframe = self._latest_dataframe(pair)
        if dataframe is None:
            return proposed_stake
        action, delta_pct, _, target_pct = self._latest_native_signal(dataframe)
        if action == "buy":
            delta_pct = self._clamp_delta_to_actual_position(
                action=action,
                native_delta_pct=delta_pct,
                native_target_pct=target_pct,
                actual_current_pct=0.0,
            )
        elif self._is_bootstrap_entry(action=action, entry_tag=entry_tag):
            delta_pct = self._latest_bootstrap_delta_pct(dataframe)
        else:
            return 0.0
        if delta_pct < self.min_delta_pct:
            return 0.0
        stake = self._pair_stake_amount(pair, max_stake) * max(0.0, delta_pct)
        if min_stake is not None and 0 < stake < min_stake:
            return 0.0
        return min(stake, max_stake)

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        pair = trade.pair
        dataframe = self._latest_dataframe(pair)
        if dataframe is None or self._already_adjusted_this_candle(trade, current_time):
            return None

        action, delta_pct, reason, target_pct = self._latest_native_signal(dataframe)
        pair_stake = self._pair_stake_amount(pair, max_stake)
        position_value = float(getattr(trade, "amount", 0.0) or 0.0) * current_rate
        actual_current_pct = position_value / pair_stake if pair_stake > 0 else 0.0
        delta_pct = self._clamp_delta_to_actual_position(
            action=action,
            native_delta_pct=delta_pct,
            native_target_pct=target_pct,
            actual_current_pct=actual_current_pct,
        )
        if abs(delta_pct) < self.min_delta_pct:
            return None
        if action == "buy":
            stake = min(max_stake, pair_stake * delta_pct)
            return stake, reason
        if action == "sell":
            sell_value = pair_stake * abs(delta_pct)
            return -min(position_value, sell_value), reason
        return None

    def _latest_dataframe(self, pair: str) -> pd.DataFrame | None:
        data_provider = getattr(self, "dp", None)
        if data_provider is None:
            return None
        dataframe, _ = data_provider.get_analyzed_dataframe(pair, self.timeframe)
        return dataframe if dataframe is not None and not dataframe.empty else None

    @staticmethod
    def _latest_native_action(dataframe: pd.DataFrame) -> tuple[str, float]:
        action, delta_pct, _, _ = CryptoSpotV26._latest_native_signal(dataframe)
        return action, delta_pct

    @staticmethod
    def _latest_native_signal(dataframe: pd.DataFrame) -> tuple[str, float, str, float | None]:
        latest = dataframe.iloc[-1]
        action = str(latest.get("native_action", "hold"))
        delta_pct = float(latest.get("native_delta_pct", 0.0) or 0.0)
        reason = str(latest.get("native_reason", "") or "")[:255]
        target_raw = latest.get("native_target_pct", None)
        target_pct = None
        if target_raw is not None and not pd.isna(target_raw):
            target_pct = float(target_raw)
        return action, delta_pct, reason, target_pct

    def _latest_bootstrap_delta_pct(self, dataframe: pd.DataFrame) -> float:
        latest = dataframe.iloc[-1]
        current_raw = latest.get("native_current_pct", 0.0)
        native_current_pct = 0.0 if pd.isna(current_raw) else float(current_raw or 0.0)
        return min(
            max(0.0, native_current_pct),
            max(0.0, float(self.bootstrap_position_alignment_max_pct)),
        )

    def _is_bootstrap_entry(self, *, action: str, entry_tag: str | None) -> bool:
        return bool(
            self.bootstrap_position_alignment_enable
            and action == "hold"
            and entry_tag == self.bootstrap_position_alignment_tag
        )

    @staticmethod
    def _clamp_delta_to_actual_position(
        *,
        action: str,
        native_delta_pct: float,
        native_target_pct: float | None,
        actual_current_pct: float,
    ) -> float:
        if native_target_pct is None:
            return native_delta_pct
        actual_current_pct = max(0.0, min(1.0, float(actual_current_pct)))
        native_target_pct = max(0.0, min(1.0, float(native_target_pct)))
        if action == "buy":
            actual_gap = max(0.0, native_target_pct - actual_current_pct)
            return min(max(0.0, native_delta_pct), actual_gap)
        if action == "sell":
            actual_excess = max(0.0, actual_current_pct - native_target_pct)
            return -min(abs(native_delta_pct), actual_excess)
        return native_delta_pct

    def _total_stake_amount(self, fallback: float) -> float:
        wallets = getattr(self, "wallets", None)
        if wallets is None:
            return fallback
        total = getattr(wallets, "get_total_stake_amount", lambda: fallback)()
        return float(total or fallback)

    def _pair_stake_amount(self, pair: str, fallback: float) -> float:
        return self._total_stake_amount(fallback) * self._pair_allocation(pair)

    def _pair_allocation(self, pair: str) -> float:
        allocations = getattr(self, "pair_allocations", {}) or {}
        if not isinstance(allocations, dict):
            return 1.0
        if pair in allocations:
            return max(0.0, float(allocations[pair]))
        return 1.0

    @staticmethod
    def _already_adjusted_this_candle(trade: Trade, current_time) -> bool:
        if not hasattr(trade, "date_last_filled_utc"):
            return False
        last_fill = getattr(trade, "date_last_filled_utc", None)
        if last_fill is None:
            return False
        return pd.to_datetime(last_fill, utc=True) >= pd.to_datetime(current_time, utc=True)
