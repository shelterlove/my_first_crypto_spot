"""Freqtrade shell for the native crypto_spot_v1 v2_6 strategy."""

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
from crypto_spot_v1.freqtrade_adapter import build_target_position_decision


class CryptoSpotV26(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = False
    position_adjustment_enable = True
    startup_candle_count = 220

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False

    strategy_name = "v2_6"
    decision_capital = 100.0
    fee_rate = 0.001
    reserve = 20.0
    min_delta_pct = 0.02

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        frame = dataframe.copy()
        if "timestamp" not in frame.columns:
            frame["timestamp"] = frame["date"] if "date" in frame.columns else frame.index
        return strategy_utils.compute_indicators(frame)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        if dataframe.empty:
            return dataframe

        decision = self._decision(metadata["pair"], dataframe, current_position_pct=0.0)
        if decision.action == "buy" and decision.delta_pct >= self.min_delta_pct:
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1
            dataframe.loc[dataframe.index[-1], "enter_tag"] = decision.reason[:255]
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
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
        decision = self._decision(pair, dataframe, current_position_pct=0.0)
        stake = self._total_stake_amount(max_stake) * max(0.0, decision.delta_pct)
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
        if dataframe is None or self._already_adjusted_this_candle(trade, dataframe):
            return None

        current_pct = self._trade_position_pct(trade, current_rate, max_stake)
        decision = self._decision(pair, dataframe, current_position_pct=current_pct)
        if abs(decision.delta_pct) < self.min_delta_pct:
            return None
        if decision.action == "buy":
            return min(max_stake, self._total_stake_amount(max_stake) * decision.delta_pct)
        if decision.action == "sell":
            position_value = float(getattr(trade, "amount", 0.0) or 0.0) * current_rate
            sell_value = self._total_stake_amount(max_stake) * abs(decision.delta_pct)
            return -min(position_value, sell_value)
        return None

    def _decision(self, pair: str, dataframe: pd.DataFrame, current_position_pct: float):
        return build_target_position_decision(
            pair=pair,
            dataframe=dataframe,
            current_position_pct=current_position_pct,
            strategy_name=self.strategy_name,
            capital=self.decision_capital,
            reserve=self.reserve,
            fee_rate=self.fee_rate,
        )

    def _latest_dataframe(self, pair: str) -> pd.DataFrame | None:
        data_provider = getattr(self, "dp", None)
        if data_provider is None:
            return None
        dataframe, _ = data_provider.get_analyzed_dataframe(pair, self.timeframe)
        return dataframe if dataframe is not None and not dataframe.empty else None

    def _trade_position_pct(self, trade: Trade, current_rate: float, fallback_total: float) -> float:
        amount = float(getattr(trade, "amount", 0.0) or 0.0)
        value = amount * current_rate
        denom = max(self._total_stake_amount(fallback_total), 1e-9)
        return max(0.0, min(1.0, value / denom))

    def _total_stake_amount(self, fallback: float) -> float:
        wallets = getattr(self, "wallets", None)
        if wallets is None:
            return fallback
        total = getattr(wallets, "get_total_stake_amount", lambda: fallback)()
        return float(total or fallback)

    @staticmethod
    def _already_adjusted_this_candle(trade: Trade, dataframe: pd.DataFrame) -> bool:
        if not hasattr(trade, "date_last_filled_utc"):
            return False
        last_fill = getattr(trade, "date_last_filled_utc", None)
        if last_fill is None:
            return False
        candle_time = pd.to_datetime(dataframe.iloc[-1].get("timestamp", dataframe.index[-1]), utc=True)
        return pd.to_datetime(last_fill, utc=True) >= candle_time
