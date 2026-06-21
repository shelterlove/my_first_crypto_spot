from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_ROOT / "freqtrade_user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from CryptoSpotV26 import CryptoSpotV26  # noqa: E402


class FakeDataProvider:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return self.frame, None


def test_partial_native_sell_does_not_emit_full_exit_signal() -> None:
    strategy = CryptoSpotV26()
    frame = pd.DataFrame(
        {
            "native_action": ["sell", "sell"],
            "native_delta_pct": [-0.10, -0.59],
            "native_current_pct": [0.60, 0.60],
            "native_reason": ["partial", "full"],
        }
    )

    result = strategy.populate_exit_trend(frame, {"pair": "BNB/USDT"})

    assert result.loc[0, "exit_long"] == 0
    assert result.loc[0, "exit_tag"] == ""
    assert result.loc[1, "exit_long"] == 1
    assert result.loc[1, "exit_tag"] == "full"


def test_bootstrap_alignment_emits_entry_on_native_hold_position() -> None:
    strategy = CryptoSpotV26()
    frame = pd.DataFrame(
        {
            "native_action": ["hold"],
            "native_delta_pct": [0.0],
            "native_current_pct": [0.26],
            "native_reason": [""],
        }
    )

    result = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT"})

    assert result.loc[0, "enter_long"] == 1
    assert result.loc[0, "enter_tag"] == "bootstrap-position-align"


def test_bootstrap_alignment_ignores_tiny_native_position() -> None:
    strategy = CryptoSpotV26()
    frame = pd.DataFrame(
        {
            "native_action": ["hold"],
            "native_delta_pct": [0.0],
            "native_current_pct": [0.01],
            "native_reason": [""],
        }
    )

    result = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT"})

    assert result.loc[0, "enter_long"] == 0
    assert result.loc[0, "enter_tag"] == ""


def test_latest_native_signal_preserves_adjustment_reason() -> None:
    frame = pd.DataFrame(
        {
            "native_action": ["sell"],
            "native_delta_pct": [-0.10],
            "native_reason": ["v2_21E_sell_target-reduce_r0"],
        }
    )

    assert CryptoSpotV26._latest_native_signal(frame) == (
        "sell",
        -0.10,
        "v2_21E_sell_target-reduce_r0",
        None,
    )


def test_clamp_sell_delta_to_actual_excess() -> None:
    assert CryptoSpotV26._clamp_delta_to_actual_position(
        action="sell",
        native_delta_pct=-0.25,
        native_target_pct=0.56,
        actual_current_pct=0.40,
    ) == -0.0
    assert CryptoSpotV26._clamp_delta_to_actual_position(
        action="sell",
        native_delta_pct=-0.25,
        native_target_pct=0.56,
        actual_current_pct=0.90,
    ) == -0.25
    assert round(CryptoSpotV26._clamp_delta_to_actual_position(
        action="sell",
        native_delta_pct=-0.25,
        native_target_pct=0.56,
        actual_current_pct=0.66,
    ), 6) == -0.10


def test_clamp_buy_delta_to_actual_gap() -> None:
    assert CryptoSpotV26._clamp_delta_to_actual_position(
        action="buy",
        native_delta_pct=0.30,
        native_target_pct=0.72,
        actual_current_pct=0.80,
    ) == 0.0
    assert round(CryptoSpotV26._clamp_delta_to_actual_position(
        action="buy",
        native_delta_pct=0.30,
        native_target_pct=0.72,
        actual_current_pct=0.50,
    ), 6) == 0.22


def test_pair_allocation_defaults_to_fixed_sleeves() -> None:
    strategy = CryptoSpotV26()

    assert strategy._pair_allocation("BTC/USDT") == 0.333
    assert strategy._pair_allocation("ETH/USDT") == 0.333
    assert strategy._pair_allocation("BNB/USDT") == 0.334
    assert strategy._pair_allocation("SOL/USDT") == 1.0


def test_pair_stake_uses_pair_allocation() -> None:
    strategy = CryptoSpotV26()

    assert round(strategy._pair_stake_amount("BNB/USDT", 1000), 6) == 334.0


def test_bootstrap_delta_uses_native_current_with_cap() -> None:
    strategy = CryptoSpotV26()
    frame = pd.DataFrame({"native_current_pct": [0.50]})

    assert strategy._latest_bootstrap_delta_pct(frame) == 0.35


def test_bootstrap_custom_stake_uses_pair_sleeve() -> None:
    strategy = CryptoSpotV26()
    strategy.dp = FakeDataProvider(pd.DataFrame({
        "native_action": ["hold"],
        "native_delta_pct": [0.0],
        "native_current_pct": [0.26],
        "native_reason": [""],
    }))

    stake = strategy.custom_stake_amount(
        pair="BTC/USDT",
        current_time=None,
        current_rate=100.0,
        proposed_stake=1000.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="bootstrap-position-align",
        side="long",
    )

    assert round(stake, 6) == 86.58


def test_bootstrap_custom_stake_requires_bootstrap_tag() -> None:
    strategy = CryptoSpotV26()
    strategy.dp = FakeDataProvider(pd.DataFrame({
        "native_action": ["hold"],
        "native_delta_pct": [0.0],
        "native_current_pct": [0.26],
        "native_reason": [""],
    }))

    stake = strategy.custom_stake_amount(
        pair="BTC/USDT",
        current_time=None,
        current_rate=100.0,
        proposed_stake=1000.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="unexpected-entry",
        side="long",
    )

    assert stake == 0.0


def test_bootstrap_custom_stake_does_not_buy_on_native_sell() -> None:
    strategy = CryptoSpotV26()
    strategy.dp = FakeDataProvider(pd.DataFrame({
        "native_action": ["sell"],
        "native_delta_pct": [-0.10],
        "native_current_pct": [0.26],
        "native_reason": ["risk-reduce"],
    }))

    stake = strategy.custom_stake_amount(
        pair="BTC/USDT",
        current_time=None,
        current_rate=100.0,
        proposed_stake=1000.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="bootstrap-position-align",
        side="long",
    )

    assert stake == 0.0


if __name__ == "__main__":
    test_partial_native_sell_does_not_emit_full_exit_signal()
    test_bootstrap_alignment_emits_entry_on_native_hold_position()
    test_bootstrap_alignment_ignores_tiny_native_position()
    test_latest_native_signal_preserves_adjustment_reason()
    test_clamp_sell_delta_to_actual_excess()
    test_clamp_buy_delta_to_actual_gap()
    test_pair_allocation_defaults_to_fixed_sleeves()
    test_pair_stake_uses_pair_allocation()
    test_bootstrap_delta_uses_native_current_with_cap()
    test_bootstrap_custom_stake_uses_pair_sleeve()
    test_bootstrap_custom_stake_requires_bootstrap_tag()
    test_bootstrap_custom_stake_does_not_buy_on_native_sell()
    print("Freqtrade shell tests passed")
