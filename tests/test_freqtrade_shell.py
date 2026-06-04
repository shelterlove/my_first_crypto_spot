from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_ROOT / "freqtrade_user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from CryptoSpotV26 import CryptoSpotV26  # noqa: E402


def test_partial_native_sell_does_not_emit_full_exit_signal() -> None:
    strategy = CryptoSpotV26({})
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
    )


if __name__ == "__main__":
    test_partial_native_sell_does_not_emit_full_exit_signal()
    test_latest_native_signal_preserves_adjustment_reason()
    print("Freqtrade shell tests passed")
