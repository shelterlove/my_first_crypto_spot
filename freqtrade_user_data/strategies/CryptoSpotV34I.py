"""Freqtrade shell for the native crypto_spot_v1 v3_4I strategy."""

from __future__ import annotations

from CryptoSpotV26 import CryptoSpotV26


class CryptoSpotV34I(CryptoSpotV26):
    """Thin Freqtrade adapter for the current paper-trading candidate v3_4I."""

    strategy_name = "v3_4I"
    min_notional = 0.0
