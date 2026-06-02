"""Freqtrade shell for the native crypto_spot_v1 v2_20A strategy."""

from __future__ import annotations

from CryptoSpotV26 import CryptoSpotV26


class CryptoSpotV220A(CryptoSpotV26):
    """Thin Freqtrade adapter for the v2_20A native strategy."""

    strategy_name = "v2_20A"
    min_notional = 0.0
