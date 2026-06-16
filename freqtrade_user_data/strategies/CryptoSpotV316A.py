"""Freqtrade shell for the native crypto_spot_v1 v3_16A strategy."""

from __future__ import annotations

from CryptoSpotV26 import CryptoSpotV26


class CryptoSpotV316A(CryptoSpotV26):
    """Thin Freqtrade adapter for the v3_16A native strategy."""

    strategy_name = "v3_16A"
    min_notional = 0.0
