"""Freqtrade shell for the native crypto_spot_v1 v2_21E strategy."""

from __future__ import annotations

from CryptoSpotV26 import CryptoSpotV26


class CryptoSpotV221E(CryptoSpotV26):
    """Thin Freqtrade adapter for the v2_21E native strategy."""

    strategy_name = "v2_21E"
    min_notional = 0.0
