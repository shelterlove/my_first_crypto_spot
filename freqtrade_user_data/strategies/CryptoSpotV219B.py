"""Freqtrade shell for the native crypto_spot_v1 v2_19B strategy."""

from __future__ import annotations

from CryptoSpotV26 import CryptoSpotV26


class CryptoSpotV219B(CryptoSpotV26):
    """Thin Freqtrade adapter for the validated v2_19B native strategy."""

    strategy_name = "v2_19B"
    min_notional = 0.0
