"""Freqtrade shell for the native crypto_spot_v1 v2_22A strategy."""

from __future__ import annotations

from CryptoSpotV221E import CryptoSpotV221E


class CryptoSpotV222A(CryptoSpotV221E):
    """Thin Freqtrade adapter for the v2_22A native strategy."""

    strategy_name = "v2_22A"
