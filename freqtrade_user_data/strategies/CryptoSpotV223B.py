"""Freqtrade shell for the native crypto_spot_v1 v2_23B strategy."""

from __future__ import annotations

from CryptoSpotV221E import CryptoSpotV221E


class CryptoSpotV223B(CryptoSpotV221E):
    """Thin Freqtrade adapter for the v2_23B native strategy."""

    strategy_name = "v2_23B"
