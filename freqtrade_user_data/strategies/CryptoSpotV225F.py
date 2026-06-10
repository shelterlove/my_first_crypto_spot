"""Freqtrade shell for the native crypto_spot_v1 v2_25F strategy."""

from __future__ import annotations

from CryptoSpotV221E import CryptoSpotV221E


class CryptoSpotV225F(CryptoSpotV221E):
    """Thin Freqtrade adapter for the v2_25F native strategy."""

    strategy_name = "v2_25F"
