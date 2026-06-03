"""Indicator and market-state helpers for V1 spot strategy code."""
import pandas as pd

from .indicators import add_atr


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators for daily-position-management strategies.

    Core indicators:
      - EMA24/72/168 (1-month / 1-quarter / half-year trends)
      - EMA168 slope (20-bar pct change — long-term trend direction)
      - ATR14 + rolling 365d percentile rank (volatility regime)
      - Donchian position 120d (price location in long-term range)
      - 120d high drawdown (peak erosion)
      - Volume MA20/MA60 ratio (breakout quality, optional column)

    The atr_pct_80 column is retained because accepted V1 behavior was measured
    with this indicator set available.
    """
    df = df.copy()

    # ── Trend ──
    df["ema24"] = df["close"].ewm(span=24, adjust=False).mean()
    df["ema72"] = df["close"].ewm(span=72, adjust=False).mean()
    df["ema168"] = df["close"].ewm(span=168, adjust=False).mean()
    df["ema168_slope"] = df["ema168"].pct_change(20)

    # ── Momentum (for early bull detection — PRE_BULL) ──
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)
    df["ema24_slope"] = df["ema24"].pct_change(5)
    df["ema72_slope"] = df["ema72"].pct_change(5)

    # ── ATR ──
    df = add_atr(df, period=14)
    df["atr_pct"] = df["atr14"] / df["close"]
    # Retained for compatibility with the accepted indicator set.
    df["atr_pct_80"] = (
        df["atr_pct"].rolling(window=200, min_periods=50).quantile(0.8)
    )
    # new: rolling 365d percentile rank
    df["atr_pct_rank"] = (
        df["atr_pct"].rolling(365, min_periods=100).rank(pct=True)
    )

    # ── Donchian position 120d ──
    high_120 = df["high"].rolling(120).max()
    low_120 = df["low"].rolling(120).min()
    denom = (high_120 - low_120).replace(0, float("nan"))
    df["donchian_pos"] = (df["close"] - low_120) / denom
    df["donchian_pos"] = df["donchian_pos"].fillna(0.5)

    # ── 120d high drawdown ──
    df["high_120"] = high_120
    df["dd_from_120d_high"] = 1 - df["close"] / df["high_120"]

    # ── Rolling 365d price position (for V3 cost-awareness) ──
    df["high_365"] = df["high"].rolling(365).max()
    df["low_365"] = df["low"].rolling(365).min()
    denom_365 = (df["high_365"] - df["low_365"]).replace(0, float("nan"))
    df["rolling_365d_pos"] = (df["close"] - df["low_365"]) / denom_365
    df["rolling_365d_pos"] = df["rolling_365d_pos"].fillna(0.5)

    # ── 180d high drawdown ──
    df["high_180"] = df["high"].rolling(180).max()
    df["dd_from_180d_high"] = 1 - df["close"] / df["high_180"]

    # ── Volume (if quote_volume available) ──
    volume_col = "quote_volume" if "quote_volume" in df.columns else "volume" if "volume" in df.columns else None
    if volume_col is not None:
        df["volume_ma20"] = (
            df[volume_col].rolling(20).mean()
        )
        df["volume_ma60"] = (
            df[volume_col].rolling(60).mean()
        )
        ratio = df["volume_ma20"] / df["volume_ma60"].replace(0, float("nan"))
        df["volume_strength"] = ratio.fillna(1.0)

    return df


def resolve_symbol(candles_by_symbol: dict[str, pd.DataFrame]) -> str | None:
    for s in candles_by_symbol:
        return s
    return None


def detect_market_state(latest: pd.Series) -> str:
    """Identify market state: BULL / BEAR / MIXED based on EMA24/72/168 arrangement."""
    if pd.isna(latest.get("ema24")) or pd.isna(latest.get("ema72")) or pd.isna(latest.get("ema168")):
        return "MIXED"
    ema24 = latest["ema24"]
    ema72 = latest["ema72"]
    ema168 = latest["ema168"]
    if ema24 > ema72 > ema168:
        return "BULL"
    if ema24 < ema72 < ema168:
        return "BEAR"
    return "MIXED"


def compute_btc_regime(df: pd.DataFrame) -> pd.Series:
    """Compute BTC regime (STRONG_BULL/BULL/RANGE/BEAR) for each bar.

    ETH/BNB use this BTC state as a market-context input.
    Returns a Series of regime strings indexed by timestamp.
    """
    result = pd.Series("RANGE", index=df.index, dtype=str)

    # Ensure indicators exist
    if "ema24" not in df.columns:
        df = df.copy()
        df["ema24"] = df["close"].ewm(span=24, adjust=False).mean()
        df["ema72"] = df["close"].ewm(span=72, adjust=False).mean()
        df["ema168"] = df["close"].ewm(span=168, adjust=False).mean()
        df["ema168_slope"] = df["ema168"].pct_change(20)
        high_120 = df["high"].rolling(120).max()
        low_120 = df["low"].rolling(120).min()
        denom = (high_120 - low_120).replace(0, float("nan"))
        df["donchian_pos"] = (df["close"] - low_120) / denom
    else:
        df = df

    for i in range(len(df)):
        row = df.iloc[i]
        ema24 = row.get("ema24")
        ema72 = row.get("ema72")
        ema168 = row.get("ema168")
        ema168_slope = row.get("ema168_slope")
        donchian_pos = row.get("donchian_pos")
        close = row["close"]

        if pd.isna(ema24) or pd.isna(ema72) or pd.isna(ema168):
            continue
        if pd.isna(ema168_slope):
            slope = 0.0
        else:
            slope = ema168_slope
        if pd.isna(donchian_pos):
            dp = 0.5
        else:
            dp = donchian_pos

        if ema24 > ema72 > ema168 and close > ema24 and slope > 0 and dp > 0.75:
            result.iloc[i] = "STRONG_BULL"
        elif ema24 > ema72 and close > ema72 and slope >= 0:
            result.iloc[i] = "BULL"
        elif close < ema168 and ema24 < ema72:
            result.iloc[i] = "BEAR"
        else:
            result.iloc[i] = "RANGE"

    return result


def check_take_profit(profit_pct: float, levels: list[tuple[float, float]]) -> float:
    """Fixed take-profit: return the highest sell fraction for which profit_pct >= threshold.

    levels: list of (threshold_multiplier, sell_fraction), e.g. [(1.06, 0.25), (1.10, 0.25)]
    """
    frac = 0.0
    for threshold, sell_frac in levels:
        if profit_pct >= threshold - 1:
            frac = max(frac, sell_frac)
    return frac


def check_trend_breakdown(latest: pd.Series, price: float) -> float:
    """Trend breakdown: close < EMA24 and EMA24 < EMA72 → sell 50%."""
    ema24 = latest.get("ema24")
    ema72 = latest.get("ema72")
    if pd.isna(ema24) or pd.isna(ema72):
        return 0.0
    if price < ema24 and ema24 < ema72:
        return 0.50
    return 0.0


def check_atr_stop(latest: pd.Series, price: float) -> float:
    """ATR stop: close < EMA24 and close < EMA24 - ATR*2 → sell 100%."""
    ema24 = latest.get("ema24")
    atr = latest.get("atr14")
    if pd.isna(ema24) or pd.isna(atr) or atr <= 0:
        return 0.0
    if price < ema24 and price < ema24 - atr * 2:
        return 1.0
    return 0.0
