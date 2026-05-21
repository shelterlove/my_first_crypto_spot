import pandas as pd


def add_atr(df: pd.DataFrame, period: int = 14, col: str = "atr14") -> pd.DataFrame:
    """Add Average True Range column."""
    df = df.copy()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df[col] = tr.rolling(window=period, min_periods=period).mean()
    return df


def add_moving_averages(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    short_col: str = "ma_short",
    long_col: str = "ma_long",
) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    df = df.copy()

    df[short_col] = (
        df["close"].rolling(window=short_window, min_periods=short_window).mean()
    )
    df[long_col] = (
        df["close"].rolling(window=long_window, min_periods=long_window).mean()
    )

    return df


def add_signal(
    df: pd.DataFrame,
    short_col: str = "ma_short",
    long_col: str = "ma_long",
) -> pd.DataFrame:
    df = df.copy()

    df["signal"] = 0

    valid = df[short_col].notna() & df[long_col].notna()

    df.loc[valid & (df[short_col] > df[long_col]), "signal"] = 1
    df.loc[valid & (df[short_col] <= df[long_col]), "signal"] = 0

    return df
