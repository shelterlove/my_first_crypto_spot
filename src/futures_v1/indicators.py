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
