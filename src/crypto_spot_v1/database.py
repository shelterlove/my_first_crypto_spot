"""Minimal candle loader for the V1 migration project."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from .config import get_db_url


def load_candles_from_db(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
) -> pd.DataFrame:
    engine = create_engine(get_db_url())
    sql = text(
        """
        SELECT exchange, symbol, timeframe, timestamp,
               open, high, low, close, volume
        FROM candles
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        ORDER BY timestamp
        """
    )
    df = pd.read_sql_query(
        sql,
        engine,
        params={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
    )
    if df.empty:
        raise ValueError(f"No candles loaded for {exchange} {symbol} {timeframe}.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df
