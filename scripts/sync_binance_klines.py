#!/usr/bin/env python3
"""Sync Binance Vision daily klines into the local candles table."""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_v1.config import get_db_url  # noqa: E402


BINANCE_VISION_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,BNB/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="Inclusive end date. Defaults to current UTC month.")
    parser.add_argument("--exchange", default="binance_um_futures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeframe != "1d":
        raise SystemExit("Only 1d klines are supported by this deployment sync script.")
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now("UTC")

    engine = create_engine(get_db_url())
    ensure_candles_table(engine)

    total_rows = 0
    for symbol in symbols:
        frame = load_binance_vision_monthly(symbol=symbol, timeframe=args.timeframe, start=start, end=end)
        if frame.empty:
            print(f"{symbol}: no rows")
            continue
        frame.insert(0, "timeframe", args.timeframe)
        frame.insert(0, "symbol", symbol)
        frame.insert(0, "exchange", args.exchange)
        upsert_candles(engine, frame)
        total_rows += len(frame)
        print(f"{symbol}: synced {len(frame)} rows {frame['timestamp'].min()} -> {frame['timestamp'].max()}")
    print(f"total_rows={total_rows}")


def ensure_candles_table(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS candles (
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        open DOUBLE PRECISION NOT NULL,
        high DOUBLE PRECISION NOT NULL,
        low DOUBLE PRECISION NOT NULL,
        close DOUBLE PRECISION NOT NULL,
        volume DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (exchange, symbol, timeframe, timestamp)
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def load_binance_vision_monthly(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    binance_symbol = symbol.replace("/", "")
    months = pd.period_range(start.tz_convert(None).to_period("M"), end.tz_convert(None).to_period("M"), freq="M")
    rows = []
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for month in months:
        url = f"{BINANCE_VISION_MONTHLY}/{binance_symbol}/{timeframe}/{binance_symbol}-{timeframe}-{month}.zip"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            rows.append(pd.read_csv(zf.open(name), header=None, names=columns))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(rows, ignore_index=True)
    open_time = pd.to_numeric(out["open_time"], errors="coerce")
    out = out[open_time.notna()].copy()
    open_time = open_time[open_time.notna()]
    open_time_ms = open_time.where(open_time < 1e14, open_time / 1000.0)
    out["timestamp"] = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].copy()
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="raise")
    return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def upsert_candles(engine, frame: pd.DataFrame) -> None:
    sql = text(
        """
        INSERT INTO candles (
            exchange, symbol, timeframe, timestamp, open, high, low, close, volume
        ) VALUES (
            :exchange, :symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume
        )
        ON CONFLICT (exchange, symbol, timeframe, timestamp)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
    )
    rows = frame.to_dict("records")
    with engine.begin() as conn:
        conn.execute(sql, rows)


if __name__ == "__main__":
    main()
