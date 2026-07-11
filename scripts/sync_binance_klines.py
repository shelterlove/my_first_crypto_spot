#!/usr/bin/env python3
"""Sync Binance Vision daily klines into the local candles table."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
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
BINANCE_VISION_DAILY = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_FUTURES_REST_KLINES = "https://fapi.binance.com/fapi/v1/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,BNB/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None, help="Inclusive end date. Defaults to current UTC month.")
    parser.add_argument("--exchange", default="binance_um_futures")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore the database high-water mark and refetch the full requested range.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeframe != "1d":
        raise SystemExit("Only 1d klines are supported by this deployment sync script.")
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    today_utc = pd.Timestamp.now("UTC").normalize()
    if args.end:
        end_exclusive = pd.Timestamp(args.end, tz="UTC").normalize() + pd.Timedelta(days=1)
        if end_exclusive > today_utc:
            raise SystemExit("--end must be an already completed UTC date (yesterday or earlier).")
    else:
        # Binance daily candles close at 00:00 UTC. Never ingest today's open candle.
        end_exclusive = today_utc

    engine = create_engine(get_db_url())
    ensure_candles_table(engine)

    total_rows = 0
    for symbol in symbols:
        effective_start = start
        if not args.full_refresh:
            latest = latest_candle_timestamp(engine, args.exchange, symbol, args.timeframe)
            if latest is not None:
                effective_start = max(start, latest)
        frame = load_binance_vision_klines(
            symbol=symbol,
            timeframe=args.timeframe,
            start=effective_start,
            end_exclusive=end_exclusive,
        )
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


def latest_candle_timestamp(engine, exchange: str, symbol: str, timeframe: str) -> pd.Timestamp | None:
    sql = text(
        """
        SELECT MAX(timestamp) AS latest
        FROM candles
        WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe
        """
    )
    with engine.connect() as conn:
        value = conn.execute(
            sql,
            {"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ).scalar_one_or_none()
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def load_binance_vision_klines(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    binance_symbol = symbol.replace("/", "")
    last_day = end_exclusive - pd.Timedelta(days=1)
    if last_day < start:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    months = pd.period_range(
        start.tz_convert(None).to_period("M"),
        last_day.tz_convert(None).to_period("M"),
        freq="M",
    )
    rows = []
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for month in months:
        url = f"{BINANCE_VISION_MONTHLY}/{binance_symbol}/{timeframe}/{binance_symbol}-{timeframe}-{month}.zip"
        content = download_archive(url)
        if content is not None:
            rows.append(read_kline_archive(content, columns))
            continue

        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC").normalize() + pd.Timedelta(days=1)
        daily_start = max(start.normalize(), month_start)
        daily_end = min(end_exclusive, month_end)
        for day in pd.date_range(daily_start, daily_end, inclusive="left", freq="D"):
            date_label = day.strftime("%Y-%m-%d")
            daily_url = (
                f"{BINANCE_VISION_DAILY}/{binance_symbol}/{timeframe}/"
                f"{binance_symbol}-{timeframe}-{date_label}.zip"
            )
            daily_content = download_archive(daily_url)
            if daily_content is not None:
                rows.append(read_kline_archive(daily_content, columns))
    archive_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    archive_open_time = pd.to_numeric(archive_rows.get("open_time"), errors="coerce").dropna()
    if archive_open_time.empty:
        rest_start = start
    else:
        archive_last_ms = float(archive_open_time.max())
        if archive_last_ms > 1e14:
            archive_last_ms /= 1000.0
        archive_last = pd.to_datetime(archive_last_ms, unit="ms", utc=True)
        rest_start = max(start, archive_last + pd.Timedelta(days=1))
    rest_rows = load_binance_rest_klines(
        symbol=binance_symbol,
        timeframe=timeframe,
        start=rest_start,
        end_exclusive=end_exclusive,
        columns=columns,
    )
    if not rest_rows.empty:
        rows.append(rest_rows)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(rows, ignore_index=True)
    open_time = pd.to_numeric(out["open_time"], errors="coerce")
    out = out[open_time.notna()].copy()
    open_time = open_time[open_time.notna()]
    open_time_ms = open_time.where(open_time < 1e14, open_time / 1000.0)
    out["timestamp"] = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    out = out[(out["timestamp"] >= start) & (out["timestamp"] < end_exclusive)].copy()
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    validate_daily_frame(out, symbol=symbol, end_exclusive=end_exclusive)
    return out


def download_archive(url: str, *, attempts: int = 3) -> bytes | None:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt + 1 >= attempts:
                raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        time.sleep(0.5 * (2 ** attempt))
    return None


def load_binance_rest_klines(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    columns: list[str],
) -> pd.DataFrame:
    if start >= end_exclusive:
        return pd.DataFrame(columns=columns)
    rows = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end_exclusive.timestamp() * 1000) - 1
    while cursor_ms <= end_ms:
        query = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": timeframe,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": 1500,
        })
        url = f"{BINANCE_FUTURES_REST_KLINES}?{query}"
        payload = download_json(url)
        if not payload:
            break
        frame = pd.DataFrame(payload, columns=columns)
        rows.append(frame)
        last_open = int(payload[-1][0])
        next_cursor = last_open + 1
        if next_cursor <= cursor_ms:
            raise RuntimeError("Binance futures kline pagination did not advance.")
        cursor_ms = next_cursor
        if len(payload) < 1500:
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def download_json(url: str, *, attempts: int = 3) -> list:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Binance response from {url}")
            return payload
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(f"Failed to load Binance futures klines from {url}: {exc}") from exc
            time.sleep(0.5 * (2 ** attempt))
    return []


def read_kline_archive(content: bytes, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in Binance archive, found {len(names)}")
        return pd.read_csv(zf.open(names[0]), header=None, names=columns)


def validate_daily_frame(frame: pd.DataFrame, *, symbol: str, end_exclusive: pd.Timestamp) -> None:
    if frame.empty:
        return
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted daily candles for {symbol}.")
    if (timestamps >= end_exclusive).any():
        raise ValueError(f"Incomplete daily candle detected for {symbol}.")
    gaps = timestamps.diff().dropna()
    bad = gaps[gaps != pd.Timedelta(days=1)]
    if not bad.empty:
        idx = int(bad.index[0])
        raise ValueError(
            f"Missing daily candles for {symbol}: "
            f"{timestamps.iloc[idx - 1].isoformat()} -> {timestamps.iloc[idx].isoformat()}"
        )


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
