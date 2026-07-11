from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sync_binance_klines as sync  # noqa: E402
from scripts.binance_futures_testnet_executor import (  # noqa: E402
    append_next_open_execution_bar,
    validate_fresh_daily_data,
)


def frame(timestamps: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1.0,
    })


def archive_for(day: str) -> bytes:
    open_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    close_ms = open_ms + 86_399_999
    csv = f"{open_ms},100,101,99,100,1,{close_ms},1,1,1,1,0\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("row.csv", csv)
    return buffer.getvalue()


def test_current_month_falls_back_to_daily_archives(monkeypatch) -> None:
    requested = []

    def fake_download(url: str, *, attempts: int = 3):
        requested.append(url)
        if "/monthly/" in url:
            return None
        day = url.rsplit("-", 3)[-3] + "-" + url.rsplit("-", 3)[-2] + "-" + url.rsplit("-", 3)[-1].removesuffix(".zip")
        return archive_for(day)

    monkeypatch.setattr(sync, "download_archive", fake_download)
    result = sync.load_binance_vision_klines(
        symbol="ETH/USDT",
        timeframe="1d",
        start=pd.Timestamp("2026-07-01", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-07-03", tz="UTC"),
    )
    assert list(result["timestamp"]) == list(pd.date_range("2026-07-01", periods=2, tz="UTC"))
    assert any("/monthly/" in url for url in requested)
    assert sum("/daily/" in url for url in requested) == 2


def test_daily_gap_is_rejected() -> None:
    try:
        sync.validate_daily_frame(
            frame(["2026-07-01", "2026-07-03"]),
            symbol="ETH/USDT",
            end_exclusive=pd.Timestamp("2026-07-04", tz="UTC"),
        )
    except ValueError as exc:
        assert "Missing daily candles" in str(exc)
    else:
        raise AssertionError("expected daily gap to fail")


def test_rest_tail_fills_archive_publication_delay(monkeypatch) -> None:
    first_ms = int(pd.Timestamp("2026-07-08", tz="UTC").timestamp() * 1000)
    second_ms = int(pd.Timestamp("2026-07-09", tz="UTC").timestamp() * 1000)

    def fake_archive(url: str, *, attempts: int = 3):
        if "/monthly/" in url:
            return None
        return archive_for("2026-07-08") if "2026-07-08" in url else None

    def fake_json(url: str, *, attempts: int = 3):
        assert "startTime=" in url
        return [[second_ms, "100", "101", "99", "100", "1", second_ms + 86_399_999, "1", 1, "1", "1", "0"]]

    monkeypatch.setattr(sync, "download_archive", fake_archive)
    monkeypatch.setattr(sync, "download_json", fake_json)
    result = sync.load_binance_vision_klines(
        symbol="ETH/USDT",
        timeframe="1d",
        start=pd.Timestamp(first_ms, unit="ms", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-07-10", tz="UTC"),
    )
    assert list(result["timestamp"]) == list(pd.date_range("2026-07-08", periods=2, tz="UTC"))


def test_incomplete_daily_bar_is_rejected(monkeypatch) -> None:
    now = pd.Timestamp.now("UTC").normalize()
    data = frame([str((now - pd.Timedelta(days=1)).date()), str(now.date())])
    try:
        validate_fresh_daily_data({"ETH/USDT": data}, ["ETH/USDT"])
    except SystemExit as exc:
        assert "incomplete_or_non_daily" in str(exc)
    else:
        raise AssertionError("expected incomplete daily bar to fail")


def test_next_open_bar_preserves_signal_boundary() -> None:
    data = frame(["2026-07-08", "2026-07-09"])
    out = append_next_open_execution_bar(
        data,
        timestamp=pd.Timestamp("2026-07-10", tz="UTC"),
        execution_price=__import__("decimal").Decimal("123.45"),
    )
    assert out.iloc[-2]["timestamp"] == pd.Timestamp("2026-07-09", tz="UTC")
    assert out.iloc[-1]["timestamp"] == pd.Timestamp("2026-07-10", tz="UTC")
    assert out.iloc[-1]["open"] == 123.45
    assert out.iloc[-1]["high"] == out.iloc[-1]["low"] == out.iloc[-1]["close"]
