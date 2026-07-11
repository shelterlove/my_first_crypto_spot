#!/usr/bin/env python3
"""Overlay historical USD-M funding on a capped V1 backtest.

This is a research approximation: funding is applied to daily closing exposure
and does not rerun subsequent sizing after each funding debit.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from futures_v1.benchmark import V1BenchmarkRunner  # noqa: E402
from scripts.render_strategy_review_chart import (  # noqa: E402
    SYMBOL_ORDER,
    _as_utc,
    _load_review_data,
    run_full_window,
)


BASE_URL = "https://fapi.binance.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-18")
    parser.add_argument("--target-gross-cap", type=float, default=1.25)
    parser.add_argument("--multipliers", default="0,1,2,3")
    parser.add_argument("--output", default="results/research/funding_stress.csv")
    return parser.parse_args()


@lru_cache(maxsize=16)
def fetch_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cache = PROJECT_ROOT / "results" / "research" / "funding_cache" / f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if cache.exists():
        return normalize_funding_frame(pd.read_csv(cache))
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    while cursor <= end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{BASE_URL}/fapi/v1/fundingRate?{query}", timeout=30) as response:
                    page = json.loads(response.read().decode("utf-8"))
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"Funding pagination did not advance for {symbol}.")
        cursor = next_cursor
        if len(page) < 1000:
            break
        time.sleep(0.05)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([row["fundingTime"] for row in rows], unit="ms", utc=True),
        "funding_rate": [float(row["fundingRate"]) for row in rows],
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def normalize_funding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed", errors="raise")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="raise")
    return out


def attach_daily_funding(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    out = {symbol: frame.copy() for symbol, frame in data.items()}
    for symbol in SYMBOL_ORDER:
        if symbol not in out:
            continue
        funding = fetch_funding(symbol.replace("/", ""), start, end)
        daily = funding.assign(day=funding["timestamp"].dt.normalize()).groupby("day")["funding_rate"].sum()
        timestamps = pd.to_datetime(out[symbol]["timestamp"], utc=True).dt.normalize()
        out[symbol]["funding_rate_daily"] = timestamps.map(daily).fillna(0.0).astype(float) * float(multiplier)
    return out


def curve_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annual_return": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe_daily": float(returns.mean() / returns.std(ddof=1) * np.sqrt(365.25)) if returns.std(ddof=1) > 0 else 0.0,
    }


def funding_overlay_metrics(
    report: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    multiplier: float,
) -> dict[str, float]:
    curves = []
    annual_drag = 0.0
    for symbol in SYMBOL_ORDER:
        rows = report["equity"].loc[report["equity"]["symbol"] == symbol].copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
        rows = rows.set_index("timestamp").sort_index()
        funding = fetch_funding(symbol.replace("/", ""), start, end)
        daily_funding = funding.assign(day=funding["timestamp"].dt.normalize()).groupby("day")["funding_rate"].sum()
        rows["funding_rate"] = daily_funding.reindex(rows.index, fill_value=0.0)
        rows["base_return"] = rows["total_value"].pct_change().fillna(0.0)
        rows["funding_return"] = rows["position_pct"].shift(1).fillna(0.0) * rows["funding_rate"]
        adjusted_return = rows["base_return"] - multiplier * rows["funding_return"]
        curves.append((1.0 + adjusted_return).cumprod().rename(symbol))
        years = max((rows.index[-1] - rows.index[0]).days / 365.25, 1 / 365.25)
        annual_drag += float(rows["funding_return"].sum() / years) / len(SYMBOL_ORDER)
    combined = pd.concat(curves, axis=1).sort_index().ffill().fillna(1.0).mean(axis=1)
    return {"approx_annual_funding_drag": multiplier * annual_drag, **curve_metrics(combined)}


def main() -> None:
    args = parse_args()
    multipliers = [float(value) for value in args.multipliers.split(",")]
    start, end = _as_utc(args.start), _as_utc(args.end)
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    data = _load_review_data(runner, SYMBOL_ORDER + ["BTC/USDT"], start, end)
    report = run_full_window(
        "eth_bnb_futures_v1", data, runner, start, end, target_gross_cap=args.target_gross_cap
    )
    output_rows = []
    for multiplier in multipliers:
        metrics = funding_overlay_metrics(report, start, end, multiplier)
        output_rows.append({
            "target_gross_cap": args.target_gross_cap,
            "funding_multiplier": multiplier,
            **metrics,
        })
        print(
            f"funding={multiplier:.1f}x annual={metrics['annual_return']:.2%} "
            f"mdd={metrics['max_drawdown']:.2%} drag={metrics['approx_annual_funding_drag']:.2%}",
            flush=True,
        )
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output, index=False)
    print(f"funding_stress={output}")


if __name__ == "__main__":
    main()
