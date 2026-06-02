#!/usr/bin/env python3
"""Analyze target-reduce exits and their forward price outcomes."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"
DEFAULT_RUN = PROJECT_ROOT / "results" / "freqtrade_eval" / "baseline_v2_19B_fixed_adj_20260602"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics"
TARGET_TAG_RE = re.compile(
    r"_r(?P<risk>\d+)_tr(?P<trend>\d+)_dd(?P<drawdown>\d+)_raw(?P<raw>[A-Z]+)_conf(?P<conf>[A-Z]+)_t(?P<target>\d+)%"
)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_candles()
    events = extract_target_reduce_events(run_dir)
    rows = [enrich_event(event, candles) for event in events]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No target-reduce events found in {run_dir}")

    summary = summarize(df)
    df.to_csv(output_dir / "target_reduce_regret_events.csv", index=False)
    summary.to_csv(output_dir / "target_reduce_regret_summary.csv", index=False)
    (output_dir / "target_reduce_regret_report.md").write_text(
        render_report(run_dir, df, summary),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="target_reduce_regret_v2_19B_20260602")
    return parser.parse_args()


def load_candles() -> dict[str, pd.DataFrame]:
    from crypto_spot_v1 import strategy_utils

    btc = load_pair("BTC/USDT")
    btc = strategy_utils.compute_indicators(btc)
    btc_regime = strategy_utils.compute_btc_regime(btc)

    result: dict[str, pd.DataFrame] = {}
    for pair in ("BTC/USDT", "ETH/USDT", "BNB/USDT"):
        df = load_pair(pair)
        df = strategy_utils.compute_indicators(df)
        df["btc_regime"] = btc_regime.reindex(df.index).ffill().fillna("RANGE")
        df["market_state"] = df.apply(strategy_utils.detect_market_state, axis=1)
        result[pair] = df
    return result


def load_pair(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair.replace('/', '_')}-1d.feather"
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").set_index("date")


def extract_target_reduce_events(run_dir: Path) -> list[dict]:
    events: list[dict] = []
    for result_zip in sorted(run_dir.rglob("backtest-result-*.zip")):
        orders_by_pair: dict[str, list[dict]] = {}
        for trade in parse_trades(result_zip):
            pair = str(trade.get("pair"))
            orders_by_pair.setdefault(pair, []).extend(normalize_orders(trade, pair))

        for pair, orders in orders_by_pair.items():
            orders = sorted(orders, key=lambda item: item["date"])
            for i, order in enumerate(orders):
                tag = str(order.get("ft_order_tag", ""))
                if order.get("ft_is_entry") or "target-reduce" not in tag:
                    continue
                next_buy = next(
                    (later for later in orders[i + 1:] if later.get("ft_is_entry")),
                    None,
                )
                events.append({
                    "result_zip": str(result_zip.relative_to(PROJECT_ROOT)),
                    "pair": pair,
                    "date": order["date"],
                    "price": float(order.get("safe_price") or 0.0),
                    "amount": float(order.get("amount") or 0.0),
                    "cost": float(order.get("cost") or 0.0),
                    "tag": tag,
                    "next_buy_date": next_buy["date"] if next_buy else pd.NaT,
                })
    return events


def parse_trades(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = next(
            n for n in zf.namelist()
            if n.endswith(".json") and not n.endswith("_config.json") and not n.endswith(".meta.json")
        )
        data = json.loads(zf.read(name))
    strategy = next(iter(data["strategy"].values()))
    return list(strategy.get("trades") or [])


def normalize_orders(trade: dict, pair: str) -> list[dict]:
    orders = []
    for order in trade.get("orders") or []:
        timestamp = order.get("order_filled_timestamp")
        if timestamp is None:
            continue
        normalized = dict(order)
        normalized["pair"] = pair
        normalized["date"] = pd.to_datetime(timestamp, unit="ms", utc=True)
        orders.append(normalized)
    return sorted(orders, key=lambda item: item["date"])


def enrich_event(event: dict, candles: dict[str, pd.DataFrame]) -> dict:
    df = candles[event["pair"]]
    date = pd.Timestamp(event["date"])
    if date not in df.index:
        date = df.index[df.index.searchsorted(date)]
    idx = df.index.get_loc(date)
    latest = df.iloc[idx]
    row = dict(event)
    row["date"] = date.strftime("%Y-%m-%d")
    row["next_buy_date"] = fmt_date(row["next_buy_date"])
    row["next_buy_bars"] = bars_between(df, date, event["next_buy_date"])

    parsed = parse_tag(event["tag"])
    row.update(parsed)
    row.update({
        "close": latest["close"],
        "raw_state_calc": latest["market_state"],
        "ema72_gt_ema168": bool(latest["ema72"] > latest["ema168"]),
        "ema168_slope": latest["ema168_slope"],
        "price_gt_ema24": bool(latest["close"] > latest["ema24"]),
        "roc_5": latest["roc_5"],
        "roc_10": latest["roc_10"],
        "atr_pct_rank": latest["atr_pct_rank"],
        "donchian_pos": latest["donchian_pos"],
        "btc_regime": latest["btc_regime"],
    })
    row["long_structure_intact"] = bool(
        row["ema72_gt_ema168"]
        and pd.notna(row["ema168_slope"])
        and row["ema168_slope"] >= 0
        and row["btc_regime"] != "BEAR"
    )
    row["low_risk"] = bool(row.get("risk_score", 99) <= 2 and row.get("trend_risk", 99) <= 2)

    for horizon in (3, 5, 10, 20):
        row[f"ret_{horizon}d_pct"] = forward_return(df, idx, horizon)
    row["max_up_10d_pct"] = forward_extreme(df, idx, 10, "max")
    row["max_down_10d_pct"] = forward_extreme(df, idx, 10, "min")
    row["regret_10d"] = bool(row["ret_10d_pct"] > 0)
    row["regret_20d"] = bool(row["ret_20d_pct"] > 0)
    return row


def parse_tag(tag: str) -> dict:
    match = TARGET_TAG_RE.search(tag)
    if not match:
        return {
            "risk_score": None,
            "trend_risk": None,
            "drawdown_risk": None,
            "raw_state": "",
            "confirmed_state": "",
            "target_pct": None,
        }
    groups = match.groupdict()
    return {
        "risk_score": int(groups["risk"]),
        "trend_risk": int(groups["trend"]),
        "drawdown_risk": int(groups["drawdown"]),
        "raw_state": groups["raw"],
        "confirmed_state": groups["conf"],
        "target_pct": int(groups["target"]),
    }


def fmt_date(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def bars_between(df: pd.DataFrame, start: pd.Timestamp, end: object) -> int | None:
    if pd.isna(end):
        return None
    end_ts = pd.Timestamp(end)
    if end_ts not in df.index:
        pos = df.index.searchsorted(end_ts)
        if pos >= len(df):
            return None
        end_ts = df.index[pos]
    return int(df.index.get_loc(end_ts) - df.index.get_loc(start))


def forward_return(df: pd.DataFrame, idx: int, horizon: int) -> float | None:
    future_idx = idx + horizon
    if future_idx >= len(df):
        return None
    base = float(df.iloc[idx]["close"])
    future = float(df.iloc[future_idx]["close"])
    return (future / base - 1.0) * 100


def forward_extreme(df: pd.DataFrame, idx: int, horizon: int, direction: str) -> float | None:
    end = min(idx + horizon, len(df) - 1)
    if end <= idx:
        return None
    base = float(df.iloc[idx]["close"])
    values = df.iloc[idx + 1:end + 1]["close"]
    price = values.max() if direction == "max" else values.min()
    return (float(price) / base - 1.0) * 100


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("all", df),
        ("low_risk_structural_intact", df[df["low_risk"] & df["long_structure_intact"]]),
        ("other", df[~(df["low_risk"] & df["long_structure_intact"])]),
    ]
    rows = [summary_row(name, group) for name, group in groups if not group.empty]
    rows.extend(
        summary_row(f"pair={pair}", group)
        for pair, group in df.groupby("pair", sort=True)
        if not group.empty
    )
    rows.extend(
        summary_row(f"raw={state}", group)
        for state, group in df.groupby("raw_state", sort=True)
        if not group.empty
    )
    return pd.DataFrame(rows)


def summary_row(name: str, group: pd.DataFrame) -> dict:
    return {
        "segment": name,
        "events": len(group),
        "regret_10d_rate_pct": (group["regret_10d"].mean() * 100),
        "regret_20d_rate_pct": (group["regret_20d"].mean() * 100),
        "mean_ret_10d_pct": group["ret_10d_pct"].mean(),
        "median_ret_10d_pct": group["ret_10d_pct"].median(),
        "mean_ret_20d_pct": group["ret_20d_pct"].mean(),
        "median_ret_20d_pct": group["ret_20d_pct"].median(),
        "mean_next_buy_bars": group["next_buy_bars"].dropna().mean(),
    }


def render_report(run_dir: Path, df: pd.DataFrame, summary: pd.DataFrame) -> str:
    worst = df.sort_values("ret_10d_pct", ascending=False).head(10)
    return "\n".join([
        "# Target-Reduce Regret Report",
        "",
        f"Source: `{run_dir}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Highest 10d Regret Events",
        "",
        worst[[
            "pair", "date", "ret_10d_pct", "ret_20d_pct", "risk_score",
            "trend_risk", "raw_state", "confirmed_state", "long_structure_intact",
            "next_buy_bars", "tag",
        ]].to_markdown(index=False, floatfmt=".2f"),
        "",
    ])


if __name__ == "__main__":
    main()
