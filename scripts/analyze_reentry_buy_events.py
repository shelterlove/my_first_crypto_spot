#!/usr/bin/env python3
"""Diagnose executed re-entry buys from Freqtrade result zips."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from scripts.analyze_mixed_regime_labels import label_mixed_row  # noqa: E402


DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "freqtrade_eval" / "rolling_v2_21E_dev_deltafix_quick_20260603"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"
PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
TAG_RE = re.compile(
    r"_r(?P<risk>\d+)_tr(?P<trend>\d+)_dd(?P<drawdown>\d+)_raw(?P<raw>[A-Z]+)_conf(?P<conf>[A-Z]+)_t(?P<target>\d+)%"
)


def main() -> None:
    args = parse_args()
    start, end = parse_timerange(args.timerange)
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_candles()
    rows = build_buy_rows(Path(args.run_dir), candles, start, end)
    summary = summarize(rows, ["setup", "raw_state", "confirmed_state"])
    mixed_summary = summarize(rows[rows["raw_state"] == "MIXED"], ["mixed_label", "btc_regime", "btc_bear_guard"])
    pair_summary = summarize(rows, ["pair", "setup", "btc_bear_guard"])

    rows.to_csv(output_dir / "reentry_buy_events.csv", index=False)
    summary.to_csv(output_dir / "reentry_buy_summary.csv", index=False)
    mixed_summary.to_csv(output_dir / "reentry_mixed_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "reentry_pair_summary.csv", index=False)
    (output_dir / "reentry_buy_report.md").write_text(
        render_report(args, summary, mixed_summary, pair_summary),
        encoding="utf-8",
    )

    print("Buy summary")
    print(summary.to_string(index=False) if not summary.empty else "No buy events")
    print("\nMIXED buy summary")
    print(mixed_summary.to_string(index=False) if not mixed_summary.empty else "No MIXED buy events")
    print("\nPair summary")
    print(pair_summary.to_string(index=False) if not pair_summary.empty else "No pair buy events")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timerange", default="20180630-20241231")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default="reentry_buy_v2_21E_dev_20260603")
    return parser.parse_args()


def parse_timerange(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = value.split("-", 1)
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


def load_candles() -> dict[str, pd.DataFrame]:
    btc = load_pair("BTC/USDT")
    btc = strategy_utils.compute_indicators(btc)
    btc_regime = strategy_utils.compute_btc_regime(btc)

    result = {}
    for pair in PAIRS:
        frame = load_pair(pair)
        frame = strategy_utils.compute_indicators(frame)
        frame["btc_regime"] = btc_regime.reindex(frame.index).ffill().fillna("RANGE")
        frame["market_state"] = frame.apply(strategy_utils.detect_market_state, axis=1)
        frame["mixed_label"] = frame.apply(label_mixed_row, axis=1)
        result[pair] = frame
    return result


def load_pair(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair.replace('/', '_')}-1d.feather"
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date").set_index("date")


def build_buy_rows(
    run_dir: Path,
    candles: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for result_zip in sorted(run_dir.resolve().rglob("backtest-result-*.zip")):
        for trade in parse_trades(result_zip):
            pair = str(trade.get("pair", ""))
            if pair not in candles:
                continue
            for order in trade.get("orders") or []:
                if not order.get("ft_is_entry"):
                    continue
                ts_raw = order.get("order_filled_timestamp")
                if ts_raw is None:
                    continue
                ts = pd.to_datetime(ts_raw, unit="ms", utc=True)
                if ts < start or ts > end:
                    continue
                rows.append(enrich_buy(pair, ts, str(order.get("ft_order_tag", "")), result_zip, candles[pair]))
    return pd.DataFrame(rows)


def parse_trades(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = next(
            item for item in zf.namelist()
            if item.endswith(".json") and not item.endswith("_config.json") and not item.endswith(".meta.json")
        )
        payload = json.loads(zf.read(name))
    strategy = next(iter(payload["strategy"].values()))
    return list(strategy.get("trades") or [])


def enrich_buy(pair: str, ts: pd.Timestamp, tag: str, result_zip: Path, frame: pd.DataFrame) -> dict:
    if ts not in frame.index:
        pos = frame.index.searchsorted(ts)
        if pos >= len(frame):
            pos = len(frame) - 1
        ts = frame.index[pos]
    loc = frame.index.get_loc(ts)
    row = frame.iloc[loc]
    out = {
        "pair": pair,
        "date": ts.strftime("%Y-%m-%d"),
        "tag": tag,
        "setup": parse_setup(tag),
        "btc_bear_guard": "btc-bear-tgap" in tag,
        "mixed_label": row["mixed_label"],
        "btc_regime": row["btc_regime"],
        "close": row["close"],
        "result_zip": relative_path(result_zip),
    }
    out.update(parse_tag(tag))
    for horizon in (20, 60, 90):
        out[f"ret_{horizon}d_pct"] = forward_return(frame, loc, horizon)
        out[f"max_down_{horizon}d_pct"] = forward_extreme(frame, loc, horizon, "min")
    return out


def parse_setup(tag: str) -> str:
    for setup in ("safe-recovery", "pullback", "trend-cont", "target-gap"):
        if setup in tag:
            return setup
    return "other"


def parse_tag(tag: str) -> dict:
    match = TAG_RE.search(tag)
    if not match:
        return {"risk_score": None, "trend_risk": None, "drawdown_risk": None, "raw_state": "", "confirmed_state": "", "target_pct": None}
    groups = match.groupdict()
    return {
        "risk_score": int(groups["risk"]),
        "trend_risk": int(groups["trend"]),
        "drawdown_risk": int(groups["drawdown"]),
        "raw_state": groups["raw"],
        "confirmed_state": groups["conf"],
        "target_pct": int(groups["target"]),
    }


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "count": len(group),
            "ret_20d_median_pct": group["ret_20d_pct"].median(),
            "ret_60d_median_pct": group["ret_60d_pct"].median(),
            "ret_90d_median_pct": group["ret_90d_pct"].median(),
            "ret_60d_positive_rate_pct": (group["ret_60d_pct"] > 0).mean() * 100,
            "max_down_60d_median_pct": group["max_down_60d_pct"].median(),
            "median_risk_score": group["risk_score"].median(),
            "median_target_pct": group["target_pct"].median(),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def forward_return(frame: pd.DataFrame, loc: int, horizon: int) -> float | None:
    future = loc + horizon
    if future >= len(frame):
        return None
    return (float(frame.iloc[future]["close"]) / float(frame.iloc[loc]["close"]) - 1.0) * 100


def forward_extreme(frame: pd.DataFrame, loc: int, horizon: int, direction: str) -> float | None:
    end = min(loc + horizon, len(frame) - 1)
    if end <= loc:
        return None
    values = frame.iloc[loc + 1:end + 1]["close"]
    value = values.min() if direction == "min" else values.max()
    return (float(value) / float(frame.iloc[loc]["close"]) - 1.0) * 100


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def render_report(args: argparse.Namespace, summary: pd.DataFrame, mixed_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> str:
    return "\n".join([
        "# Re-entry Buy Diagnostic",
        "",
        f"- Timerange: `{args.timerange}`",
        f"- Strategy run directory: `{args.run_dir}`",
        "",
        "## Buy Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No buy events.",
        "",
        "## MIXED Buy Summary",
        "",
        mixed_summary.to_markdown(index=False) if not mixed_summary.empty else "No MIXED buy events.",
        "",
        "## Pair Summary",
        "",
        pair_summary.to_markdown(index=False) if not pair_summary.empty else "No pair buy events.",
        "",
    ])


if __name__ == "__main__":
    main()
