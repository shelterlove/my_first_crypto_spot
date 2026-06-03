#!/usr/bin/env python3
"""Diagnose MIXED market subtypes before changing strategy rules."""

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

from crypto_spot_v1 import strategy_utils  # noqa: E402


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
    mixed_rows = build_mixed_rows(candles, start, end)
    bar_summary = summarize_bars(mixed_rows)
    pair_bar_summary = summarize_bars_by_pair(mixed_rows)

    event_rows = build_target_reduce_rows(Path(args.run_dir), candles, start, end)
    event_summary = summarize_target_reduce(event_rows)
    pair_event_summary = summarize_target_reduce_by_pair(event_rows)

    mixed_rows.to_csv(output_dir / "mixed_regime_bars.csv", index=False)
    bar_summary.to_csv(output_dir / "mixed_regime_bar_summary.csv", index=False)
    pair_bar_summary.to_csv(output_dir / "mixed_regime_pair_summary.csv", index=False)
    event_rows.to_csv(output_dir / "mixed_target_reduce_events.csv", index=False)
    event_summary.to_csv(output_dir / "mixed_target_reduce_summary.csv", index=False)
    pair_event_summary.to_csv(output_dir / "mixed_target_reduce_pair_summary.csv", index=False)
    (output_dir / "mixed_regime_report.md").write_text(
        render_report(args, bar_summary, event_summary),
        encoding="utf-8",
    )

    print("MIXED bar summary")
    print(bar_summary.to_string(index=False))
    print("\nMIXED pair summary")
    print(pair_bar_summary.to_string(index=False))
    print("\nTarget-reduce summary")
    print(event_summary.to_string(index=False) if not event_summary.empty else "No events")
    print("\nTarget-reduce pair summary")
    print(pair_event_summary.to_string(index=False) if not pair_event_summary.empty else "No events")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timerange", default="20180630-20241231")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default="mixed_regime_v2_21E_dev_20260603")
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


def label_mixed_row(row: pd.Series) -> str:
    if row.get("market_state") != "MIXED":
        return "NOT_MIXED"
    price = row.get("close", 0.0)
    ema24 = row.get("ema24")
    ema72 = row.get("ema72")
    ema168 = row.get("ema168")
    ema168_slope = row.get("ema168_slope")
    rolling_pos = row.get("rolling_365d_pos")
    dd_120 = row.get("dd_from_120d_high")
    dd_180 = row.get("dd_from_180d_high")
    donchian_pos = row.get("donchian_pos")
    roc20 = row.get("roc_20")
    roc10 = row.get("roc_10")
    ema24_slope = row.get("ema24_slope")
    volume_strength = row.get("volume_strength")

    if any(pd.isna(v) for v in (ema24, ema72, ema168, ema168_slope)):
        return "NEUTRAL_MIXED"

    long_intact = price > ema168 and ema72 > ema168 and ema168_slope > 0
    high_zone = (
        (not pd.isna(rolling_pos) and rolling_pos >= 0.75)
        or (not pd.isna(donchian_pos) and donchian_pos >= 0.80)
    )
    meaningful_pullback = (
        (not pd.isna(dd_120) and dd_120 >= 0.08)
        or (not pd.isna(dd_180) and dd_180 >= 0.12)
    )
    short_weak = (price < ema24) or (not pd.isna(ema24_slope) and ema24_slope < 0)

    negative_momentum = not pd.isna(roc20) and roc20 < 0
    positive_momentum = not pd.isna(roc20) and roc20 > 0
    sharp_short_drop = not pd.isna(roc10) and roc10 <= -0.08
    high_volume = not pd.isna(volume_strength) and volume_strength >= 1.15

    if price < ema168 or (ema72 < ema168 and ema168_slope < 0):
        return "BREAKDOWN_MIXED"
    if high_zone and meaningful_pullback and short_weak and (negative_momentum or sharp_short_drop or high_volume):
        return "DISTRIBUTION_MIXED"
    if str(row.get("btc_regime", "")) == "BEAR":
        return "DEFENSIVE_MIXED"
    if long_intact and price > ema72 and not negative_momentum:
        return "ABOVE_EMA72_MIXED"
    if long_intact and ema168 < price <= ema72 and not sharp_short_drop:
        return "EMA72_PULLBACK_MIXED"
    if long_intact and price > ema168 and (negative_momentum or sharp_short_drop):
        return "EMA168_TEST_MIXED"
    if long_intact and not high_zone and price > ema24 and positive_momentum:
        return "RECOVERY_MIXED"
    return "NEUTRAL_MIXED"


def build_mixed_rows(
    candles: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for pair, frame in candles.items():
        sample = frame[(frame.index >= start) & (frame.index <= end) & (frame["market_state"] == "MIXED")]
        for idx, row in sample.iterrows():
            out = {
                "pair": pair,
                "date": idx.strftime("%Y-%m-%d"),
                "mixed_label": row["mixed_label"],
                "close": row["close"],
                "btc_regime": row["btc_regime"],
                "rolling_365d_pos": row["rolling_365d_pos"],
                "donchian_pos": row["donchian_pos"],
                "dd_from_120d_high": row["dd_from_120d_high"],
                "dd_from_180d_high": row["dd_from_180d_high"],
                "ema168_slope": row["ema168_slope"],
                "price_to_ema24": row["close"] / row["ema24"] if row["ema24"] else None,
                "price_to_ema72": row["close"] / row["ema72"] if row["ema72"] else None,
                "price_to_ema168": row["close"] / row["ema168"] if row["ema168"] else None,
                "roc_5": row["roc_5"],
                "roc_10": row["roc_10"],
                "roc_20": row["roc_20"],
                "atr_pct_rank": row["atr_pct_rank"],
                "volume_strength": row.get("volume_strength", 1.0),
            }
            loc = frame.index.get_loc(idx)
            for horizon in (20, 60, 90):
                out[f"ret_{horizon}d_pct"] = forward_return(frame, loc, horizon)
                out[f"max_down_{horizon}d_pct"] = forward_extreme(frame, loc, horizon, "min")
            rows.append(out)
    return pd.DataFrame(rows)


def build_target_reduce_rows(
    run_dir: Path,
    candles: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        return pd.DataFrame()

    for result_zip in sorted(run_dir.rglob("backtest-result-*.zip")):
        for trade in parse_trades(result_zip):
            pair = str(trade.get("pair", ""))
            if pair not in candles:
                continue
            for order in trade.get("orders") or []:
                tag = str(order.get("ft_order_tag", ""))
                if order.get("ft_is_entry") or "target-reduce" not in tag:
                    continue
                ts_raw = order.get("order_filled_timestamp")
                if ts_raw is None:
                    continue
                ts = pd.to_datetime(ts_raw, unit="ms", utc=True)
                if ts < start or ts > end:
                    continue
                rows.append(enrich_target_reduce(pair, ts, tag, result_zip, candles[pair]))
    if not rows:
        return pd.DataFrame()
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


def enrich_target_reduce(pair: str, ts: pd.Timestamp, tag: str, result_zip: Path, frame: pd.DataFrame) -> dict:
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
        "mixed_label": row["mixed_label"],
        "tag": tag,
        "result_zip": relative_path(result_zip),
        "close": row["close"],
        "btc_regime": row["btc_regime"],
        "rolling_365d_pos": row["rolling_365d_pos"],
        "donchian_pos": row["donchian_pos"],
        "dd_from_120d_high": row["dd_from_120d_high"],
        "dd_from_180d_high": row["dd_from_180d_high"],
        "price_to_ema24": row["close"] / row["ema24"] if row["ema24"] else None,
        "price_to_ema72": row["close"] / row["ema72"] if row["ema72"] else None,
        "price_to_ema168": row["close"] / row["ema168"] if row["ema168"] else None,
        "roc_5": row["roc_5"],
        "roc_10": row["roc_10"],
        "roc_20": row["roc_20"],
        "atr_pct_rank": row["atr_pct_rank"],
        "volume_strength": row.get("volume_strength", 1.0),
    }
    out.update(parse_tag(tag))
    for horizon in (10, 20, 60, 90):
        out[f"ret_{horizon}d_pct"] = forward_return(frame, loc, horizon)
        out[f"max_down_{horizon}d_pct"] = forward_extreme(frame, loc, horizon, "min")
        out[f"regret_{horizon}d"] = bool((out[f"ret_{horizon}d_pct"] or 0.0) > 0)
    return out


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def summarize_bars(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return summarize_group(rows, ["mixed_label"])


def summarize_bars_by_pair(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return summarize_group(rows, ["pair", "mixed_label"])


def summarize_target_reduce(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return summarize_group(rows, ["mixed_label", "raw_state", "confirmed_state"])


def summarize_target_reduce_by_pair(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return summarize_group(rows, ["pair", "mixed_label", "raw_state", "confirmed_state"])


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "count": len(group),
            "ret_20d_median_pct": group.get("ret_20d_pct", pd.Series(dtype=float)).median(),
            "ret_60d_median_pct": group.get("ret_60d_pct", pd.Series(dtype=float)).median(),
            "ret_90d_median_pct": group.get("ret_90d_pct", pd.Series(dtype=float)).median(),
            "ret_60d_positive_rate_pct": (group.get("ret_60d_pct", pd.Series(dtype=float)) > 0).mean() * 100,
            "max_down_60d_median_pct": group.get("max_down_60d_pct", pd.Series(dtype=float)).median(),
        })
        if "regret_20d" in group:
            row["regret_20d_rate_pct"] = group["regret_20d"].mean() * 100
        if "regret_60d" in group:
            row["regret_60d_rate_pct"] = group["regret_60d"].mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def render_report(args: argparse.Namespace, bar_summary: pd.DataFrame, event_summary: pd.DataFrame) -> str:
    lines = [
        "# MIXED Regime Label Diagnostic",
        "",
        f"- Timerange: `{args.timerange}`",
        f"- Strategy run directory: `{args.run_dir}`",
        "",
        "## Bar Summary",
        "",
        bar_summary.to_markdown(index=False) if not bar_summary.empty else "No MIXED bars.",
        "",
        "## Target-Reduce Summary",
        "",
        event_summary.to_markdown(index=False) if not event_summary.empty else "No target-reduce events.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
