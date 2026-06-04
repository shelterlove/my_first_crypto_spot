#!/usr/bin/env python3
"""Measure post-defence position rebuild efficiency from Freqtrade result zips."""

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
from scripts.freqtrade_eval import build_equity_curve, read_feather_from_zip  # noqa: E402


DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "freqtrade_eval" / "rolling_v2_21E_dev_deltafix_quick_20260603"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics"
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"
PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
TAG_RE = re.compile(
    r"_r(?P<risk>\d+)_tr(?P<trend>\d+)_dd(?P<drawdown>\d+)_raw(?P<raw>[A-Z]+)_conf(?P<conf>[A-Z]+)_t(?P<target>\d+)%"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_candles()
    events, buys = build_recovery_events(Path(args.run_dir), args.strategy, candles, args)
    unique_events = dedupe_events(events)
    event_summary = summarize_events(events)
    unique_event_summary = summarize_events(unique_events)
    buy_summary = summarize_buys(buys)

    events.to_csv(output_dir / "recovery_events.csv", index=False)
    unique_events.to_csv(output_dir / "recovery_unique_events.csv", index=False)
    buys.to_csv(output_dir / "recovery_buy_events.csv", index=False)
    event_summary.to_csv(output_dir / "recovery_event_summary.csv", index=False)
    unique_event_summary.to_csv(output_dir / "recovery_unique_event_summary.csv", index=False)
    buy_summary.to_csv(output_dir / "recovery_buy_summary.csv", index=False)
    (output_dir / "recovery_efficiency_report.md").write_text(
        render_report(args, event_summary, unique_event_summary, buy_summary),
        encoding="utf-8",
    )

    print("Recovery event summary")
    print(event_summary.to_string(index=False) if not event_summary.empty else "No recovery events")
    print("\nUnique recovery event summary")
    print(unique_event_summary.to_string(index=False) if not unique_event_summary.empty else "No unique recovery events")
    print("\nRecovery buy summary")
    print(buy_summary.to_string(index=False) if not buy_summary.empty else "No recovery buys")
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--strategy", default="CryptoSpotV221E")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default="recovery_efficiency_v2_21E_dev_quick_20260604")
    parser.add_argument("--high-threshold", type=float, default=0.60)
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--recovery-thresholds", default="0.60,0.80")
    parser.add_argument("--prior-days", type=int, default=30)
    parser.add_argument("--max-event-days", type=int, default=120)
    return parser.parse_args()


def load_candles() -> dict[str, pd.DataFrame]:
    btc = strategy_utils.compute_indicators(load_pair("BTC/USDT"))
    btc_regime = strategy_utils.compute_btc_regime(btc)
    result = {}
    for pair in PAIRS:
        frame = strategy_utils.compute_indicators(load_pair(pair))
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


def build_recovery_events(
    run_dir: Path,
    strategy: str,
    candles: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    buy_rows = []
    thresholds = [float(item) for item in args.recovery_thresholds.split(",")]
    for result_zip in sorted(run_dir.resolve().rglob("backtest-result-*.zip")):
        pair = pair_from_zip(result_zip)
        if pair not in candles:
            continue
        result = parse_result_zip(result_zip, strategy)
        equity = build_equity_curve(result["wallet"])
        equity = equity.sort_values("date").reset_index(drop=True)
        orders = normalize_orders(result["trades"])
        pair_events = detect_events(equity, args)
        for idx, event in enumerate(pair_events, start=1):
            enriched = enrich_event(
                event=event,
                event_index=idx,
                pair=pair,
                result_zip=result_zip,
                equity=equity,
                orders=orders,
                candles=candles[pair],
                thresholds=thresholds,
                max_event_days=args.max_event_days,
            )
            event_rows.append(enriched)
            buy_rows.extend(enrich_buys_for_event(enriched, orders, candles[pair]))
    return pd.DataFrame(event_rows), pd.DataFrame(buy_rows)


def parse_result_zip(path: Path, strategy: str) -> dict:
    with zipfile.ZipFile(path) as zf:
        name = next(
            item for item in zf.namelist()
            if item.endswith(".json") and not item.endswith("_config.json") and not item.endswith(".meta.json")
        )
        payload = json.loads(zf.read(name))
        strategy_result = payload["strategy"][strategy]
        wallet_name = next(item for item in zf.namelist() if item.endswith("_wallet.feather"))
        wallet = read_feather_from_zip(zf, wallet_name)
    return {"wallet": wallet, "trades": list(strategy_result.get("trades") or [])}


def pair_from_zip(path: Path) -> str:
    name = path.parent.name
    if "_" not in name:
        return ""
    base, quote = name.split("_", 1)
    return f"{base}/{quote}"


def normalize_orders(trades: list[dict]) -> list[dict]:
    rows = []
    for trade in trades:
        pair = str(trade.get("pair", ""))
        for order in trade.get("orders") or []:
            ts_raw = order.get("order_filled_timestamp")
            if ts_raw is None:
                continue
            tag = str(order.get("ft_order_tag", ""))
            rows.append({
                "pair": pair,
                "date": pd.to_datetime(ts_raw, unit="ms", utc=True),
                "is_entry": bool(order.get("ft_is_entry")),
                "tag": tag,
                "setup": parse_setup(tag),
                "cost": float(order.get("cost") or 0.0),
            })
    return sorted(rows, key=lambda row: row["date"])


def detect_events(equity: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    events = []
    exposure = equity["exposure"].fillna(0.0)
    dates = pd.to_datetime(equity["date"], utc=True)
    in_event_until = pd.Timestamp.min.tz_localize("UTC")
    for i in range(len(equity)):
        date = dates.iloc[i]
        if date <= in_event_until or exposure.iloc[i] > args.low_threshold:
            continue
        prior = exposure.iloc[max(0, i - args.prior_days):i + 1]
        if prior.max() < args.high_threshold:
            continue
        start = max(0, i - args.prior_days)
        high_idx = prior.idxmax()
        end_idx = min(len(equity) - 1, i + args.max_event_days)
        events.append({
            "defence_date": date,
            "defence_idx": i,
            "prior_high_date": dates.iloc[high_idx],
            "prior_high_exposure": float(exposure.iloc[high_idx]),
            "low_exposure": float(exposure.iloc[i]),
            "event_end_idx": end_idx,
            "event_end_date": dates.iloc[end_idx],
            "prior_window_start_idx": start,
        })
        in_event_until = dates.iloc[end_idx]
    return events


def enrich_event(
    *,
    event: dict,
    event_index: int,
    pair: str,
    result_zip: Path,
    equity: pd.DataFrame,
    orders: list[dict],
    candles: pd.DataFrame,
    thresholds: list[float],
    max_event_days: int,
) -> dict:
    date = event["defence_date"]
    candle_date, candle = nearest_candle(candles, date)
    out = {
        "pair": pair,
        "event_index": event_index,
        "defence_date": date.strftime("%Y-%m-%d"),
        "prior_high_date": event["prior_high_date"].strftime("%Y-%m-%d"),
        "event_end_date": event["event_end_date"].strftime("%Y-%m-%d"),
        "prior_high_exposure_pct": event["prior_high_exposure"] * 100,
        "low_exposure_pct": event["low_exposure"] * 100,
        "btc_regime": candle.get("btc_regime", ""),
        "market_state": candle.get("market_state", ""),
        "mixed_label": candle.get("mixed_label", ""),
        "close": float(candle["close"]),
        "result_zip": relative_path(result_zip),
    }
    for threshold in thresholds:
        key = int(threshold * 100)
        recovery = first_recovery(equity, event["defence_idx"], event["event_end_idx"], threshold)
        if recovery is None:
            out[f"days_to_{key}_pct"] = None
            out[f"price_ret_to_{key}_pct"] = None
        else:
            rec_date, rec_idx = recovery
            rec_candle_date, rec_candle = nearest_candle(candles, rec_date)
            out[f"days_to_{key}_pct"] = int((rec_date - date).days)
            out[f"price_ret_to_{key}_pct"] = pct_return(candle["close"], rec_candle["close"])
            out[f"date_to_{key}_pct"] = rec_candle_date.strftime("%Y-%m-%d")
            out[f"exposure_at_{key}_pct"] = float(equity.iloc[rec_idx]["exposure"]) * 100
    end_date = min(date + pd.Timedelta(days=max_event_days), pd.to_datetime(equity.iloc[-1]["date"], utc=True))
    _, end_candle = nearest_candle(candles, end_date)
    out["price_ret_120d_or_window_pct"] = pct_return(candle["close"], end_candle["close"])
    for horizon in (30, 60, 90, 120):
        horizon_date = min(date + pd.Timedelta(days=horizon), pd.to_datetime(equity.iloc[-1]["date"], utc=True))
        eq_idx = equity["date"].searchsorted(horizon_date)
        if eq_idx >= len(equity):
            eq_idx = len(equity) - 1
        _, horizon_candle = nearest_candle(candles, horizon_date)
        out[f"exposure_{horizon}d_pct"] = float(equity.iloc[eq_idx]["exposure"]) * 100
        out[f"price_ret_{horizon}d_pct"] = pct_return(candle["close"], horizon_candle["close"])
    event_orders = [row for row in orders if date <= row["date"] <= event["event_end_date"]]
    out["buy_count_120d"] = sum(1 for row in event_orders if row["is_entry"])
    out["sell_count_120d"] = sum(1 for row in event_orders if not row["is_entry"])
    out["first_buy_days"] = first_order_days(event_orders, date, is_entry=True)
    out["first_buy_setup"] = first_order_setup(event_orders, is_entry=True)
    out["first_buy_tag"] = first_order_tag(event_orders, is_entry=True)
    return out


def enrich_buys_for_event(event: dict, orders: list[dict], candles: pd.DataFrame) -> list[dict]:
    start = pd.Timestamp(event["defence_date"], tz="UTC")
    end = pd.Timestamp(event["event_end_date"], tz="UTC")
    rows = []
    for order in orders:
        if not order["is_entry"] or order["date"] < start or order["date"] > end:
            continue
        _, candle = nearest_candle(candles, order["date"])
        row = {
            "pair": event["pair"],
            "event_index": event["event_index"],
            "defence_date": event["defence_date"],
            "buy_date": order["date"].strftime("%Y-%m-%d"),
            "days_after_defence": int((order["date"] - start).days),
            "setup": order["setup"],
            "tag": order["tag"],
            "btc_regime": candle.get("btc_regime", ""),
            "market_state": candle.get("market_state", ""),
            "mixed_label": candle.get("mixed_label", ""),
        }
        row.update(parse_tag(order["tag"]))
        rows.append(row)
    return rows


def first_recovery(equity: pd.DataFrame, start_idx: int, end_idx: int, threshold: float) -> tuple[pd.Timestamp, int] | None:
    frame = equity.iloc[start_idx:end_idx + 1]
    hits = frame[frame["exposure"] >= threshold]
    if hits.empty:
        return None
    idx = int(hits.index[0])
    return pd.to_datetime(equity.iloc[idx]["date"], utc=True), idx


def nearest_candle(candles: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series]:
    date = pd.Timestamp(date).tz_convert("UTC") if pd.Timestamp(date).tzinfo else pd.Timestamp(date, tz="UTC")
    pos = candles.index.searchsorted(date)
    if pos >= len(candles):
        pos = len(candles) - 1
    return candles.index[pos], candles.iloc[pos]


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


def first_order_days(orders: list[dict], start: pd.Timestamp, *, is_entry: bool) -> int | None:
    rows = [row for row in orders if row["is_entry"] == is_entry]
    if not rows:
        return None
    return int((rows[0]["date"] - start).days)


def first_order_setup(orders: list[dict], *, is_entry: bool) -> str:
    rows = [row for row in orders if row["is_entry"] == is_entry]
    return rows[0]["setup"] if rows else ""


def first_order_tag(orders: list[dict], *, is_entry: bool) -> str:
    rows = [row for row in orders if row["is_entry"] == is_entry]
    return rows[0]["tag"] if rows else ""


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    group_cols = ["pair", "btc_regime", "market_state", "mixed_label"]
    rows = []
    for keys, group in events.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update({
            "count": len(group),
            "median_low_exposure_pct": group["low_exposure_pct"].median(),
            "median_days_to_60_pct": group["days_to_60_pct"].median(),
            "hit_60_rate_pct": group["days_to_60_pct"].notna().mean() * 100,
            "median_price_ret_to_60_pct": group["price_ret_to_60_pct"].median(),
            "median_days_to_80_pct": group["days_to_80_pct"].median(),
            "hit_80_rate_pct": group["days_to_80_pct"].notna().mean() * 100,
            "median_price_ret_120d_pct": group["price_ret_120d_or_window_pct"].median(),
            "median_first_buy_days": group["first_buy_days"].median(),
            "median_buy_count_120d": group["buy_count_120d"].median(),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count", "median_price_ret_120d_pct"], ascending=[False, False])


def dedupe_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return (
        events.sort_values(["pair", "defence_date", "result_zip"])
        .drop_duplicates(["pair", "defence_date"], keep="first")
        .reset_index(drop=True)
    )


def summarize_buys(buys: pd.DataFrame) -> pd.DataFrame:
    if buys.empty:
        return pd.DataFrame()
    group_cols = ["setup", "raw_state", "confirmed_state", "btc_regime", "mixed_label"]
    rows = []
    for keys, group in buys.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update({
            "count": len(group),
            "median_days_after_defence": group["days_after_defence"].median(),
            "median_risk_score": group["risk_score"].median(),
            "median_target_pct": group["target_pct"].median(),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def pct_return(start: float, end: float) -> float:
    if pd.isna(start) or start == 0:
        return float("nan")
    return (float(end) / float(start) - 1.0) * 100


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def render_report(
    args: argparse.Namespace,
    event_summary: pd.DataFrame,
    unique_event_summary: pd.DataFrame,
    buy_summary: pd.DataFrame,
) -> str:
    return "\n".join([
        "# Recovery Efficiency Diagnostic",
        "",
        f"- Run directory: `{args.run_dir}`",
        f"- High threshold: `{args.high_threshold:.0%}`",
        f"- Low threshold: `{args.low_threshold:.0%}`",
        f"- Prior days: `{args.prior_days}`",
        f"- Max event days: `{args.max_event_days}`",
        "",
        "## Recovery Events",
        "",
        event_summary.to_markdown(index=False) if not event_summary.empty else "No recovery events.",
        "",
        "## Unique Recovery Events",
        "",
        unique_event_summary.to_markdown(index=False) if not unique_event_summary.empty else "No unique recovery events.",
        "",
        "## Recovery Buys",
        "",
        buy_summary.to_markdown(index=False) if not buy_summary.empty else "No recovery buy events.",
        "",
    ])


if __name__ == "__main__":
    main()
