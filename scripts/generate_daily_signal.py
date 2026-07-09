#!/usr/bin/env python3
"""Generate signal-only decisions for paper trading review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils
from crypto_spot_v1.benchmark import build_strategy
from crypto_spot_v1.database import load_candles_from_db
from crypto_spot_v1.decision import build_decision_record, build_strategy_manifest
from crypto_spot_v1.strategy_rebalance import PortfolioState, PositionState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="v4_8_eth_bnb", help="Registered strategy name.")
    parser.add_argument("--config", default="configs/backtest_v1.json", help="Backtest config path.")
    parser.add_argument("--output-dir", default="results/daily_signals", help="Signal output directory.")
    parser.add_argument(
        "--position-pct",
        action="append",
        default=[],
        metavar="SYMBOL=PCT",
        help="Current position percent for a symbol, e.g. BTC/USDT=0.65.",
    )
    parser.add_argument("--capital", type=float, default=100.0, help="Paper portfolio value per symbol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    position_pct = _parse_position_pct(args.position_pct)
    all_dfs = _load_data_with_btc_regime(config)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifests = {}
    for symbol in config["symbols"]:
        df = all_dfs[symbol]
        latest = df.iloc[-1]
        price = float(latest["close"])
        portfolio = _portfolio_for_symbol(symbol, price, args.capital, position_pct.get(symbol, 0.0))
        strategy = build_strategy(
            args.strategy,
            config["capital"]["initial"],
            config["capital"]["reserve"],
            config["cost"]["fee_rate"],
            min_notional=config.get("cost", {}).get("min_notional"),
        )
        setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
        candles = {symbol: df}
        actions = strategy.compute_actions(candles, portfolio, {symbol: price})
        action = actions[0] if actions else None
        no_trade_reason = _no_trade_reason(action, position_pct.get(symbol, 0.0))
        rows.append(build_decision_record(
            timestamp=latest["timestamp"],
            symbol=symbol,
            strategy_name=args.strategy,
            action=action,
            portfolio=portfolio,
            price=price,
            latest=latest,
            no_trade_reason=no_trade_reason,
        ))
        manifests[symbol] = build_strategy_manifest(strategy, config)

    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{timestamp}_{args.strategy}_signals.csv"
    json_path = output_dir / f"{timestamp}_{args.strategy}_signals.json"
    manifest_path = output_dir / f"{timestamp}_{args.strategy}_manifest.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"signals_csv={csv_path}")
    print(f"signals_json={json_path}")
    print(f"manifest_json={manifest_path}")
    print(frame[["timestamp", "symbol", "action", "current_pct", "target_pct", "setup", "raw_state", "confirmed_state", "no_trade_reason"]].to_string(index=False))


def _load_data_with_btc_regime(config: dict) -> dict[str, pd.DataFrame]:
    symbols = list(config["symbols"])
    load_symbols = list(dict.fromkeys(symbols + config.get("reference_symbols", [])))
    all_dfs = {
        symbol: strategy_utils.compute_indicators(load_candles_from_db(
            exchange="binance",
            symbol=symbol,
            timeframe=config["timeframe"],
        ))
        for symbol in load_symbols
    }
    if "BTC/USDT" not in all_dfs:
        return all_dfs
    btc_regime = strategy_utils.compute_btc_regime(all_dfs["BTC/USDT"])
    regime_map = dict(zip(all_dfs["BTC/USDT"]["timestamp"], btc_regime))
    regime_ts_map = dict(zip(all_dfs["BTC/USDT"]["timestamp"], all_dfs["BTC/USDT"]["timestamp"]))
    btc = all_dfs["BTC/USDT"]
    btc_feature_maps = {
        "btc_price_vs_ema72": dict(zip(btc["timestamp"], btc["close"] / btc["ema72"] - 1.0)),
        "btc_price_vs_ema168": dict(zip(btc["timestamp"], btc["close"] / btc["ema168"] - 1.0)),
        "btc_ema24_slope": dict(zip(btc["timestamp"], btc["ema24_slope"])),
        "btc_ema168_slope": dict(zip(btc["timestamp"], btc["ema168_slope"])),
        "btc_roc_20": dict(zip(btc["timestamp"], btc["roc_20"])),
    }
    for df in all_dfs.values():
        df["btc_regime"] = df["timestamp"].map(regime_map).ffill()
        df["btc_regime_timestamp"] = df["timestamp"].map(regime_ts_map).ffill()
        for column, values in btc_feature_maps.items():
            df[column] = df["timestamp"].map(values).ffill()
    return all_dfs


def _portfolio_for_symbol(symbol: str, price: float, capital: float, pct: float) -> PortfolioState:
    pct = max(0.0, min(1.0, pct))
    position_value = capital * pct
    cash = capital - position_value
    quantity = position_value / price if price > 0 else 0.0
    return PortfolioState(
        cash=cash,
        positions={symbol: PositionState(quantity=quantity, avg_cost=price if quantity > 0 else 0.0)},
    )


def _parse_position_pct(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --position-pct value: {value}")
        symbol, pct = value.split("=", 1)
        result[symbol.strip()] = float(pct)
    return result


def _no_trade_reason(action, current_pct: float) -> str:
    if action is not None:
        return ""
    if current_pct <= 0:
        return "no_action_from_zero_position"
    return "target_or_cooldown_not_actionable"


if __name__ == "__main__":
    main()
