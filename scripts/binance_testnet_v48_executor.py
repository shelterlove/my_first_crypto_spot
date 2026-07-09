#!/usr/bin/env python3
"""Execute V4.8 ETH/BNB signals on Binance Spot Testnet.

Default mode is dry-run. Pass --execute to submit MARKET orders to testnet.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_spot_v1.backtest_event_driven import execute_action_in_backtest, run_rebalance_backtest  # noqa: E402
from crypto_spot_v1.benchmark import build_strategy  # noqa: E402
from crypto_spot_v1.strategy_rebalance import PortfolioState, PositionState  # noqa: E402
from crypto_spot_v1.v47 import position_book as v47_position_book  # noqa: E402
from scripts.generate_daily_signal import _load_data_with_btc_regime  # noqa: E402


TESTNET_BASE_URL = "https://testnet.binance.vision/api"
DEFAULT_STRATEGY = "v4_8_eth_bnb"


@dataclass(frozen=True)
class SymbolFilters:
    base_asset: str
    quote_asset: str
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


@dataclass
class PlannedOrder:
    symbol: str
    api_symbol: str
    side: str
    quantity: Decimal | None
    quote_order_qty: Decimal | None
    price: Decimal
    notional: Decimal
    requested_notional: Decimal
    clip_reason: str
    reason: str
    current_pct: float
    sleeve_value: Decimal


class BinanceSpotTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = TESTNET_BASE_URL):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params or {}, signed=False)

    def signed_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params or {}, signed=True)

    def signed_post(self, path: str, params: dict[str, Any]) -> Any:
        return self._request("POST", path, params, signed=True)

    def _request(self, method: str, path: str, params: dict[str, Any], *, signed: bool) -> Any:
        payload = dict(params)
        headers = {"X-MBX-APIKEY": self.api_key} if signed else {}
        if signed:
            payload.setdefault("recvWindow", 5000)
            payload["timestamp"] = self.server_time()
            query = urllib.parse.urlencode(payload, doseq=True)
            signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
            query = f"{query}&signature={signature}"
        else:
            query = urllib.parse.urlencode(payload, doseq=True)

        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Binance API connection error: {exc}") from exc
        return json.loads(raw) if raw else {}

    def server_time(self) -> int:
        try:
            with urllib.request.urlopen(f"{self.base_url}/v3/time", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return int(payload["serverTime"])
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Binance API connection error: {exc}") from exc

    def ticker_price(self, symbol: str) -> Decimal:
        payload = self.public_get("/v3/ticker/price", {"symbol": symbol})
        return Decimal(str(payload["price"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest_v1.json")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--base-url", default=os.getenv("BINANCE_TESTNET_BASE_URL", TESTNET_BASE_URL))
    parser.add_argument("--api-key-env", default="BINANCE_TESTNET_API_KEY")
    parser.add_argument("--api-secret-env", default="BINANCE_TESTNET_API_SECRET")
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--sleeve-fraction", type=float, default=1.0, help="Fraction of testnet equity allocated to this strategy.")
    parser.add_argument("--bootstrap-cap", type=Decimal, default=Decimal("0.35"), help="Max initial alignment pct per sleeve from near-zero position.")
    parser.add_argument("--bootstrap-threshold-pct", type=float, default=0.02, help="Bootstrap only when current sleeve pct is below this level.")
    parser.add_argument("--min-order-usdt", type=Decimal, default=Decimal("10"))
    parser.add_argument("--max-order-usdt", type=Decimal, default=Decimal("0"), help="0 disables this safety cap.")
    parser.add_argument("--execute", action="store_true", help="Submit MARKET orders. Omit for dry-run.")
    parser.add_argument("--output-dir", default="results/binance_testnet_v48")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    api_key = os.getenv(args.api_key_env, "")
    api_secret = os.getenv(args.api_secret_env, "")
    if not api_key or not api_secret:
        raise SystemExit(
            f"Missing API credentials. Set {args.api_key_env} and {args.api_secret_env}; "
            "use Binance Spot Testnet keys only."
        )

    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    symbols = list(config["symbols"])
    validate_deployment_symbols(symbols)

    client = BinanceSpotTestnetClient(api_key, api_secret, args.base_url)
    all_dfs = _load_data_with_btc_regime(config)
    exchange_info = client.public_get("/v3/exchangeInfo", {"symbols": json.dumps([api_symbol(s) for s in symbols])})
    filters = parse_filters(exchange_info)
    account = client.signed_get("/v3/account")
    balances = parse_balances(account)

    latest_prices = {symbol: client.ticker_price(api_symbol(symbol)) for symbol in symbols}
    total_equity = account_equity_usdt(balances, latest_prices, quote_asset=args.quote_asset)
    if total_equity <= 0:
        raise SystemExit(f"No positive {args.quote_asset} equity available in Spot Testnet account.")
    deploy_equity = total_equity * Decimal(str(args.sleeve_fraction))
    sleeve_value = deploy_equity / Decimal(len(symbols))
    available_quote_start = balances.get(args.quote_asset, Decimal("0")) * Decimal(str(args.sleeve_fraction))
    available_quote = available_quote_start

    planned = []
    for symbol in symbols:
        order = plan_symbol_order(
            symbol=symbol,
            config=config,
            strategy_name=args.strategy,
            df=all_dfs[symbol],
            balances=balances,
            filters=filters[api_symbol(symbol)],
            price=latest_prices[symbol],
            sleeve_value=sleeve_value,
            available_quote=available_quote,
            bootstrap_target_pct=native_latest_position_pct(
                symbol=symbol,
                df=all_dfs[symbol],
                config=config,
                strategy_name=args.strategy,
                sleeve_value=sleeve_value,
            ),
            bootstrap_cap=args.bootstrap_cap,
            bootstrap_threshold_pct=args.bootstrap_threshold_pct,
            min_order_usdt=args.min_order_usdt,
            max_order_usdt=args.max_order_usdt,
        )
        if order is not None:
            planned.append(order)
            if order.side == "BUY":
                available_quote = max(Decimal("0"), available_quote - (order.quote_order_qty or Decimal("0")))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "mode": "execute" if args.execute else "dry_run",
        "base_url": args.base_url,
        "strategy": args.strategy,
        "symbols": symbols,
        "total_equity_usdt": str(total_equity),
        "deploy_equity_usdt": str(deploy_equity),
        "sleeve_value_usdt": str(sleeve_value),
        "available_quote_start_usdt": str(available_quote_start),
        "available_quote_after_planning_usdt": str(available_quote),
        "orders": [order_to_dict(order) for order in planned],
        "responses": [],
    }

    if args.execute:
        for order in planned:
            response = submit_market_order(client, order)
            report["responses"].append(response)

    out_path = output_dir / f"{timestamp}_{args.strategy}_{'execute' if args.execute else 'dry_run'}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"report={out_path}")
    print(pd.DataFrame(report["orders"]).to_string(index=False) if report["orders"] else "No actionable orders.")
    if args.execute:
        print(json.dumps(report["responses"], indent=2, ensure_ascii=False, default=str))


def validate_args(args: argparse.Namespace) -> None:
    if args.sleeve_fraction <= 0 or args.sleeve_fraction > 1:
        raise SystemExit("--sleeve-fraction must be > 0 and <= 1.")
    if args.bootstrap_cap < 0:
        raise SystemExit("--bootstrap-cap must be >= 0.")
    if args.bootstrap_threshold_pct < 0:
        raise SystemExit("--bootstrap-threshold-pct must be >= 0.")
    if args.min_order_usdt < 0:
        raise SystemExit("--min-order-usdt must be >= 0.")
    if args.max_order_usdt < 0:
        raise SystemExit("--max-order-usdt must be >= 0.")


def validate_deployment_symbols(symbols: list[str]) -> None:
    expected = {"ETH/USDT", "BNB/USDT"}
    actual = set(symbols)
    if actual != expected:
        raise SystemExit(f"V4.8 spot deployment symbols must be exactly {sorted(expected)}, got {symbols}.")


def plan_symbol_order(
    *,
    symbol: str,
    config: dict[str, Any],
    strategy_name: str,
    df: pd.DataFrame,
    balances: dict[str, Decimal],
    filters: SymbolFilters,
    price: Decimal,
    sleeve_value: Decimal,
    available_quote: Decimal,
    bootstrap_target_pct: float,
    bootstrap_cap: Decimal,
    bootstrap_threshold_pct: float,
    min_order_usdt: Decimal,
    max_order_usdt: Decimal,
) -> PlannedOrder | None:
    base_qty = balances.get(filters.base_asset, Decimal("0"))
    position_value = base_qty * price
    current_pct = float(position_value / sleeve_value) if sleeve_value > 0 else 0.0
    cash = max(Decimal("0"), sleeve_value - position_value)

    portfolio = PortfolioState(
        cash=float(cash),
        positions={symbol: PositionState(quantity=float(base_qty), avg_cost=float(price) if base_qty > 0 else 0.0)},
    )
    bootstrap_order = plan_bootstrap_order(
        symbol=symbol,
        filters=filters,
        price=price,
        position_value=position_value,
        current_pct=current_pct,
        sleeve_value=sleeve_value,
        available_quote=available_quote,
        bootstrap_target_pct=bootstrap_target_pct,
        bootstrap_cap=bootstrap_cap,
        bootstrap_threshold_pct=bootstrap_threshold_pct,
        min_order_usdt=min_order_usdt,
        max_order_usdt=max_order_usdt,
    )
    if bootstrap_order is not None:
        return bootstrap_order

    strategy = build_strategy(
        strategy_name,
        float(sleeve_value),
        config["capital"]["reserve"],
        config["cost"]["fee_rate"],
        min_notional=config.get("cost", {}).get("min_notional"),
    )
    setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
    decision_portfolio = PortfolioState(
        cash=float(cash),
        positions={symbol: PositionState(quantity=float(base_qty), avg_cost=float(price) if base_qty > 0 else 0.0)},
    )
    raw_actions = strategy.compute_actions({symbol: df}, decision_portfolio, {symbol: float(price)})
    for raw_action in [item for item in raw_actions if item.side == "sell"]:
        execute_action_in_backtest(raw_action, decision_portfolio, float(config["cost"]["fee_rate"]))
    for raw_action in [item for item in raw_actions if item.side == "buy"]:
        execute_action_in_backtest(raw_action, decision_portfolio, float(config["cost"]["fee_rate"]))
    actions = v47_position_book.build_rebalance_actions(
        strategy=strategy,
        decision_portfolio=decision_portfolio,
        execution_portfolio=portfolio,
        candles_by_symbol={symbol: df},
        execution_prices={symbol: float(price)},
        fee_rate=float(config["cost"]["fee_rate"]),
        raw_actions=raw_actions,
    )
    if not actions:
        return None

    action = actions[0]
    requested_notional = Decimal(str(action.quantity)) * Decimal(str(action.price))
    notional = requested_notional
    clip_reasons = []
    if max_order_usdt > 0:
        if notional > max_order_usdt:
            clip_reasons.append("max_order_usdt")
        notional = min(notional, max_order_usdt)

    if action.side == "buy":
        if notional > available_quote:
            clip_reasons.append("available_quote")
        notional = min(notional, available_quote)
        if notional < max(min_order_usdt, filters.min_notional):
            return None
        quote_order_qty = quantize_quote(notional)
        return PlannedOrder(
            symbol=symbol,
            api_symbol=api_symbol(symbol),
            side="BUY",
            quantity=None,
            quote_order_qty=quote_order_qty,
            price=price,
            notional=quote_order_qty,
            requested_notional=requested_notional,
            clip_reason=",".join(clip_reasons),
            reason=action.reason,
            current_pct=current_pct,
            sleeve_value=sleeve_value,
        )

    if notional < max(min_order_usdt, filters.min_notional):
        return None
    quantity = round_step(notional / price, filters.step_size)
    max_quantity = round_step(base_qty, filters.step_size)
    if quantity > max_quantity:
        clip_reasons.append("base_balance")
    quantity = min(quantity, max_quantity)
    if quantity < filters.min_qty or quantity * price < max(min_order_usdt, filters.min_notional):
        return None
    return PlannedOrder(
        symbol=symbol,
        api_symbol=api_symbol(symbol),
        side="SELL",
        quantity=quantity,
        quote_order_qty=None,
        price=price,
        notional=quantity * price,
        requested_notional=requested_notional,
        clip_reason=",".join(clip_reasons),
        reason=action.reason,
        current_pct=current_pct,
        sleeve_value=sleeve_value,
    )


def plan_bootstrap_order(
    *,
    symbol: str,
    filters: SymbolFilters,
    price: Decimal,
    position_value: Decimal,
    current_pct: float,
    sleeve_value: Decimal,
    available_quote: Decimal,
    bootstrap_target_pct: float,
    bootstrap_cap: Decimal,
    bootstrap_threshold_pct: float,
    min_order_usdt: Decimal,
    max_order_usdt: Decimal,
) -> PlannedOrder | None:
    if bootstrap_cap <= 0 or current_pct >= bootstrap_threshold_pct:
        return None
    target_pct = min(Decimal(str(max(0.0, bootstrap_target_pct))), bootstrap_cap)
    desired_value = sleeve_value * target_pct
    requested_notional = max(Decimal("0"), desired_value - position_value)
    if requested_notional <= 0:
        return None
    notional = requested_notional
    clip_reasons = []
    if max_order_usdt > 0 and notional > max_order_usdt:
        notional = max_order_usdt
        clip_reasons.append("max_order_usdt")
    if notional > available_quote:
        notional = available_quote
        clip_reasons.append("available_quote")
    if notional < max(min_order_usdt, filters.min_notional):
        return None
    quote_order_qty = quantize_quote(notional)
    return PlannedOrder(
        symbol=symbol,
        api_symbol=api_symbol(symbol),
        side="BUY",
        quantity=None,
        quote_order_qty=quote_order_qty,
        price=price,
        notional=quote_order_qty,
        requested_notional=requested_notional,
        clip_reason=",".join(clip_reasons),
        reason=f"{DEFAULT_STRATEGY}_bootstrap-position-align_target{target_pct:.0%}",
        current_pct=current_pct,
        sleeve_value=sleeve_value,
    )


def native_latest_position_pct(
    *,
    symbol: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    strategy_name: str,
    sleeve_value: Decimal,
) -> float:
    strategy = build_strategy(
        strategy_name,
        float(sleeve_value),
        config["capital"]["reserve"],
        config["cost"]["fee_rate"],
        min_notional=config.get("cost", {}).get("min_notional"),
    )
    setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
    result = run_rebalance_backtest(
        {symbol: df},
        strategy,
        initial_capital=float(sleeve_value),
        reserve=config["capital"]["reserve"],
        fee_rate=config["cost"]["fee_rate"],
        execution_mode=config.get("execution", {}).get("mode", "next_open"),
    )
    if result.empty:
        return 0.0
    latest = result.iloc[-1]
    total = float(latest.get("total_value", 0.0) or 0.0)
    value = float(latest.get(f"{symbol}_value", 0.0) or 0.0)
    return max(0.0, min(1.0, value / total if total > 0 else 0.0))


def submit_market_order(client: BinanceSpotTestnetClient, order: PlannedOrder) -> Any:
    params: dict[str, Any] = {
        "symbol": order.api_symbol,
        "side": order.side,
        "type": "MARKET",
    }
    if order.side == "BUY":
        params["quoteOrderQty"] = format_decimal(order.quote_order_qty)
    else:
        params["quantity"] = format_decimal(order.quantity)
    return client.signed_post("/v3/order", params)


def parse_filters(exchange_info: dict[str, Any]) -> dict[str, SymbolFilters]:
    out = {}
    for item in exchange_info["symbols"]:
        filters = {entry["filterType"]: entry for entry in item["filters"]}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
        out[item["symbol"]] = SymbolFilters(
            base_asset=item["baseAsset"],
            quote_asset=item["quoteAsset"],
            step_size=Decimal(str(lot.get("stepSize", "0.000001"))),
            min_qty=Decimal(str(lot.get("minQty", "0"))),
            min_notional=Decimal(str(notional.get("minNotional", "0"))),
        )
    return out


def parse_balances(account: dict[str, Any]) -> dict[str, Decimal]:
    return {
        row["asset"]: Decimal(str(row["free"])) + Decimal(str(row["locked"]))
        for row in account.get("balances", [])
    }


def account_equity_usdt(balances: dict[str, Decimal], prices: dict[str, Decimal], *, quote_asset: str) -> Decimal:
    total = balances.get(quote_asset, Decimal("0"))
    for symbol, price in prices.items():
        base = symbol.split("/", 1)[0]
        total += balances.get(base, Decimal("0")) * price
    return total


def round_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def quantize_quote(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_DOWN)


def format_decimal(value: Decimal | None) -> str:
    if value is None:
        raise ValueError("decimal value is required")
    return format(value.normalize(), "f")


def api_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def order_to_dict(order: PlannedOrder) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "api_symbol": order.api_symbol,
        "side": order.side,
        "quantity": format_decimal(order.quantity) if order.quantity is not None else None,
        "quote_order_qty": format_decimal(order.quote_order_qty) if order.quote_order_qty is not None else None,
        "price": str(order.price),
        "notional": str(order.notional),
        "requested_notional": str(order.requested_notional),
        "clip_reason": order.clip_reason,
        "reason": order.reason,
        "current_pct": order.current_pct,
        "sleeve_value": str(order.sleeve_value),
    }


if __name__ == "__main__":
    main()
