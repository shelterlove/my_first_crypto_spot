#!/usr/bin/env python3
"""Execute Official V1 ETH/BNB targets on Binance USD-M Futures Testnet.

The exchange leverage is an integer account setting. The strategy's actual
gross exposure is controlled separately by target_gross_cap and order sizing.

Default mode is dry-run. Pass --execute to submit MARKET orders to testnet.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from futures_v1.backtest_event_driven import run_rebalance_backtest  # noqa: E402
from futures_v1.benchmark import build_strategy  # noqa: E402
from scripts.generate_daily_signal import _load_data_with_btc_regime  # noqa: E402


FUTURES_TESTNET_BASE_URL = "https://demo-fapi.binance.com"
DEFAULT_STRATEGY = "eth_bnb_futures_v1"


@dataclass(frozen=True)
class FuturesSymbolFilters:
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class FuturesPosition:
    symbol: str
    position_amt: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Decimal
    leverage: int
    margin_type: str


@dataclass(frozen=True)
class VirtualSleeve:
    symbol: str
    api_symbol: str
    total_value: Decimal
    pnl_since_last_run: Decimal
    external_adjustment: Decimal
    previous_position_amt: Decimal
    previous_mark_price: Decimal
    current_position_amt: Decimal
    current_mark_price: Decimal
    previous_target_gross: Decimal | None


@dataclass
class PlannedFuturesOrder:
    symbol: str
    side: str
    quantity: Decimal
    reduce_only: bool
    mark_price: Decimal
    notional: Decimal
    requested_notional_delta: Decimal
    target_gross: Decimal
    current_gross: Decimal
    exchange_leverage: int
    target_gross_cap: Decimal
    clip_reason: str
    reason: str

class BinanceUsdMFuturesTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = FUTURES_TESTNET_BASE_URL):
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
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance Futures API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Binance Futures API connection error: {exc}") from exc
        return json.loads(raw) if raw else {}

    def server_time(self) -> int:
        try:
            with urllib.request.urlopen(f"{self.base_url}/fapi/v1/time", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return int(payload["serverTime"])
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Binance Futures API connection error: {exc}") from exc

    def set_margin_type_isolated(self, symbol: str) -> Any:
        try:
            return self.signed_post("/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"})
        except RuntimeError as exc:
            if "-4046" in str(exc) or "No need to change margin type" in str(exc):
                return {"symbol": symbol, "marginType": "ISOLATED", "status": "unchanged"}
            raise

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self.signed_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest_v1.json")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--base-url", default=os.getenv("BINANCE_FUTURES_TESTNET_BASE_URL", FUTURES_TESTNET_BASE_URL))
    parser.add_argument("--api-key-env", default="BINANCE_FUTURES_TESTNET_API_KEY")
    parser.add_argument("--api-secret-env", default="BINANCE_FUTURES_TESTNET_API_SECRET")
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--sleeve-fraction", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--exchange-leverage", type=int, default=3, help="Integer leverage setting sent to Binance.")
    parser.add_argument("--max-exchange-leverage", type=int, default=3, help="Hard safety limit for exchange leverage.")
    parser.add_argument("--target-gross-cap", type=Decimal, default=Decimal("3.00"), help="Fractional strategy gross cap, e.g. 3.0 means 300%.")
    parser.add_argument("--hard-target-gross-limit", type=Decimal, default=Decimal("3.00"), help="Hard safety limit for target_gross_cap.")
    parser.add_argument("--min-order-usdt", type=Decimal, default=Decimal("10"))
    parser.add_argument("--max-order-usdt", type=Decimal, default=Decimal("0"), help="0 disables this safety cap.")
    parser.add_argument("--state-file", default="runtime/futures_state.json")
    parser.add_argument("--allow-other-positions", action="store_true", help="Allow unrelated non-zero futures positions in the same account.")
    parser.add_argument("--execute", action="store_true", help="Submit MARKET futures orders. Omit for dry-run.")
    parser.add_argument("--output-dir", default="results/binance_futures_testnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_testnet_base_url(args.base_url)
    validate_leverage_args(
        exchange_leverage=args.exchange_leverage,
        max_exchange_leverage=args.max_exchange_leverage,
        target_gross_cap=args.target_gross_cap,
        hard_target_gross_limit=args.hard_target_gross_limit,
    )
    validate_deployment_args(
        sleeve_fraction=args.sleeve_fraction,
        min_order_usdt=args.min_order_usdt,
        max_order_usdt=args.max_order_usdt,
    )
    api_key = os.getenv(args.api_key_env, "")
    api_secret = os.getenv(args.api_secret_env, "")
    if not api_key or not api_secret:
        raise SystemExit(
            f"Missing API credentials. Set {args.api_key_env} and {args.api_secret_env}; "
            "use Binance USD-M Futures Testnet keys only."
        )

    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    symbols = list(config["symbols"])
    validate_deployment_symbols(symbols)

    client = BinanceUsdMFuturesTestnetClient(api_key, api_secret, args.base_url)
    ensure_one_way_position_mode(client)
    all_dfs = _load_data_with_btc_regime(config)
    validate_fresh_daily_data(all_dfs, symbols + config.get("reference_symbols", []))
    api_symbols = [api_symbol(symbol) for symbol in symbols]
    exchange_info = client.public_get("/fapi/v1/exchangeInfo")
    filters = parse_futures_filters(exchange_info, set(api_symbols))

    setup_responses = []
    setup_plan = [
        {"symbol": fsymbol, "marginType": "ISOLATED", "leverage": args.exchange_leverage}
        for fsymbol in api_symbols
    ]
    account = client.signed_get("/fapi/v2/account")
    all_positions = client.signed_get("/fapi/v2/positionRisk")
    reject_unrelated_positions(all_positions, set(api_symbols), allow=args.allow_other_positions)
    reject_short_target_positions(all_positions, set(api_symbols))

    if args.execute:
        for fsymbol in api_symbols:
            setup_responses.append(client.set_margin_type_isolated(fsymbol))
            setup_responses.append(client.set_leverage(fsymbol, args.exchange_leverage))
        account = client.signed_get("/fapi/v2/account")
        all_positions = client.signed_get("/fapi/v2/positionRisk")
        reject_unrelated_positions(all_positions, set(api_symbols), allow=args.allow_other_positions)
        reject_short_target_positions(all_positions, set(api_symbols))

    positions = parse_positions(all_positions, set(api_symbols))

    for symbol in symbols:
        fsymbol = api_symbol(symbol)
        position = positions.get(fsymbol)
        if position is None:
            mark_price = futures_mark_price(client, fsymbol)
            position = FuturesPosition(
                symbol=fsymbol,
                position_amt=Decimal("0"),
                entry_price=Decimal("0"),
                mark_price=mark_price,
                liquidation_price=Decimal("0"),
                leverage=args.exchange_leverage,
                margin_type="isolated",
            )
            positions[fsymbol] = position
        elif position.mark_price <= 0:
            position = FuturesPosition(
                symbol=position.symbol,
                position_amt=position.position_amt,
                entry_price=position.entry_price,
                mark_price=futures_mark_price(client, fsymbol),
                liquidation_price=position.liquidation_price,
                leverage=position.leverage,
                margin_type=position.margin_type,
            )
            positions[fsymbol] = position

    account_equity = futures_account_equity(account, args.quote_asset)
    if account_equity <= 0:
        raise SystemExit(f"No positive {args.quote_asset} futures account equity available.")
    deploy_equity = account_equity * args.sleeve_fraction
    state_path = PROJECT_ROOT / args.state_file
    state = load_state(state_path)
    virtual_sleeves = build_virtual_sleeves(
        state=state,
        symbols=symbols,
        positions=positions,
        deploy_equity=deploy_equity,
    )

    planned = []
    latest_targets: dict[str, Decimal] = {}
    for symbol in symbols:
        fsymbol = api_symbol(symbol)
        sleeve = virtual_sleeves[symbol]
        raw_target_gross = native_latest_gross_pct(
            symbol=symbol,
            df=all_dfs[symbol],
            config=config,
            strategy_name=args.strategy,
            sleeve_value=sleeve.total_value,
        )
        target_gross = min(Decimal(str(max(0.0, raw_target_gross))), args.target_gross_cap)
        latest_targets[fsymbol] = target_gross
        order = plan_futures_order(
            api_symbol=fsymbol,
            strategy_name=args.strategy,
            position=positions[fsymbol],
            filters=filters[fsymbol],
            sleeve_value=sleeve.total_value,
            exchange_leverage=args.exchange_leverage,
            target_gross_cap=args.target_gross_cap,
            target_gross=target_gross,
            min_order_usdt=args.min_order_usdt,
            max_order_usdt=args.max_order_usdt,
        )
        if isinstance(order, PlannedFuturesOrder):
            planned.append(order)

    responses = []
    if args.execute:
        for order in planned:
            responses.append(submit_futures_market_order(client, order))
        all_positions = client.signed_get("/fapi/v2/positionRisk")
        positions = parse_positions(all_positions, set(api_symbols))
        save_state(state_path, build_next_state(symbols, virtual_sleeves, latest_targets, positions))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "mode": "execute" if args.execute else "dry_run",
        "base_url": args.base_url,
        "strategy": args.strategy,
        "symbols": symbols,
        "wallet_balance_usdt": str(account_equity),
        "account_equity_usdt": str(account_equity),
        "deploy_equity_usdt": str(deploy_equity),
        "symbol_sleeves": {
            symbol: virtual_sleeve_to_dict(virtual_sleeves[symbol], latest_targets.get(api_symbol(symbol)))
            for symbol in symbols
        },
        "exchange_leverage": args.exchange_leverage,
        "max_exchange_leverage": args.max_exchange_leverage,
        "target_gross_cap": str(args.target_gross_cap),
        "hard_target_gross_limit": str(args.hard_target_gross_limit),
        "state_file": str(state_path),
        "positions": {
            symbol: futures_position_to_dict(positions.get(api_symbol(symbol)))
            for symbol in symbols
        },
        "setup_plan": setup_plan,
        "setup_responses": setup_responses,
        "orders": [futures_order_to_dict(order) for order in planned],
        "responses": responses,
    }
    out_path = output_dir / f"{timestamp}_{args.strategy}_{'execute' if args.execute else 'dry_run'}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"report={out_path}")
    print(pd.DataFrame(report["orders"]).to_string(index=False) if report["orders"] else "No actionable orders.")


def validate_leverage_args(
    *,
    exchange_leverage: int,
    max_exchange_leverage: int,
    target_gross_cap: Decimal,
    hard_target_gross_limit: Decimal,
) -> None:
    if exchange_leverage < 1:
        raise SystemExit("--exchange-leverage must be an integer >= 1.")
    if max_exchange_leverage < 1:
        raise SystemExit("--max-exchange-leverage must be an integer >= 1.")
    if exchange_leverage > max_exchange_leverage:
        raise SystemExit("--exchange-leverage must not exceed --max-exchange-leverage.")
    if target_gross_cap <= 0:
        raise SystemExit("--target-gross-cap must be > 0.")
    if hard_target_gross_limit <= 0:
        raise SystemExit("--hard-target-gross-limit must be > 0.")
    if target_gross_cap > hard_target_gross_limit:
        raise SystemExit("--target-gross-cap must not exceed --hard-target-gross-limit.")
    if target_gross_cap > Decimal(str(exchange_leverage)):
        raise SystemExit("--target-gross-cap must not exceed --exchange-leverage.")


def validate_deployment_args(*, sleeve_fraction: Decimal, min_order_usdt: Decimal, max_order_usdt: Decimal) -> None:
    if sleeve_fraction <= 0 or sleeve_fraction > 1:
        raise SystemExit("--sleeve-fraction must be > 0 and <= 1.")
    if min_order_usdt < 0:
        raise SystemExit("--min-order-usdt must be >= 0.")
    if max_order_usdt < 0:
        raise SystemExit("--max-order-usdt must be >= 0.")


def validate_deployment_symbols(symbols: list[str]) -> None:
    expected = {"ETH/USDT", "BNB/USDT"}
    actual = set(symbols)
    if actual != expected:
        raise SystemExit(f"Official V1 futures deployment symbols must be exactly {sorted(expected)}, got {symbols}.")


def validate_testnet_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    allowed = urllib.parse.urlparse(FUTURES_TESTNET_BASE_URL)
    if (parsed.scheme, parsed.netloc, parsed.path.rstrip("/")) != (allowed.scheme, allowed.netloc, allowed.path.rstrip("/")):
        raise SystemExit(f"Refusing non-testnet Binance Futures base URL: {base_url}")


def validate_fresh_daily_data(all_dfs: dict[str, pd.DataFrame], symbols: list[str], *, max_age_hours: float = 72.0) -> None:
    now = pd.Timestamp.now("UTC")
    stale = []
    for symbol in dict.fromkeys(symbols):
        df = all_dfs.get(symbol)
        if df is None or df.empty:
            stale.append(f"{symbol}:missing")
            continue
        latest = pd.Timestamp(df["timestamp"].iloc[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        age_hours = (now - latest).total_seconds() / 3600.0
        if age_hours > max_age_hours:
            stale.append(f"{symbol}:{latest.isoformat()} age={age_hours:.1f}h")
    if stale:
        raise SystemExit(f"Daily candle data is stale or missing: {', '.join(stale)}")


def ensure_one_way_position_mode(client: BinanceUsdMFuturesTestnetClient) -> None:
    mode = client.signed_get("/fapi/v1/positionSide/dual")
    if bool(mode.get("dualSidePosition", False)):
        raise SystemExit("Hedge Mode is enabled. Switch USD-M Futures Testnet to One-way Mode before using this executor.")


def reject_unrelated_positions(rows: list[dict[str, Any]], wanted_symbols: set[str], *, allow: bool) -> None:
    if allow:
        return
    unrelated = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if symbol in wanted_symbols:
            continue
        amount = Decimal(str(row.get("positionAmt", "0")))
        if amount != 0:
            unrelated.append(f"{symbol}:{amount}")
    if unrelated:
        raise SystemExit(
            "Unrelated non-zero futures positions detected. Close them or pass "
            f"--allow-other-positions explicitly: {', '.join(unrelated[:10])}"
        )


def reject_short_target_positions(rows: list[dict[str, Any]], wanted_symbols: set[str]) -> None:
    shorts = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if symbol not in wanted_symbols:
            continue
        amount = Decimal(str(row.get("positionAmt", "0")))
        if amount < 0:
            shorts.append(f"{symbol}:{amount}")
    if shorts:
        raise SystemExit(f"Short target positions detected. Close them before running Official V1 long-only executor: {', '.join(shorts)}")


def plan_futures_order(
    *,
    api_symbol: str,
    strategy_name: str,
    position: FuturesPosition,
    filters: FuturesSymbolFilters,
    sleeve_value: Decimal,
    exchange_leverage: int,
    target_gross_cap: Decimal,
    target_gross: Decimal,
    min_order_usdt: Decimal,
    max_order_usdt: Decimal,
) -> PlannedFuturesOrder | None:
    if position.position_amt < 0:
        raise SystemExit(f"{api_symbol} has a short position. This Official V1 executor is long-only.")
    mark_price = position.mark_price
    if mark_price <= 0:
        return None
    current_notional = position.position_amt * mark_price
    current_gross = current_notional / sleeve_value if sleeve_value > 0 else Decimal("0")
    desired_notional = sleeve_value * target_gross
    delta_notional = desired_notional - current_notional
    if abs(delta_notional) < max(min_order_usdt, filters.min_notional):
        return None
    side = "BUY" if delta_notional > 0 else "SELL"

    clip_reasons = []
    notional = abs(delta_notional)
    if max_order_usdt > 0 and notional > max_order_usdt:
        notional = max_order_usdt
        clip_reasons.append("max_order_usdt")
    quantity = round_step(notional / mark_price, filters.step_size)
    if quantity < filters.min_qty or quantity * mark_price < max(min_order_usdt, filters.min_notional):
        return None
    if delta_notional < 0:
        max_qty = round_step(position.position_amt, filters.step_size)
        if quantity > max_qty:
            quantity = max_qty
            clip_reasons.append("position_size")
        if quantity <= 0:
            return None

    return PlannedFuturesOrder(
        symbol=api_symbol,
        side=side,
        quantity=quantity,
        reduce_only=side == "SELL",
        mark_price=mark_price,
        notional=quantity * mark_price,
        requested_notional_delta=delta_notional,
        target_gross=target_gross,
        current_gross=current_gross,
        exchange_leverage=exchange_leverage,
        target_gross_cap=target_gross_cap,
        clip_reason=",".join(clip_reasons),
        reason=(
            f"{strategy_name}_futures-virtual-sleeve"
            f"_exchangeLev{exchange_leverage}x"
            f"_targetGross{target_gross:.2f}"
        ),
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"symbols": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid futures state file {path}: {exc}") from exc


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_path.replace(path)


def build_virtual_sleeves(
    *,
    state: dict[str, Any],
    symbols: list[str],
    positions: dict[str, FuturesPosition],
    deploy_equity: Decimal,
) -> dict[str, VirtualSleeve]:
    default_value = deploy_equity / Decimal(len(symbols))
    raw_sleeves = []
    for symbol in symbols:
        fsymbol = api_symbol(symbol)
        position = positions[fsymbol]
        record = state.get("symbols", {}).get(fsymbol, {})
        previous_total = decimal_field(record, "virtual_total_value_usdt", default_value)
        previous_position_amt = decimal_field(record, "last_position_amt", position.position_amt)
        previous_mark_price = decimal_field(record, "last_mark_price", position.mark_price)
        previous_target = optional_decimal_field(record, "target_gross")
        pnl = Decimal("0")
        if previous_mark_price > 0 and position.mark_price > 0:
            pnl = previous_position_amt * (position.mark_price - previous_mark_price)
        raw_sleeves.append(
            {
                "symbol": symbol,
                "api_symbol": fsymbol,
                "pre_adjust_total": previous_total + pnl,
                "pnl": pnl,
                "previous_position_amt": previous_position_amt,
                "previous_mark_price": previous_mark_price,
                "position": position,
                "previous_target": previous_target,
            }
        )

    external_delta = deploy_equity - sum(item["pre_adjust_total"] for item in raw_sleeves)
    external_adjustment = external_delta / Decimal(len(raw_sleeves))
    out: dict[str, VirtualSleeve] = {}
    for item in raw_sleeves:
        total = item["pre_adjust_total"] + external_adjustment
        if total <= 0:
            raise SystemExit(f"Virtual sleeve equity for {item['symbol']} is not positive: {total}. Stop and review account transfers/risk.")
        position = item["position"]
        out[item["symbol"]] = VirtualSleeve(
            symbol=item["symbol"],
            api_symbol=item["api_symbol"],
            total_value=total,
            pnl_since_last_run=item["pnl"],
            external_adjustment=external_adjustment,
            previous_position_amt=item["previous_position_amt"],
            previous_mark_price=item["previous_mark_price"],
            current_position_amt=position.position_amt,
            current_mark_price=position.mark_price,
            previous_target_gross=item["previous_target"],
        )
    return out


def build_next_state(
    symbols: list[str],
    sleeves: dict[str, VirtualSleeve],
    latest_targets: dict[str, Decimal],
    positions: dict[str, FuturesPosition],
) -> dict[str, Any]:
    next_symbols = {}
    for symbol in symbols:
        fsymbol = api_symbol(symbol)
        sleeve = sleeves[symbol]
        position = positions.get(fsymbol)
        position_amt = sleeve.current_position_amt if position is None else position.position_amt
        mark_price = sleeve.current_mark_price if position is None else position.mark_price
        next_symbols[fsymbol] = {
            "symbol": symbol,
            "virtual_total_value_usdt": str(sleeve.total_value),
            "target_gross": str(latest_targets[fsymbol]),
            "last_position_amt": str(position_amt),
            "last_mark_price": str(mark_price),
            "updated_at": pd.Timestamp.now("UTC").isoformat(),
        }
    return {
        "updated_at": pd.Timestamp.now("UTC").isoformat(),
        "symbols": next_symbols,
    }


def decimal_field(record: dict[str, Any], key: str, default: Decimal) -> Decimal:
    raw = record.get(key)
    if raw is None:
        return default
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise SystemExit(f"Invalid decimal field in futures state: {key}={raw!r}") from exc


def optional_decimal_field(record: dict[str, Any], key: str) -> Decimal | None:
    raw = record.get(key)
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise SystemExit(f"Invalid decimal field in futures state: {key}={raw!r}") from exc


def virtual_sleeve_to_dict(sleeve: VirtualSleeve, latest_target_gross: Decimal | None) -> dict[str, Any]:
    return {
        "api_symbol": sleeve.api_symbol,
        "virtual_total_value_usdt": str(sleeve.total_value),
        "pnl_since_last_run": str(sleeve.pnl_since_last_run),
        "external_adjustment": str(sleeve.external_adjustment),
        "previous_position_amt": str(sleeve.previous_position_amt),
        "current_position_amt": str(sleeve.current_position_amt),
        "previous_mark_price": str(sleeve.previous_mark_price),
        "current_mark_price": str(sleeve.current_mark_price),
        "previous_target_gross": None if sleeve.previous_target_gross is None else str(sleeve.previous_target_gross),
        "latest_target_gross": None if latest_target_gross is None else str(latest_target_gross),
    }


def native_latest_gross_pct(
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
    return max(0.0, value / total if total > 0 else 0.0)


def submit_futures_market_order(client: BinanceUsdMFuturesTestnetClient, order: PlannedFuturesOrder) -> Any:
    params: dict[str, Any] = {
        "symbol": order.symbol,
        "side": order.side,
        "type": "MARKET",
        "quantity": format_decimal(order.quantity),
    }
    if order.reduce_only:
        params["reduceOnly"] = "true"
    return client.signed_post("/fapi/v1/order", params)


def parse_futures_filters(exchange_info: dict[str, Any], wanted_symbols: set[str]) -> dict[str, FuturesSymbolFilters]:
    out = {}
    for item in exchange_info.get("symbols", []):
        symbol = item.get("symbol")
        if symbol not in wanted_symbols:
            continue
        filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
        out[symbol] = FuturesSymbolFilters(
            step_size=Decimal(str(lot.get("stepSize", "0.001"))),
            min_qty=Decimal(str(lot.get("minQty", "0"))),
            min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "0")))),
        )
    missing = wanted_symbols - set(out)
    if missing:
        raise SystemExit(f"Missing futures symbol filters: {sorted(missing)}")
    return out


def parse_positions(rows: list[dict[str, Any]], wanted_symbols: set[str]) -> dict[str, FuturesPosition]:
    out = {}
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if symbol not in wanted_symbols:
            continue
        out[symbol] = FuturesPosition(
            symbol=symbol,
            position_amt=Decimal(str(row.get("positionAmt", "0"))),
            entry_price=Decimal(str(row.get("entryPrice", "0"))),
            mark_price=Decimal(str(row.get("markPrice", "0"))),
            liquidation_price=Decimal(str(row.get("liquidationPrice", "0"))),
            leverage=int(row.get("leverage", 1)),
            margin_type=str(row.get("marginType", "")),
        )
    return out


def futures_account_equity(account: dict[str, Any], quote_asset: str) -> Decimal:
    total_margin_balance = account.get("totalMarginBalance")
    if total_margin_balance is not None:
        return Decimal(str(total_margin_balance))
    for asset in account.get("assets", []):
        if asset.get("asset") == quote_asset:
            margin_balance = asset.get("marginBalance")
            if margin_balance is not None:
                return Decimal(str(margin_balance))
            wallet = Decimal(str(asset.get("walletBalance", "0")))
            unrealized = Decimal(str(asset.get("unrealizedProfit", "0")))
            return wallet + unrealized
    return Decimal("0")


def futures_mark_price(client: BinanceUsdMFuturesTestnetClient, symbol: str) -> Decimal:
    payload = client.public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return Decimal(str(payload["markPrice"]))


def round_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def api_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def futures_order_to_dict(order: PlannedFuturesOrder) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "quantity": format_decimal(order.quantity),
        "reduce_only": order.reduce_only,
        "mark_price": str(order.mark_price),
        "notional": str(order.notional),
        "requested_notional_delta": str(order.requested_notional_delta),
        "target_gross": str(order.target_gross),
        "current_gross": str(order.current_gross),
        "exchange_leverage": order.exchange_leverage,
        "target_gross_cap": str(order.target_gross_cap),
        "clip_reason": order.clip_reason,
        "reason": order.reason,
    }


def futures_position_to_dict(position: FuturesPosition | None) -> dict[str, Any] | None:
    if position is None:
        return None
    liquidation_buffer_pct = None
    if position.position_amt > 0 and position.mark_price > 0 and position.liquidation_price > 0:
        liquidation_buffer_pct = (position.mark_price - position.liquidation_price) / position.mark_price
    return {
        "symbol": position.symbol,
        "position_amt": str(position.position_amt),
        "entry_price": str(position.entry_price),
        "mark_price": str(position.mark_price),
        "liquidation_price": str(position.liquidation_price),
        "liquidation_buffer_pct": str(liquidation_buffer_pct) if liquidation_buffer_pct is not None else None,
        "leverage": position.leverage,
        "margin_type": position.margin_type,
    }


if __name__ == "__main__":
    main()
