#!/usr/bin/env python3
"""Execute Official V1 ETH/BNB targets on Binance USD-M Futures Testnet.

The exchange leverage is an integer account setting. The strategy's actual
gross exposure is controlled separately by target_gross_cap and order sizing.

Default mode is dry-run. Pass --execute to submit MARKET orders to testnet.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import hmac
import json
import os
import re
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
    max_qty: Decimal = Decimal("Infinity")


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


@dataclass(frozen=True)
class NativeTargetSnapshot:
    symbol: str
    signal_timestamp: pd.Timestamp
    execution_timestamp: pd.Timestamp
    execution_price: Decimal
    target_gross: Decimal


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
    client_order_id: str

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
    parser.add_argument("--exchange-leverage", type=int, default=2, help="Integer leverage setting sent to Binance.")
    parser.add_argument("--max-exchange-leverage", type=int, default=2, help="Hard safety limit for exchange leverage.")
    parser.add_argument("--target-gross-cap", type=Decimal, default=Decimal("1.25"), help="Soft strategy target gross cap.")
    parser.add_argument("--hard-target-gross-limit", type=Decimal, default=Decimal("1.50"), help="Hard configuration limit for target_gross_cap.")
    parser.add_argument("--hard-account-gross-limit", type=Decimal, default=Decimal("1.50"), help="Hard projected gross limit relative to deployed equity.")
    parser.add_argument("--hard-symbol-gross-limit", type=Decimal, default=Decimal("1.50"), help="Hard projected gross limit for each virtual sleeve.")
    parser.add_argument("--max-deploy-usdt", type=Decimal, default=Decimal("1000"), help="Maximum account equity assigned to this strategy; 0 disables the cap.")
    parser.add_argument("--margin-buffer-fraction", type=Decimal, default=Decimal("0.25"), help="Fraction of available initial margin reserved from BUY orders.")
    parser.add_argument("--min-liquidation-buffer", type=Decimal, default=Decimal("0.30"), help="Minimum mark-to-liquidation distance required before any BUY.")
    parser.add_argument("--min-order-usdt", type=Decimal, default=Decimal("10"))
    parser.add_argument("--max-order-usdt", type=Decimal, default=Decimal("250"), help="Per-symbol order notional cap; 0 disables it.")
    parser.add_argument("--state-file", default="runtime/futures_state.json")
    parser.add_argument("--lock-file", default="runtime/futures_executor.lock")
    parser.add_argument(
        "--allow-nonzero-bootstrap",
        action="store_true",
        help="Explicitly initialize a missing state file while target positions are non-zero.",
    )
    parser.add_argument(
        "--allow-state-rebind",
        action="store_true",
        help="Explicitly adopt a legacy state file that has no deployment binding.",
    )
    parser.add_argument("--allow-other-positions", action="store_true", help="Allow unrelated non-zero futures positions in the same account.")
    parser.add_argument("--execute", action="store_true", help="Submit MARKET futures orders. Omit for dry-run.")
    parser.add_argument("--output-dir", default="results/binance_futures_testnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acquire_process_lock(PROJECT_ROOT / args.lock_file)
    validate_testnet_base_url(args.base_url)
    validate_leverage_args(
        exchange_leverage=args.exchange_leverage,
        max_exchange_leverage=args.max_exchange_leverage,
        target_gross_cap=args.target_gross_cap,
        hard_target_gross_limit=args.hard_target_gross_limit,
        hard_account_gross_limit=args.hard_account_gross_limit,
        hard_symbol_gross_limit=args.hard_symbol_gross_limit,
    )
    validate_deployment_args(
        sleeve_fraction=args.sleeve_fraction,
        min_order_usdt=args.min_order_usdt,
        max_order_usdt=args.max_order_usdt,
        max_deploy_usdt=args.max_deploy_usdt,
        margin_buffer_fraction=args.margin_buffer_fraction,
        min_liquidation_buffer=args.min_liquidation_buffer,
        hard_account_gross_limit=args.hard_account_gross_limit,
        hard_symbol_gross_limit=args.hard_symbol_gross_limit,
    )
    api_key = os.getenv(args.api_key_env, "")
    api_secret = os.getenv(args.api_secret_env, "")
    if not api_key or not api_secret:
        raise SystemExit(
            f"Missing API credentials. Set {args.api_key_env} and {args.api_secret_env}; "
            "use Binance USD-M Futures Testnet keys only."
        )

    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    symbols = list(config["symbols"])
    validate_deployment_symbols(symbols)

    client = BinanceUsdMFuturesTestnetClient(api_key, api_secret, args.base_url)
    ensure_one_way_position_mode(client)
    all_dfs = _load_data_with_btc_regime(config)
    validate_fresh_daily_data(all_dfs, symbols + config.get("reference_symbols", []))
    api_symbols = [api_symbol(symbol) for symbol in symbols]
    exchange_info = client.public_get("/fapi/v1/exchangeInfo")
    filters = parse_futures_filters(exchange_info, set(api_symbols))
    maintenance_brackets = parse_maintenance_brackets(
        client.signed_get("/fapi/v1/leverageBracket"),
        set(api_symbols),
    )

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
    if args.execute:
        validate_position_settings(
            positions,
            set(api_symbols),
            expected_leverage=args.exchange_leverage,
        )

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
    state_path = PROJECT_ROOT / args.state_file
    state = load_state(state_path)
    deploy_equity, unallocated_account_equity = restore_deployment_equity(
        account_equity=account_equity,
        state=state,
        positions=positions,
        symbols=symbols,
        sleeve_fraction=args.sleeve_fraction,
        max_deploy_usdt=args.max_deploy_usdt,
    )
    deployment_id = build_deployment_id(args.base_url, api_key)
    validate_state_binding(
        state,
        deployment_id=deployment_id,
        account_alias=str(account.get("accountAlias", "") or ""),
        config_hash=config_hash,
        allow_state_rebind=args.allow_state_rebind,
    )
    income_audit = fetch_new_income_events(
        client,
        state=state,
        wanted_symbols=set(api_symbols),
        quote_asset=args.quote_asset,
    )
    virtual_sleeves = build_virtual_sleeves(
        state=state,
        symbols=symbols,
        positions=positions,
        deploy_equity=deploy_equity,
        allow_nonzero_bootstrap=args.allow_nonzero_bootstrap,
        known_external_adjustments=income_adjustments_by_symbol(income_audit["events"]),
    )

    planned = []
    latest_targets: dict[str, Decimal] = {}
    target_snapshots: dict[str, NativeTargetSnapshot] = {}
    for symbol in symbols:
        fsymbol = api_symbol(symbol)
        sleeve = virtual_sleeves[symbol]
        target_snapshot = native_latest_target_snapshot(
            symbol=symbol,
            df=all_dfs[symbol],
            config=config,
            strategy_name=args.strategy,
            sleeve_value=sleeve.total_value,
            execution_price=positions[fsymbol].mark_price,
        )
        target_snapshots[fsymbol] = target_snapshot
        target_gross = min(max(Decimal("0"), target_snapshot.target_gross), args.target_gross_cap)
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
            client_order_id=build_client_order_id(
                strategy_name=args.strategy,
                api_symbol=fsymbol,
                signal_timestamp=target_snapshot.signal_timestamp,
                target_gross=target_gross,
            ),
        )
        if isinstance(order, PlannedFuturesOrder):
            planned.append(order)

    planned.sort(key=lambda item: 0 if item.side == "SELL" else 1)
    risk_before = validate_projected_account_gross(
        positions=positions,
        orders=planned,
        deploy_equity=deploy_equity,
        hard_account_gross_limit=args.hard_account_gross_limit,
    )
    symbol_risk_before = validate_projected_symbol_gross(
        positions=positions,
        orders=planned,
        sleeve_values={api_symbol(symbol): sleeve.total_value for symbol, sleeve in virtual_sleeves.items()},
        hard_symbol_gross_limit=args.hard_symbol_gross_limit,
    )
    validate_liquidation_buffers(
        positions,
        planned,
        min_liquidation_buffer=args.min_liquidation_buffer,
    )
    validate_planned_buying_power(
        planned,
        account,
        args.quote_asset,
        margin_buffer_fraction=args.margin_buffer_fraction,
    )
    responses = []
    pre_submit_position_checks = []
    risk_after = risk_before
    symbol_risk_after = symbol_risk_before
    post_trade_income_audit = {"events": [], "cursor_ms": income_audit["cursor_ms"]}
    if args.execute:
        for order in planned:
            live_position = validate_live_position_before_order(
                client,
                order=order,
                planned_position=positions[order.symbol],
                expected_leverage=args.exchange_leverage,
            )
            pre_submit_position_checks.append(futures_position_to_dict(live_position))
            responses.append(submit_futures_market_order(client, order))
        account = client.signed_get("/fapi/v2/account")
        all_positions = client.signed_get("/fapi/v2/positionRisk")
        reject_unrelated_positions(all_positions, set(api_symbols), allow=args.allow_other_positions)
        reject_short_target_positions(all_positions, set(api_symbols))
        positions = parse_positions(all_positions, set(api_symbols))
        validate_position_settings(
            positions,
            set(api_symbols),
            expected_leverage=args.exchange_leverage,
        )
        account_equity = futures_account_equity(account, args.quote_asset)
        deploy_equity = account_equity - unallocated_account_equity
        if deploy_equity <= 0:
            raise SystemExit("Post-trade deployed equity is not positive. Stop and review account risk.")
        risk_after = account_gross_snapshot(positions, deploy_equity)
        post_trade_income_audit = fetch_new_income_events(
            client,
            state={
                "income_cursor_ms": income_audit["query_end_ms"],
                "processed_income_ids": income_audit["processed_ids"],
            },
            wanted_symbols=set(api_symbols),
            quote_asset=args.quote_asset,
        )
        virtual_sleeves = reconcile_post_trade_sleeves(
            symbols=symbols,
            sleeves=virtual_sleeves,
            positions=positions,
            deploy_equity=deploy_equity,
            known_external_adjustments=income_adjustments_by_symbol(post_trade_income_audit["events"]),
        )
        symbol_risk_after = symbol_gross_snapshot(
            positions,
            {api_symbol(symbol): sleeve.total_value for symbol, sleeve in virtual_sleeves.items()},
            hard_symbol_gross_limit=args.hard_symbol_gross_limit,
        )
        save_state(state_path, build_next_state(
            symbols,
            virtual_sleeves,
            latest_targets,
            positions,
            deployment_id=deployment_id,
            account_alias=str(account.get("accountAlias", "") or ""),
            config_hash=config_hash,
            income_cursor_ms=post_trade_income_audit["cursor_ms"],
            processed_income_ids=post_trade_income_audit["processed_ids"],
            unallocated_account_equity=unallocated_account_equity,
        ))

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
        "target_snapshots": {
            symbol: native_target_snapshot_to_dict(target_snapshots[api_symbol(symbol)])
            for symbol in symbols
        },
        "exchange_leverage": args.exchange_leverage,
        "max_exchange_leverage": args.max_exchange_leverage,
        "target_gross_cap": str(args.target_gross_cap),
        "hard_target_gross_limit": str(args.hard_target_gross_limit),
        "hard_account_gross_limit": str(args.hard_account_gross_limit),
        "hard_symbol_gross_limit": str(args.hard_symbol_gross_limit),
        "max_deploy_usdt": str(args.max_deploy_usdt),
        "unallocated_account_equity_usdt": str(unallocated_account_equity),
        "margin_buffer_fraction": str(args.margin_buffer_fraction),
        "min_liquidation_buffer": str(args.min_liquidation_buffer),
        "account_gross_before": risk_before,
        "account_gross_after": risk_after,
        "symbol_gross_before": symbol_risk_before,
        "symbol_gross_after": symbol_risk_after,
        "income_audit": {
            "pre_trade_events": income_audit["events"],
            "post_trade_events": post_trade_income_audit["events"],
            "excluded_income_types": ["REALIZED_PNL"],
        },
        "state_file": str(state_path),
        "state_updated": bool(args.execute),
        "deployment_id": deployment_id,
        "account_alias": str(account.get("accountAlias", "") or ""),
        "config_hash": config_hash,
        "positions": {
            symbol: futures_position_to_dict(positions.get(api_symbol(symbol)))
            for symbol in symbols
        },
        "maintenance_brackets": maintenance_brackets,
        "setup_plan": setup_plan,
        "setup_responses": setup_responses,
        "orders": [futures_order_to_dict(order) for order in planned],
        "responses": responses,
        "pre_submit_position_checks": pre_submit_position_checks,
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
    hard_account_gross_limit: Decimal = Decimal("1.50"),
    hard_symbol_gross_limit: Decimal = Decimal("1.50"),
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
    if target_gross_cap > hard_account_gross_limit:
        raise SystemExit("--target-gross-cap must not exceed --hard-account-gross-limit.")
    if hard_account_gross_limit > Decimal(str(exchange_leverage)):
        raise SystemExit("--hard-account-gross-limit must not exceed --exchange-leverage.")
    if target_gross_cap > hard_symbol_gross_limit:
        raise SystemExit("--target-gross-cap must not exceed --hard-symbol-gross-limit.")
    if hard_symbol_gross_limit > Decimal(str(exchange_leverage)):
        raise SystemExit("--hard-symbol-gross-limit must not exceed --exchange-leverage.")


def validate_deployment_args(
    *,
    sleeve_fraction: Decimal,
    min_order_usdt: Decimal,
    max_order_usdt: Decimal,
    max_deploy_usdt: Decimal = Decimal("0"),
    margin_buffer_fraction: Decimal = Decimal("0.25"),
    min_liquidation_buffer: Decimal = Decimal("0.30"),
    hard_account_gross_limit: Decimal = Decimal("1.50"),
    hard_symbol_gross_limit: Decimal = Decimal("1.50"),
) -> None:
    if sleeve_fraction <= 0 or sleeve_fraction > 1:
        raise SystemExit("--sleeve-fraction must be > 0 and <= 1.")
    if min_order_usdt < 0:
        raise SystemExit("--min-order-usdt must be >= 0.")
    if max_order_usdt < 0:
        raise SystemExit("--max-order-usdt must be >= 0.")
    if max_deploy_usdt < 0:
        raise SystemExit("--max-deploy-usdt must be >= 0.")
    if margin_buffer_fraction < 0 or margin_buffer_fraction >= 1:
        raise SystemExit("--margin-buffer-fraction must be >= 0 and < 1.")
    if min_liquidation_buffer < 0 or min_liquidation_buffer >= 1:
        raise SystemExit("--min-liquidation-buffer must be >= 0 and < 1.")
    if hard_account_gross_limit <= 0:
        raise SystemExit("--hard-account-gross-limit must be > 0.")
    if hard_symbol_gross_limit <= 0:
        raise SystemExit("--hard-symbol-gross-limit must be > 0.")


def acquire_process_lock(path: Path, *, stale_after_hours: float = 6.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age_hours = (pd.Timestamp.now("UTC").timestamp() - path.stat().st_mtime) / 3600.0
        existing_pid = lock_file_pid(path)
        if existing_pid is not None and not process_is_running(existing_pid):
            path.unlink()
        elif existing_pid is None and age_hours > stale_after_hours:
            path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"Another futures executor process holds lock: {path}") from exc
    payload = f"pid={os.getpid()} started_at={pd.Timestamp.now('UTC').isoformat()}\n"
    os.write(fd, payload.encode("utf-8"))
    os.close(fd)

    def release() -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release)


def lock_file_pid(path: Path) -> int | None:
    try:
        match = re.search(r"\bpid=(\d+)\b", path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return int(match.group(1)) if match else None


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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


def validate_fresh_daily_data(all_dfs: dict[str, pd.DataFrame], symbols: list[str], *, max_age_hours: float = 48.0) -> None:
    now = pd.Timestamp.now("UTC")
    stale = []
    for symbol in dict.fromkeys(symbols):
        df = all_dfs.get(symbol)
        if df is None or df.empty:
            stale.append(f"{symbol}:missing")
            continue
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            stale.append(f"{symbol}:invalid_timestamps")
            continue
        gaps = timestamps.diff().dropna()
        if (gaps != pd.Timedelta(days=1)).any():
            stale.append(f"{symbol}:missing_daily_candles")
            continue
        latest = pd.Timestamp(timestamps.iloc[-1])
        if latest != latest.normalize() or latest >= now.normalize():
            stale.append(f"{symbol}:incomplete_or_non_daily:{latest.isoformat()}")
            continue
        age_hours = (now - (latest + pd.Timedelta(days=1))).total_seconds() / 3600.0
        if age_hours > max_age_hours:
            stale.append(f"{symbol}:{latest.isoformat()} close_age={age_hours:.1f}h")
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
    client_order_id: str = "",
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
    if delta_notional > 0 and delta_notional < max(min_order_usdt, filters.min_notional):
        return None
    side = "BUY" if delta_notional > 0 else "SELL"

    clip_reasons = []
    notional = abs(delta_notional)
    if max_order_usdt > 0 and notional > max_order_usdt:
        notional = max_order_usdt
        clip_reasons.append("max_order_usdt")
    quantity = round_step(notional / mark_price, filters.step_size)
    if quantity < filters.min_qty:
        return None
    if delta_notional > 0 and quantity * mark_price < max(min_order_usdt, filters.min_notional):
        return None
    if quantity > filters.max_qty:
        quantity = round_step(filters.max_qty, filters.step_size)
        clip_reasons.append("max_market_qty")
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
        client_order_id=client_order_id,
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
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        backup_tmp = backup.with_name(backup.name + ".tmp")
        backup_tmp.write_bytes(path.read_bytes())
        backup_tmp.replace(backup)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_path.replace(path)


def build_virtual_sleeves(
    *,
    state: dict[str, Any],
    symbols: list[str],
    positions: dict[str, FuturesPosition],
    deploy_equity: Decimal,
    allow_nonzero_bootstrap: bool = False,
    known_external_adjustments: dict[str, Decimal] | None = None,
) -> dict[str, VirtualSleeve]:
    state_symbols = state.get("symbols", {}) if isinstance(state, dict) else {}
    if not state_symbols and not allow_nonzero_bootstrap:
        nonzero = [
            f"{fsymbol}:{positions[fsymbol].position_amt}"
            for fsymbol in (api_symbol(symbol) for symbol in symbols)
            if positions[fsymbol].position_amt != 0
        ]
        if nonzero:
            raise SystemExit(
                "Futures state is missing while target positions are non-zero. "
                "Restore the state file or pass --allow-nonzero-bootstrap after an audited review: "
                + ", ".join(nonzero)
            )
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

    known_external_adjustments = known_external_adjustments or {}
    external_delta = deploy_equity - sum(item["pre_adjust_total"] for item in raw_sleeves)
    known_total = sum(
        (known_external_adjustments.get(item["api_symbol"], Decimal("0")) for item in raw_sleeves),
        Decimal("0"),
    )
    shared_adjustment = (external_delta - known_total) / Decimal(len(raw_sleeves))
    out: dict[str, VirtualSleeve] = {}
    for item in raw_sleeves:
        external_adjustment = shared_adjustment + known_external_adjustments.get(item["api_symbol"], Decimal("0"))
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


def reconcile_post_trade_sleeves(
    *,
    symbols: list[str],
    sleeves: dict[str, VirtualSleeve],
    positions: dict[str, FuturesPosition],
    deploy_equity: Decimal,
    known_external_adjustments: dict[str, Decimal] | None = None,
) -> dict[str, VirtualSleeve]:
    rows = []
    for symbol in symbols:
        sleeve = sleeves[symbol]
        position = positions[api_symbol(symbol)]
        mark_pnl = sleeve.current_position_amt * (position.mark_price - sleeve.current_mark_price)
        rows.append((symbol, sleeve, position, sleeve.total_value + mark_pnl, mark_pnl))
    known_external_adjustments = known_external_adjustments or {}
    residual = deploy_equity - sum(row[3] for row in rows)
    known_total = sum(
        (known_external_adjustments.get(api_symbol(row[0]), Decimal("0")) for row in rows),
        Decimal("0"),
    )
    shared_adjustment = (residual - known_total) / Decimal(len(rows))
    out = {}
    for symbol, sleeve, position, pre_adjust, mark_pnl in rows:
        adjustment = shared_adjustment + known_external_adjustments.get(api_symbol(symbol), Decimal("0"))
        total = pre_adjust + adjustment
        if total <= 0:
            raise SystemExit(f"Post-trade virtual sleeve equity for {symbol} is not positive: {total}.")
        out[symbol] = VirtualSleeve(
            symbol=sleeve.symbol,
            api_symbol=sleeve.api_symbol,
            total_value=total,
            pnl_since_last_run=sleeve.pnl_since_last_run + mark_pnl,
            external_adjustment=sleeve.external_adjustment + adjustment,
            previous_position_amt=sleeve.previous_position_amt,
            previous_mark_price=sleeve.previous_mark_price,
            current_position_amt=position.position_amt,
            current_mark_price=position.mark_price,
            previous_target_gross=sleeve.previous_target_gross,
        )
    return out


def build_next_state(
    symbols: list[str],
    sleeves: dict[str, VirtualSleeve],
    latest_targets: dict[str, Decimal],
    positions: dict[str, FuturesPosition],
    *,
    deployment_id: str = "",
    account_alias: str = "",
    config_hash: str = "",
    income_cursor_ms: int = 0,
    processed_income_ids: list[str] | None = None,
    unallocated_account_equity: Decimal = Decimal("0"),
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
        "schema_version": 3,
        "deployment_id": deployment_id,
        "account_alias": account_alias,
        "config_hash": config_hash,
        "income_cursor_ms": income_cursor_ms,
        "processed_income_ids": list((processed_income_ids or [])[-2000:]),
        "unallocated_account_equity_usdt": str(unallocated_account_equity),
        "updated_at": pd.Timestamp.now("UTC").isoformat(),
        "symbols": next_symbols,
    }


def build_deployment_id(base_url: str, api_key: str) -> str:
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{urllib.parse.urlparse(base_url).netloc}:{fingerprint}"


def validate_state_binding(
    state: dict[str, Any],
    *,
    deployment_id: str,
    allow_state_rebind: bool = False,
    account_alias: str = "",
    config_hash: str = "",
) -> None:
    if not isinstance(state, dict):
        raise SystemExit("Invalid futures state file: root value must be an object.")
    symbols = state.get("symbols", {})
    if not isinstance(symbols, dict):
        raise SystemExit("Invalid futures state file: symbols must be an object.")
    if not symbols:
        return
    existing = str(state.get("deployment_id", "") or "")
    if not existing:
        if not allow_state_rebind:
            raise SystemExit(
                "Legacy futures state has no deployment binding. Restore a v2 state or pass "
                "--allow-state-rebind once after verifying the account."
            )
        return
    if not hmac.compare_digest(existing, deployment_id):
        raise SystemExit(
            f"Futures state belongs to a different deployment ({existing}); current deployment is {deployment_id}."
        )
    existing_alias = str(state.get("account_alias", "") or "")
    if existing_alias and account_alias and not hmac.compare_digest(existing_alias, account_alias):
        raise SystemExit("Futures state account alias does not match the authenticated account.")
    existing_config = str(state.get("config_hash", "") or "")
    if existing_config and config_hash and not hmac.compare_digest(existing_config, config_hash):
        if not allow_state_rebind:
            raise SystemExit(
                "Futures state config hash does not match the active config. Pass --allow-state-rebind "
                "once only after reviewing the resulting targets."
            )


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


def native_latest_target_snapshot(
    *,
    symbol: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    strategy_name: str,
    sleeve_value: Decimal,
    execution_price: Decimal,
) -> NativeTargetSnapshot:
    if df.empty or execution_price <= 0:
        raise SystemExit(f"Cannot build target snapshot for {symbol}: missing data or execution price.")
    signal_timestamp = pd.Timestamp(df["timestamp"].iloc[-1])
    if signal_timestamp.tzinfo is None:
        signal_timestamp = signal_timestamp.tz_localize("UTC")
    else:
        signal_timestamp = signal_timestamp.tz_convert("UTC")
    execution_timestamp = signal_timestamp + pd.Timedelta(days=1)
    execution_df = append_next_open_execution_bar(
        df,
        timestamp=execution_timestamp,
        execution_price=execution_price,
    )
    strategy = build_strategy(
        strategy_name,
        float(sleeve_value),
        config["capital"]["reserve"],
        config["cost"]["fee_rate"],
        min_notional=config.get("cost", {}).get("min_notional"),
    )
    setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
    result = run_rebalance_backtest(
        {symbol: execution_df},
        strategy,
        initial_capital=float(sleeve_value),
        reserve=config["capital"]["reserve"],
        fee_rate=config["cost"]["fee_rate"],
        execution_mode=config.get("execution", {}).get("mode", "next_open"),
    )
    if result.empty:
        gross = Decimal("0")
    else:
        latest = result.iloc[-1]
        total = Decimal(str(latest.get("total_value", 0.0) or 0.0))
        value = Decimal(str(latest.get(f"{symbol}_value", 0.0) or 0.0))
        gross = max(Decimal("0"), value / total if total > 0 else Decimal("0"))
    return NativeTargetSnapshot(
        symbol=symbol,
        signal_timestamp=signal_timestamp,
        execution_timestamp=execution_timestamp,
        execution_price=execution_price,
        target_gross=gross,
    )


def append_next_open_execution_bar(
    df: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    execution_price: Decimal,
) -> pd.DataFrame:
    out = df.copy()
    row = {column: pd.NA for column in out.columns}
    row.update({
        "timestamp": timestamp,
        "open": float(execution_price),
        "high": float(execution_price),
        "low": float(execution_price),
        "close": float(execution_price),
        "volume": 0.0,
    })
    return pd.concat([out, pd.DataFrame([row], columns=out.columns)], ignore_index=True)


def native_target_snapshot_to_dict(snapshot: NativeTargetSnapshot) -> dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "signal_timestamp": snapshot.signal_timestamp.isoformat(),
        "execution_timestamp": snapshot.execution_timestamp.isoformat(),
        "execution_price": str(snapshot.execution_price),
        "target_gross": str(snapshot.target_gross),
    }


def submit_futures_market_order(client: BinanceUsdMFuturesTestnetClient, order: PlannedFuturesOrder) -> Any:
    existing = query_order_if_exists(client, order.symbol, order.client_order_id)
    if existing is not None:
        return validate_market_order_result(existing, order.client_order_id)
    params: dict[str, Any] = {
        "symbol": order.symbol,
        "side": order.side,
        "type": "MARKET",
        "quantity": format_decimal(order.quantity),
        "newClientOrderId": order.client_order_id,
        "newOrderRespType": "RESULT",
    }
    if order.reduce_only:
        params["reduceOnly"] = "true"
    try:
        response = client.signed_post("/fapi/v1/order", params)
    except RuntimeError:
        reconciled = query_order_if_exists(client, order.symbol, order.client_order_id)
        if reconciled is None:
            raise
        response = reconciled
    return validate_market_order_result(response, order.client_order_id)


def query_order_if_exists(
    client: BinanceUsdMFuturesTestnetClient,
    symbol: str,
    client_order_id: str,
) -> dict[str, Any] | None:
    try:
        result = client.signed_get(
            "/fapi/v1/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
        )
    except RuntimeError as exc:
        if "-2013" in str(exc) or "Order does not exist" in str(exc):
            return None
        raise
    return result if isinstance(result, dict) else None


def validate_market_order_result(response: dict[str, Any], client_order_id: str) -> dict[str, Any]:
    status = str(response.get("status", "") or "").upper()
    if status != "FILLED":
        raise RuntimeError(
            f"Market order {client_order_id} was not confirmed FILLED; status={status or 'missing'}."
        )
    return response


def build_client_order_id(
    *,
    strategy_name: str,
    api_symbol: str,
    signal_timestamp: pd.Timestamp,
    target_gross: Decimal,
) -> str:
    normalized_target = format(target_gross.normalize(), "f")
    digest_input = f"{strategy_name}|{api_symbol}|{signal_timestamp.isoformat()}|{normalized_target}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    date_label = signal_timestamp.strftime("%Y%m%d")
    symbol_label = api_symbol.replace("USDT", "")[:5].lower()
    return f"fv1-{date_label}-{symbol_label}-{digest}"[:36]


def parse_futures_filters(exchange_info: dict[str, Any], wanted_symbols: set[str]) -> dict[str, FuturesSymbolFilters]:
    out = {}
    for item in exchange_info.get("symbols", []):
        symbol = item.get("symbol")
        if symbol not in wanted_symbols:
            continue
        if str(item.get("status", "")) != "TRADING" or str(item.get("contractType", "")) != "PERPETUAL":
            raise SystemExit(
                f"Futures symbol is not a trading perpetual contract: {symbol} "
                f"status={item.get('status')} contractType={item.get('contractType')}"
            )
        filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
        out[symbol] = FuturesSymbolFilters(
            step_size=Decimal(str(lot.get("stepSize", "0.001"))),
            min_qty=Decimal(str(lot.get("minQty", "0"))),
            min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "0")))),
            max_qty=Decimal(str(lot.get("maxQty", "Infinity"))),
        )
    missing = wanted_symbols - set(out)
    if missing:
        raise SystemExit(f"Missing futures symbol filters: {sorted(missing)}")
    return out


def parse_maintenance_brackets(payload: Any, wanted_symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    rows = payload if isinstance(payload, list) else [payload]
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("symbol", "")) not in wanted_symbols:
            continue
        symbol = str(row["symbol"])
        brackets = []
        for bracket in row.get("brackets", []):
            brackets.append({
                "bracket": int(bracket.get("bracket", 0)),
                "initial_leverage": int(bracket.get("initialLeverage", 0)),
                "notional_floor": str(Decimal(str(bracket.get("notionalFloor", "0")))),
                "notional_cap": str(Decimal(str(bracket.get("notionalCap", "0")))),
                "maintenance_margin_ratio": str(Decimal(str(bracket.get("maintMarginRatio", "0")))),
                "cumulative_maintenance_amount": str(Decimal(str(bracket.get("cum", "0")))),
            })
        if not brackets:
            raise SystemExit(f"Binance returned no maintenance brackets for {symbol}.")
        out[symbol] = brackets
    missing = wanted_symbols - set(out)
    if missing:
        raise SystemExit(f"Missing Binance maintenance brackets: {sorted(missing)}")
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


def validate_position_settings(
    positions: dict[str, FuturesPosition],
    wanted_symbols: set[str],
    *,
    expected_leverage: int,
) -> None:
    missing = wanted_symbols - set(positions)
    if missing:
        raise SystemExit(f"Missing position settings after Binance setup: {sorted(missing)}")
    invalid = []
    for symbol in sorted(wanted_symbols):
        position = positions[symbol]
        if position.margin_type.lower() != "isolated" or position.leverage != expected_leverage:
            invalid.append(
                f"{symbol}:margin={position.margin_type},leverage={position.leverage}"
            )
    if invalid:
        raise SystemExit("Binance position settings were not applied: " + ", ".join(invalid))


def validate_live_position_before_order(
    client: BinanceUsdMFuturesTestnetClient,
    *,
    order: PlannedFuturesOrder,
    planned_position: FuturesPosition,
    expected_leverage: int,
) -> FuturesPosition:
    rows = client.signed_get("/fapi/v2/positionRisk", {"symbol": order.symbol})
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise RuntimeError(f"Binance positionRisk response for {order.symbol} must be a list or object.")
    parsed = parse_positions(rows, {order.symbol})
    position = parsed.get(order.symbol)
    if position is None:
        raise SystemExit(f"Missing live position snapshot immediately before {order.symbol} order.")
    if position.position_amt < 0:
        raise SystemExit(f"Live {order.symbol} position became short before order submission; aborting.")
    if position.margin_type.lower() != "isolated" or position.leverage != expected_leverage:
        raise SystemExit(
            f"Live {order.symbol} settings changed before order submission: "
            f"margin={position.margin_type}, leverage={position.leverage}."
        )
    if position.position_amt != planned_position.position_amt:
        raise SystemExit(
            f"Live {order.symbol} position changed after planning "
            f"({planned_position.position_amt} -> {position.position_amt}); abort and rerun the cycle."
        )
    if order.side == "SELL" and order.quantity > position.position_amt:
        raise SystemExit(
            f"Planned reduce-only {order.symbol} quantity {order.quantity} exceeds live position "
            f"{position.position_amt}."
        )
    return position


def fetch_new_income_events(
    client: BinanceUsdMFuturesTestnetClient,
    *,
    state: dict[str, Any],
    wanted_symbols: set[str],
    quote_asset: str,
) -> dict[str, Any]:
    query_end_ms = client.server_time()
    cursor_raw = state.get("income_cursor_ms")
    if cursor_raw is None and state.get("updated_at"):
        try:
            cursor_raw = int(pd.Timestamp(state["updated_at"]).timestamp() * 1000)
        except (TypeError, ValueError):
            cursor_raw = None
    start_ms = int(cursor_raw) if cursor_raw is not None else query_end_ms
    start_ms = min(start_ms, query_end_ms)
    prior_processed = [str(value) for value in state.get("processed_income_ids", [])]
    processed = set(prior_processed)
    events: list[dict[str, Any]] = []
    window_ms = 6 * 24 * 60 * 60 * 1000
    for symbol in sorted(wanted_symbols):
        for income_type in ("COMMISSION", "FUNDING_FEE"):
            window_start = start_ms
            while window_start <= query_end_ms:
                window_end = min(query_end_ms, window_start + window_ms)
                page_start = window_start
                while page_start <= window_end:
                    rows = client.signed_get(
                        "/fapi/v1/income",
                        {
                            "symbol": symbol,
                            "incomeType": income_type,
                            "startTime": page_start,
                            "endTime": window_end,
                            "limit": 1000,
                        },
                    )
                    if not isinstance(rows, list):
                        raise RuntimeError("Binance income history response must be a list.")
                    for row in rows:
                        if str(row.get("asset", "")) != quote_asset:
                            continue
                        event = normalize_income_event(row)
                        if event["event_id"] not in processed:
                            events.append(event)
                            processed.add(event["event_id"])
                    if len(rows) < 1000:
                        break
                    latest_time = max(int(row.get("time", page_start)) for row in rows)
                    if latest_time < page_start:
                        raise RuntimeError("Binance income pagination did not advance.")
                    page_start = latest_time + 1
                if window_end >= query_end_ms:
                    break
                window_start = window_end + 1
    events.sort(key=lambda item: (item["time_ms"], item["event_id"]))
    return {
        "events": events,
        "query_start_ms": start_ms,
        "query_end_ms": query_end_ms,
        "cursor_ms": query_end_ms,
        "processed_ids": list(dict.fromkeys(prior_processed + [event["event_id"] for event in events]))[-2000:],
    }


def normalize_income_event(row: dict[str, Any]) -> dict[str, Any]:
    stable = "|".join(
        str(row.get(key, ""))
        for key in ("incomeType", "tranId", "tradeId", "symbol", "time", "income")
    )
    return {
        "event_id": hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24],
        "symbol": str(row.get("symbol", "")),
        "income_type": str(row.get("incomeType", "")),
        "income": str(Decimal(str(row.get("income", "0")))),
        "asset": str(row.get("asset", "")),
        "time_ms": int(row.get("time", 0)),
        "tran_id": str(row.get("tranId", "")),
        "trade_id": str(row.get("tradeId", "")),
    }


def income_adjustments_by_symbol(events: list[dict[str, Any]]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for event in events:
        symbol = str(event.get("symbol", ""))
        if not symbol:
            continue
        out[symbol] = out.get(symbol, Decimal("0")) + Decimal(str(event.get("income", "0")))
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


def calculate_deploy_equity(
    account_equity: Decimal,
    sleeve_fraction: Decimal,
    max_deploy_usdt: Decimal,
) -> Decimal:
    deploy_equity = account_equity * sleeve_fraction
    if max_deploy_usdt > 0:
        deploy_equity = min(deploy_equity, max_deploy_usdt)
    if deploy_equity <= 0:
        raise SystemExit("Calculated deployed equity must be positive.")
    return deploy_equity


def restore_deployment_equity(
    *,
    account_equity: Decimal,
    state: dict[str, Any],
    positions: dict[str, FuturesPosition],
    symbols: list[str],
    sleeve_fraction: Decimal,
    max_deploy_usdt: Decimal,
) -> tuple[Decimal, Decimal]:
    unallocated_raw = state.get("unallocated_account_equity_usdt")
    if unallocated_raw is not None:
        try:
            unallocated = Decimal(str(unallocated_raw))
        except (InvalidOperation, ValueError) as exc:
            raise SystemExit("Invalid unallocated_account_equity_usdt in futures state.") from exc
        deploy_equity = account_equity - unallocated
    elif state.get("symbols"):
        estimated_sleeves = Decimal("0")
        for symbol in symbols:
            fsymbol = api_symbol(symbol)
            record = state["symbols"].get(fsymbol, {})
            position = positions[fsymbol]
            prior_total = decimal_field(record, "virtual_total_value_usdt", Decimal("0"))
            prior_qty = decimal_field(record, "last_position_amt", position.position_amt)
            prior_mark = decimal_field(record, "last_mark_price", position.mark_price)
            estimated_sleeves += prior_total + prior_qty * (position.mark_price - prior_mark)
        deploy_equity = estimated_sleeves
        unallocated = account_equity - deploy_equity
    else:
        deploy_equity = calculate_deploy_equity(account_equity, sleeve_fraction, max_deploy_usdt)
        unallocated = account_equity - deploy_equity
    if deploy_equity <= 0:
        raise SystemExit(
            f"Restored deployed equity is not positive ({deploy_equity}); stop and audit the futures account/state."
        )
    return deploy_equity, unallocated


def account_gross_snapshot(
    positions: dict[str, FuturesPosition],
    deploy_equity: Decimal,
) -> dict[str, Any]:
    if deploy_equity <= 0:
        raise SystemExit("Cannot calculate account gross with non-positive deployed equity.")
    notionals = {
        symbol: abs(position.position_amt) * position.mark_price
        for symbol, position in positions.items()
    }
    gross_notional = sum(notionals.values(), Decimal("0"))
    return {
        "deploy_equity_usdt": str(deploy_equity),
        "gross_notional_usdt": str(gross_notional),
        "gross_ratio": str(gross_notional / deploy_equity),
        "symbol_notionals_usdt": {symbol: str(value) for symbol, value in notionals.items()},
    }


def validate_projected_account_gross(
    *,
    positions: dict[str, FuturesPosition],
    orders: list[PlannedFuturesOrder],
    deploy_equity: Decimal,
    hard_account_gross_limit: Decimal,
) -> dict[str, Any]:
    current = account_gross_snapshot(positions, deploy_equity)
    projected_qty = {symbol: position.position_amt for symbol, position in positions.items()}
    marks = {symbol: position.mark_price for symbol, position in positions.items()}
    for order in orders:
        signed_qty = order.quantity if order.side == "BUY" else -order.quantity
        projected_qty[order.symbol] = max(Decimal("0"), projected_qty.get(order.symbol, Decimal("0")) + signed_qty)
        marks[order.symbol] = order.mark_price
    projected_notional = sum(
        (abs(quantity) * marks[symbol] for symbol, quantity in projected_qty.items()),
        Decimal("0"),
    )
    current_ratio = Decimal(current["gross_ratio"])
    projected_ratio = projected_notional / deploy_equity
    has_buy = any(order.side == "BUY" for order in orders)
    if has_buy and current_ratio > hard_account_gross_limit:
        raise SystemExit(
            f"Current account gross {current_ratio:.4f} exceeds hard limit "
            f"{hard_account_gross_limit:.4f}; BUY orders are forbidden until exposure is reduced."
        )
    if projected_ratio > hard_account_gross_limit and projected_ratio >= current_ratio:
        raise SystemExit(
            f"Projected account gross {projected_ratio:.4f} exceeds hard limit "
            f"{hard_account_gross_limit:.4f}. Reduce target gross or order size."
        )
    return {
        **current,
        "projected_gross_notional_usdt": str(projected_notional),
        "projected_gross_ratio": str(projected_ratio),
        "hard_limit": str(hard_account_gross_limit),
        "hard_limit_breached": current_ratio > hard_account_gross_limit,
    }


def symbol_gross_snapshot(
    positions: dict[str, FuturesPosition],
    sleeve_values: dict[str, Decimal],
    *,
    hard_symbol_gross_limit: Decimal | None = None,
) -> dict[str, Any]:
    out = {}
    for symbol, sleeve_value in sleeve_values.items():
        if sleeve_value <= 0:
            raise SystemExit(f"Cannot calculate {symbol} gross with non-positive sleeve equity.")
        position = positions[symbol]
        notional = abs(position.position_amt) * position.mark_price
        ratio = notional / sleeve_value
        out[symbol] = {
            "sleeve_equity_usdt": str(sleeve_value),
            "gross_notional_usdt": str(notional),
            "gross_ratio": str(ratio),
        }
        if hard_symbol_gross_limit is not None:
            out[symbol]["hard_limit"] = str(hard_symbol_gross_limit)
            out[symbol]["hard_limit_breached"] = ratio > hard_symbol_gross_limit
    return out


def validate_projected_symbol_gross(
    *,
    positions: dict[str, FuturesPosition],
    orders: list[PlannedFuturesOrder],
    sleeve_values: dict[str, Decimal],
    hard_symbol_gross_limit: Decimal,
) -> dict[str, Any]:
    snapshots = symbol_gross_snapshot(positions, sleeve_values)
    orders_by_symbol = {order.symbol: order for order in orders}
    for symbol, snapshot in snapshots.items():
        position = positions[symbol]
        current_ratio = Decimal(snapshot["gross_ratio"])
        order = orders_by_symbol.get(symbol)
        projected_qty = position.position_amt
        if order is not None:
            projected_qty += order.quantity if order.side == "BUY" else -order.quantity
            projected_qty = max(Decimal("0"), projected_qty)
        projected_notional = projected_qty * position.mark_price
        projected_ratio = projected_notional / sleeve_values[symbol]
        if order is not None and order.side == "BUY" and current_ratio > hard_symbol_gross_limit:
            raise SystemExit(
                f"{symbol} current sleeve gross {current_ratio:.4f} exceeds hard limit "
                f"{hard_symbol_gross_limit:.4f}; BUY is forbidden until exposure is reduced."
            )
        if projected_ratio > hard_symbol_gross_limit and projected_ratio >= current_ratio:
            raise SystemExit(
                f"{symbol} projected sleeve gross {projected_ratio:.4f} exceeds hard limit "
                f"{hard_symbol_gross_limit:.4f}."
            )
        snapshot.update({
            "projected_gross_notional_usdt": str(projected_notional),
            "projected_gross_ratio": str(projected_ratio),
            "hard_limit": str(hard_symbol_gross_limit),
            "hard_limit_breached": current_ratio > hard_symbol_gross_limit,
        })
    return snapshots


def validate_liquidation_buffers(
    positions: dict[str, FuturesPosition],
    orders: list[PlannedFuturesOrder],
    *,
    min_liquidation_buffer: Decimal,
) -> None:
    if not any(order.side == "BUY" for order in orders):
        return
    unsafe = []
    for symbol, position in sorted(positions.items()):
        if position.position_amt <= 0:
            continue
        if position.mark_price <= 0 or position.liquidation_price <= 0:
            unsafe.append(f"{symbol}:unknown")
            continue
        buffer = (position.mark_price - position.liquidation_price) / position.mark_price
        if buffer < min_liquidation_buffer:
            unsafe.append(f"{symbol}:{buffer:.4f}")
    if unsafe:
        raise SystemExit(
            "BUY orders are forbidden because liquidation buffer is below "
            f"{min_liquidation_buffer:.2%}: {', '.join(unsafe)}"
        )


def futures_available_balance(account: dict[str, Any], quote_asset: str) -> Decimal | None:
    direct = account.get("availableBalance")
    if direct is not None:
        return Decimal(str(direct))
    for asset in account.get("assets", []):
        if asset.get("asset") == quote_asset and asset.get("availableBalance") is not None:
            return Decimal(str(asset["availableBalance"]))
    return None


def validate_planned_buying_power(
    orders: list[PlannedFuturesOrder],
    account: dict[str, Any],
    quote_asset: str,
    *,
    margin_buffer_fraction: Decimal = Decimal("0.25"),
) -> None:
    buys = [order for order in orders if order.side == "BUY"]
    if not buys:
        return
    available = futures_available_balance(account, quote_asset)
    if available is None:
        raise SystemExit(f"Binance account response has no available {quote_asset} balance.")
    released = sum(
        (order.notional / Decimal(order.exchange_leverage) for order in orders if order.side == "SELL"),
        Decimal("0"),
    )
    required = sum(
        (order.notional / Decimal(order.exchange_leverage) for order in buys),
        Decimal("0"),
    )
    usable = max(Decimal("0"), available + released) * (Decimal("1") - margin_buffer_fraction)
    if required > usable:
        raise SystemExit(
            f"Planned BUY initial margin {required:.4f} {quote_asset} exceeds buffered available "
            f"margin {usable:.4f} {quote_asset}. Reduce target gross or free margin."
        )


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
        "client_order_id": order.client_order_id,
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
