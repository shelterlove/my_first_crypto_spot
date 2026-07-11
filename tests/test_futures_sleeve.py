from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.binance_futures_testnet_executor import (  # noqa: E402
    FuturesPosition,
    FuturesSymbolFilters,
    PlannedFuturesOrder,
    acquire_process_lock,
    calculate_deploy_equity,
    fetch_new_income_events,
    build_client_order_id,
    build_deployment_id,
    build_virtual_sleeves,
    load_state,
    income_adjustments_by_symbol,
    plan_futures_order,
    parse_maintenance_brackets,
    restore_deployment_equity,
    save_state,
    submit_futures_market_order,
    validate_planned_buying_power,
    validate_liquidation_buffers,
    validate_live_position_before_order,
    validate_projected_account_gross,
    validate_projected_symbol_gross,
    validate_state_binding,
)


SYMBOLS = ["ETH/USDT", "BNB/USDT"]


def position(symbol: str, amount: str, mark: str) -> FuturesPosition:
    return FuturesPosition(
        symbol=symbol,
        position_amt=Decimal(amount),
        entry_price=Decimal(mark),
        mark_price=Decimal(mark),
        liquidation_price=Decimal("0"),
        leverage=3,
        margin_type="isolated",
    )


def base_state() -> dict:
    return {
        "symbols": {
            "ETHUSDT": {
                "symbol": "ETH/USDT",
                "virtual_total_value_usdt": "100",
                "last_position_amt": "0.1",
                "last_mark_price": "1000",
            },
            "BNBUSDT": {
                "symbol": "BNB/USDT",
                "virtual_total_value_usdt": "100",
                "last_position_amt": "0.2",
                "last_mark_price": "300",
            },
        }
    }


def test_missing_state_initializes() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing.json"
        assert load_state(path) == {"symbols": {}}


def test_dead_process_lock_is_recovered_immediately() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "executor.lock"
        path.write_text("pid=999999999 started_at=2026-01-01T00:00:00+00:00\n", encoding="utf-8")
        acquire_process_lock(path)
        assert f"pid={__import__('os').getpid()}" in path.read_text(encoding="utf-8")
        path.unlink()


def test_corrupt_state_exits() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        path.write_text("{bad json", encoding="utf-8")
        try:
            load_state(path)
        except SystemExit as exc:
            assert "Invalid futures state file" in str(exc)
        else:
            raise AssertionError("expected corrupt state to exit")


def test_symbol_pnl_stays_in_own_sleeve() -> None:
    sleeves = build_virtual_sleeves(
        state=base_state(),
        symbols=SYMBOLS,
        positions={
            "ETHUSDT": position("ETHUSDT", "0.1", "1100"),
            "BNBUSDT": position("BNBUSDT", "0.2", "250"),
        },
        deploy_equity=Decimal("200"),
    )

    assert sleeves["ETH/USDT"].pnl_since_last_run == Decimal("10.0")
    assert sleeves["BNB/USDT"].pnl_since_last_run == Decimal("-10.0")
    assert sleeves["ETH/USDT"].total_value == Decimal("110.0")
    assert sleeves["BNB/USDT"].total_value == Decimal("90.0")


def test_deposit_is_split_evenly() -> None:
    sleeves = build_virtual_sleeves(
        state=base_state(),
        symbols=SYMBOLS,
        positions={
            "ETHUSDT": position("ETHUSDT", "0.1", "1000"),
            "BNBUSDT": position("BNBUSDT", "0.2", "300"),
        },
        deploy_equity=Decimal("300"),
    )

    assert sleeves["ETH/USDT"].external_adjustment == Decimal("50.0")
    assert sleeves["BNB/USDT"].external_adjustment == Decimal("50.0")
    assert sleeves["ETH/USDT"].total_value == Decimal("150.0")
    assert sleeves["BNB/USDT"].total_value == Decimal("150.0")


def test_withdrawal_is_split_evenly() -> None:
    sleeves = build_virtual_sleeves(
        state=base_state(),
        symbols=SYMBOLS,
        positions={
            "ETHUSDT": position("ETHUSDT", "0.1", "1000"),
            "BNBUSDT": position("BNBUSDT", "0.2", "300"),
        },
        deploy_equity=Decimal("160"),
    )

    assert sleeves["ETH/USDT"].external_adjustment == Decimal("-20.0")
    assert sleeves["BNB/USDT"].external_adjustment == Decimal("-20.0")
    assert sleeves["ETH/USDT"].total_value == Decimal("80.0")
    assert sleeves["BNB/USDT"].total_value == Decimal("80.0")


def test_commission_is_charged_to_own_symbol_sleeve() -> None:
    sleeves = build_virtual_sleeves(
        state=base_state(),
        symbols=SYMBOLS,
        positions={
            "ETHUSDT": position("ETHUSDT", "0.1", "1000"),
            "BNBUSDT": position("BNBUSDT", "0.2", "300"),
        },
        deploy_equity=Decimal("196"),
        known_external_adjustments={"ETHUSDT": Decimal("-4")},
    )
    assert sleeves["ETH/USDT"].external_adjustment == Decimal("-4")
    assert sleeves["BNB/USDT"].external_adjustment == Decimal("0")
    assert sleeves["ETH/USDT"].total_value == Decimal("96")
    assert sleeves["BNB/USDT"].total_value == Decimal("100")


def test_income_history_is_filtered_and_deduplicated() -> None:
    class Client:
        def server_time(self):
            return 2000

        def signed_get(self, path, params):
            assert path == "/fapi/v1/income"
            if params["symbol"] == "ETHUSDT" and params["incomeType"] == "COMMISSION":
                return [{
                    "symbol": "ETHUSDT",
                    "incomeType": "COMMISSION",
                    "income": "-0.25",
                    "asset": "USDT",
                    "time": 1500,
                    "tranId": 9,
                    "tradeId": "12",
                }]
            return []

    first = fetch_new_income_events(
        Client(),
        state={"income_cursor_ms": 1000, "processed_income_ids": []},
        wanted_symbols={"ETHUSDT", "BNBUSDT"},
        quote_asset="USDT",
    )
    assert income_adjustments_by_symbol(first["events"]) == {"ETHUSDT": Decimal("-0.25")}
    second = fetch_new_income_events(
        Client(),
        state={"income_cursor_ms": 1000, "processed_income_ids": first["processed_ids"]},
        wanted_symbols={"ETHUSDT", "BNBUSDT"},
        quote_asset="USDT",
    )
    assert second["events"] == []


def test_invalid_decimal_state_exits() -> None:
    state = base_state()
    state["symbols"]["ETHUSDT"]["virtual_total_value_usdt"] = "not-a-number"
    try:
        build_virtual_sleeves(
            state=state,
            symbols=SYMBOLS,
            positions={
                "ETHUSDT": position("ETHUSDT", "0.1", "1000"),
                "BNBUSDT": position("BNBUSDT", "0.2", "300"),
            },
            deploy_equity=Decimal("200"),
        )
    except SystemExit as exc:
        assert "Invalid decimal field" in str(exc)
    else:
        raise AssertionError("expected invalid decimal state to exit")


def test_save_state_round_trips() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state = {"symbols": {"ETHUSDT": {"virtual_total_value_usdt": "100"}}}
        save_state(path, state)
        assert json.loads(path.read_text(encoding="utf-8")) == state
        assert not path.with_name(path.name + ".tmp").exists()


def test_missing_state_rejects_nonzero_positions() -> None:
    try:
        build_virtual_sleeves(
            state={"symbols": {}},
            symbols=SYMBOLS,
            positions={
                "ETHUSDT": position("ETHUSDT", "0.1", "1000"),
                "BNBUSDT": position("BNBUSDT", "0", "300"),
            },
            deploy_equity=Decimal("200"),
        )
    except SystemExit as exc:
        assert "--allow-nonzero-bootstrap" in str(exc)
    else:
        raise AssertionError("expected non-zero bootstrap to exit")


def test_state_binding_rejects_different_key() -> None:
    state = {"deployment_id": build_deployment_id("https://demo-fapi.binance.com", "key-a"), "symbols": {"ETHUSDT": {}}}
    try:
        validate_state_binding(
            state,
            deployment_id=build_deployment_id("https://demo-fapi.binance.com", "key-b"),
        )
    except SystemExit as exc:
        assert "different deployment" in str(exc)
    else:
        raise AssertionError("expected deployment mismatch to exit")


def test_small_reduce_only_exit_is_not_blocked_by_min_notional() -> None:
    order = plan_futures_order(
        api_symbol="BNBUSDT",
        strategy_name="eth_bnb_futures_v1",
        position=position("BNBUSDT", "0.01", "600"),
        filters=FuturesSymbolFilters(
            step_size=Decimal("0.01"),
            min_qty=Decimal("0.01"),
            min_notional=Decimal("5"),
        ),
        sleeve_value=Decimal("100"),
        exchange_leverage=3,
        target_gross_cap=Decimal("3"),
        target_gross=Decimal("0"),
        min_order_usdt=Decimal("10"),
        max_order_usdt=Decimal("0"),
        client_order_id="fv1-test",
    )
    assert order is not None
    assert order.side == "SELL"
    assert order.reduce_only is True
    assert order.quantity == Decimal("0.01")


def planned_buy(notional: str = "90") -> PlannedFuturesOrder:
    return PlannedFuturesOrder(
        symbol="ETHUSDT",
        side="BUY",
        quantity=Decimal("0.03"),
        reduce_only=False,
        mark_price=Decimal("3000"),
        notional=Decimal(notional),
        requested_notional_delta=Decimal(notional),
        target_gross=Decimal("1"),
        current_gross=Decimal("0"),
        exchange_leverage=3,
        target_gross_cap=Decimal("3"),
        clip_reason="",
        reason="test",
        client_order_id="fv1-test",
    )


def test_buying_power_uses_buffered_available_balance() -> None:
    validate_planned_buying_power([planned_buy("90")], {"availableBalance": "40"}, "USDT")
    try:
        validate_planned_buying_power([planned_buy("120")], {"availableBalance": "40"}, "USDT")
    except SystemExit as exc:
        assert "exceeds buffered available" in str(exc)
    else:
        raise AssertionError("expected insufficient margin to exit")


def test_deployed_equity_is_capped() -> None:
    assert calculate_deploy_equity(Decimal("5000"), Decimal("0.5"), Decimal("1000")) == Decimal("1000")
    assert calculate_deploy_equity(Decimal("500"), Decimal("0.5"), Decimal("1000")) == Decimal("250.0")


def test_unallocated_balance_preserves_deployment_pnl() -> None:
    deploy, unallocated = restore_deployment_equity(
        account_equity=Decimal("9990"),
        state={"unallocated_account_equity_usdt": "9000", "symbols": {}},
        positions={
            "ETHUSDT": position("ETHUSDT", "0", "100"),
            "BNBUSDT": position("BNBUSDT", "0", "100"),
        },
        symbols=SYMBOLS,
        sleeve_fraction=Decimal("1"),
        max_deploy_usdt=Decimal("1000"),
    )
    assert deploy == Decimal("990")
    assert unallocated == Decimal("9000")


def test_projected_account_gross_rejects_buy_above_hard_limit() -> None:
    positions = {"ETHUSDT": position("ETHUSDT", "1", "100")}
    order = planned_buy("60")
    order.quantity = Decimal("0.6")
    order.mark_price = Decimal("100")
    try:
        validate_projected_account_gross(
            positions=positions,
            orders=[order],
            deploy_equity=Decimal("100"),
            hard_account_gross_limit=Decimal("1.5"),
        )
    except SystemExit as exc:
        assert "Projected account gross" in str(exc)
    else:
        raise AssertionError("expected projected gross breach to exit")


def test_concentrated_symbol_gross_is_checked_separately() -> None:
    positions = {
        "ETHUSDT": position("ETHUSDT", "0.7", "100"),
        "BNBUSDT": position("BNBUSDT", "0", "100"),
    }
    order = planned_buy("10")
    order.quantity = Decimal("0.1")
    order.mark_price = Decimal("100")
    try:
        validate_projected_symbol_gross(
            positions=positions,
            orders=[order],
            sleeve_values={"ETHUSDT": Decimal("50"), "BNBUSDT": Decimal("50")},
            hard_symbol_gross_limit=Decimal("1.5"),
        )
    except SystemExit as exc:
        assert "ETHUSDT projected sleeve gross" in str(exc)
    else:
        raise AssertionError("expected concentrated sleeve gross breach to exit")


def test_over_limit_account_may_submit_reduce_only_order() -> None:
    positions = {"ETHUSDT": position("ETHUSDT", "2", "100")}
    order = planned_buy("60")
    order.side = "SELL"
    order.reduce_only = True
    order.quantity = Decimal("0.6")
    order.mark_price = Decimal("100")
    snapshot = validate_projected_account_gross(
        positions=positions,
        orders=[order],
        deploy_equity=Decimal("100"),
        hard_account_gross_limit=Decimal("1.5"),
    )
    assert snapshot["hard_limit_breached"] is True
    assert snapshot["projected_gross_ratio"] == "1.4"


def test_low_liquidation_buffer_blocks_buy() -> None:
    risky = FuturesPosition(
        symbol="ETHUSDT",
        position_amt=Decimal("1"),
        entry_price=Decimal("100"),
        mark_price=Decimal("100"),
        liquidation_price=Decimal("75"),
        leverage=2,
        margin_type="isolated",
    )
    try:
        validate_liquidation_buffers(
            {"ETHUSDT": risky},
            [planned_buy()],
            min_liquidation_buffer=Decimal("0.30"),
        )
    except SystemExit as exc:
        assert "liquidation buffer" in str(exc)
    else:
        raise AssertionError("expected low liquidation buffer to exit")


def test_existing_filled_client_order_is_not_submitted_again() -> None:
    class Client:
        post_called = False

        def signed_get(self, path, params):
            return {"status": "FILLED", "clientOrderId": params["origClientOrderId"]}

        def signed_post(self, path, params):
            self.post_called = True
            raise AssertionError("duplicate post")

    client = Client()
    response = submit_futures_market_order(client, planned_buy())
    assert response["status"] == "FILLED"
    assert client.post_called is False


def test_live_position_change_aborts_before_order() -> None:
    class Client:
        def signed_get(self, path, params):
            assert path == "/fapi/v2/positionRisk"
            return [{
                "symbol": "ETHUSDT",
                "positionAmt": "0.2",
                "entryPrice": "100",
                "markPrice": "100",
                "liquidationPrice": "50",
                "leverage": "2",
                "marginType": "isolated",
            }]

    try:
        validate_live_position_before_order(
            Client(),
            order=planned_buy(),
            planned_position=position("ETHUSDT", "0.1", "100"),
            expected_leverage=2,
        )
    except SystemExit as exc:
        assert "changed after planning" in str(exc)
    else:
        raise AssertionError("expected changed live position to abort")


def test_client_order_id_is_deterministic() -> None:
    timestamp = __import__("pandas").Timestamp("2026-07-09", tz="UTC")
    first = build_client_order_id(
        strategy_name="eth_bnb_futures_v1",
        api_symbol="ETHUSDT",
        signal_timestamp=timestamp,
        target_gross=Decimal("1.25"),
    )
    second = build_client_order_id(
        strategy_name="eth_bnb_futures_v1",
        api_symbol="ETHUSDT",
        signal_timestamp=timestamp,
        target_gross=Decimal("1.2500"),
    )
    assert first == second
    assert len(first) <= 36


def test_maintenance_brackets_are_normalized() -> None:
    payload = [{
        "symbol": "ETHUSDT",
        "brackets": [{
            "bracket": 1,
            "initialLeverage": 75,
            "notionalFloor": 0,
            "notionalCap": 10000,
            "maintMarginRatio": "0.005",
            "cum": 0,
        }],
    }]
    result = parse_maintenance_brackets(payload, {"ETHUSDT"})
    assert result["ETHUSDT"][0]["maintenance_margin_ratio"] == "0.005"
    assert result["ETHUSDT"][0]["initial_leverage"] == 75


def main() -> None:
    test_missing_state_initializes()
    test_corrupt_state_exits()
    test_symbol_pnl_stays_in_own_sleeve()
    test_deposit_is_split_evenly()
    test_withdrawal_is_split_evenly()
    test_invalid_decimal_state_exits()
    test_save_state_round_trips()
    print("Official V1 futures sleeve tests passed")


if __name__ == "__main__":
    main()
    acquire_process_lock,
