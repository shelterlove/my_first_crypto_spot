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
    build_virtual_sleeves,
    load_state,
    save_state,
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
