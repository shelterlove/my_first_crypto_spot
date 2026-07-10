from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_monitor_dashboard_data import build_payload  # noqa: E402


def test_empty_results_builds_futures_only_payload() -> None:
    with TemporaryDirectory() as tmp:
        payload = build_payload(results_dir=Path(tmp), recent_limit=20)

    assert payload["strategy"] == "eth_bnb_futures_v1"
    assert payload["symbols"] == ["ETH/USDT", "BNB/USDT"]
    assert payload["latest_futures_report"] is None
    assert any(alert["message"] == "No executor report found." for alert in payload["alerts"])


def test_recent_futures_report_is_loaded() -> None:
    with TemporaryDirectory() as tmp:
        results_dir = Path(tmp)
        report_dir = results_dir / "binance_futures_testnet"
        report_dir.mkdir(parents=True)
        report = {
            "timestamp": "20260709_010000",
            "mode": "dry_run",
            "strategy": "eth_bnb_futures_v1",
            "wallet_balance_usdt": "100",
            "account_equity_usdt": "100",
            "target_gross_cap": "3.00",
            "exchange_leverage": 3,
            "orders": [],
            "positions": {},
        }
        (report_dir / "sample.json").write_text(json.dumps(report), encoding="utf-8")
        payload = build_payload(results_dir=results_dir, recent_limit=20)

    assert payload["latest_futures_report"]["mode"] == "dry_run"
    assert payload["latest_futures_report"]["order_count"] == 0
    assert len(payload["recent_futures_reports"]) == 1


def main() -> None:
    test_empty_results_builds_futures_only_payload()
    test_recent_futures_report_is_loaded()
    print("Monitor data tests passed")


if __name__ == "__main__":
    main()
