#!/usr/bin/env python3
"""Build static monitor data for the Official V1 deployment dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="web/monitor/data/monitor.json")
    parser.add_argument("--recent-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    payload = build_payload(results_dir=results_dir, recent_limit=args.recent_limit)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"monitor_data={output}")


def build_payload(*, results_dir: Path, recent_limit: int) -> dict[str, Any]:
    futures_reports = report_payloads(results_dir / "binance_futures_testnet", recent_limit)
    latest_futures = futures_reports[0] if futures_reports else None
    daemon_status = read_json(PROJECT_ROOT / "runtime" / "daemon_status.json")
    if not isinstance(daemon_status, dict):
        daemon_status = None
    alerts = build_alerts(
        latest_futures=latest_futures,
        daemon_status=daemon_status,
    )
    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "strategy": "eth_bnb_futures_v1",
        "symbols": ["ETH/USDT", "BNB/USDT"],
        "alerts": alerts,
        "daemon_status": daemon_status,
        "latest_futures_report": latest_futures,
        "recent_futures_reports": futures_reports,
    }


def report_payloads(report_dir: Path, limit: int) -> list[dict[str, Any]]:
    files = sorted(report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for path in files[: max(0, limit)]:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        out.append(normalize_report(path, data))
    return out


def normalize_report(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    orders = data.get("orders") if isinstance(data.get("orders"), list) else []
    responses = data.get("responses") if isinstance(data.get("responses"), list) else []
    positions = data.get("positions") if isinstance(data.get("positions"), dict) else {}
    filled_count = sum(
        1 for response in responses
        if isinstance(response, dict) and str(response.get("status", "")).upper() == "FILLED"
    )
    return {
        "path": rel(path),
        "mtime": mtime(path),
        "timestamp": data.get("timestamp"),
        "mode": data.get("mode", "unknown"),
        "base_url": data.get("base_url"),
        "strategy": data.get("strategy"),
        "wallet_balance_usdt": data.get("wallet_balance_usdt"),
        "account_equity_usdt": data.get("account_equity_usdt"),
        "total_equity_usdt": data.get("total_equity_usdt"),
        "deploy_equity_usdt": data.get("deploy_equity_usdt"),
        "sleeve_value_usdt": data.get("sleeve_value_usdt"),
        "symbol_sleeves": data.get("symbol_sleeves", {}),
        "target_snapshots": data.get("target_snapshots", {}),
        "exchange_leverage": data.get("exchange_leverage"),
        "target_gross_cap": data.get("target_gross_cap"),
        "hard_account_gross_limit": data.get("hard_account_gross_limit"),
        "hard_symbol_gross_limit": data.get("hard_symbol_gross_limit"),
        "account_gross_before": data.get("account_gross_before", {}),
        "account_gross_after": data.get("account_gross_after", {}),
        "symbol_gross_after": data.get("symbol_gross_after", {}),
        "orders": orders,
        "order_count": len(orders),
        "responses": responses,
        "response_count": len(responses),
        "filled_count": filled_count,
        "positions": positions,
        "setup_plan": data.get("setup_plan", []),
        "setup_responses": data.get("setup_responses", []),
        "state_updated": bool(data.get("state_updated", False)),
        "risk": risk_summary(data),
    }


def risk_summary(data: dict[str, Any]) -> dict[str, Any]:
    positions = data.get("positions") if isinstance(data.get("positions"), dict) else {}
    buffers = []
    notionals = []
    for position in positions.values():
        if not isinstance(position, dict):
            continue
        buffer_raw = position.get("liquidation_buffer_pct")
        if buffer_raw not in (None, ""):
            try:
                buffers.append(float(buffer_raw))
            except (TypeError, ValueError):
                pass
        try:
            amount = abs(float(position.get("position_amt", 0) or 0))
            mark = float(position.get("mark_price", 0) or 0)
            notionals.append(amount * mark)
        except (TypeError, ValueError):
            pass
    orders = data.get("orders") if isinstance(data.get("orders"), list) else []
    clipped = [order for order in orders if str(order.get("clip_reason", "") or "")]
    return {
        "min_liquidation_buffer_pct": min(buffers) if buffers else None,
        "gross_notional_usdt": sum(notionals),
        "gross_ratio": nested_float(data, "account_gross_after", "gross_ratio"),
        "projected_gross_ratio": nested_float(data, "account_gross_before", "projected_gross_ratio"),
        "hard_account_gross_limit": float(data.get("hard_account_gross_limit", 0) or 0),
        "symbol_gross_breached": any(
            bool(item.get("hard_limit_breached"))
            for item in (data.get("symbol_gross_after", {}) or {}).values()
            if isinstance(item, dict)
        ),
        "clipped_order_count": len(clipped),
    }


def nested_float(data: dict[str, Any], parent: str, key: str) -> float | None:
    value = data.get(parent)
    if not isinstance(value, dict) or value.get(key) in (None, ""):
        return None
    try:
        return float(value[key])
    except (TypeError, ValueError):
        return None


def build_alerts(
    *,
    latest_futures: dict[str, Any] | None,
    daemon_status: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    alerts = []
    now = pd.Timestamp.now("UTC")
    if daemon_status is not None:
        daemon_state = str(daemon_status.get("status", "") or "")
        if daemon_state == "failed":
            alerts.append({
                "level": "danger",
                "message": f"Latest daemon cycle failed: {daemon_status.get('command', '')}",
            })
        daemon_age = age_hours(daemon_status.get("updated_at"), now)
        if daemon_age is not None and daemon_age > 36:
            alerts.append({"level": "warn", "message": f"Daemon status is stale: {daemon_age:.1f} hours old."})

    if latest_futures is None:
        alerts.append({"level": "warn", "message": "No executor report found."})
        return alerts

    report_age = age_hours(latest_futures.get("mtime"), now)
    if report_age is not None and report_age > 36:
        alerts.append({"level": "warn", "message": f"Futures report is stale: {report_age:.1f} hours old."})
    if latest_futures.get("mode") == "dry_run":
        alerts.append({"level": "warn", "message": "Latest executor report is dry-run; no orders were submitted."})
    if latest_futures.get("mode") == "execute" and not latest_futures.get("state_updated"):
        alerts.append({"level": "danger", "message": "Execute report did not confirm a state update."})
    if (
        latest_futures.get("mode") == "execute"
        and latest_futures.get("filled_count", 0) != latest_futures.get("order_count", 0)
    ):
        alerts.append({"level": "danger", "message": "Confirmed fill count does not match planned order count."})
    risk = latest_futures.get("risk") if isinstance(latest_futures.get("risk"), dict) else {}
    min_buffer = risk.get("min_liquidation_buffer_pct")
    if min_buffer is not None and float(min_buffer) < 0.20:
        alerts.append({"level": "danger", "message": "Futures liquidation buffer below 20%."})
    hard_gross = risk.get("hard_account_gross_limit")
    actual_gross = risk.get("gross_ratio")
    if hard_gross and actual_gross is not None and float(actual_gross) > float(hard_gross):
        alerts.append({"level": "danger", "message": "Actual account gross exceeds the configured hard limit."})
    if risk.get("symbol_gross_breached"):
        alerts.append({"level": "danger", "message": "A virtual sleeve exceeds its hard gross limit."})
    if risk.get("clipped_order_count", 0):
        alerts.append({"level": "info", "message": "Futures has clipped orders. Check clip_reason."})
    return alerts


def age_hours(timestamp: Any, now: pd.Timestamp) -> float | None:
    if not timestamp:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def mtime(path: Path) -> str:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


if __name__ == "__main__":
    main()
