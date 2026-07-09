#!/usr/bin/env python3
"""Build static monitor data for the V4.8 deployment dashboard."""

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
    latest_signal = latest_signal_payload(results_dir)
    futures_reports = report_payloads(results_dir / "binance_futures_testnet_v48", recent_limit)
    spot_reports = report_payloads(results_dir / "binance_testnet_v48", recent_limit)
    latest_futures = futures_reports[0] if futures_reports else None
    latest_spot = spot_reports[0] if spot_reports else None
    alerts = build_alerts(latest_signal=latest_signal, latest_futures=latest_futures, latest_spot=latest_spot)
    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "strategy": "v4_8_eth_bnb",
        "symbols": ["ETH/USDT", "BNB/USDT"],
        "alerts": alerts,
        "latest_signal": latest_signal,
        "latest_futures_report": latest_futures,
        "latest_spot_report": latest_spot,
        "recent_futures_reports": futures_reports,
        "recent_spot_reports": spot_reports,
    }


def latest_signal_payload(results_dir: Path) -> dict[str, Any] | None:
    files = sorted(results_dir.glob("daily_signals*/**/*_signals.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    path = files[0]
    rows = read_json(path)
    if not isinstance(rows, list):
        rows = []
    return {
        "path": rel(path),
        "mtime": mtime(path),
        "rows": rows,
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
    return {
        "path": rel(path),
        "mtime": mtime(path),
        "timestamp": data.get("timestamp"),
        "mode": data.get("mode", "unknown"),
        "base_url": data.get("base_url"),
        "strategy": data.get("strategy"),
        "wallet_balance_usdt": data.get("wallet_balance_usdt"),
        "total_equity_usdt": data.get("total_equity_usdt"),
        "deploy_equity_usdt": data.get("deploy_equity_usdt"),
        "sleeve_value_usdt": data.get("sleeve_value_usdt"),
        "exchange_leverage": data.get("exchange_leverage"),
        "target_gross_cap": data.get("target_gross_cap"),
        "orders": orders,
        "order_count": len(orders),
        "responses": responses,
        "response_count": len(responses),
        "positions": positions,
        "setup_plan": data.get("setup_plan", []),
        "setup_responses": data.get("setup_responses", []),
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
        "clipped_order_count": len(clipped),
    }


def build_alerts(
    *,
    latest_signal: dict[str, Any] | None,
    latest_futures: dict[str, Any] | None,
    latest_spot: dict[str, Any] | None,
) -> list[dict[str, str]]:
    alerts = []
    now = pd.Timestamp.now("UTC")
    if latest_signal is None:
        alerts.append({"level": "warn", "message": "No daily signal file found."})
    else:
        signal_age = age_hours(latest_signal.get("mtime"), now)
        if signal_age is not None and signal_age > 36:
            alerts.append({"level": "warn", "message": f"Latest signal is stale: {signal_age:.1f} hours old."})

    if latest_futures is None and latest_spot is None:
        alerts.append({"level": "warn", "message": "No executor report found."})
    for label, report in (("Futures", latest_futures), ("Spot", latest_spot)):
        if report is None:
            continue
        report_age = age_hours(report.get("mtime"), now)
        if report_age is not None and report_age > 36:
            alerts.append({"level": "warn", "message": f"{label} report is stale: {report_age:.1f} hours old."})
        risk = report.get("risk") if isinstance(report.get("risk"), dict) else {}
        min_buffer = risk.get("min_liquidation_buffer_pct")
        if min_buffer is not None and float(min_buffer) < 0.20:
            alerts.append({"level": "danger", "message": f"{label} liquidation buffer below 20%."})
        if risk.get("clipped_order_count", 0):
            alerts.append({"level": "info", "message": f"{label} has clipped orders. Check clip_reason."})
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
