#!/usr/bin/env python3
"""Run the Official V1 deployment loop as a long-lived process.

The loop syncs daily candles, runs the Binance Futures Testnet executor, then
builds monitor dashboard data. It is intended to run inside tmux or a process
manager on the VPS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-at-utc", default="01:10", help="Daily run time in HH:MM UTC.")
    parser.add_argument("--run-on-start", action="store_true", help="Run immediately before waiting for the next schedule.")
    parser.add_argument("--execute", action="store_true", help="Pass --execute to the futures executor.")
    parser.add_argument("--config", default="configs/backtest_v1.json")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,BNB/USDT")
    parser.add_argument("--sync-start", default="2020-01-01")
    parser.add_argument("--exchange-leverage", default="3")
    parser.add_argument("--target-gross-cap", default="3.00")
    parser.add_argument("--max-order-usdt", default="0")
    parser.add_argument("--log-file", default="logs/futures_daemon.log")
    parser.add_argument("--stop-after-one", action="store_true", help="Run one cycle and exit. Useful for smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = PROJECT_ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "runtime").mkdir(parents=True, exist_ok=True)

    log(log_path, "daemon started")
    if args.run_on_start or args.stop_after_one:
        run_cycle(args, log_path)
        if args.stop_after_one:
            log(log_path, "stop-after-one complete")
            return

    while True:
        sleep_seconds = seconds_until(args.run_at_utc)
        next_time = datetime.now(timezone.utc).timestamp() + sleep_seconds
        log(log_path, f"sleeping {sleep_seconds}s until {datetime.fromtimestamp(next_time, timezone.utc).isoformat()}")
        time.sleep(sleep_seconds)
        run_cycle(args, log_path)


def run_cycle(args: argparse.Namespace, log_path: Path) -> None:
    log(log_path, "cycle started")
    commands = [
        [
            sys.executable,
            "scripts/sync_binance_klines.py",
            "--symbols",
            args.symbols,
            "--timeframe",
            "1d",
            "--start",
            args.sync_start,
        ],
        executor_command(args),
        [sys.executable, "scripts/build_monitor_dashboard_data.py"],
    ]
    for command in commands:
        if not run_command(command, log_path):
            log(log_path, "cycle aborted after command failure")
            return
    log(log_path, "cycle finished")


def executor_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/binance_futures_testnet_executor.py",
        "--config",
        args.config,
        "--exchange-leverage",
        str(args.exchange_leverage),
        "--target-gross-cap",
        str(args.target_gross_cap),
        "--max-order-usdt",
        str(args.max_order_usdt),
    ]
    if args.execute:
        command.append("--execute")
    return command


def run_command(command: list[str], log_path: Path) -> bool:
    log(log_path, f"running: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        log(log_path, f"command failed returncode={proc.returncode}: {' '.join(command)}")
        return False
    log(log_path, f"command ok: {' '.join(command)}")
    return True


def seconds_until(hhmm: str) -> int:
    try:
        hour_raw, minute_raw = hhmm.split(":", 1)
        hour = int(hour_raw)
        minute = int(minute_raw)
    except ValueError as exc:
        raise SystemExit("--run-at-utc must use HH:MM format") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise SystemExit("--run-at-utc must be a valid UTC time")

    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return max(1, int((target - now).total_seconds()))


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


if __name__ == "__main__":
    main()
