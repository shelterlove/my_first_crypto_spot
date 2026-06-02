#!/usr/bin/env python3
"""Run and summarize Freqtrade backtests for fixed-allocation strategy review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
DEFAULT_ALLOCATION = [333.0, 333.0, 334.0]
ROLLING_PRESETS = {
    "quick": [(365, 180)],
    "standard": [(365, 90), (730, 120), (1095, 180)],
}


@dataclass(frozen=True)
class BacktestRun:
    pair: str
    wallet: float
    directory: Path
    result_zip: Path


def main() -> None:
    args = parse_args()
    pairs = args.pairs
    allocation = args.allocation
    if len(allocation) != len(pairs):
        raise SystemExit("--allocation must have the same length as --pairs")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rolling_windows or args.rolling_preset:
        run_rolling_eval(args, output_dir)
        return

    run_single_eval(args, output_dir, timerange=args.timerange, report_window=args.report_window)


def run_single_eval(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    timerange: str,
    report_window: str,
) -> list[dict]:
    runs: list[BacktestRun] = []
    for pair, wallet in zip(args.pairs, args.allocation):
        run_dir = output_dir / "single" / safe_name(pair)
        runs.append(run_backtest(args, [pair], wallet, run_dir, pair=pair, timerange=timerange))

    rows: list[dict] = []
    trade_rows: list[dict] = []
    single_equities: list[pd.DataFrame] = []
    for run in runs:
        result = parse_result_zip(run.result_zip, args.strategy)
        equity = build_equity_curve(result.wallet)
        rows.append(summarize_run(
            args=args,
            run=run,
            result=result,
            equity=equity,
            pairs=[run.pair],
            allocation=[run.wallet],
            timerange=timerange,
            report_window=report_window,
        ))
        trade_rows.extend(extract_trade_rows(run, result.trades))
        single_equities.append(equity.rename(columns={
            "equity": f"{run.pair}:equity",
            "cash_value": f"{run.pair}:cash",
        })[["date", f"{run.pair}:equity", f"{run.pair}:cash"]])

    if single_equities:
        rows.append(summarize_fixed_single_portfolio(
            args=args,
            equities=single_equities,
            pairs=args.pairs,
            allocation=args.allocation,
            output_dir=output_dir,
            timerange=timerange,
            report_window=report_window,
        ))

    rows = json_safe_rows(rows)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(output_dir / "trades.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown_report(args, rows), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {output_dir}")
    return rows


def run_rolling_eval(args: argparse.Namespace, output_dir: Path) -> None:
    detail_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for spec_index, (window_days, step_days) in enumerate(rolling_specs(args), start=1):
        windows = rolling_windows(args.timerange, window_days, step_days)
        if not windows:
            continue
        for window_index, (start, end) in enumerate(windows, start=1):
            timerange = f"{fmt_date(start)}-{fmt_date(end)}"
            window_dir = (
                output_dir
                / f"{window_days}d_step{step_days}d"
                / f"window_{window_index:03d}_{fmt_date(start)}_{fmt_date(end)}"
            )
            rows = run_single_eval(args, window_dir, timerange=timerange, report_window=timerange)
            for row in rows:
                detail_rows.append(rolling_detail_row(
                    row=row,
                    spec_index=spec_index,
                    window_index=window_index,
                    window_days=window_days,
                    step_days=step_days,
                    window_start=start,
                    window_end=end,
                    result_dir=window_dir,
                ))
            aggregate = next(row for row in rows if row["mode"] == "single_fixed_aggregate")
            aggregate_rows.append(rolling_aggregate_row(
                row=aggregate,
                spec_index=spec_index,
                window_index=window_index,
                window_days=window_days,
                step_days=step_days,
                window_start=start,
                window_end=end,
                result_dir=window_dir,
            ))

    if not aggregate_rows:
        raise SystemExit("No rolling windows generated. Check --timerange and rolling settings.")

    pd.DataFrame(detail_rows).to_csv(output_dir / "rolling_detail.csv", index=False)
    rolling = pd.DataFrame(aggregate_rows)
    rolling.to_csv(output_dir / "rolling_summary.csv", index=False)
    (output_dir / "rolling_report.md").write_text(render_rolling_report(args, aggregate_rows), encoding="utf-8")
    print(rolling.to_string(index=False))
    print(f"\nWrote rolling evaluation {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="CryptoSpotV219B")
    parser.add_argument("--config", default="freqtrade_user_data/config/config.dryrun.example.json")
    parser.add_argument("--userdir", default="freqtrade_user_data")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--timerange", default="20240601-20260601")
    parser.add_argument("--report-window", default="20260301-20260601")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--allocation", nargs="+", type=float, default=DEFAULT_ALLOCATION)
    parser.add_argument("--cache", default="none")
    parser.add_argument("--output-dir", default="results/freqtrade_eval")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-run", action="store_true", help="Parse latest zip files in the output directory.")
    parser.add_argument("--verbose", action="store_true", help="Print full Freqtrade output instead of writing it to backtest.log.")
    parser.add_argument("--rolling-windows", action="store_true", help="Run lightweight fixed-allocation rolling-window evaluation.")
    parser.add_argument("--rolling-preset", choices=sorted(ROLLING_PRESETS), default="")
    parser.add_argument("--rolling-window-days", type=int, default=365)
    parser.add_argument("--rolling-step-days", type=int, default=90)
    return parser.parse_args()


def run_backtest(
    args: argparse.Namespace,
    pairs: list[str],
    wallet: float,
    run_dir: Path,
    *,
    pair: str,
    timerange: str,
) -> BacktestRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.no_run:
        result_zip = latest_result_zip(run_dir)
        return BacktestRun(pair=pair, wallet=wallet, directory=run_dir, result_zip=result_zip)

    cmd = [
        sys.executable,
        "-m",
        "freqtrade",
        "backtesting",
        "--userdir",
        args.userdir,
        "--config",
        args.config,
        "--strategy",
        args.strategy,
        "--timerange",
        timerange,
        "--timeframe",
        args.timeframe,
        "--cache",
        args.cache,
        "--dry-run-wallet",
        f"{wallet:g}",
        "--backtest-directory",
        str(run_dir),
        "--pairs",
        *pairs,
    ]
    print("Running:", " ".join(cmd))
    (run_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
    if args.verbose:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    else:
        log_path = run_dir / "backtest.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            print_log_tail(log_path)
            raise subprocess.CalledProcessError(completed.returncode, cmd)
    return BacktestRun(pair=pair, wallet=wallet, directory=run_dir, result_zip=latest_result_zip(run_dir))


def latest_result_zip(directory: Path) -> Path:
    files = sorted(directory.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No Freqtrade result zip found in {directory}")
    return files[0]


@dataclass
class ParsedResult:
    strategy: dict
    wallet: pd.DataFrame
    trades: list[dict]


def parse_result_zip(path: Path, strategy: str) -> ParsedResult:
    with zipfile.ZipFile(path) as zf:
        result_json = None
        for name in zf.namelist():
            if not name.endswith(".json") or name.endswith("_config.json") or name.endswith(".meta.json"):
                continue
            candidate = json.loads(zf.read(name))
            if isinstance(candidate, dict) and "strategy" in candidate:
                result_json = candidate
                break
        if result_json is None:
            raise ValueError(f"Missing strategy result JSON in {path}")
        strategy_result = result_json["strategy"][strategy]
        wallet_name = next(name for name in zf.namelist() if name.endswith("_wallet.feather"))
        wallet = read_feather_from_zip(zf, wallet_name)
    return ParsedResult(
        strategy=strategy_result,
        wallet=wallet,
        trades=list(strategy_result.get("trades", [])),
    )


def read_feather_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as handle:
        handle.write(zf.read(name))
        temp_path = Path(handle.name)
    try:
        return pd.read_feather(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_equity_curve(wallet: pd.DataFrame) -> pd.DataFrame:
    frame = wallet.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    grouped = frame.groupby("date", as_index=False).agg(equity=("total_quote", "sum"))
    cash = frame[frame["currency"] == "USDT"].groupby("date", as_index=False)["total_quote"].sum()
    cash = cash.rename(columns={"total_quote": "cash_value"})
    grouped = grouped.merge(cash, on="date", how="left")
    grouped["cash_value"] = grouped["cash_value"].fillna(0.0)
    grouped["exposure"] = (grouped["equity"] - grouped["cash_value"]) / grouped["equity"].clip(lower=1e-9)
    return grouped.sort_values("date")


def summarize_run(
    *,
    args: argparse.Namespace,
    run: BacktestRun,
    result: ParsedResult,
    equity: pd.DataFrame,
    pairs: list[str],
    allocation: list[float],
    timerange: str,
    report_window: str,
) -> dict:
    window_start, window_end = parse_window(report_window)
    full_start, full_end = first_last_equity(equity)
    window = slice_equity(equity, window_start, window_end)
    bh_full = buyhold_return(pairs, allocation, full_start, full_end, args)
    bh_window = buyhold_return(pairs, allocation, window["date"].iloc[0], window["date"].iloc[-1], args)
    trades = result.trades
    return {
        "mode": "single",
        "pair": run.pair,
        "timerange": timerange,
        "report_window": report_window,
        "start_wallet": run.wallet,
        "final_equity": float(equity["equity"].iloc[-1]),
        "total_return_pct": pct_return(equity["equity"].iloc[0], equity["equity"].iloc[-1]),
        "buyhold_total_return_pct": bh_full * 100,
        "total_excess_pct": pct_return(equity["equity"].iloc[0], equity["equity"].iloc[-1]) - bh_full * 100,
        "max_drawdown_pct": max_drawdown(equity["equity"]) * 100,
        "underwater_days": underwater_days(equity),
        "avg_exposure_pct": float(equity["exposure"].mean() * 100),
        "trade_count": len(trades),
        "win_rate_pct": win_rate(trades) * 100,
        "window_start_equity": float(window["equity"].iloc[0]),
        "window_end_equity": float(window["equity"].iloc[-1]),
        "window_return_pct": pct_return(window["equity"].iloc[0], window["equity"].iloc[-1]),
        "window_buyhold_return_pct": bh_window * 100,
        "window_excess_pct": pct_return(window["equity"].iloc[0], window["equity"].iloc[-1]) - bh_window * 100,
        "window_max_drawdown_pct": max_drawdown(window["equity"]) * 100,
        "window_underwater_days": underwater_days(window),
        "window_avg_exposure_pct": float(window["exposure"].mean() * 100),
        "window_trades_opened": count_trades(trades, window_start, window_end, "open"),
        "window_trades_closed": count_trades(trades, window_start, window_end, "close"),
        "active_at_window_start": active_at_start(trades, window_start),
        "result_zip": str(run.result_zip),
    }


def summarize_fixed_single_portfolio(
    *,
    args: argparse.Namespace,
    equities: list[pd.DataFrame],
    pairs: list[str],
    allocation: list[float],
    output_dir: Path,
    timerange: str,
    report_window: str,
) -> dict:
    merged = equities[0]
    for item in equities[1:]:
        merged = merged.merge(item, on="date", how="outer")
    merged = merged.sort_values("date").ffill()
    equity_cols = [f"{pair}:equity" for pair in pairs]
    cash_cols = [f"{pair}:cash" for pair in pairs]
    merged["equity"] = merged[equity_cols].sum(axis=1)
    merged["cash_value"] = merged[cash_cols].sum(axis=1)
    merged["exposure"] = (merged["equity"] - merged["cash_value"]) / merged["equity"].clip(lower=1e-9)
    merged[["date", "equity"]].to_csv(output_dir / "single_fixed_portfolio_equity.csv", index=False)

    window_start, window_end = parse_window(report_window)
    full_start, full_end = first_last_equity(merged)
    window = slice_equity(merged, window_start, window_end)
    bh_full = buyhold_return(pairs, allocation, full_start, full_end, args)
    bh_window = buyhold_return(pairs, allocation, window["date"].iloc[0], window["date"].iloc[-1], args)
    return {
        "mode": "single_fixed_aggregate",
        "pair": "PORTFOLIO",
        "timerange": timerange,
        "report_window": report_window,
        "start_wallet": sum(allocation),
        "final_equity": float(merged["equity"].iloc[-1]),
        "total_return_pct": pct_return(merged["equity"].iloc[0], merged["equity"].iloc[-1]),
        "buyhold_total_return_pct": bh_full * 100,
        "total_excess_pct": pct_return(merged["equity"].iloc[0], merged["equity"].iloc[-1]) - bh_full * 100,
        "max_drawdown_pct": max_drawdown(merged["equity"]) * 100,
        "underwater_days": underwater_days(merged),
        "avg_exposure_pct": float(merged["exposure"].mean() * 100),
        "trade_count": None,
        "win_rate_pct": None,
        "window_start_equity": float(window["equity"].iloc[0]),
        "window_end_equity": float(window["equity"].iloc[-1]),
        "window_return_pct": pct_return(window["equity"].iloc[0], window["equity"].iloc[-1]),
        "window_buyhold_return_pct": bh_window * 100,
        "window_excess_pct": pct_return(window["equity"].iloc[0], window["equity"].iloc[-1]) - bh_window * 100,
        "window_max_drawdown_pct": max_drawdown(window["equity"]) * 100,
        "window_underwater_days": underwater_days(window),
        "window_avg_exposure_pct": float(window["exposure"].mean() * 100),
        "window_trades_opened": None,
        "window_trades_closed": None,
        "active_at_window_start": None,
        "result_zip": "",
    }


def buyhold_return(pairs: list[str], allocation: list[float], start: pd.Timestamp, end: pd.Timestamp, args: argparse.Namespace) -> float:
    initial = sum(allocation)
    final = 0.0
    for pair, stake in zip(pairs, allocation):
        prices = load_pair_prices(pair, args)
        start_price = price_on_or_after(prices, start)
        end_price = price_on_or_before(prices, end)
        final += stake * end_price / start_price
    return final / initial - 1.0 if initial else 0.0


def load_pair_prices(pair: str, args: argparse.Namespace) -> pd.DataFrame:
    data_dir = PROJECT_ROOT / args.userdir / "data" / "binance"
    path = data_dir / f"{pair.replace('/', '_')}-{args.timeframe}.feather"
    if not path.exists():
        raise FileNotFoundError(f"Missing OHLCV data: {path}")
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date")


def price_on_or_after(prices: pd.DataFrame, date: pd.Timestamp) -> float:
    rows = prices[prices["date"] >= date]
    if rows.empty:
        raise ValueError(f"No price on or after {date}")
    return float(rows.iloc[0]["close"])


def price_on_or_before(prices: pd.DataFrame, date: pd.Timestamp) -> float:
    rows = prices[prices["date"] <= date]
    if rows.empty:
        raise ValueError(f"No price on or before {date}")
    return float(rows.iloc[-1]["close"])


def parse_window(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = value.split("-", 1)
    return parse_date(start), parse_date(end)


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def rolling_windows(timerange: str, window_days: int, step_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if window_days <= 0 or step_days <= 0:
        raise ValueError("rolling window and step days must be positive")
    start, end = parse_window(timerange)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current + pd.Timedelta(days=window_days) <= end:
        window_end = current + pd.Timedelta(days=window_days)
        windows.append((current, window_end))
        current += pd.Timedelta(days=step_days)
    return windows


def rolling_specs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.rolling_preset:
        return ROLLING_PRESETS[args.rolling_preset]
    return [(args.rolling_window_days, args.rolling_step_days)]


def rolling_detail_row(
    *,
    row: dict,
    spec_index: int,
    window_index: int,
    window_days: int,
    step_days: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    result_dir: Path,
) -> dict:
    return {
        "spec_index": spec_index,
        "window_index": window_index,
        "window_days": window_days,
        "step_days": step_days,
        "window_start": fmt_date(window_start),
        "window_end": fmt_date(window_end),
        "mode": row["mode"],
        "pair": row["pair"],
        "return_pct": row["total_return_pct"],
        "buyhold_return_pct": row["buyhold_total_return_pct"],
        "excess_return_pct": row["total_excess_pct"],
        "max_drawdown_pct": row["max_drawdown_pct"],
        "underwater_days": row["underwater_days"],
        "avg_exposure_pct": row["avg_exposure_pct"],
        "trade_count": row["trade_count"],
        "win_rate_pct": row["win_rate_pct"],
        "result_dir": str(result_dir),
    }


def rolling_aggregate_row(
    *,
    row: dict,
    spec_index: int,
    window_index: int,
    window_days: int,
    step_days: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    result_dir: Path,
) -> dict:
    detail = rolling_detail_row(
        row=row,
        spec_index=spec_index,
        window_index=window_index,
        window_days=window_days,
        step_days=step_days,
        window_start=window_start,
        window_end=window_end,
        result_dir=result_dir,
    )
    return {
        "spec_index": detail["spec_index"],
        "window_index": detail["window_index"],
        "window_days": detail["window_days"],
        "step_days": detail["step_days"],
        "window_start": detail["window_start"],
        "window_end": detail["window_end"],
        "portfolio_return_pct": detail["return_pct"],
        "buyhold_return_pct": detail["buyhold_return_pct"],
        "excess_return_pct": detail["excess_return_pct"],
        "max_drawdown_pct": detail["max_drawdown_pct"],
        "underwater_days": detail["underwater_days"],
        "avg_exposure_pct": detail["avg_exposure_pct"],
        "result_dir": detail["result_dir"],
    }


def fmt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def first_last_equity(equity: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(equity["date"].iloc[0]), pd.Timestamp(equity["date"].iloc[-1])


def slice_equity(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = equity[(equity["date"] >= start) & (equity["date"] <= end)].copy()
    if rows.empty:
        raise ValueError(f"No equity rows in report window {start} - {end}")
    return rows


def pct_return(start: float, end: float) -> float:
    return (float(end) / float(start) - 1.0) * 100 if start else 0.0


def max_drawdown(series: Iterable[float]) -> float:
    values = pd.Series(series, dtype=float).dropna()
    if values.empty:
        return 0.0
    return float((values / values.cummax() - 1.0).min())


def underwater_days(equity: pd.DataFrame) -> int:
    values = equity[["date", "equity"]].dropna().copy()
    if values.empty:
        return 0
    values["underwater"] = values["equity"] < values["equity"].cummax()
    longest = current = 0
    for is_underwater in values["underwater"]:
        current = current + 1 if bool(is_underwater) else 0
        longest = max(longest, current)
    return int(longest)


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if float(trade.get("profit_ratio") or 0.0) > 0)
    return wins / len(trades)


def count_trades(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp, field: str) -> int:
    count = 0
    for trade in trades:
        date = trade_date(trade, field)
        if date is not None and start <= date <= end:
            count += 1
    return count


def active_at_start(trades: list[dict], start: pd.Timestamp) -> bool:
    for trade in trades:
        open_date = trade_date(trade, "open")
        close_date = trade_date(trade, "close")
        if open_date is not None and open_date < start and (close_date is None or close_date >= start):
            return True
    return False


def trade_date(trade: dict, field: str) -> pd.Timestamp | None:
    keys = {
        "open": ("open_date", "open_date_utc"),
        "close": ("close_date", "close_date_utc"),
    }[field]
    for key in keys:
        value = trade.get(key)
        if value:
            date = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.notna(date):
                return pd.Timestamp(date)
    return None


def extract_trade_rows(run: BacktestRun, trades: list[dict]) -> list[dict]:
    rows = []
    for trade in trades:
        rows.append({
            "mode": "single",
            "run_pair": run.pair,
            "pair": trade.get("pair"),
            "open_date": trade.get("open_date") or trade.get("open_date_utc"),
            "close_date": trade.get("close_date") or trade.get("close_date_utc"),
            "profit_pct": float(trade.get("profit_ratio") or 0.0) * 100,
            "profit_abs": trade.get("profit_abs"),
            "stake_amount": trade.get("stake_amount"),
            "open_rate": trade.get("open_rate"),
            "close_rate": trade.get("close_rate"),
            "enter_tag": trade.get("enter_tag"),
            "exit_reason": trade.get("exit_reason"),
        })
    return rows


def safe_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def json_safe_rows(rows: list[dict]) -> list[dict]:
    safe: list[dict] = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                out[key] = None
            else:
                out[key] = value
        safe.append(out)
    return safe


def print_log_tail(path: Path, lines: int = 80) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\nLast {min(lines, len(content))} lines from {path}:")
    for line in content[-lines:]:
        print(line)


def render_markdown_report(args: argparse.Namespace, rows: list[dict]) -> str:
    lines = [
        "# Freqtrade Evaluation Report",
        "",
        f"- Strategy: `{args.strategy}`",
        f"- Timerange: `{args.timerange}`",
        f"- Report window: `{args.report_window}`",
        f"- Pairs: `{', '.join(args.pairs)}`",
        f"- Allocation: `{', '.join(f'{item:g}' for item in args.allocation)}`",
        "",
        "## Summary",
        "",
        "| Mode | Pair | Total | Buy&Hold | Excess | Max DD | Avg Exposure | Window | Window BH | Window Excess | Window DD | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {pair} | {total_return_pct} | {buyhold_total_return_pct} | "
            "{total_excess_pct} | {max_drawdown_pct} | {avg_exposure_pct} | "
            "{window_return_pct} | {window_buyhold_return_pct} | {window_excess_pct} | "
            "{window_max_drawdown_pct} | {trade_count} |".format(
                mode=row["mode"],
                pair=row["pair"],
                total_return_pct=fmt_pct(row.get("total_return_pct")),
                buyhold_total_return_pct=fmt_pct(row.get("buyhold_total_return_pct")),
                total_excess_pct=fmt_pct(row.get("total_excess_pct")),
                max_drawdown_pct=fmt_pct(row.get("max_drawdown_pct")),
                avg_exposure_pct=fmt_pct(row.get("avg_exposure_pct")),
                window_return_pct=fmt_pct(row.get("window_return_pct")),
                window_buyhold_return_pct=fmt_pct(row.get("window_buyhold_return_pct")),
                window_excess_pct=fmt_pct(row.get("window_excess_pct")),
                window_max_drawdown_pct=fmt_pct(row.get("window_max_drawdown_pct")),
                trade_count="" if row.get("trade_count") is None else f"{row['trade_count']:.0f}",
            )
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `summary.csv`: compact decision metrics",
        "- `summary.json`: machine-readable summary",
        "- `trades.csv`: trade-level diagnostics",
        "- `single_fixed_portfolio_equity.csv`: fixed-allocation aggregate equity curve",
        "- `backtest.log`: raw Freqtrade output in each run directory",
    ])
    return "\n".join(lines) + "\n"


def render_rolling_report(args: argparse.Namespace, rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    lines = [
        "# Freqtrade Rolling Evaluation",
        "",
        f"- Strategy: `{args.strategy}`",
        f"- Timerange: `{args.timerange}`",
        f"- Rolling preset: `{args.rolling_preset or 'custom'}`",
        f"- Rolling specs: `{', '.join(f'{days}d/{step}d' for days, step in rolling_specs(args))}`",
        f"- Pairs: `{', '.join(args.pairs)}`",
        f"- Allocation: `{', '.join(f'{item:g}' for item in args.allocation)}`",
        "",
        "## Aggregate",
        "",
    ]
    if frame.empty:
        lines.append("No windows generated.")
        return "\n".join(lines) + "\n"

    lines.extend([
        f"- Windows: `{len(frame)}`",
        f"- Mean return: `{frame['portfolio_return_pct'].mean():.2f}%`",
        f"- Median return: `{frame['portfolio_return_pct'].median():.2f}%`",
        f"- Mean Buy&Hold: `{frame['buyhold_return_pct'].mean():.2f}%`",
        f"- Median excess: `{frame['excess_return_pct'].median():.2f}%`",
        f"- Win rate vs Buy&Hold: `{(frame['excess_return_pct'] > 0).mean() * 100:.1f}%`",
        f"- Worst return: `{frame['portfolio_return_pct'].min():.2f}%`",
        f"- Worst max drawdown: `{frame['max_drawdown_pct'].min():.2f}%`",
        "",
        "## By Length",
        "",
        "| Length | Step | Windows | Median Return | Median Excess | Win Rate | Worst Return | Worst DD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for (window_days, step_days), group in frame.groupby(["window_days", "step_days"], sort=True):
        lines.append(
            f"| {window_days}d | {step_days}d | {len(group)} | "
            f"{group['portfolio_return_pct'].median():.2f}% | "
            f"{group['excess_return_pct'].median():.2f}% | "
            f"{(group['excess_return_pct'] > 0).mean() * 100:.1f}% | "
            f"{group['portfolio_return_pct'].min():.2f}% | "
            f"{group['max_drawdown_pct'].min():.2f}% |"
        )
    lines.extend([
        "",
        "## Windows",
        "",
        "| Length | Window | Strategy | Buy&Hold | Excess | Max DD | Avg Exposure |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['window_days']}d | {row['window_start']}~{row['window_end']} | "
            f"{row['portfolio_return_pct']:.2f}% | "
            f"{row['buyhold_return_pct']:.2f}% | "
            f"{row['excess_return_pct']:.2f}% | "
            f"{row['max_drawdown_pct']:.2f}% | "
            f"{row['avg_exposure_pct']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def fmt_pct(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
