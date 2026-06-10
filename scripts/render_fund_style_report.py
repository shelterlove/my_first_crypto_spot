#!/usr/bin/env python3
"""Render a fund-style one-page chart for a registered strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.backtest_event_driven import run_rebalance_backtest  # noqa: E402
from crypto_spot_v1.benchmark import V1BenchmarkRunner, build_strategy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="v3")
    parser.add_argument("--start", default="", help="Optional explicit start date/timestamp.")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--end", default="", help="Optional end date/timestamp. Defaults to latest common DB timestamp.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "reports" / "v3_fund_style_report.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", PROJECT_ROOT / "results")
    data = runner._inject_btc_regime()
    if not data:
        raise SystemExit("No DB data loaded.")

    latest_common = min(df["timestamp"].max() for df in data.values() if not df.empty)
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else latest_common
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    if args.start:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = end_ts - pd.DateOffset(years=args.years) + pd.Timedelta(days=1)

    report = run_strategy_window(args.strategy, data, runner, start_ts, end_ts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_report(report, args.strategy, start_ts, end_ts, output)
    print(f"Wrote {output}")


def run_strategy_window(
    strategy_name: str,
    data: dict[str, pd.DataFrame],
    runner: V1BenchmarkRunner,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    config = runner.config
    capital = config["capital"]["initial"]
    reserve = config["capital"]["reserve"]
    fee = config["cost"]["fee_rate"]
    min_notional = config.get("cost", {}).get("min_notional")
    execution_mode = config.get("execution", {}).get("mode", "next_open")
    equities: dict[str, pd.DataFrame] = {}
    prices: dict[str, pd.DataFrame] = {}
    actions: dict[str, pd.DataFrame] = {}

    for symbol, df in data.items():
        starts = df.index[df["timestamp"] >= start_ts].tolist()
        ends = df.index[df["timestamp"] <= end_ts].tolist()
        if not starts or not ends:
            continue
        eval_start = starts[0]
        eval_end = ends[-1] + 1
        backtest_start = max(0, eval_start - 1 if execution_mode != "same_close" else eval_start)
        window_df = df.iloc[eval_start:eval_end].reset_index(drop=True)
        backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)

        strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
        setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
        result = run_rebalance_backtest(
            {symbol: backtest_df},
            strategy,
            initial_capital=capital,
            reserve=reserve,
            fee_rate=fee,
            execution_mode=execution_mode,
        )
        action_log = result.attrs.get("action_log")
        result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
        action_log = pd.DataFrame() if action_log is None else action_log
        if not action_log.empty:
            action_log = action_log[action_log["timestamp"] >= start_ts].reset_index(drop=True)
        equities[symbol] = result[["timestamp", "total_value"]].copy()
        prices[symbol] = window_df[["timestamp", "close"]].copy()
        actions[symbol] = action_log

    return {
        "equities": equities,
        "prices": prices,
        "actions": actions,
        "metrics": portfolio_metrics(equities, actions, start_ts, end_ts),
    }


def portfolio_metrics(
    equities: dict[str, pd.DataFrame],
    actions: dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, float]:
    normalized = []
    for frame in equities.values():
        values = frame["total_value"].astype(float)
        normalized.append(pd.Series((values / values.iloc[0]).to_numpy()))
    if not normalized:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "trade_count": 0}
    composite = pd.concat(normalized, axis=1).mean(axis=1)
    total_return = float(composite.iloc[-1] - 1.0)
    years = max((end_ts - start_ts).days / 365.25, 1e-9)
    annual_return = float(composite.iloc[-1] ** (1 / years) - 1.0)
    drawdown = composite / composite.cummax() - 1.0
    trade_count = sum(len(frame) for frame in actions.values())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "trade_count": trade_count,
    }


def render_report(report: dict, strategy: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, output: Path) -> None:
    equities = report["equities"]
    prices = report["prices"]
    actions = report["actions"]
    metrics = report["metrics"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 10), dpi=160)
    grid = fig.add_gridspec(4, 4, height_ratios=[0.85, 1.25, 1.25, 1.25], hspace=0.42, wspace=0.25)
    fig.suptitle(f"{strategy.upper()} Crypto Spot Strategy Report", fontsize=20, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.935, f"{start_ts.date()} to {end_ts.date()} | Composite is equal-weight average of single-pair backtests", fontsize=10, color="#555555")

    metric_items = [
        ("Total Return", metrics["total_return"]),
        ("Annual Return", metrics["annual_return"]),
        ("Max Drawdown", metrics["max_drawdown"]),
        ("Trades", metrics["trade_count"]),
    ]
    metric_axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    for ax, (label, value) in zip(metric_axes, metric_items):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="#d9dee7", linewidth=1.0))
        text = f"{value:.1%}" if label != "Trades" else f"{int(value)}"
        color = "#0f766e" if label != "Max Drawdown" else "#b91c1c"
        ax.text(0.05, 0.62, text, fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.05, 0.24, label, fontsize=10, color="#475569", transform=ax.transAxes)

    palette = {"BTC/USDT": "#f59e0b", "ETH/USDT": "#2563eb", "BNB/USDT": "#10b981"}
    ordered_symbols = [symbol for symbol in ["BTC/USDT", "ETH/USDT", "BNB/USDT"] if symbol in prices]
    ordered_symbols.extend(symbol for symbol in sorted(prices) if symbol not in ordered_symbols)
    for row, symbol in enumerate(ordered_symbols, start=1):
        ax = fig.add_subplot(grid[row, :])
        price = prices[symbol].copy()
        price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
        price["normalized_close"] = price["close"].astype(float) / float(price["close"].iloc[0])
        ax.plot(price["timestamp"], price["normalized_close"], color=palette.get(symbol, "#334155"), linewidth=1.6, label=f"{symbol} price")

        equity = equities.get(symbol)
        if equity is not None and not equity.empty:
            eq = equity.copy()
            eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
            eq_norm = eq["total_value"].astype(float) / float(eq["total_value"].iloc[0])
            ax.plot(eq["timestamp"], eq_norm, color="#111827", linewidth=1.1, alpha=0.75, label="strategy equity")

        plot_actions(ax, actions.get(symbol, pd.DataFrame()), price)
        ax.set_title(symbol, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized")
        ax.legend(loc="upper left", ncols=3, fontsize=8, frameon=True)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_actions(ax, actions: pd.DataFrame, price: pd.DataFrame) -> None:
    if actions.empty:
        return
    first_close = float(price["close"].iloc[0])
    actions = actions.copy()
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    actions["normalized_price"] = actions["price"].astype(float) / first_close
    buys = actions[actions["side"] == "buy"]
    sells = actions[actions["side"] == "sell"]
    if not buys.empty:
        ax.scatter(buys["timestamp"], buys["normalized_price"], marker="^", s=34, color="#16a34a", edgecolor="white", linewidth=0.5, label="buy", zorder=5)
    if not sells.empty:
        ax.scatter(sells["timestamp"], sells["normalized_price"], marker="v", s=34, color="#dc2626", edgecolor="white", linewidth=0.5, label="sell", zorder=5)


if __name__ == "__main__":
    main()
