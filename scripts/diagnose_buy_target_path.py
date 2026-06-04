#!/usr/bin/env python3
"""Diagnose daily buy-target recovery paths for a native strategy."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from crypto_spot_v1.benchmark import build_strategy  # noqa: E402
from crypto_spot_v1.freqtrade_adapter import _execute_synthetic_action  # noqa: E402
from crypto_spot_v1.freqtrade_adapter import _portfolio_value  # noqa: E402
from crypto_spot_v1.strategy_rebalance import PortfolioState, PositionState  # noqa: E402


PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
WALLETS = {"BTC/USDT": 333.0, "ETH/USDT": 333.0, "BNB/USDT": 334.0}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    btc_regime = load_btc_regime(Path(args.datadir))
    for pair in PAIRS:
        frame = load_pair(Path(args.datadir), pair)
        frame["btc_regime"] = btc_regime.reindex(frame.index).ffill().fillna("RANGE")
        rows.extend(run_pair_diagnostic(pair, frame, args))

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "buy_target_path_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(output_dir / "buy_target_path_summary.csv", index=False)
    report = render_report(args, summary)
    (output_dir / "buy_target_path_report.md").write_text(report, encoding="utf-8")

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-name", default="v2_21E")
    parser.add_argument("--datadir", default=str(PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="buy_target_path_v2_21E_2023_2024_20260604")
    parser.add_argument("--timerange-start", default="2023-01-01")
    parser.add_argument("--timerange-end", default="2024-12-01")
    parser.add_argument("--startup-candles", type=int, default=220)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-notional", type=float, default=0.0)
    parser.add_argument("--reset-at-start", action="store_true")
    return parser.parse_args()


def load_btc_regime(datadir: Path) -> pd.Series:
    btc = load_pair(datadir, "BTC/USDT")
    return strategy_utils.compute_btc_regime(btc)


def load_pair(datadir: Path, pair: str) -> pd.DataFrame:
    path = datadir / f"{pair.replace('/', '_')}-1d.feather"
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").set_index("date")
    return strategy_utils.compute_indicators(frame)


def run_pair_diagnostic(pair: str, frame: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    capital = WALLETS[pair]
    strategy = build_strategy(
        args.strategy_name,
        capital=capital,
        reserve=0.0,
        fee=args.fee_rate,
        min_notional=args.min_notional,
    )
    setattr(strategy, "TARGET_ALLOC", {pair: 1.0})
    portfolio = PortfolioState(cash=capital, positions={pair: PositionState()})
    rows = []

    report_start = pd.Timestamp(args.timerange_start, tz="UTC")
    report_end = pd.Timestamp(args.timerange_end, tz="UTC")
    start = max(1, int(args.startup_candles))
    if args.reset_at_start:
        start = max(start, int(frame.index.searchsorted(report_start)))
    for idx in range(start, len(frame)):
        history = frame.iloc[: idx + 1]
        latest = history.iloc[-1]
        if latest.name < report_start:
            if not args.reset_at_start:
                action = strategy.compute_actions({pair: history}, portfolio, {pair: float(latest["close"])})
                if action:
                    _execute_synthetic_action(action[0], portfolio, args.fee_rate)
            continue
        if latest.name > report_end:
            break

        price = float(latest["close"])
        context = compute_target_context(copy.deepcopy(strategy), pair, history, portfolio, price)
        action = strategy.compute_actions({pair: history}, portfolio, {pair: price})
        action_obj = action[0] if action else None
        if action_obj is not None:
            _execute_synthetic_action(action_obj, portfolio, args.fee_rate)

        rows.append({
            "date": latest.name,
            "pair": pair,
            "price": price,
            "action": action_obj.side if action_obj is not None else "hold",
            "action_reason": action_obj.reason if action_obj is not None else "",
            **context,
        })
    return rows


def compute_target_context(strategy, pair: str, history: pd.DataFrame, portfolio: PortfolioState, price: float) -> dict:
    strategy._call_count += 1
    latest = history.iloc[-1]
    pos = portfolio.positions.get(pair, PositionState())
    position_value = pos.quantity * price
    total_value = portfolio.cash + position_value
    current_pct = position_value / total_value if total_value > 0 else 0.0

    if current_pct < 0.20:
        strategy._peak_price = price
    elif pos.quantity > 1e-12:
        strategy._peak_price = max(strategy._peak_price, price)

    raw_state = strategy._detect_market_state(latest)
    confirmed_state = strategy._apply_state_confirmation(raw_state)
    trend_risk = strategy._calculate_trend_risk(latest, price)
    drawdown_risk = strategy._calculate_drawdown_risk(latest, pos, price)
    risk_score = min(trend_risk + drawdown_risk, 5)

    recovery_override = strategy._is_recovery_override_setup(
        df=history,
        latest=latest,
        price=price,
        raw_state=raw_state,
        confirmed_state=confirmed_state,
        trend_risk=trend_risk,
        risk_score=risk_score,
    )
    effective_risk_score = (
        max(risk_score - strategy.RECOVERY_RISK_SCORE_REDUCTION, 0)
        if recovery_override and hasattr(strategy, "RECOVERY_RISK_SCORE_REDUCTION")
        else risk_score
    )

    sell_target = strategy._lookup_target(raw_state, risk_score)
    buy_target = strategy._lookup_target(confirmed_state, effective_risk_score)
    vol_multiplier = strategy._get_directional_vol_multiplier(latest, price)
    sell_target = max(0.0, min(1.0, sell_target * vol_multiplier))
    buy_target = max(0.0, min(1.0, buy_target * vol_multiplier))

    btc_adjust = strategy._get_btc_adjust(latest, pair)
    sell_target = max(0.0, min(1.0, sell_target + btc_adjust))
    buy_target = max(0.0, min(1.0, buy_target + btc_adjust))

    sell_target = strategy._compose_target(
        symbol=pair,
        tactical_target=sell_target,
        raw_state=raw_state,
        trend_risk=trend_risk,
        drawdown_risk=drawdown_risk,
        latest=latest,
        price=price,
        side="sell",
    )
    buy_target = strategy._compose_target(
        symbol=pair,
        tactical_target=buy_target,
        raw_state=raw_state,
        trend_risk=trend_risk,
        drawdown_risk=drawdown_risk,
        latest=latest,
        price=price,
        side="buy",
    )

    trend_continuation = strategy._is_trend_continuation_setup(confirmed_state, latest, price, trend_risk)
    if trend_continuation:
        buy_target = min(strategy._target_cap(), buy_target + strategy.TREND_CONTINUATION_BOOST)

    bull_guard = False
    if hasattr(strategy, "_is_bull_guard_setup"):
        bull_guard = strategy._is_bull_guard_setup(
            latest=latest,
            price=price,
            confirmed_state=confirmed_state,
            trend_risk=trend_risk,
            risk_score=risk_score,
        )
        if bull_guard:
            buy_target = max(buy_target, strategy.BULL_GUARD_MIN_POSITION_PCT)

    sell_target = max(0.0, min(strategy._target_cap(), sell_target))
    buy_target = max(0.0, min(strategy._target_cap(), buy_target))

    strategy._track_recovery(confirmed_state)

    pullback_buy = strategy._is_safe_pullback_buy(confirmed_state, latest, price, trend_risk)
    safe_recovery = strategy._is_safe_recovery_buy(latest, price, trend_risk)
    buy_setup = strategy._classify_buy_setup(
        trend_continuation,
        safe_recovery or recovery_override,
        pullback_buy,
    )

    cfg = strategy._state_config[confirmed_state]
    if trend_continuation:
        base_max_buy = min(cfg.get("max_buy", 0.25) * strategy.TREND_CONTINUATION_MAX_BUY_MULT, max(buy_target - current_pct, 0.0))
    elif safe_recovery:
        base_max_buy = min(cfg.get("max_buy", 0.25) * 2.0, max(buy_target - current_pct, 0.0))
    elif recovery_override and hasattr(strategy, "RECOVERY_BUY_SIZE_MULT"):
        base_max_buy = min(cfg.get("max_buy", 0.25) * strategy.RECOVERY_BUY_SIZE_MULT, max(buy_target - current_pct, 0.0))
    elif pullback_buy:
        base_max_buy = min(cfg.get("max_buy", 0.25) * 1.5, max(buy_target - current_pct, 0.0))
    else:
        base_max_buy = cfg.get("max_buy", 0.25)
    adjusted_max_buy, buy_guard = strategy._adjust_buy_execution(
        latest=latest,
        price=price,
        raw_state=raw_state,
        buy_setup=buy_setup,
        max_buy=base_max_buy,
        confirmed_state=confirmed_state,
    )

    gap = buy_target - current_pct
    cooldown = strategy._compute_buy_cooldown(confirmed_state, cfg, effective_risk_score)
    cooldown, cooldown_guard = strategy._adjust_buy_cooldown(buy_setup, cooldown)
    days_since_buy = strategy._call_count - strategy._last_buy_call

    return {
        "portfolio_value": _portfolio_value(portfolio, pair, price),
        "current_pct": current_pct,
        "sell_target": sell_target,
        "buy_target": buy_target,
        "target_gap": gap,
        "base_max_buy": base_max_buy,
        "adjusted_max_buy": adjusted_max_buy,
        "executable_buy_pct": max(0.0, min(gap, adjusted_max_buy)),
        "raw_state": raw_state,
        "confirmed_state": confirmed_state,
        "trend_risk": trend_risk,
        "drawdown_risk": drawdown_risk,
        "risk_score": risk_score,
        "effective_risk_score": effective_risk_score,
        "btc_regime": str(latest.get("btc_regime", "")),
        "buy_setup": buy_setup,
        "recovery_override": recovery_override,
        "trend_continuation": trend_continuation,
        "bull_guard": bull_guard,
        "buy_cooldown": cooldown,
        "days_since_buy": days_since_buy,
        "cooldown_blocked": days_since_buy < cooldown,
        "buy_guard": buy_guard,
        "cooldown_guard": cooldown_guard,
        "ema24": latest.get("ema24"),
        "ema72": latest.get("ema72"),
        "ema72_slope": latest.get("ema72_slope"),
        "ema168": latest.get("ema168"),
        "ema168_slope": latest.get("ema168_slope"),
        "roc_10": latest.get("roc_10"),
        "roc_20": latest.get("roc_20"),
        "atr_pct_rank": latest.get("atr_pct_rank"),
        "donchian_pos": latest.get("donchian_pos"),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, group in detail.groupby("pair"):
        recovery_like = group[
            (group["confirmed_state"].isin(["MIXED", "BULL"]))
            & (group["raw_state"] != "BEAR")
            & (group["price"] > group["ema24"])
            & (group["ema72_slope"] > 0)
        ]
        actionable = group[group["target_gap"] > 0.02]
        rows.append({
            "pair": pair,
            "days": len(group),
            "mean_current_pct": group["current_pct"].mean(),
            "mean_buy_target": group["buy_target"].mean(),
            "mean_target_gap": group["target_gap"].mean(),
            "days_target_ge_70": int((group["buy_target"] >= 0.70).sum()),
            "days_target_ge_90": int((group["buy_target"] >= 0.90).sum()),
            "actionable_days": len(actionable),
            "cooldown_blocked_actionable_days": int(actionable["cooldown_blocked"].sum()),
            "recovery_like_days": len(recovery_like),
            "recovery_like_mean_buy_target": recovery_like["buy_target"].mean(),
            "recovery_like_mean_current_pct": recovery_like["current_pct"].mean(),
            "recovery_like_mean_gap": recovery_like["target_gap"].mean(),
            "recovery_like_target_ge_70_pct": pct((recovery_like["buy_target"] >= 0.70).mean()),
            "recovery_like_current_ge_70_pct": pct((recovery_like["current_pct"] >= 0.70).mean()),
        })
    return pd.DataFrame(rows)


def pct(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value * 100.0)


def render_report(args: argparse.Namespace, summary: pd.DataFrame) -> str:
    return "\n".join([
        f"# Buy Target Path Diagnostic: {args.strategy_name}",
        "",
        f"Timerange: {args.timerange_start} to {args.timerange_end}",
        "",
        summary.to_markdown(index=False),
        "",
        "Interpretation focus:",
        "- If recovery-like buy_target is low, target recovery is the bottleneck.",
        "- If buy_target is high but current_pct stays low with many cooldown blocks, execution is the bottleneck.",
        "- If target and current are both high while returns lag, sell sensitivity or target-reduce churn is likely upstream.",
        "",
    ])


if __name__ == "__main__":
    main()
