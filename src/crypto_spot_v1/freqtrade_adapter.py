"""Small adapter between native target-position decisions and Freqtrade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark import build_strategy
from .decision import build_decision_record
from . import strategy_utils
from .strategy_rebalance import Action
from .strategy_rebalance import PortfolioState, PositionState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"
BTC_PAIR = "BTC/USDT"
DEFAULT_STRATEGY_NAME = "v4_8_eth_bnb"
BTC_FEATURE_COLUMNS = (
    "btc_price_vs_ema72",
    "btc_price_vs_ema168",
    "btc_ema24_slope",
    "btc_ema168_slope",
    "btc_roc_20",
)


@dataclass(frozen=True)
class TargetPositionDecision:
    pair: str
    timestamp: Any
    action: str
    target_pct: float | None
    current_pct: float
    delta_pct: float
    price: float
    order_notional: float
    reason: str
    state: str
    risk_score: int | None


def build_target_position_decision(
    *,
    pair: str,
    dataframe: pd.DataFrame,
    current_position_pct: float,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    capital: float = 100.0,
    reserve: float = 20.0,
    fee_rate: float = 0.001,
    min_notional: float = 0.0,
) -> TargetPositionDecision:
    """Run the native strategy once and return a compact target-position decision."""
    if dataframe.empty:
        raise ValueError("dataframe must not be empty")

    latest = dataframe.iloc[-1]
    price = float(latest["close"])
    portfolio = _portfolio_for_pair(pair, price, capital, current_position_pct)
    strategy = build_strategy(strategy_name, capital, reserve, fee_rate, min_notional=min_notional)
    setattr(strategy, "TARGET_ALLOC", {pair: 1.0})

    actions = strategy.compute_actions({pair: dataframe}, portfolio, {pair: price})
    action = actions[0] if actions else None
    record = build_decision_record(
        timestamp=latest.get("timestamp", dataframe.index[-1]),
        symbol=pair,
        strategy_name=strategy_name,
        action=action,
        portfolio=portfolio,
        price=price,
        latest=latest,
        no_trade_reason="" if action else "target_or_cooldown_not_actionable",
    )
    delta_pct = _action_delta_pct(record["side"], record["notional"], capital)
    return TargetPositionDecision(
        pair=pair,
        timestamp=record["timestamp"],
        action=record["action"],
        target_pct=record["target_pct"],
        current_pct=float(record["current_pct"]),
        delta_pct=delta_pct,
        price=price,
        order_notional=float(record["notional"]),
        reason=record["reason"],
        state=record["confirmed_state"] or record["raw_state"],
        risk_score=record["risk_score"],
    )


def decision_as_dict(decision: TargetPositionDecision) -> dict[str, Any]:
    return asdict(decision)


def build_native_signal_frame(
    *,
    pair: str,
    dataframe: pd.DataFrame,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    capital: float = 100.0,
    reserve: float = 20.0,
    fee_rate: float = 0.001,
    min_notional: float = 0.0,
    startup_candle_count: int = 220,
) -> pd.DataFrame:
    """Generate stateful native actions over a full Freqtrade dataframe.

    This is the backtest/dry-run bridge for Freqtrade. It keeps one native
    strategy instance and one synthetic portfolio for the whole dataframe, so
    confirmation bars, cooldowns, and last-trade state are preserved.
    """
    out = dataframe.copy()
    out["native_action"] = "hold"
    out["native_reason"] = ""
    out["native_delta_pct"] = 0.0
    out["native_current_pct"] = 0.0
    out["native_target_pct"] = pd.NA

    if out.empty:
        return out
    out = _with_btc_regime(out, pair)

    strategy = build_strategy(strategy_name, capital, reserve, fee_rate, min_notional=min_notional)
    setattr(strategy, "TARGET_ALLOC", {pair: 1.0})
    portfolio = PortfolioState(cash=capital, positions={pair: PositionState()})

    start = max(1, int(startup_candle_count))
    for idx in range(start, len(out)):
        frame = out.iloc[: idx + 1]
        latest = frame.iloc[-1]
        price = float(latest["close"])
        current_pct = _current_position_pct(portfolio, pair, price)
        actions = strategy.compute_actions({pair: frame}, portfolio, {pair: price})
        action = actions[0] if actions else None
        record = build_decision_record(
            timestamp=latest.get("timestamp", frame.index[-1]),
            symbol=pair,
            strategy_name=strategy_name,
            action=action,
            portfolio=portfolio,
            price=price,
            latest=latest,
            no_trade_reason="" if action else "target_or_cooldown_not_actionable",
        )
        row_index = out.index[idx]
        out.loc[row_index, "native_action"] = record["action"]
        out.loc[row_index, "native_reason"] = record["reason"]
        out.loc[row_index, "native_delta_pct"] = _action_delta_pct(
            record["side"],
            record["notional"],
            _portfolio_value(portfolio, pair, price),
        )
        out.loc[row_index, "native_current_pct"] = current_pct
        if record["target_pct"] is not None:
            out.loc[row_index, "native_target_pct"] = record["target_pct"]

        for item in actions:
            _execute_synthetic_action(item, portfolio, fee_rate)

    return out


def _portfolio_for_pair(pair: str, price: float, capital: float, pct: float) -> PortfolioState:
    pct = max(0.0, min(1.0, float(pct)))
    position_value = capital * pct
    quantity = position_value / price if price > 0 else 0.0
    return PortfolioState(
        cash=capital - position_value,
        positions={pair: PositionState(quantity=quantity, avg_cost=price if quantity > 0 else 0.0)},
    )


def _with_btc_regime(dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
    out = dataframe.copy()
    if "timestamp" not in out.columns:
        out["timestamp"] = out["date"] if "date" in out.columns else out.index
    timestamps = pd.to_datetime(out["timestamp"], utc=True)

    if pair == BTC_PAIR:
        source = out.copy()
        source.index = timestamps
        btc_features = strategy_utils.compute_indicators(source)
        btc_features["btc_regime"] = strategy_utils.compute_btc_regime(btc_features)
    else:
        btc_features = _cached_btc_features()

    aligned = btc_features.reindex(timestamps).ffill()
    out["btc_regime"] = aligned["btc_regime"].fillna("RANGE").to_numpy()
    out["btc_regime_timestamp"] = timestamps.to_numpy()
    for column in BTC_FEATURE_COLUMNS:
        out[column] = aligned[column].to_numpy()
    return out


@lru_cache(maxsize=1)
def _cached_btc_features() -> pd.DataFrame:
    path = DATA_DIR / "BTC_USDT-1d.feather"
    if not path.exists():
        raise FileNotFoundError(f"BTC regime source not found: {path}")
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").set_index("date")
    frame = strategy_utils.compute_indicators(frame)
    out = pd.DataFrame(index=frame.index)
    out["btc_regime"] = strategy_utils.compute_btc_regime(frame)
    out["btc_price_vs_ema72"] = frame["close"] / frame["ema72"] - 1.0
    out["btc_price_vs_ema168"] = frame["close"] / frame["ema168"] - 1.0
    out["btc_ema24_slope"] = frame["ema24_slope"]
    out["btc_ema168_slope"] = frame["ema168_slope"]
    out["btc_roc_20"] = frame["roc_20"]
    return out


def _current_position_pct(portfolio: PortfolioState, pair: str, price: float) -> float:
    pos = portfolio.positions.get(pair, PositionState())
    position_value = pos.quantity * price
    total_value = max(portfolio.cash + position_value, 1e-9)
    return max(0.0, min(1.0, position_value / total_value))


def _portfolio_value(portfolio: PortfolioState, pair: str, price: float) -> float:
    pos = portfolio.positions.get(pair, PositionState())
    return max(portfolio.cash + pos.quantity * price, 1e-9)


def _execute_synthetic_action(action: Action, portfolio: PortfolioState, fee_rate: float) -> None:
    pos = portfolio.positions.setdefault(action.symbol, PositionState())
    notional = action.quantity * action.price
    fee = notional * fee_rate
    if action.side == "buy":
        total_cost = pos.avg_cost * pos.quantity + notional
        pos.quantity += action.quantity
        pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0.0
        portfolio.cash -= notional + fee
        return

    portfolio.cash += notional - fee
    pos.quantity -= action.quantity
    if pos.quantity <= 1e-12:
        pos.quantity = 0.0
        pos.avg_cost = 0.0


def _action_delta_pct(side: str, notional: float, capital: float) -> float:
    if capital <= 0 or not side:
        return 0.0
    sign = 1.0 if side == "buy" else -1.0
    return sign * float(notional) / capital
