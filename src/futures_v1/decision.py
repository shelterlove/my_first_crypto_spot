"""Structured decision records for strategy actions and signal review."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .strategy_rebalance import Action, PortfolioState, PositionState


@dataclass
class DecisionRecord:
    timestamp: Any
    symbol: str
    strategy: str
    action: str
    side: str
    quantity: float
    price: float
    notional: float
    current_pct: float
    raw_state: str
    confirmed_state: str
    target_pct: float | None
    risk_score: int | None
    trend_risk: int | None
    drawdown_risk: int | None
    setup: str
    guards: str
    reason: str
    no_trade_reason: str


def parse_action_reason(reason: str | None) -> dict[str, Any]:
    text = str(reason or "")
    parsed: dict[str, Any] = {
        "version": "",
        "side": "",
        "setup": "",
        "risk_score": None,
        "trend_risk": None,
        "drawdown_risk": None,
        "raw_state": "",
        "confirmed_state": "",
        "target_pct": None,
        "guards": "",
    }
    match = re.match(
        r"^(?P<version>.+?)_(?P<side>buy|sell)_(?P<setup>.*?)"
        r"_r(?P<risk>\d+)_tr(?P<trend>\d+)_dd(?P<drawdown>\d+)"
        r"_raw(?P<raw>[A-Z]+)_conf(?P<confirmed>[A-Z]+)_t(?P<target>\d+)%",
        text,
    )
    if not match:
        return parsed
    parsed.update({
        "version": match.group("version"),
        "side": match.group("side"),
        "setup": match.group("setup"),
        "risk_score": int(match.group("risk")),
        "trend_risk": int(match.group("trend")),
        "drawdown_risk": int(match.group("drawdown")),
        "raw_state": match.group("raw"),
        "confirmed_state": match.group("confirmed"),
        "target_pct": int(match.group("target")) / 100.0,
    })
    prefix = match.group(0)
    guards = text[len(prefix):].strip("_")
    parsed["guards"] = guards
    return parsed


def build_decision_record(
    *,
    timestamp: Any,
    symbol: str,
    strategy_name: str,
    action: Action | None,
    portfolio: PortfolioState,
    price: float,
    latest: pd.Series | None = None,
    no_trade_reason: str = "",
) -> dict[str, Any]:
    pos = portfolio.positions.get(symbol, PositionState())
    position_value = pos.quantity * price
    total_value = portfolio.cash + position_value
    current_pct = position_value / total_value if total_value > 0 else 0.0
    reason = action.reason if action is not None else ""
    parsed = parse_action_reason(reason)
    record = DecisionRecord(
        timestamp=timestamp,
        symbol=symbol,
        strategy=strategy_name,
        action=action.side if action is not None else "hold",
        side=action.side if action is not None else "",
        quantity=float(action.quantity) if action is not None else 0.0,
        price=float(price),
        notional=float(action.quantity * action.price) if action is not None else 0.0,
        current_pct=float(current_pct),
        raw_state=parsed["raw_state"],
        confirmed_state=parsed["confirmed_state"],
        target_pct=parsed["target_pct"],
        risk_score=parsed["risk_score"],
        trend_risk=parsed["trend_risk"],
        drawdown_risk=parsed["drawdown_risk"],
        setup=parsed["setup"],
        guards=parsed["guards"],
        reason=reason,
        no_trade_reason=no_trade_reason,
    )
    data = asdict(record)
    if latest is not None:
        for field in [
            "close", "ema24", "ema72", "ema168", "ema168_slope",
            "roc_5", "roc_10", "roc_20", "atr_pct_rank", "donchian_pos",
            "btc_regime",
        ]:
            if field in latest:
                data[field] = _json_value(latest.get(field))
    return data


def build_strategy_manifest(strategy: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "strategy_name": getattr(strategy, "name", strategy.__class__.__name__),
        "class_name": strategy.__class__.__name__,
        "version_label": getattr(strategy, "VERSION_LABEL", ""),
        "git_commit": _git_commit_hash(),
        "target_table": _json_value(getattr(strategy, "TARGET_TABLE", {})),
        "state_config": _json_value(getattr(strategy, "STATE_CONFIG", {})),
        "confirm_bars": _json_value(getattr(strategy, "CONFIRM_BARS", {})),
        "key_constants": _strategy_constants(strategy),
        "config": _json_value(config or {}),
    }


def write_strategy_manifest(path: str, strategy: Any, config: dict[str, Any] | None = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_strategy_manifest(strategy, config), fh, indent=2, ensure_ascii=False)


def _strategy_constants(strategy: Any) -> dict[str, Any]:
    names = [
        "MIN_ADJUST_THRESHOLD", "LONG_TRIM_MAX_SELL", "LONG_SELL_THRESHOLD",
        "VOL_SCALE_HIGH", "VOL_SCALE_EXTREME", "BUY_REDUCTION_FLOOR",
        "COST_AWARE_BUY_PROTECTION_MULT", "COST_AWARE_BUY_MODERATE_MULT",
    ]
    return {name: _json_value(getattr(strategy, name)) for name in names if hasattr(strategy, name)}


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value
