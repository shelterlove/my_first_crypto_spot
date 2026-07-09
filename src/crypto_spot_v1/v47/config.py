"""V4.7 configuration.

The first clean extraction keeps raw decision behavior in the legacy chain and
centralizes only the execution-layer parameters that make V4.7 distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class V47ExecutionConfig:
    borrow_apr: float = 0.10
    min_target_gap: float = 0.005
    min_notional: float = 5.0
    maintenance_margin: float = 0.25
    warning_gross: float = 1.85
    low_recovery_mult: float = 2.60
    low_recovery_cap: float = 2.00
    trend_mult: float = 1.75
    trend_cap: float = 1.75


@dataclass(frozen=True)
class V47OuterConfig:
    target_pct: dict[str, float] = field(default_factory=lambda: {
        "BTC/USDT": 0.28,
        "ETH/USDT": 0.28,
        "BNB/USDT": 0.20,
    })
    min_raw: float = 0.05
    min_history: int = 180
    min_hold_calls: int = 120
    hard_stop: float = 0.70
    deep_only_entry: bool = False
    entry_rolling365_pos: float = 0.15
    entry_dd365: float = -0.60
    entry_dd180: float = -0.42
    entry_rebound20: float = 0.08
    entry_roc5: float = -0.04
    entry_roc20: float = -0.18
    deep_rolling365_pos: float = 0.12
    deep_dd365: float = -0.60
    waterfall_roc20: float = -0.18
    waterfall_rebound20: float = 0.08


@dataclass(frozen=True)
class V47Config:
    execution: V47ExecutionConfig = field(default_factory=V47ExecutionConfig)
    outer: V47OuterConfig = field(default_factory=V47OuterConfig)
