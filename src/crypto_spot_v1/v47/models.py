"""Small V4.7 execution-layer models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V47ExecutionDecision:
    symbol: str
    raw_pct: float
    main_target_pct: float
    outer_quantity: float
    final_target_pct: float
    reason: str


@dataclass
class V47OuterLot:
    state: str = "IDLE"
    quantity: float = 0.0
    overlay_pct: float = 0.0
    entry_price: float = 0.0
    entry_low: float = 0.0
    entry_call: int = 0
