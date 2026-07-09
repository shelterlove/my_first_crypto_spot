from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategy_rebalance import PositionState


class V42Record:
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class V42Context(V42Record):
    symbol: str
    df: pd.DataFrame
    latest: pd.Series
    price: float
    pos: PositionState
    total_value: float
    current_pct: float
    raw_state: str
    confirmed_state: str
    trend_risk: int
    drawdown_risk: int
    risk_score: int


@dataclass
class V42Regime(V42Record):
    regime: str
    reason: str
    btc_regime: str
    atr_rank: float
    price_vs_ema72: float
    price_vs_ema168: float
    structural_bear: bool


@dataclass
class V42Signals:
    starter: bool
    value_recovery: bool
    trend_continuation: bool
    distribution_exhaustion: bool
    recovery_signal: bool
    strong_recovery_signal: bool
    recovery_quality_ok: bool

    @property
    def accumulation(self) -> bool:
        return self.starter or self.value_recovery or self.trend_continuation


@dataclass
class V42Lifecycle(V42Record):
    phase: str = "NORMAL"
    episode_state: str = "NORMAL"
    reason: str = "normal"


@dataclass
class V42RiskGate(V42Record):
    allow_main_buy: bool = True
    allow_recovery_buy: bool = True
    allow_base_buy: bool = True
    allow_main_sell: bool = True
    allow_base_sell: bool = True
    allow_distribution_sell: bool = True
    force_defense: bool = False
    force_exit: bool = False
    reason: str = ""


@dataclass
class V42RiskAssessment(V42Record):
    risk_mode: str = "NORMAL"
    severity: int = 0
    expected_intent: str = "HOLD"
    allow_main_buy: bool = True
    allow_recovery_buy: bool = True
    allow_base_buy: bool = True
    allow_main_sell: bool = True
    allow_base_sell: bool = True
    force_defense: bool = False
    force_exit: bool = False
    episode_override: str = ""
    reason: str = ""


@dataclass
class V42BearBaseProposal(V42Record):
    allowed: bool = False
    layer: int = 0
    target: float = 0.0
    blocked_reason: str = ""


@dataclass
class MainSleeveState(V42Record):
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    invested_capital: float = 0.0


@dataclass
class BaseSleeveState(V42Record):
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    invested_capital: float = 0.0


@dataclass
class V42Sizing(V42Record):
    side: str = ""
    setup: str = ""
    quantity: float = 0.0
    target: float = 0.0
    guard: str = ""
    blocked_reason: str = ""
    actual_position_before: float = 0.0
    actual_position_after: float = 0.0
    target_gap_before: float = 0.0
    actual_step_pct: float = 0.0
    remaining_gap_after: float = 0.0


@dataclass
class V42TargetPlan(V42Record):
    bear_base_target: float = 0.0
    bear_base_floor: float = 0.0
    bear_base_exit_target: float = 0.0
    base_accumulate_needed: bool = False
    base_exit_distribute: bool = False
    tactical_capacity: float = 1.0
    tactical_current: float = 0.0
    tactical_current_ratio: float = 0.0
    desired_base: float = 0.0
    tactical_target: float = 0.0
    mature_target: float = 0.0
    phase_target: float = 0.0
    execution_target_today: float = 0.0


@dataclass
class V42SleevePlan(V42Record):
    sleeve: str = ""
    side: str = ""
    setup: str = ""
    target: float = 0.0
    guard: str = ""
    priority: int = 0
    allowed: bool = False
    blocked_reason: str = ""


@dataclass
class V42DecisionPlan(V42Record):
    intent: str = "HOLD"
    target: float = 0.0
    mature_target: float = 0.0
    phase_target: float = 0.0
    execution_target_today: float = 0.0
    target_plan: V42TargetPlan | None = None
    primary_sleeve: V42SleevePlan | None = None
    sleeve_plans: tuple[V42SleevePlan, ...] = ()


@dataclass
class V42DecisionSnapshot(V42Record):
    context: V42Context
    regime: V42Regime
    episode: dict
    lifecycle: V42Lifecycle
    risk_gate: V42RiskGate
    risk_assessment: V42RiskAssessment
    signals: V42Signals
    decision: V42DecisionPlan


@dataclass
class V42RecoveryPlan(V42Record):
    allowed: bool = False
    blocked_reason: str | None = None
    target: float | None = None
    max_buy: float | None = None
    drawdown: float | None = None
    remaining_budget: float | None = None
    guard: str | None = None
    fraction: float | None = None
    quality: str | None = None

    def get(self, key: str, default=None):
        value = getattr(self, key, None)
        return default if value is None else value
