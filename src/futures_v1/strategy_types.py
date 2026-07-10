from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategy_rebalance import PositionState


class StrategyRecord:
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class StrategyContext(StrategyRecord):
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
class StrategyRegime(StrategyRecord):
    regime: str
    reason: str
    btc_regime: str
    atr_rank: float
    price_vs_ema72: float
    price_vs_ema168: float
    structural_bear: bool


@dataclass
class StrategySignals:
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
class StrategyLifecycle(StrategyRecord):
    phase: str = "NORMAL"
    episode_state: str = "NORMAL"
    reason: str = "normal"


@dataclass
class StrategyRiskGate(StrategyRecord):
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
class StrategyRiskAssessment(StrategyRecord):
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
class StrategyBearBaseProposal(StrategyRecord):
    allowed: bool = False
    layer: int = 0
    target: float = 0.0
    blocked_reason: str = ""


@dataclass
class MainSleeveState(StrategyRecord):
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    invested_capital: float = 0.0


@dataclass
class BaseSleeveState(StrategyRecord):
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    invested_capital: float = 0.0


@dataclass
class StrategySizing(StrategyRecord):
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
class StrategyTargetPlan(StrategyRecord):
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
class StrategySleevePlan(StrategyRecord):
    sleeve: str = ""
    side: str = ""
    setup: str = ""
    target: float = 0.0
    guard: str = ""
    priority: int = 0
    allowed: bool = False
    blocked_reason: str = ""


@dataclass
class StrategyDecisionPlan(StrategyRecord):
    intent: str = "HOLD"
    target: float = 0.0
    mature_target: float = 0.0
    phase_target: float = 0.0
    execution_target_today: float = 0.0
    target_plan: StrategyTargetPlan | None = None
    primary_sleeve: StrategySleevePlan | None = None
    sleeve_plans: tuple[StrategySleevePlan, ...] = ()


@dataclass
class StrategyDecisionSnapshot(StrategyRecord):
    context: StrategyContext
    regime: StrategyRegime
    episode: dict
    lifecycle: StrategyLifecycle
    risk_gate: StrategyRiskGate
    risk_assessment: StrategyRiskAssessment
    signals: StrategySignals
    decision: StrategyDecisionPlan


@dataclass
class StrategyRecoveryPlan(StrategyRecord):
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
