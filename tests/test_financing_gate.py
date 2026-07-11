import pandas as pd

from futures_v1.benchmark import build_strategy


def gate_strategy(**values):
    strategy = build_strategy("eth_bnb_futures_v1", 100.0, 20.0, 0.001)
    strategy.RESEARCH_FINANCING_GATE_ENABLED = True
    strategy.RESEARCH_FINANCING_GATE_NON_STRONG_BULL_CAP = 1.0
    strategy.RESEARCH_FINANCING_GATE_STRONG_BULL_CAP = 1.75
    for name, value in values.items():
        setattr(strategy, name, value)
    return strategy


def frame(regime: str, funding: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "close": [100.0 + index for index in range(40)],
        "btc_regime": [regime] * 40,
        "funding_rate_daily": [0.0] * 39 + [funding],
    })


def test_gate_only_allows_full_financing_in_strong_bull() -> None:
    strategy = gate_strategy()
    assert strategy._apply_research_financing_gate(1.6, frame("RANGE"))[0] == 1.0
    assert strategy._apply_research_financing_gate(1.6, frame("STRONG_BULL"))[0] == 1.6


def test_high_funding_uses_prior_history_without_lookahead() -> None:
    strategy = gate_strategy(
        RESEARCH_FINANCING_GATE_NON_STRONG_BULL_CAP=1.75,
        RESEARCH_FINANCING_GATE_FUNDING_QUANTILE=0.8,
        RESEARCH_FINANCING_GATE_FUNDING_CAP=1.0,
    )
    target, reason = strategy._apply_research_financing_gate(1.6, frame("STRONG_BULL", funding=0.01))
    assert target == 1.0
    assert "funding-high" in reason
