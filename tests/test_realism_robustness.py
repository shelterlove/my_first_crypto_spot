import pandas as pd

from scripts.research_realism_robustness import SCENARIOS, grouped_return_stats, stats_from_returns


def test_scenarios_include_baseline_stress_and_delay() -> None:
    assert SCENARIOS["a_realistic"]["trend_mult"] == 1.75
    assert SCENARIOS["b_stress"]["funding"] == 2.0
    assert SCENARIOS["b_stress"]["fill_ratio"] < 1.0
    assert SCENARIOS["b_btc_delay_3d"]["btc_delay"] == 3


def test_stats_from_returns_uses_observation_count() -> None:
    returns = pd.Series([0.01] * 365)
    stats = stats_from_returns(returns)
    assert stats["annual_return"] > 36.0
    assert stats["max_drawdown"] == 0.0


def test_grouped_stats_do_not_annualize_sparse_bucket() -> None:
    stats = grouped_return_stats(pd.Series([0.25]))
    assert stats["mean_daily_return"] == 0.25
    assert stats["positive_rate"] == 1.0
    assert "annual_return" not in stats
