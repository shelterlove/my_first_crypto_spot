import pandas as pd

from scripts.research_execution_leverage import VARIANTS, audit_metrics, calendar_returns
from futures_v1.benchmark import build_strategy
from scripts.render_strategy_review_chart import apply_execution_overrides


def test_variants_keep_caps_consistent() -> None:
    for values in VARIANTS.values():
        assert values["low_recovery_mult"] > 0
        assert values["low_recovery_cap"] <= 2.30
        assert values["trend_cap"] <= 2.30


def test_execution_overrides_rebuild_engine_config() -> None:
    strategy = build_strategy("eth_bnb_futures_v1", 100.0, 20.0, 0.001)
    apply_execution_overrides(strategy, {"trend_mult": 2.2, "trend_cap": 2.3})
    assert strategy._core_execution_engine.config.execution.trend_mult == 2.2
    assert strategy._core_execution_engine.config.execution.trend_cap == 2.3


def test_calendar_and_audit_metrics() -> None:
    composite = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"], utc=True),
        "strategy_equity": [1.0, 1.2, 1.2, 1.08],
    })
    returns = calendar_returns(composite)
    assert round(returns[2024], 6) == 0.2
    assert round(returns[2025], 6) == -0.1
    audit = pd.DataFrame({
        "gross_position": [1.0, 2.1],
        "transform_reason": ["trend_confirmed+outer_qty_idle", "low_recovery+outer_qty_hold"],
    })
    metrics = audit_metrics(audit)
    assert metrics["symbol_days_gross_gt_2"] == 1
    assert metrics["trend_confirmed_rows"] == 1
    assert metrics["low_recovery_rows"] == 1
