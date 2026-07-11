import pandas as pd

from scripts.research_funding_stress import attach_daily_funding, curve_metrics, normalize_funding_frame


def test_curve_metrics_reports_drawdown_and_positive_return() -> None:
    index = pd.date_range("2025-01-01", periods=366, freq="D", tz="UTC")
    equity = pd.Series([1.0 + index_value / 365 for index_value in range(366)], index=index)
    equity.iloc[200] = equity.iloc[199] * 0.8
    metrics = curve_metrics(equity)
    assert metrics["annual_return"] > 0.9
    assert metrics["max_drawdown"] <= -0.19


def test_funding_cache_accepts_mixed_timestamp_precision() -> None:
    frame = pd.DataFrame({
        "timestamp": ["2020-01-01 00:00:00+00:00", "2020-01-01 08:00:00.009000+00:00"],
        "funding_rate": ["0.0001", "-0.0002"],
    })
    normalized = normalize_funding_frame(frame)
    assert str(normalized["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert normalized["funding_rate"].sum() == -0.0001


def test_attach_daily_funding_maps_rates_without_mutating_input(monkeypatch) -> None:
    candles = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)})
    funding = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"], utc=True),
        "funding_rate": [0.001, 0.002],
    })
    monkeypatch.setattr("scripts.research_funding_stress.fetch_funding", lambda *args: funding)
    source = {"ETH/USDT": candles}
    result = attach_daily_funding(
        source,
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
        2.0,
    )
    assert "funding_rate_daily" not in source["ETH/USDT"]
    assert result["ETH/USDT"]["funding_rate_daily"].tolist() == [0.006, 0.0]
