"""Shared performance math for the V1 event-driven backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365 * 24


def infer_periods_per_year(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        periods = 365 * 24 * 60 // int(tf.rstrip("m"))
    elif tf.endswith("h"):
        periods = 365 * 24 // int(tf.rstrip("h"))
    elif tf.endswith("d"):
        periods = 365 // int(tf.rstrip("d"))
    elif tf.endswith("w"):
        periods = 52
    else:
        periods = PERIODS_PER_YEAR
    return max(periods, 1)


def calculate_annual_return(
    total_return: float,
    periods: int,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    if periods <= 0:
        return np.nan
    years = periods / periods_per_year
    if years <= 0 or total_return <= -1:
        return np.nan if total_return <= -1 else -1.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_annual_volatility(
    return_series: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    return return_series.std() * np.sqrt(periods_per_year)


def calculate_sharpe(
    return_series: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    std = return_series.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return return_series.mean() / std * np.sqrt(periods_per_year)


def calculate_max_drawdown(return_series: pd.Series) -> float:
    equity = (1 + return_series).cumprod()
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()
