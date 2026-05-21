"""Window metrics, summary aggregation, and V1 scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WindowMetrics:
    strategy_name: str
    symbol: str
    window_label: str
    window_start: str
    window_end: str
    total_return: float
    buy_hold_return: float
    excess_return: float
    cagr: float
    bh_cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    annual_volatility: float
    downside_volatility: float
    win_vs_bh: bool
    excess_return_pct: float
    dd_vs_bh: float
    trade_count: int
    avg_action_per_bar: float
    total_fee_cost: float
    avg_exposure: float
    turnover: float
    final_position_pct: float
    bull_capture: float | None = None
    bear_protection: float | None = None
    market_regime: str | None = None


@dataclass
class StratPerf:
    strategy_name: str
    symbol: str
    window_label: str
    windows: list[WindowMetrics] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def _returns(self) -> list[float]:
        return [w.total_return for w in self.windows]

    def _excess(self) -> list[float]:
        return [w.excess_return for w in self.windows]

    def mean_return(self) -> float:
        return float(np.mean(self._returns())) if self.windows else 0.0

    def median_return(self) -> float:
        return float(np.median(self._returns())) if self.windows else 0.0

    def mean_excess_return(self) -> float:
        return float(np.mean(self._excess())) if self.windows else 0.0

    def median_excess_return(self) -> float:
        return float(np.median(self._excess())) if self.windows else 0.0

    def win_rate_vs_bh(self) -> float:
        return sum(w.win_vs_bh for w in self.windows) / len(self.windows) if self.windows else 0.0

    def mean_max_drawdown(self) -> float:
        return float(np.mean([w.max_drawdown for w in self.windows])) if self.windows else 0.0

    def mean_calmar(self) -> float:
        return _finite_mean(w.calmar for w in self.windows)

    def mean_sharpe(self) -> float:
        return _finite_mean(w.sharpe for w in self.windows)

    def mean_sortino(self) -> float:
        return _finite_mean(w.sortino for w in self.windows)

    def mean_trade_count(self) -> float:
        return float(np.mean([w.trade_count for w in self.windows])) if self.windows else 0.0

    def mean_cagr(self) -> float:
        return _finite_mean(w.cagr for w in self.windows)

    def mean_bh_cagr(self) -> float:
        return _finite_mean(w.bh_cagr for w in self.windows)

    def mean_exposure(self) -> float:
        return float(np.mean([w.avg_exposure for w in self.windows])) if self.windows else 0.0

    def mean_dd_vs_bh(self) -> float:
        return _finite_mean(w.dd_vs_bh for w in self.windows)


@dataclass
class StrategySummary:
    strategy_name: str
    perfs: list[StratPerf] = field(default_factory=list)

    @property
    def all_windows(self) -> list[WindowMetrics]:
        return [w for perf in self.perfs for w in perf.windows]

    def mean_return(self) -> float:
        return float(np.mean([w.total_return for w in self.all_windows])) if self.all_windows else 0.0

    def median_return(self) -> float:
        return float(np.median([w.total_return for w in self.all_windows])) if self.all_windows else 0.0

    def mean_excess_return(self) -> float:
        return float(np.mean([w.excess_return for w in self.all_windows])) if self.all_windows else 0.0

    def median_excess_return(self) -> float:
        return float(np.median([w.excess_return for w in self.all_windows])) if self.all_windows else 0.0

    def mean_cagr(self) -> float:
        return _finite_mean(w.cagr for w in self.all_windows)

    def mean_bh_cagr(self) -> float:
        return _finite_mean(w.bh_cagr for w in self.all_windows)

    def mean_max_drawdown(self) -> float:
        return float(np.mean([w.max_drawdown for w in self.all_windows])) if self.all_windows else 0.0

    def mean_annual_volatility(self) -> float:
        return _finite_mean(w.annual_volatility for w in self.all_windows)

    def mean_sharpe(self) -> float:
        return _finite_mean(w.sharpe for w in self.all_windows)

    def mean_sortino(self) -> float:
        return _finite_mean(w.sortino for w in self.all_windows)

    def mean_calmar(self) -> float:
        return _finite_mean(w.calmar for w in self.all_windows)

    def win_rate_vs_bh(self) -> float:
        return sum(w.win_vs_bh for w in self.all_windows) / len(self.all_windows) if self.all_windows else 0.0

    def mean_dd_vs_bh(self) -> float:
        return _finite_mean(w.dd_vs_bh for w in self.all_windows)

    def retention_ratio(self, bh_summary: StrategySummary | None = None) -> float:
        bh_ret = bh_summary.mean_cagr() if bh_summary else self.mean_bh_cagr()
        strat_ret = self.mean_cagr()
        if bh_ret <= 0 or strat_ret <= 0:
            return 1.0 if strat_ret >= bh_ret else strat_ret / bh_ret if bh_ret < 0 else 1.0
        return min(2.0, strat_ret / bh_ret)

    def drawdown_reduction(self, bh_summary: StrategySummary | None = None) -> float:
        bh_dd = bh_summary.mean_max_drawdown() if bh_summary else self._bh_mdd()
        strat_dd = self.mean_max_drawdown()
        return (strat_dd - bh_dd) / abs(bh_dd) if bh_dd < 0 else 0.0

    def calmar_vs_bh(self, bh_summary: StrategySummary | None = None) -> float:
        bh_cal = bh_summary.mean_calmar() if bh_summary else self._bh_calmar()
        strat_cal = self.mean_calmar()
        if bh_cal <= 0 and strat_cal > 0:
            return min(3.0, strat_cal / 2.0)
        if bh_cal <= 0:
            return 1.0
        return min(3.0, strat_cal / bh_cal)

    def excess_return_consistency(self) -> float:
        return sum(w.excess_return > 0 for w in self.all_windows) / len(self.all_windows) if self.all_windows else 0.0

    def mean_trade_count(self) -> float:
        return float(np.mean([w.trade_count for w in self.all_windows])) if self.all_windows else 0.0

    def mean_exposure(self) -> float:
        return float(np.mean([w.avg_exposure for w in self.all_windows])) if self.all_windows else 0.0

    def mean_turnover(self) -> float:
        return _finite_mean(w.turnover for w in self.all_windows)

    def mean_fee_cost(self) -> float:
        return float(np.mean([w.total_fee_cost for w in self.all_windows])) if self.all_windows else 0.0

    def regime_breakdown(self) -> dict[str, dict]:
        regimes: dict[str, dict] = {}
        for window in self.all_windows:
            regime = window.market_regime or "unknown"
            entry = regimes.setdefault(regime, {"count": 0, "wins": 0, "returns": [], "excess": [], "dds": []})
            entry["count"] += 1
            entry["wins"] += int(window.win_vs_bh)
            entry["returns"].append(window.total_return)
            entry["excess"].append(window.excess_return)
            entry["dds"].append(window.max_drawdown)
        return {
            regime: {
                "count": data["count"],
                "win_rate": data["wins"] / data["count"],
                "mean_return": float(np.mean(data["returns"])),
                "mean_excess": float(np.mean(data["excess"])),
                "mean_dd": float(np.mean(data["dds"])),
            }
            for regime, data in regimes.items()
        }

    def classify_strategy(self, bh_summary: StrategySummary | None = None) -> str:
        bh_return = (bh_summary.mean_cagr() if bh_summary else 0) or 0.3
        bh_mdd = (bh_summary.mean_max_drawdown() if bh_summary else 0) or -0.35
        strat_return = self.mean_cagr() or self.mean_return()
        dd_red = self.drawdown_reduction(bh_summary)

        if bh_return < 0.05 and bh_mdd > -0.05:
            bh_return = max(bh_return, 0.3)
            bh_mdd = min(bh_mdd, -0.35)
            dd_red = self.mean_dd_vs_bh() / abs(bh_mdd) if bh_mdd < 0 else 0

        retention = strat_return / bh_return if bh_return > 0 else 1.0
        if retention >= 1.05 and dd_red >= 0:
            return "return_enhanced"
        if retention >= 0.80 and dd_red >= 0.20:
            return "risk_controlled"
        if retention >= 0.70 and dd_red >= 0.10:
            return "balanced"
        if retention >= 0.60 or dd_red >= 0.05:
            return "marginal"
        return "no_value"

    def total_window_count(self) -> int:
        return len(self.all_windows)

    def per_symbol(self, symbol: str) -> StratPerf | None:
        return next((perf for perf in self.perfs if perf.symbol == symbol), None)

    def per_window(self, window_label: str) -> StratPerf | None:
        return next((perf for perf in self.perfs if perf.window_label == window_label), None)

    def _bh_mdd(self) -> float:
        bh_windows = [w.buy_hold_return for w in self.all_windows]
        return min(min(bh_windows), -0.3) if bh_windows else -0.35

    def _bh_calmar(self) -> float:
        bh_mdd = self._bh_mdd()
        cal = self.mean_bh_cagr() / abs(bh_mdd) if bh_mdd != 0 else 0.5
        return max(0.2, cal)


def _finite_mean(values) -> float:
    vals = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    return float(np.mean(vals)) if vals else 0.0


def calc_cagr(total_return: float, window_days: int) -> float:
    if window_days <= 0 or total_return <= -1:
        return float("nan")
    years = window_days / 365.0
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return


def calc_sortino(result_df: pd.DataFrame, annual_return: float, periods_per_year: int) -> float:
    if "total_value" not in result_df.columns or len(result_df) < 5:
        return float("nan")
    daily_ret = result_df["total_value"].pct_change().dropna()
    if len(daily_ret) < 5:
        return float("nan")
    downside = daily_ret[daily_ret < 0]
    if len(downside) < 2:
        return float("inf") if annual_return > 0 else 0.0
    downside_std = downside.std() * math.sqrt(periods_per_year)
    if downside_std == 0:
        return float("inf") if annual_return > 0 else 0.0
    return annual_return / downside_std


def calc_downside_vol(result_df: pd.DataFrame, periods_per_year: int) -> float:
    if "total_value" not in result_df.columns or len(result_df) < 5:
        return float("nan")
    daily_ret = result_df["total_value"].pct_change().dropna()
    if len(daily_ret) < 5:
        return float("nan")
    downside = daily_ret[daily_ret < 0]
    if len(downside) < 2:
        return 0.0
    return downside.std() * math.sqrt(periods_per_year)


def calc_exposure(result_df: pd.DataFrame) -> float:
    value_cols = [c for c in result_df.columns if c.endswith("_value") and c != "total_value"]
    if not value_cols or "total_value" not in result_df.columns:
        return 0.0
    invested = result_df[value_cols].sum(axis=1)
    return float((invested / result_df["total_value"]).mean())


def calc_turnover(result_df: pd.DataFrame, initial_capital: float) -> float:
    action_log = result_df.attrs.get("action_log")
    if action_log is not None and not action_log.empty and "notional" in action_log.columns:
        avg_equity = result_df["total_value"].mean() if "total_value" in result_df.columns else initial_capital
        return float(action_log["notional"].sum() / avg_equity) if avg_equity > 0 else 0.0
    return 0.0


def calc_final_position_pct(result_df: pd.DataFrame) -> float:
    value_cols = [c for c in result_df.columns if c.endswith("_value") and c != "total_value"]
    if not value_cols or "total_value" not in result_df.columns:
        return 0.0
    invested = result_df[value_cols].sum(axis=1)
    final_equity = result_df["total_value"].iloc[-1]
    return float(invested.iloc[-1] / final_equity) if final_equity > 0 else 0.0


def classify_market_regime(bh_return: float) -> str:
    if bh_return > 0.30:
        return "bull"
    if bh_return < -0.20:
        return "bear"
    return "sideways"


def compute_score(summary: StrategySummary, bh_summary: StrategySummary | None = None) -> float:
    components = compute_score_components(summary, bh_summary)
    return round(sum(v["weighted"] for v in components.values()), 4)


def compute_score_components(
    summary: StrategySummary,
    bh_summary: StrategySummary | None = None,
) -> dict[str, dict[str, float]]:
    bh_return = (bh_summary.mean_cagr() if bh_summary else 0) or 0.3
    bh_mdd = (bh_summary.mean_max_drawdown() if bh_summary else 0) or -0.35
    bh_calmar = (bh_summary.mean_calmar() if bh_summary else 0) or 0.5
    if bh_return < 0.05 and bh_mdd > -0.05:
        bh_return = max(bh_return, 0.3)
        bh_mdd = min(bh_mdd, -0.35)
        bh_calmar = max(bh_calmar, 0.5)

    median_excess = summary.median_excess_return()
    median_excess = median_excess if not (math.isnan(median_excess) or math.isinf(median_excess)) else 0
    excess_score = 1.0 / (1.0 + math.exp(-3 * median_excess))

    strat_ret = summary.mean_cagr()
    strat_ret = strat_ret if not (math.isnan(strat_ret) or math.isinf(strat_ret)) else 0
    if bh_return > 0 and strat_ret > 0:
        retention = min(1.0, strat_ret / bh_return)
    elif bh_return > 0:
        retention = max(0, strat_ret / bh_return)
    else:
        retention = 0.5
    return_score = 0.6 * excess_score + 0.4 * retention

    strat_mdd = summary.mean_max_drawdown()
    strat_mdd = strat_mdd if not math.isnan(strat_mdd) else -0.3
    dd_score = max(0.0, min(1.0, (strat_mdd - bh_mdd) / abs(bh_mdd))) if bh_mdd < 0 else 0.5

    calmar = summary.mean_calmar()
    calmar = calmar if not (math.isnan(calmar) or math.isinf(calmar)) else 0
    if bh_calmar > 0:
        calmar_score = min(1.0, max(0.0, calmar / bh_calmar))
    elif calmar > 0:
        calmar_score = min(1.0, calmar / 2.0)
    else:
        calmar_score = 0.0

    trade_count = summary.mean_trade_count()
    trade_count = trade_count if not (math.isnan(trade_count) or math.isinf(trade_count)) else 0
    if trade_count <= 10:
        freq_score = 1.0
    elif trade_count <= 60:
        freq_score = 1.0 - 0.5 * (trade_count - 10) / 50
    elif trade_count <= 150:
        freq_score = 0.5 - 0.5 * (trade_count - 60) / 90
    else:
        freq_score = 0.0
    freq_score = max(0.0, freq_score)

    return {
        "return": {"score": return_score, "weight": 0.30, "weighted": 0.30 * return_score},
        "drawdown": {"score": dd_score, "weight": 0.30, "weighted": 0.30 * dd_score},
        "calmar": {"score": calmar_score, "weight": 0.20, "weighted": 0.20 * calmar_score},
        "consistency": {
            "score": summary.excess_return_consistency(),
            "weight": 0.10,
            "weighted": 0.10 * summary.excess_return_consistency(),
        },
        "frequency": {"score": freq_score, "weight": 0.10, "weighted": 0.10 * freq_score},
    }


def make_window_metrics(
    strategy_name: str,
    symbol: str,
    window_label: str,
    window_start: str,
    window_end: str,
    result_df: pd.DataFrame,
    perf: dict,
    initial_capital: float,
    window_days: int = 365,
) -> WindowMetrics:
    bh_ret = perf.get("bh_total_return", 0.0)
    strat_ret = perf.get("total_return", 0.0)
    sharpe = perf.get("sharpe", float("nan"))
    mdd = perf.get("max_drawdown", 0.0)
    is_buy_hold = strategy_name == "buy_hold"
    if is_buy_hold:
        strat_ret = bh_ret

    periods_per_year = perf.get("periods_per_year", 365)
    n_bars = len(result_df)
    bh_cagr = calc_cagr(bh_ret, n_bars)
    annual_ret = bh_cagr if is_buy_hold else perf.get("annual_return", float("nan"))
    cagr = annual_ret if not (math.isnan(annual_ret) or math.isinf(annual_ret)) else calc_cagr(strat_ret, n_bars)

    downside_vol = calc_downside_vol(result_df, periods_per_year)
    sortino = calc_sortino(result_df, cagr, periods_per_year)
    exposure = calc_exposure(result_df)
    turnover = calc_turnover(result_df, initial_capital)
    final_pct = calc_final_position_pct(result_df)

    win = strat_ret >= bh_ret if not (math.isnan(strat_ret) or math.isnan(bh_ret)) else False
    excess_pct = (strat_ret - bh_ret) / abs(bh_ret) if bh_ret != 0 else 0.0

    bh_mdd = perf.get("bh_max_drawdown")
    if bh_mdd is not None and not (math.isnan(bh_mdd) or math.isinf(bh_mdd)):
        dd_vs_bh = mdd - bh_mdd
    else:
        dd_vs_bh = mdd - min(bh_ret, -0.1)
    if is_buy_hold and bh_mdd is not None and not (math.isnan(bh_mdd) or math.isinf(bh_mdd)):
        mdd = bh_mdd
        dd_vs_bh = 0.0
        sharpe = perf.get("bh_sharpe", sharpe)

    calmar = cagr / abs(mdd) if mdd < 0 and not math.isnan(cagr) else float("nan")

    return WindowMetrics(
        strategy_name=strategy_name,
        symbol=symbol,
        window_label=window_label,
        window_start=window_start,
        window_end=window_end,
        total_return=strat_ret,
        buy_hold_return=bh_ret,
        excess_return=0.0 if is_buy_hold else strat_ret - bh_ret,
        cagr=cagr if not math.isnan(cagr) else 0.0,
        bh_cagr=bh_cagr if not math.isnan(bh_cagr) else 0.0,
        sharpe=sharpe,
        sortino=sortino if not (math.isnan(sortino) or math.isinf(sortino)) else 0.0,
        max_drawdown=mdd,
        calmar=calmar if not (math.isnan(calmar) or math.isinf(calmar)) else 0.0,
        annual_volatility=perf.get("annual_volatility", 0.0),
        downside_volatility=downside_vol if not (math.isnan(downside_vol) or math.isinf(downside_vol)) else 0.0,
        win_vs_bh=False if is_buy_hold else win,
        excess_return_pct=0.0 if is_buy_hold else excess_pct,
        dd_vs_bh=dd_vs_bh,
        trade_count=perf.get("trade_count", 0),
        avg_action_per_bar=perf.get("avg_action_per_bar", 0.0),
        total_fee_cost=perf.get("total_fee_cost", 0.0),
        avg_exposure=exposure,
        turnover=turnover,
        final_position_pct=final_pct,
        bull_capture=perf.get("bull_capture_ratio"),
        bear_protection=perf.get("bear_protection"),
        market_regime=classify_market_regime(bh_ret),
    )
