"""标准化回测指标定义和计算 — 5 维度评估体系。

评估框架：
  [收益]      Total Return, CAGR, Excess Return, Retention Ratio
  [风险]      Max Drawdown, Volatility, Downside Vol, Drawdown Reduction vs BH
  [风险收益比]   Sharpe, Sortino, Calmar, Calmar vs BH
  [稳定性]     Window Win Rate vs BH, Excess Return Consistency, Regime Breakdown
  [行为]      Trade Count, Turnover, Avg Exposure, Fee Impact

策略类型分类:
  - return_enhanced: 收益超过 BH + 回撤不差于 BH
  - risk_controlled: 收益接近 BH + 回撤显著小于 BH
  - balanced:       收益略低 BH + 回撤好于 BH
  - marginal:      单维度有微弱优势
  - no_value:      收益和回撤都不如 BH
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class WindowMetrics:
    """单个滚动窗口的完整回测指标。"""
    # 标识
    strategy_name: str
    symbol: str
    window_label: str
    window_start: str
    window_end: str

    # ── 收益 ──
    total_return: float
    buy_hold_return: float
    excess_return: float
    cagr: float                     # 年化收益率
    bh_cagr: float                  # BH 年化

    # ── 风险 ──
    sharpe: float
    sortino: float                  # 下行风险调整收益
    max_drawdown: float
    calmar: float                   # CAGR / |MaxDD|
    annual_volatility: float
    downside_volatility: float

    # ── 对比 BH ──
    win_vs_bh: bool                 # 本窗口是否跑赢 BH
    excess_return_pct: float        # 超额收益百分比 (超额/BH收益)
    dd_vs_bh: float                 # 回撤对比: strat_mdd - bh_mdd (正=更好)

    # ── 交易活动 ──
    trade_count: int
    avg_action_per_bar: float
    total_fee_cost: float

    # ── 仓位暴露 ──
    avg_exposure: float
    turnover: float
    final_position_pct: float

    # ── 市场拆分 ──
    bull_capture: Optional[float] = None
    bear_protection: Optional[float] = None

    # ── 市场状态标签 ──
    market_regime: Optional[str] = None  # "bull" / "bear" / "sideways"


@dataclass
class StratPerf:
    """一个策略在一个币种×窗口上的汇总表现。"""
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
        if not self.windows:
            return 0.0
        return sum(1 for w in self.windows if w.win_vs_bh) / len(self.windows)

    def mean_max_drawdown(self) -> float:
        return float(np.mean([w.max_drawdown for w in self.windows])) if self.windows else 0.0

    def mean_calmar(self) -> float:
        vals = [w.calmar for w in self.windows if not (math.isnan(w.calmar) or math.isinf(w.calmar))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_sharpe(self) -> float:
        vals = [w.sharpe for w in self.windows if not (math.isnan(w.sharpe) or math.isinf(w.sharpe))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_sortino(self) -> float:
        vals = [w.sortino for w in self.windows if not (math.isnan(w.sortino) or math.isinf(w.sortino))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_trade_count(self) -> float:
        return float(np.mean([w.trade_count for w in self.windows])) if self.windows else 0.0

    def mean_cagr(self) -> float:
        vals = [w.cagr for w in self.windows if not (math.isnan(w.cagr) or math.isinf(w.cagr))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_bh_cagr(self) -> float:
        vals = [w.bh_cagr for w in self.windows if not (math.isnan(w.bh_cagr) or math.isinf(w.bh_cagr))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_exposure(self) -> float:
        return float(np.mean([w.avg_exposure for w in self.windows])) if self.windows else 0.0

    def mean_dd_vs_bh(self) -> float:
        """平均回撤对比: 正值表示回撤小于 BH。"""
        vals = [w.dd_vs_bh for w in self.windows if not (math.isnan(w.dd_vs_bh) or math.isinf(w.dd_vs_bh))]
        return float(np.mean(vals)) if vals else 0.0

    def avg_bull_capture(self) -> Optional[float]:
        vals = [w.bull_capture for w in self.windows if w.bull_capture is not None
                and not (math.isnan(w.bull_capture) or math.isinf(w.bull_capture))]
        return float(np.mean(vals)) if vals else None

    def avg_bear_protection(self) -> Optional[float]:
        vals = [w.bear_protection for w in self.windows if w.bear_protection is not None
                and not (math.isnan(w.bear_protection) or math.isinf(w.bear_protection))]
        return float(np.mean(vals)) if vals else None

    # ── 市场状态拆分 ──

    def _filter_by_regime(self, regime: str) -> list[WindowMetrics]:
        return [w for w in self.windows if w.market_regime == regime]

    def regime_breakdown(self) -> dict[str, dict]:
        """返回 {regime: {count, win_rate, mean_return, mean_excess}}。"""
        regimes = {}
        for regime in ("bull", "bear", "sideways"):
            ws = self._filter_by_regime(regime)
            if not ws:
                continue
            regimes[regime] = {
                "count": len(ws),
                "win_rate": sum(1 for w in ws if w.win_vs_bh) / len(ws),
                "mean_return": float(np.mean([w.total_return for w in ws])),
                "mean_excess": float(np.mean([w.excess_return for w in ws])),
                "mean_dd": float(np.mean([w.max_drawdown for w in ws])),
            }
        return regimes


@dataclass
class StrategySummary:
    """一个策略在所有币种×窗口上的整体汇总。"""
    strategy_name: str
    perfs: list[StratPerf] = field(default_factory=list)

    @property
    def all_windows(self) -> list[WindowMetrics]:
        return [w for p in self.perfs for w in p.windows]

    # ── 收益 ──

    def mean_return(self) -> float:
        return float(np.mean([w.total_return for w in self.all_windows])) if self.all_windows else 0.0

    def median_return(self) -> float:
        return float(np.median([w.total_return for w in self.all_windows])) if self.all_windows else 0.0

    def mean_excess_return(self) -> float:
        return float(np.mean([w.excess_return for w in self.all_windows])) if self.all_windows else 0.0

    def median_excess_return(self) -> float:
        return float(np.median([w.excess_return for w in self.all_windows])) if self.all_windows else 0.0

    def mean_cagr(self) -> float:
        vals = [w.cagr for w in self.all_windows if not (math.isnan(w.cagr) or math.isinf(w.cagr))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_bh_cagr(self) -> float:
        vals = [w.bh_cagr for w in self.all_windows if not (math.isnan(w.bh_cagr) or math.isinf(w.bh_cagr))]
        return float(np.mean(vals)) if vals else 0.0

    # ── 风险 ──

    def mean_max_drawdown(self) -> float:
        return float(np.mean([w.max_drawdown for w in self.all_windows])) if self.all_windows else 0.0

    def mean_annual_volatility(self) -> float:
        vals = [w.annual_volatility for w in self.all_windows
                if not (math.isnan(w.annual_volatility) or math.isinf(w.annual_volatility))]
        return float(np.mean(vals)) if vals else 0.0

    # ── 风险收益比 ──

    def mean_sharpe(self) -> float:
        vals = [w.sharpe for w in self.all_windows
                if not (math.isnan(w.sharpe) or math.isinf(w.sharpe))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_sortino(self) -> float:
        vals = [w.sortino for w in self.all_windows
                if not (math.isnan(w.sortino) or math.isinf(w.sortino))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_calmar(self) -> float:
        vals = [w.calmar for w in self.all_windows
                if not (math.isnan(w.calmar) or math.isinf(w.calmar))]
        return float(np.mean(vals)) if vals else 0.0

    # ── 对比 BH ──

    def win_rate_vs_bh(self) -> float:
        if not self.all_windows:
            return 0.0
        return sum(1 for w in self.all_windows if w.win_vs_bh) / len(self.all_windows)

    def mean_dd_vs_bh(self) -> float:
        """平均回撤对比: 正值 = 回撤小于 BH。"""
        vals = [w.dd_vs_bh for w in self.all_windows
                if not (math.isnan(w.dd_vs_bh) or math.isinf(w.dd_vs_bh))]
        return float(np.mean(vals)) if vals else 0.0

    def retention_ratio(self, bh_summary: Optional["StrategySummary"] = None) -> float:
        """收益保留率: 策略收益 / BH 收益。1.0 = 和 BH 一样。"""
        bh_ret = bh_summary.mean_cagr() if bh_summary else self.mean_bh_cagr()
        strat_ret = self.mean_cagr()
        if bh_ret <= 0 or strat_ret <= 0:
            return 1.0 if strat_ret >= bh_ret else strat_ret / bh_ret if bh_ret < 0 else 1.0
        return min(2.0, strat_ret / bh_ret)

    def drawdown_reduction(self, bh_summary: Optional["StrategySummary"] = None) -> float:
        """回撤缩减: 0 = 和 BH 一样, 0.3 = 比 BH 少 30% 回撤。"""
        bh_dd = bh_summary.mean_max_drawdown() if bh_summary else self._bh_mdd()
        strat_dd = self.mean_max_drawdown()
        if bh_dd >= 0:
            return 0.0
        return (strat_dd - bh_dd) / abs(bh_dd)

    def calmar_vs_bh(self, bh_summary: Optional["StrategySummary"] = None) -> float:
        """Calmar 对比 BH: 1.0 = 和 BH 一样好。"""
        bh_cal = bh_summary.mean_calmar() if bh_summary else self._bh_calmar()
        strat_cal = self.mean_calmar()
        if bh_cal <= 0 and strat_cal > 0:
            return min(3.0, strat_cal / 2.0)
        if bh_cal <= 0:
            return 1.0
        return min(3.0, strat_cal / bh_cal)

    def excess_return_consistency(self) -> float:
        """超额收益一致性: 有多少窗口有正的超额收益。"""
        if not self.all_windows:
            return 0.0
        return sum(1 for w in self.all_windows if w.excess_return > 0) / len(self.all_windows)

    # ── 交易行为 ──

    def mean_trade_count(self) -> float:
        return float(np.mean([w.trade_count for w in self.all_windows])) if self.all_windows else 0.0

    def mean_exposure(self) -> float:
        return float(np.mean([w.avg_exposure for w in self.all_windows])) if self.all_windows else 0.0

    def mean_turnover(self) -> float:
        vals = [w.turnover for w in self.all_windows if not (math.isnan(w.turnover) or math.isinf(w.turnover))]
        return float(np.mean(vals)) if vals else 0.0

    def mean_fee_cost(self) -> float:
        return float(np.mean([w.total_fee_cost for w in self.all_windows])) if self.all_windows else 0.0

    # ── 市场状态拆分 ──

    def regime_breakdown(self) -> dict[str, dict]:
        """返回 {regime: {count, win_rate, mean_return, mean_excess, mean_dd}}。"""
        regimes = {}
        for w in self.all_windows:
            r = w.market_regime or "unknown"
            if r not in regimes:
                regimes[r] = {"count": 0, "wins": 0, "returns": [], "excess": [], "dds": []}
            regimes[r]["count"] += 1
            regimes[r]["wins"] += 1 if w.win_vs_bh else 0
            regimes[r]["returns"].append(w.total_return)
            regimes[r]["excess"].append(w.excess_return)
            regimes[r]["dds"].append(w.max_drawdown)
        return {
            r: {
                "count": d["count"],
                "win_rate": d["wins"] / d["count"],
                "mean_return": float(np.mean(d["returns"])),
                "mean_excess": float(np.mean(d["excess"])),
                "mean_dd": float(np.mean(d["dds"])),
            }
            for r, d in regimes.items()
        }

    # ── 策略类型分类 ──

    STRATEGY_TYPE_LABELS = {
        "return_enhanced": "收益增强型",
        "risk_controlled": "风险控制型",
        "balanced": "均衡型",
        "marginal": "边缘型",
        "no_value": "无明显价值",
    }

    def classify_strategy(self, bh_summary: Optional[StrategySummary] = None) -> str:
        """按 5 类分型判断策略价值。

        比较基准: BH summary (全仓买入持有).
        如果 BH 数据不可靠 (如 BH 有 bug), 用内置保守参数降级判断。
        """
        bh_return = (bh_summary.mean_cagr() if bh_summary else 0) or 0.3
        bh_mdd = (bh_summary.mean_max_drawdown() if bh_summary else 0) or -0.35

        strat_return = self.mean_cagr() or self.mean_return()
        strat_mdd = self.mean_max_drawdown()
        wr = self.win_rate_vs_bh()
        dd_red = self.drawdown_reduction(bh_summary)

        # BH 数据不可靠时的软检测
        bh_seems_broken = bh_return < 0.05 and bh_mdd > -0.05

        if bh_seems_broken:
            # BH 数据异常, 用保守估计
            bh_return = max(bh_return, 0.3)
            bh_mdd = min(bh_mdd, -0.35)
            dd_red = self.mean_dd_vs_bh() / abs(bh_mdd) if bh_mdd < 0 else 0

        retention = strat_return / bh_return if bh_return > 0 else 1.0

        # ── 分型判定 ──
        if retention >= 1.05 and dd_red >= 0:
            return "return_enhanced"

        if retention >= 0.80 and dd_red >= 0.20:
            return "risk_controlled"

        if retention >= 0.70 and dd_red >= 0.10:
            return "balanced"

        if retention >= 0.60 or dd_red >= 0.05:
            return "marginal"

        return "no_value"

    # ── 私有辅助 ──

    def _bh_mdd(self) -> float:
        """估算 BH 的最大回撤 (用于 drawdown_reduction)。"""
        dds = [w.max_drawdown for w in self.all_windows if w.max_drawdown < 0]
        # BH 回撤通常是窗口中最深的
        bh_windows = [w.buy_hold_return for w in self.all_windows]
        if bh_windows:
            worst_bh = min(bh_windows)
            return min(worst_bh, -0.3)  # 至少 -30%
        return -0.35

    def _bh_calmar(self) -> float:
        """估算 BH 的 Calmar。"""
        cal = self.mean_bh_cagr() / abs(self._bh_mdd()) if self._bh_mdd() != 0 else 0.5
        return max(0.2, cal)

    # ── 兼容旧接口 ──

    def total_window_count(self) -> int:
        return len(self.all_windows)

    def per_symbol(self, symbol: str) -> Optional[StratPerf]:
        for p in self.perfs:
            if p.symbol == symbol:
                return p
        return None

    def per_window(self, window_label: str) -> Optional[StratPerf]:
        for p in self.perfs:
            if p.window_label == window_label:
                return p
        return None


# ============================================================
# 指标计算
# ============================================================

def calc_cagr(total_return: float, window_days: int) -> float:
    """从总收益率和窗口天数估算年化。"""
    if window_days <= 0 or total_return <= -1:
        return float("nan")
    years = window_days / 365.0
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return


def calc_sortino(result_df: pd.DataFrame, annual_return: float, periods_per_year: int) -> float:
    """计算 Sortino 比率 (下行波动调整收益)。"""
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
    return (annual_return - 0) / downside_std


def calc_downside_vol(result_df: pd.DataFrame, periods_per_year: int) -> float:
    """计算年化下行波动率。"""
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
    """平均仓位暴露比例。"""
    total_cols = [
        c for c in result_df.columns
        if c.endswith("_value") and c != "total_value"
    ]
    if not total_cols or "total_value" not in result_df.columns:
        return 0.0
    invested = result_df[total_cols].sum(axis=1)
    ratios = invested / result_df["total_value"]
    return float(ratios.mean())


def calc_turnover(result_df: pd.DataFrame, initial_capital: float) -> float:
    """换手率 = 累计单边成交额 / 平均权益。"""
    action_log = result_df.attrs.get("action_log")
    if action_log is not None and not action_log.empty and "notional" in action_log.columns:
        avg_equity = result_df["total_value"].mean() if "total_value" in result_df.columns else initial_capital
        return float(action_log["notional"].sum() / avg_equity) if avg_equity > 0 else 0.0

    action_col = "action_summary"
    if action_col not in result_df.columns:
        return 0.0
    price_cols = [c for c in result_df.columns if c.endswith("_price")]
    qty_cols = [c for c in result_df.columns if c.endswith("_qty")]
    if not price_cols or not qty_cols:
        return 0.0
    total_volume = 0.0
    for p_col, q_col in zip(price_cols, qty_cols):
        qty = result_df[q_col].diff().abs().sum()
        avg_price = result_df[p_col].mean()
        total_volume += qty * avg_price
    avg_equity = result_df["total_value"].mean() if "total_value" in result_df.columns else initial_capital
    return total_volume / avg_equity if avg_equity > 0 else 0.0


def calc_final_position_pct(result_df: pd.DataFrame) -> float:
    """最终仓位比例。"""
    total_cols = [
        c for c in result_df.columns
        if c.endswith("_value") and c != "total_value"
    ]
    if not total_cols or "total_value" not in result_df.columns:
        return 0.0
    invested = result_df[total_cols].sum(axis=1)
    return float(invested.iloc[-1] / result_df["total_value"].iloc[-1]) if result_df["total_value"].iloc[-1] > 0 else 0.0


def classify_market_regime(bh_return: float) -> str:
    """根据 BH 收益分类市场状态。"""
    if bh_return > 0.30:
        return "bull"
    elif bh_return < -0.20:
        return "bear"
    else:
        return "sideways"


# ============================================================
# 综合评分 (5 维度)
# ============================================================

def compute_score(
    summary: StrategySummary,
    bh_summary: StrategySummary | None = None,
) -> float:
    """5 维度综合评分 [0, 1]。

    权重:
      收益 30% — excess return + retention ratio
      风险 30% — drawdown reduction vs BH
      风险收益比 20% — Calmar vs BH
      稳定性 10% — excess return consistency
      行为 10% — trade count reasonableness
    """
    components = compute_score_components(summary, bh_summary)
    score = sum(v["weighted"] for v in components.values())
    return round(score, 4)


def compute_score_components(
    summary: StrategySummary,
    bh_summary: StrategySummary | None = None,
) -> dict[str, dict[str, float]]:
    """Return the exact normalized components used by compute_score."""
    bh_return = (bh_summary.mean_cagr() if bh_summary else 0) or 0.3
    bh_mdd = (bh_summary.mean_max_drawdown() if bh_summary else 0) or -0.35
    bh_calmar = (bh_summary.mean_calmar() if bh_summary else 0) or 0.5
    bh_seems_broken = bh_return < 0.05 and bh_mdd > -0.05
    if bh_seems_broken:
        bh_return = max(bh_return, 0.3)
        bh_mdd = min(bh_mdd, -0.35)
        bh_calmar = max(bh_calmar, 0.5)

    # ── 1. 收益 (30%) ──
    mer = summary.median_excess_return()
    mer = mer if not (math.isnan(mer) or math.isinf(mer)) else 0
    excess_score = 1.0 / (1.0 + math.exp(-3 * mer))

    strat_ret = summary.mean_cagr()
    strat_ret = strat_ret if not (math.isnan(strat_ret) or math.isinf(strat_ret)) else 0
    retention = min(1.0, strat_ret / bh_return) if bh_return > 0 and strat_ret > 0 else max(0, strat_ret / bh_return) if bh_return > 0 else 0.5

    return_score = 0.6 * excess_score + 0.4 * retention

    # ── 2. 风险 (30%) ──
    strat_mdd = summary.mean_max_drawdown()
    strat_mdd = strat_mdd if not math.isnan(strat_mdd) else -0.3
    if bh_mdd < 0:
        dd_improv = (strat_mdd - bh_mdd) / abs(bh_mdd)
        dd_score = max(0.0, min(1.0, dd_improv))
    else:
        dd_score = 0.5

    # ── 3. 风险收益比 (20%) ──
    cal = summary.mean_calmar()
    cal = cal if not (math.isnan(cal) or math.isinf(cal)) else 0
    if bh_calmar > 0:
        calmar_score = min(1.0, max(0.0, cal / bh_calmar))
    elif cal > 0:
        calmar_score = min(1.0, cal / 2.0)
    else:
        calmar_score = 0.0

    # ── 4. 稳定性 (10%) ──
    consistency = summary.excess_return_consistency()

    # ── 5. 行为 (10%) ──
    tc = summary.mean_trade_count()
    tc = tc if not (math.isnan(tc) or math.isinf(tc)) else 0
    if tc <= 10:
        freq_score = 1.0
    elif tc <= 60:
        freq_score = 1.0 - 0.5 * (tc - 10) / 50
    elif tc <= 150:
        freq_score = 0.5 - 0.5 * (tc - 60) / 90
    else:
        freq_score = 0.0
    freq_score = max(0.0, freq_score)

    return {
        "return": {"score": return_score, "weight": 0.30, "weighted": 0.30 * return_score},
        "drawdown": {"score": dd_score, "weight": 0.30, "weighted": 0.30 * dd_score},
        "calmar": {"score": calmar_score, "weight": 0.20, "weighted": 0.20 * calmar_score},
        "consistency": {"score": consistency, "weight": 0.10, "weighted": 0.10 * consistency},
        "frequency": {"score": freq_score, "weight": 0.10, "weighted": 0.10 * freq_score},
    }


# ============================================================
# 窗口指标构建
# ============================================================

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
    """从回测结果构建 WindowMetrics。"""
    bh_ret = perf.get("bh_total_return", 0.0)
    strat_ret = perf.get("total_return", 0.0)

    sharpe = perf.get("sharpe", float("nan"))
    mdd = perf.get("max_drawdown", 0.0)

    # Buy & Hold is the benchmark itself. Use the theoretical BH path for the
    # benchmark row so report columns do not compare executable BH to itself.
    is_buy_hold = strategy_name == "buy_hold"
    if is_buy_hold:
        strat_ret = bh_ret

    # 年化
    ppy = perf.get("periods_per_year", 365)
    n_bars = len(result_df)
    bh_cagr = calc_cagr(bh_ret, n_bars)
    annual_ret = bh_cagr if is_buy_hold else perf.get("annual_return", float("nan"))
    cagr = annual_ret if not (math.isnan(annual_ret) or math.isinf(annual_ret)) else calc_cagr(strat_ret, n_bars)

    # 下行波动 + Sortino
    downside_vol = calc_downside_vol(result_df, ppy)
    sortino = calc_sortino(result_df, cagr if not (math.isnan(annual_ret) or math.isinf(annual_ret)) else annual_ret, ppy)

    # 暴露
    exposure = calc_exposure(result_df)
    turnover = calc_turnover(result_df, initial_capital)
    final_pct = calc_final_position_pct(result_df)

    # BH 对比
    win = strat_ret >= bh_ret if not (math.isnan(strat_ret) or math.isnan(bh_ret)) else False
    dd_vs_bh = mdd - mdd  # 需要 BH mdd
    excess_pct = (strat_ret - bh_ret) / abs(bh_ret) if bh_ret != 0 else 0.0

    # BH mdd
    bh_mdd = perf.get("bh_max_drawdown")
    if bh_mdd is not None and not (math.isnan(bh_mdd) or math.isinf(bh_mdd)):
        dd_vs_bh = mdd - bh_mdd
    else:
        dd_vs_bh = mdd - min(bh_ret, -0.1)  # 保守估计
    if is_buy_hold and bh_mdd is not None and not (math.isnan(bh_mdd) or math.isinf(bh_mdd)):
        mdd = bh_mdd
        dd_vs_bh = 0.0
        sharpe = perf.get("bh_sharpe", sharpe)

    # Calmar depends on the final MDD, so compute it after possible BH override.
    calmar = float("nan")
    if mdd < 0 and not math.isnan(cagr):
        calmar = cagr / abs(mdd)

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
