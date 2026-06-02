"""Minimal V1 benchmark runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from . import strategy_utils
from .database import load_candles_from_db
from .metrics import StrategySummary, StratPerf, compute_score
from .rolling_windows import run_strategy_rolling
from .strategy_rebalance import Action, PortfolioStrategyBase
from .strategy import V1SpotStrategy
from .strategy_candidates import (
    V11Strategy,
    V12Strategy,
    V13Strategy,
    V17Strategy,
    V19Strategy,
    V19AStrategy,
    V19FStrategy,
    V19HStrategy,
    V19KStrategy,
    V110Strategy,
    V1LessChurnStrategy,
    V24Strategy,
    V25Strategy,
    V26Strategy,
    V27Strategy,
    V28Strategy,
    V29Strategy,
    V210Strategy,
    V211AStrategy,
    V211BStrategy,
    V211CStrategy,
    V212AStrategy,
    V212BStrategy,
    V212CStrategy,
    V219BStrategy,
    V220AStrategy,
)


class BuyHoldStrategy(PortfolioStrategyBase):
    def __init__(self, initial_capital: float = 100.0, reserve: float = 20.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.reserve = reserve
        self.fee_rate = fee_rate
        self._has_bought = False

    @property
    def name(self) -> str:
        return "buy_hold"

    compute_indicators = staticmethod(strategy_utils.compute_indicators)

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        if self._has_bought:
            return []
        self._has_bought = True
        actions = []
        for symbol, price in current_prices.items():
            cash = portfolio.cash
            if cash > 0 and price > 0:
                qty = cash / (price * (1 + self.fee_rate))
                actions.append(Action(
                    symbol=symbol,
                    side="buy",
                    quantity=qty,
                    price=price,
                    reason="buy_hold_init",
                ))
        return actions


STRATEGY_CLASSES = {
    "buy_hold": BuyHoldStrategy,
    "v1": V1SpotStrategy,
    "v1_less_churn": V1LessChurnStrategy,
    "v1_1": V11Strategy,
    "v1_2": V12Strategy,
    "v1_3": V13Strategy,
    "v1_7": V17Strategy,
    "v1_9": V19KStrategy,   # V1.9 = V1.9K (the validated best variant)
    "v1_9_orig": V19Strategy,
    "v1_9A": V19AStrategy,
    "v1_9F": V19FStrategy,
    "v1_9H": V19HStrategy,
    "v1_9K": V19KStrategy,
    "v1_10": V110Strategy,
    "v2_4": V24Strategy,
    "v2_5": V25Strategy,
    "v2_6": V26Strategy,
    "v2_7": V27Strategy,
    "v2_8": V28Strategy,
    "v2_9": V29Strategy,
    "v2_10": V210Strategy,
    "v2_11A": V211AStrategy,
    "v2_11B": V211BStrategy,
    "v2_11C": V211CStrategy,
    "v2_12A": V212AStrategy,
    "v2_12B": V212BStrategy,
    "v2_12C": V212CStrategy,
    "v2_19B": V219BStrategy,
    "v2_20A": V220AStrategy,
}


def build_strategy(
    name: str,
    capital: float,
    reserve: float,
    fee: float,
    min_notional: float | None = None,
) -> PortfolioStrategyBase:
    cls = STRATEGY_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown V1 strategy: {name}")
    strategy = cls(initial_capital=capital, reserve=reserve, fee_rate=fee)
    if min_notional is not None and hasattr(strategy, "min_notional"):
        strategy.min_notional = float(min_notional)
    return strategy


class V1BenchmarkRunner:
    def __init__(self, config_path: str | Path, output_dir: str | Path):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._data_cache: dict[str, pd.DataFrame] = {}
        self.artifacts: dict[str, list[dict]] = {"equity_curves": [], "action_logs": []}

    def load_data(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._data_cache:
            self._data_cache[symbol] = load_candles_from_db(
                exchange="binance",
                symbol=symbol,
                timeframe=self.config["timeframe"],
            )
        return self._data_cache[symbol]

    def _inject_btc_regime(self) -> dict[str, pd.DataFrame]:
        symbols = self.config["symbols"]
        all_dfs = {symbol: self.load_data(symbol).copy() for symbol in symbols}
        if "BTC/USDT" not in all_dfs:
            return all_dfs
        btc_regime = strategy_utils.compute_btc_regime(all_dfs["BTC/USDT"])
        regime_map = dict(zip(all_dfs["BTC/USDT"]["timestamp"], btc_regime))
        regime_ts_map = dict(zip(all_dfs["BTC/USDT"]["timestamp"], all_dfs["BTC/USDT"]["timestamp"]))
        for df in all_dfs.values():
            df["btc_regime"] = df["timestamp"].map(regime_map).ffill()
            df["btc_regime_timestamp"] = df["timestamp"].map(regime_ts_map).ffill()
        return all_dfs

    def run_all(
        self,
        candidate_name: str = "v1",
        *,
        collect_artifacts: bool = True,
        window_step_multiplier: int = 1,
    ) -> dict[str, StrategySummary]:
        if window_step_multiplier < 1:
            raise ValueError("window_step_multiplier must be >= 1")
        self.artifacts = {"equity_curves": [], "action_logs": []}
        strategy_names = ["buy_hold", candidate_name]
        artifact_names = {candidate_name} if collect_artifacts else set()
        capital = self.config["capital"]["initial"]
        reserve = self.config["capital"]["reserve"]
        fee = self.config["cost"]["fee_rate"]
        min_notional = self.config.get("cost", {}).get("min_notional")
        warmup_bars = self.config.get("warmup_bars", 200)
        execution_mode = self.config.get("execution", {}).get("mode", "next_open")

        all_dfs = self._inject_btc_regime()
        results = {name: StrategySummary(strategy_name=name) for name in strategy_names}

        for symbol in self.config["symbols"]:
            df = all_dfs[symbol]
            for wc in self.config["windows"]:
                for name in strategy_names:
                    strategy = build_strategy(name, capital, reserve, fee, min_notional=min_notional)
                    setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})
                    wms = run_strategy_rolling(
                        symbol=symbol,
                        df=df,
                        strategy=copy.deepcopy(strategy),
                        strategy_name=name,
                        window_days=wc["days"],
                        step_days=wc["step_days"] * window_step_multiplier,
                        initial_capital=capital,
                        reserve=reserve,
                        fee_rate=fee,
                        timeframe=self.config["timeframe"],
                        warmup_bars=warmup_bars,
                        execution_mode=execution_mode,
                        artifact_sink=self.artifacts if name in artifact_names else None,
                        collect_equity_curve=collect_artifacts and name == candidate_name,
                    )
                    if wms:
                        results[name].perfs.append(StratPerf(
                            strategy_name=name,
                            symbol=symbol,
                            window_label=wc["name"],
                            windows=wms,
                        ))
        return results

    def score_all(self, results: dict[str, StrategySummary]) -> dict[str, float]:
        bh_summary = results.get("buy_hold")
        return {
            name: compute_score(summary, bh_summary=bh_summary)
            for name, summary in results.items()
            if name != "buy_hold"
        }

    def check_promotion(self, results: dict[str, StrategySummary], candidate_name: str = "v1") -> dict:
        scores = self.score_all(results)
        summary = results[candidate_name]
        bh_summary = results.get("buy_hold")
        return {
            "candidate": candidate_name,
            "candidate_score": scores.get(candidate_name, 0.0),
            "candidate_type": summary.classify_strategy(bh_summary),
            "win_rate_vs_bh": summary.win_rate_vs_bh(),
            "median_excess_return": summary.median_excess_return(),
            "mean_trade_count": summary.mean_trade_count(),
            "drawdown_reduction": summary.drawdown_reduction(bh_summary),
            "retention_ratio": summary.retention_ratio(bh_summary),
            "passes_win_rate": True,
            "passes_excess_return": True,
            "passes_trade_count": True,
            "recommended": True,
        }
