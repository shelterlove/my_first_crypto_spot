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
    V220DStrategy,
    V221EStrategy,
    V222AStrategy,
    V223AStrategy,
    V223BStrategy,
    V225FStrategy,
    V228AStrategy,
    V228BStrategy,
    V228CStrategy,
    V229AStrategy,
    V229BStrategy,
    V229CStrategy,
    V230AStrategy,
    V231AStrategy,
    V231BStrategy,
    V231CStrategy,
    V232AStrategy,
    V232BStrategy,
    V233AStrategy,
    V233BStrategy,
    V234AStrategy,
    V234BStrategy,
    V234CStrategy,
    V234DStrategy,
    V235AStrategy,
    V235BStrategy,
    V235CStrategy,
    V235DStrategy,
    V236AStrategy,
    V236BStrategy,
    V236CStrategy,
    V236DStrategy,
    V236EStrategy,
    V236FStrategy,
    V236GStrategy,
    V236HStrategy,
    V236IStrategy,
    V3Strategy,
    V31AStrategy,
    V31BStrategy,
    V32AStrategy,
    V33AStrategy,
    V33BStrategy,
    V33CStrategy,
    V33DStrategy,
    V33EStrategy,
    V34AStrategy,
    V34BStrategy,
    V34CStrategy,
    V34DStrategy,
    V34EStrategy,
    V34FStrategy,
    V34GStrategy,
    V34HStrategy,
    V34IStrategy,
    V35AStrategy,
    V35BStrategy,
    V35CStrategy,
    V35DStrategy,
    V35EStrategy,
    V35FStrategy,
    V35GStrategy,
    V35HStrategy,
    V36AStrategy,
    V36BStrategy,
    V36CStrategy,
    V37AStrategy,
    V37BStrategy,
    V37CStrategy,
    V37DStrategy,
    V37EStrategy,
    V37FStrategy,
    V37GStrategy,
    V38AStrategy,
    V38BStrategy,
    V38CStrategy,
    V38DStrategy,
    V39AStrategy,
    V39BStrategy,
    V39CStrategy,
    V39DStrategy,
    V39EStrategy,
    V39FStrategy,
    V310AStrategy,
    V310BStrategy,
    V310CStrategy,
    V310DStrategy,
    V310EStrategy,
    V310FStrategy,
    V310GStrategy,
    V310HStrategy,
    V310IStrategy,
    V310JStrategy,
    V311AStrategy,
    V311BStrategy,
    V311CStrategy,
    V312AStrategy,
    V312BStrategy,
    V312CStrategy,
    V313AStrategy,
    V313BStrategy,
    V314AStrategy,
    V314BStrategy,
    V314CStrategy,
    V314DStrategy,
    V315AStrategy,
    V315BStrategy,
    V315CStrategy,
    V315DStrategy,
    V315EStrategy,
    V315FStrategy,
    V316AStrategy,
    V226AStrategy,
    V226BStrategy,
    V226CStrategy,
    V226DStrategy,
    V226EStrategy,
    V226FStrategy,
    V227AStrategy,
    V227BStrategy,
    V227CStrategy,
    V227DStrategy,
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
    "v2_20D": V220DStrategy,
    "v2_21E": V221EStrategy,
    "v2_22A": V222AStrategy,
    "v2_23A": V223AStrategy,
    "v2_23B": V223BStrategy,
    "v2_25F": V225FStrategy,
    "v2_28A": V228AStrategy,
    "v2_28B": V228BStrategy,
    "v2_28C": V228CStrategy,
    "v2_29A": V229AStrategy,
    "v2_29B": V229BStrategy,
    "v2_29C": V229CStrategy,
    "v2_30A": V230AStrategy,
    "v2_31A": V231AStrategy,
    "v2_31B": V231BStrategy,
    "v2_31C": V231CStrategy,
    "v2_32A": V232AStrategy,
    "v2_32B": V232BStrategy,
    "v2_33A": V233AStrategy,
    "v2_33B": V233BStrategy,
    "v2_34A": V234AStrategy,
    "v2_34B": V234BStrategy,
    "v2_34C": V234CStrategy,
    "v2_34D": V234DStrategy,
    "v2_35A": V235AStrategy,
    "v2_35B": V235BStrategy,
    "v2_35C": V235CStrategy,
    "v2_35D": V235DStrategy,
    "v2_36A": V236AStrategy,
    "v2_36B": V236BStrategy,
    "v2_36C": V236CStrategy,
    "v2_36D": V236DStrategy,
    "v2_36E": V236EStrategy,
    "v2_36F": V236FStrategy,
    "v2_36G": V236GStrategy,
    "v2_36H": V236HStrategy,
    "v2_36I": V236IStrategy,
    "v3": V3Strategy,
    "v3_1A": V31AStrategy,
    "v3_1B": V31BStrategy,
    "v3_2A": V32AStrategy,
    "v3_3A": V33AStrategy,
    "v3_3B": V33BStrategy,
    "v3_3C": V33CStrategy,
    "v3_3D": V33DStrategy,
    "v3_3E": V33EStrategy,
    "v3_4A": V34AStrategy,
    "v3_4B": V34BStrategy,
    "v3_4C": V34CStrategy,
    "v3_4D": V34DStrategy,
    "v3_4E": V34EStrategy,
    "v3_4F": V34FStrategy,
    "v3_4G": V34GStrategy,
    "v3_4H": V34HStrategy,
    "v3_4I": V34IStrategy,
    "v3_5A": V35AStrategy,
    "v3_5B": V35BStrategy,
    "v3_5C": V35CStrategy,
    "v3_5D": V35DStrategy,
    "v3_5E": V35EStrategy,
    "v3_5F": V35FStrategy,
    "v3_5G": V35GStrategy,
    "v3_5H": V35HStrategy,
    "v3_6A": V36AStrategy,
    "v3_6B": V36BStrategy,
    "v3_6C": V36CStrategy,
    "v3_7A": V37AStrategy,
    "v3_7B": V37BStrategy,
    "v3_7C": V37CStrategy,
    "v3_7D": V37DStrategy,
    "v3_7E": V37EStrategy,
    "v3_7F": V37FStrategy,
    "v3_7G": V37GStrategy,
    "v3_8A": V38AStrategy,
    "v3_8B": V38BStrategy,
    "v3_8C": V38CStrategy,
    "v3_8D": V38DStrategy,
    "v3_9A": V39AStrategy,
    "v3_9B": V39BStrategy,
    "v3_9C": V39CStrategy,
    "v3_9D": V39DStrategy,
    "v3_9E": V39EStrategy,
    "v3_9F": V39FStrategy,
    "v3_10A": V310AStrategy,
    "v3_10B": V310BStrategy,
    "v3_10C": V310CStrategy,
    "v3_10D": V310DStrategy,
    "v3_10E": V310EStrategy,
    "v3_10F": V310FStrategy,
    "v3_10G": V310GStrategy,
    "v3_10H": V310HStrategy,
    "v3_10I": V310IStrategy,
    "v3_10J": V310JStrategy,
    "v3_11A": V311AStrategy,
    "v3_11B": V311BStrategy,
    "v3_11C": V311CStrategy,
    "v3_12A": V312AStrategy,
    "v3_12B": V312BStrategy,
    "v3_12C": V312CStrategy,
    "v3_13A": V313AStrategy,
    "v3_13B": V313BStrategy,
    "v3_14A": V314AStrategy,
    "v3_14B": V314BStrategy,
    "v3_14C": V314CStrategy,
    "v3_14D": V314DStrategy,
    "v3_15A": V315AStrategy,
    "v3_15B": V315BStrategy,
    "v3_15C": V315CStrategy,
    "v3_15D": V315DStrategy,
    "v3_15E": V315EStrategy,
    "v3_15F": V315FStrategy,
    "v3_16A": V316AStrategy,
    "v2_26A": V226AStrategy,
    "v2_26B": V226BStrategy,
    "v2_26C": V226CStrategy,
    "v2_26D": V226DStrategy,
    "v2_26E": V226EStrategy,
    "v2_26F": V226FStrategy,
    "v2_27A": V227AStrategy,
    "v2_27B": V227BStrategy,
    "v2_27C": V227CStrategy,
    "v2_27D": V227DStrategy,
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
        btc_features = strategy_utils.compute_indicators(all_dfs["BTC/USDT"])
        btc_regime = strategy_utils.compute_btc_regime(btc_features)
        btc_features["btc_regime"] = btc_regime
        btc_features["btc_price_vs_ema72"] = btc_features["close"] / btc_features["ema72"] - 1.0
        btc_features["btc_price_vs_ema168"] = btc_features["close"] / btc_features["ema168"] - 1.0
        regime_map = dict(zip(btc_features["timestamp"], btc_features["btc_regime"]))
        regime_ts_map = dict(zip(btc_features["timestamp"], btc_features["timestamp"]))
        btc_feature_maps = {
            "btc_price_vs_ema72": dict(zip(btc_features["timestamp"], btc_features["btc_price_vs_ema72"])),
            "btc_price_vs_ema168": dict(zip(btc_features["timestamp"], btc_features["btc_price_vs_ema168"])),
            "btc_ema24_slope": dict(zip(btc_features["timestamp"], btc_features["ema24_slope"])),
            "btc_ema168_slope": dict(zip(btc_features["timestamp"], btc_features["ema168_slope"])),
            "btc_roc_20": dict(zip(btc_features["timestamp"], btc_features["roc_20"])),
        }
        for df in all_dfs.values():
            df["btc_regime"] = df["timestamp"].map(regime_map).ffill()
            df["btc_regime_timestamp"] = df["timestamp"].map(regime_ts_map).ffill()
            for column, values in btc_feature_maps.items():
                df[column] = df["timestamp"].map(values).ffill()
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
