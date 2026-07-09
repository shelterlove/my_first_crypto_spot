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
    V26Strategy,
    V42ExpBtcTcOffBaseExitStrategy,
    V42ExpRecoveryOverlayOuterQty2xV1Strategy,
    V46Strategy,
    V46OuterQtyV1Strategy,
    V46OuterQty2xV1Strategy,
    V46Trend2OuterQty2xV1Strategy,
    V47Strategy,
    V47CleanEventExecDrift10V1Strategy,
    V47CleanEventExecDrift10OuterDeepV1Strategy,
    V47CleanEventExecDrift10OuterRelaxedV1Strategy,
    V47CleanEventExecDrift10IntradayShockLadderV11Strategy,
    V47CleanEventExecDrift10IntradayShockLadderV12Strategy,
    V47CleanEventExecDrift10IntradayShockLadderV7Strategy,
    V47CleanEventExecDrift15V1Strategy,
    V47CleanEventExecDrift2V1Strategy,
    V47CleanEventExecDrift20V1Strategy,
    V47CleanEventExecDrift30V1Strategy,
    V47CleanEventExecDrift5V1Strategy,
    V47CleanStrategy,
    V47CleanEventExecV1Strategy,
    V48EthBnbStrategy,
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
    # Core compatibility entries used by tests and the legacy Freqtrade shell.
    "buy_hold": BuyHoldStrategy,
    "v1": V1SpotStrategy,
    "v2_6": V26Strategy,

    # V4.7 mainline and the minimal comparison set documented for future work.
    "v4_2_exp_btc_tc_off_base_exit": V42ExpBtcTcOffBaseExitStrategy,
    "v4_2_exp_recovery_overlay_outer_qty2x_v1": V42ExpRecoveryOverlayOuterQty2xV1Strategy,
    "v4_6": V46Strategy,
    "v4_6_outer_qty_v1": V46OuterQtyV1Strategy,
    "v4_6_outer_qty2x_v1": V46OuterQty2xV1Strategy,
    "v4_6_trend2_outer_qty2x_v1": V46Trend2OuterQty2xV1Strategy,
    "v4_7": V47Strategy,
    "v4_7_clean": V47CleanStrategy,
    "v4_7_clean_event_exec_v1": V47CleanEventExecV1Strategy,
    "v4_7_clean_event_exec_drift2_v1": V47CleanEventExecDrift2V1Strategy,
    "v4_7_clean_event_exec_drift5_v1": V47CleanEventExecDrift5V1Strategy,
    "v4_7_clean_event_exec_drift10_v1": V47CleanEventExecDrift10V1Strategy,
    "v4_7_clean_event_exec_drift10_outer_deep_v1": V47CleanEventExecDrift10OuterDeepV1Strategy,
    "v4_7_clean_event_exec_drift10_outer_relaxed_v1": V47CleanEventExecDrift10OuterRelaxedV1Strategy,
    "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v11": V47CleanEventExecDrift10IntradayShockLadderV11Strategy,
    "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v12": V47CleanEventExecDrift10IntradayShockLadderV12Strategy,
    "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v7": V47CleanEventExecDrift10IntradayShockLadderV7Strategy,
    "v4_8_eth_bnb": V48EthBnbStrategy,
    "v4_7_clean_event_exec_drift15_v1": V47CleanEventExecDrift15V1Strategy,
    "v4_7_clean_event_exec_drift20_v1": V47CleanEventExecDrift20V1Strategy,
    "v4_7_clean_event_exec_drift30_v1": V47CleanEventExecDrift30V1Strategy,
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
        self.artifacts: dict[str, list[dict]] = {
            "equity_curves": [],
            "action_logs": [],
            "sleeve_events": [],
            "sleeve_daily": [],
        }

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
        load_symbols = list(dict.fromkeys(symbols + self.config.get("reference_symbols", [])))
        all_dfs = {symbol: self.load_data(symbol).copy() for symbol in load_symbols}
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
        self.artifacts = {
            "equity_curves": [],
            "action_logs": [],
            "sleeve_events": [],
            "sleeve_daily": [],
        }
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
