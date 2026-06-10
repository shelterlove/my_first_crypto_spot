#!/usr/bin/env python3
"""Screen path-level counterfactual rule variants before promoting candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from crypto_spot_v1.backtest_engine import infer_periods_per_year  # noqa: E402
from crypto_spot_v1.backtest_event_driven import calculate_portfolio_performance, run_rebalance_backtest  # noqa: E402
from crypto_spot_v1.benchmark import build_strategy  # noqa: E402
from crypto_spot_v1.strategy_candidates import V228CStrategy, V231AStrategy  # noqa: E402
from crypto_spot_v1.strategy_rebalance import Action, PositionState  # noqa: E402


PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
SMOKE_WINDOWS = (
    ("strong_bull", "2019-02-25", "2021-02-24"),
    ("post_covid", "2020-03-21", "2021-03-21"),
    ("path_pollution", "2018-06-30", "2021-06-29"),
    ("bear_rally", "2022-08-01", "2022-12-31"),
    ("bear_defence", "2021-12-11", "2022-12-11"),
    ("btc_2023_recovery", "2023-05-01", "2023-08-31"),
    ("eth_2024_recovery", "2024-04-01", "2024-07-31"),
    ("full_dev_tail", "2023-01-01", "2024-12-31"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "freqtrade_user_data" / "data" / "binance"))
    parser.add_argument(
        "--families",
        default="",
        help="Optional comma-separated family filter, for example post_recovery_fade_cap.",
    )
    parser.add_argument(
        "--names",
        default="",
        help="Optional comma-separated strategy-name filter after family filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"path_counterfactual_rule_screen_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_all_candles(Path(args.data_dir))
    specs = build_rule_specs()
    if args.families:
        allowed = {item.strip() for item in args.families.split(",") if item.strip()}
        specs = [spec for spec in specs if spec["family"] in allowed]
    if args.names:
        allowed_names = {item.strip() for item in args.names.split(",") if item.strip()}
        specs = [spec for spec in specs if spec["name"] in allowed_names]
    summary, deltas, actions = run_screen(candles, specs)
    summary.to_csv(output_dir / "screen_summary.csv", index=False)
    deltas.to_csv(output_dir / "screen_deltas.csv", index=False)
    actions.to_csv(output_dir / "screen_actions.csv", index=False)

    passed = summary[
        (summary["negative_return_deltas"] == 0)
        & (summary["worse_drawdown_deltas"] == 0)
        & (summary["return_delta_sum"] > 0)
    ].sort_values("return_delta_sum", ascending=False)

    report = render_report(summary, passed, output_dir)
    (output_dir / "rule_screen_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_dir}")


def build_rule_specs() -> list[dict]:
    specs: list[dict] = []
    for rolling_min in (0.68, 0.72, 0.76):
        for donchian_max in (0.55, 0.60):
            specs.append({
                "family": "safe_recovery_veto",
                "name": f"safe_veto_r{rolling_min:.2f}_d{donchian_max:.2f}",
                "rolling_min": rolling_min,
                "donchian_max": donchian_max,
            })
    for atr_min in (0.85, 0.90, 0.95):
        for require_overheat in (True, False):
            specs.append({
                "family": "high_atr_target_gap_veto",
                "name": f"atr_veto_a{atr_min:.2f}_{'hot' if require_overheat else 'all'}",
                "atr_min": atr_min,
                "require_overheat": require_overheat,
            })
    for current_max in (0.08, 0.10, 0.12):
        for donchian_min in (0.55, 0.70, 0.85):
            specs.append({
                "family": "bnb_bear_probe",
                "name": f"bnb_probe_c{current_max:.2f}_d{donchian_min:.2f}",
                "current_max": current_max,
                "donchian_min": donchian_min,
            })
    for cap in (0.30, 0.35, 0.40):
        for rolling_drop in (0.01, 0.02, 0.03):
            specs.append({
                "family": "post_recovery_fade_cap",
                "name": f"post_fade_cap{cap:.2f}_d{rolling_drop:.2f}",
                "cap": cap,
                "rolling_drop": rolling_drop,
                "symbol": "",
            })
            specs.append({
                "family": "post_recovery_fade_cap",
                "name": f"post_fade_bnb_cap{cap:.2f}_d{rolling_drop:.2f}",
                "cap": cap,
                "rolling_drop": rolling_drop,
                "symbol": "BNB/USDT",
            })
            for btc_slope_min in (0.02, 0.03, 0.05):
                specs.append({
                    "family": "post_recovery_fade_cap",
                    "name": f"post_fade_bnb_btcup{btc_slope_min:.2f}_cap{cap:.2f}_d{rolling_drop:.2f}",
                    "cap": cap,
                    "rolling_drop": rolling_drop,
                    "symbol": "BNB/USDT",
                    "btc_ema168_slope_min": btc_slope_min,
                })
    return specs


def make_strategy(spec: dict, capital: float, reserve: float, fee: float):
    family = spec["family"]
    if family == "safe_recovery_veto":
        return SafeRecoveryVetoStrategy(spec, capital, reserve, fee)
    if family == "high_atr_target_gap_veto":
        return HighAtrTargetGapVetoStrategy(spec, capital, reserve, fee)
    if family == "bnb_bear_probe":
        return BnbBearProbeStrategy(spec, capital, reserve, fee)
    if family == "post_recovery_fade_cap":
        return PostRecoveryFadeCapStrategy(spec, capital, reserve, fee)
    raise ValueError(f"Unknown family: {family}")


class OverlayStateMixin:
    def _track_overlay_state(self, candles_by_symbol, portfolio, current_prices) -> None:
        self._overlay_symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        self._overlay_current_pct = 0.0
        if self._overlay_symbol is None:
            return
        price = current_prices.get(self._overlay_symbol, 0.0)
        pos = portfolio.positions.get(self._overlay_symbol, PositionState())
        position_value = pos.quantity * price if price > 0 else 0.0
        total_value = portfolio.cash + position_value
        if total_value > 0:
            self._overlay_current_pct = position_value / total_value

    @staticmethod
    def _value(latest: pd.Series, column: str, default: float = float("nan")) -> float:
        value = latest.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    @classmethod
    def _ratio(cls, latest: pd.Series, numerator: str, denominator: str) -> float:
        num = cls._value(latest, numerator)
        den = cls._value(latest, denominator)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return num / den - 1.0

    @classmethod
    def _price_vs(cls, latest: pd.Series, price: float, column: str) -> float:
        den = cls._value(latest, column)
        if pd.isna(den) or den <= 0:
            return float("nan")
        return price / den - 1.0


class SafeRecoveryVetoStrategy(OverlayStateMixin, V228CStrategy):
    def __init__(self, spec: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec["name"]

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._track_overlay_state(candles_by_symbol, portfolio, current_prices)
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(self, latest, price, raw_state, buy_setup, max_buy, confirmed_state=None):
        max_buy, guard = super()._adjust_buy_execution(latest, price, raw_state, buy_setup, max_buy, confirmed_state)
        if not self._should_veto(latest, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        return 0.0, self._join_guard(guard, f"{self.name}_veto")

    def _should_veto(self, latest, raw_state, confirmed_state, buy_setup) -> bool:
        if self._overlay_symbol != "BNB/USDT" or buy_setup != "safe-recovery":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        rolling_pos = self._value(latest, "rolling_365d_pos", 0.5)
        donchian_pos = self._value(latest, "donchian_pos", 0.5)
        roc_10 = self._value(latest, "roc_10", 0.0)
        roc_20 = self._value(latest, "roc_20", 0.0)
        volume_strength = self._value(latest, "volume_strength", 1.0)
        return bool(
            rolling_pos >= self.spec["rolling_min"]
            and donchian_pos <= self.spec["donchian_max"]
            and (roc_10 < 0 or roc_20 < 0 or volume_strength < 0.85)
        )


class HighAtrTargetGapVetoStrategy(OverlayStateMixin, V228CStrategy):
    def __init__(self, spec: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec["name"]

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._track_overlay_state(candles_by_symbol, portfolio, current_prices)
        return super().compute_actions(candles_by_symbol, portfolio, current_prices)

    def _adjust_buy_execution(self, latest, price, raw_state, buy_setup, max_buy, confirmed_state=None):
        max_buy, guard = super()._adjust_buy_execution(latest, price, raw_state, buy_setup, max_buy, confirmed_state)
        if not self._should_veto(latest, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        return 0.0, self._join_guard(guard, f"{self.name}_veto")

    def _should_veto(self, latest, raw_state, confirmed_state, buy_setup) -> bool:
        if buy_setup != "target-gap" or raw_state != "BULL" or confirmed_state != "BULL":
            return False
        if self._value(latest, "atr_pct_rank", 0.0) < self.spec["atr_min"]:
            return False
        if not self.spec["require_overheat"]:
            return True
        return bool(
            self._value(latest, "roc_20", 0.0) >= 0.20
            or self._value(latest, "volume_strength", 1.0) >= 1.20
            or self._value(latest, "donchian_pos", 0.5) >= 0.85
        )


class BnbBearProbeStrategy(OverlayStateMixin, V231AStrategy):
    def __init__(self, spec: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec
        self._last_probe_call = -10_000

    @property
    def name(self) -> str:
        return self.spec["name"]

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._track_overlay_state(candles_by_symbol, portfolio, current_prices)
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        if actions and f"{self.name}_probe" in str(getattr(actions[0], "reason", "")):
            self._last_probe_call = self._call_count
        return actions

    def _adjust_buy_execution(self, latest, price, raw_state, buy_setup, max_buy, confirmed_state=None):
        max_buy, guard = V228CStrategy._adjust_buy_execution(self, latest, price, raw_state, buy_setup, max_buy, confirmed_state)
        if not self._should_probe(latest, price, raw_state, confirmed_state, buy_setup):
            return max_buy, guard
        return max(max_buy, 0.05), self._join_guard(guard, f"{self.name}_probe")

    def _should_probe(self, latest, price, raw_state, confirmed_state, buy_setup) -> bool:
        if self._overlay_symbol != "BNB/USDT" or buy_setup != "target-gap":
            return False
        if self._overlay_current_pct >= self.spec["current_max"]:
            return False
        if self._call_count - self._last_probe_call < 30:
            return False
        if str(latest.get("btc_regime", "")) != "BEAR":
            return False
        if raw_state != "MIXED" or confirmed_state != "MIXED":
            return False
        volume_strength = self._value(latest, "volume_strength", 1.0)
        roc_20 = self._value(latest, "roc_20", 0.0)
        donchian_pos = self._value(latest, "donchian_pos", 0.5)
        rolling_pos = self._value(latest, "rolling_365d_pos", 0.5)
        ema72_vs_ema168 = self._ratio(latest, "ema72", "ema168")
        price_vs_ema168 = self._price_vs(latest, price, "ema168")
        return bool(
            volume_strength >= 1.15
            and roc_20 > 0
            and donchian_pos >= self.spec["donchian_min"]
            and rolling_pos < 0.55
            and ema72_vs_ema168 < -0.03
            and price_vs_ema168 >= 0
        )


class PostRecoveryFadeCapStrategy(OverlayStateMixin, V228CStrategy):
    """Temporary external cap after a recovery buy starts to fade."""

    def __init__(self, spec: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec
        self._entry_call = -10_000
        self._entry_rolling_pos = float("nan")

    @property
    def name(self) -> str:
        return self.spec["name"]

    def compute_actions(self, candles_by_symbol, portfolio, current_prices):
        self._track_overlay_state(candles_by_symbol, portfolio, current_prices)
        actions = super().compute_actions(candles_by_symbol, portfolio, current_prices)
        latest = self._latest_row(candles_by_symbol)
        if latest is None:
            return actions

        if actions:
            reason = str(getattr(actions[0], "reason", ""))
            if actions[0].side == "buy" and ("safe-recovery" in reason or "target-gap" in reason):
                self._entry_call = self._call_count
                self._entry_rolling_pos = self._value(latest, "rolling_365d_pos", 0.5)
            if actions[0].side == "sell":
                return actions
            if actions[0].side == "buy" and self._cap_active(latest) and self._overlay_current_pct >= self.spec["cap"]:
                return []
            return actions

        if not self._cap_active(latest):
            return []
        cap = self.spec["cap"]
        if self._overlay_current_pct <= cap + 0.03:
            return []
        price = current_prices.get(self._overlay_symbol, 0.0)
        if price <= 0:
            return []
        pos = portfolio.positions.get(self._overlay_symbol, PositionState())
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        sell_pct = min(self._overlay_current_pct - cap, 0.12)
        sell_qty = min(total_value * sell_pct / price, pos.quantity)
        if sell_qty <= 1e-12:
            return []
        return [
            Action(
                symbol=self._overlay_symbol,
                side="sell",
                quantity=sell_qty,
                price=price,
                reason=(
                    f"{self.name}_sell_external-fade-cap"
                    f"_t{cap:.0%}_drop{self.spec['rolling_drop']:.0%}"
                ),
            )
        ]

    @staticmethod
    def _latest_row(candles_by_symbol) -> pd.Series | None:
        symbol = strategy_utils.resolve_symbol(candles_by_symbol)
        if symbol is None:
            return None
        frame = candles_by_symbol.get(symbol)
        if frame is None or frame.empty:
            return None
        return frame.iloc[-1]

    def _cap_active(self, latest: pd.Series) -> bool:
        if self.spec.get("symbol") and self._overlay_symbol != self.spec["symbol"]:
            return False
        if "btc_ema168_slope_min" in self.spec:
            if self._value(latest, "btc_ema168_slope", 0.0) < self.spec["btc_ema168_slope_min"]:
                return False
        age = self._call_count - self._entry_call
        if age < 5 or age > 30 or pd.isna(self._entry_rolling_pos):
            return False
        rolling_delta = self._value(latest, "rolling_365d_pos", 0.5) - self._entry_rolling_pos
        if rolling_delta > -self.spec["rolling_drop"]:
            return False
        weak_momentum = self._value(latest, "roc_10", 0.0) <= -0.03 or self._value(latest, "roc_20", 0.0) <= -0.05
        weak_structure = self._value(latest, "price_vs_ema72", 0.0) < 0 or self._value(latest, "donchian_pos", 0.5) < 0.45
        return bool(weak_momentum and weak_structure)


def load_all_candles(data_dir: Path) -> dict[str, pd.DataFrame]:
    all_dfs = {pair: load_pair(data_dir, pair) for pair in PAIRS}
    btc = all_dfs["BTC/USDT"].copy()
    btc["btc_regime"] = strategy_utils.compute_btc_regime(btc)
    btc["btc_price_vs_ema72"] = btc["close"] / btc["ema72"] - 1.0
    btc["btc_price_vs_ema168"] = btc["close"] / btc["ema168"] - 1.0
    btc_features = (
        btc.set_index("timestamp")[
            ["btc_regime", "btc_price_vs_ema72", "btc_price_vs_ema168", "ema24_slope", "ema168_slope"]
        ]
        .rename(columns={"ema24_slope": "btc_ema24_slope", "ema168_slope": "btc_ema168_slope"})
        .reset_index()
    )
    for pair, frame in all_dfs.items():
        merged = pd.merge_asof(
            frame.sort_values("timestamp"),
            btc_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged["btc_regime"] = merged["btc_regime"].ffill().fillna("RANGE")
        all_dfs[pair] = merged.reset_index(drop=True)
    return all_dfs


def load_pair(data_dir: Path, pair: str) -> pd.DataFrame:
    path = data_dir / f"{pair.replace('/', '_')}-1d.feather"
    frame = pd.read_feather(path)
    frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.drop(columns=["date"]).sort_values("timestamp").reset_index(drop=True)
    return strategy_utils.compute_indicators(frame)


def run_screen(candles: dict[str, pd.DataFrame], specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capital = 100_000.0
    reserve = 0.2
    fee = 0.001
    ppy = infer_periods_per_year("1d")
    rows: list[dict] = []
    action_rows: list[dict] = []
    strategy_names = ["v2_28C", *[spec["name"] for spec in specs]]

    for window_name, start, end in SMOKE_WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        for pair, frame in candles.items():
            start_matches = frame.index[frame["timestamp"] >= start_ts].tolist()
            end_matches = frame.index[frame["timestamp"] <= end_ts].tolist()
            if not start_matches or not end_matches:
                continue
            eval_start = start_matches[0]
            eval_end = end_matches[-1] + 1
            backtest_start = max(0, eval_start - 1)
            window_df = frame.iloc[eval_start:eval_end].reset_index(drop=True)
            backtest_df = frame.iloc[backtest_start:eval_end].reset_index(drop=True)
            for strategy_name in strategy_names:
                if strategy_name == "v2_28C":
                    strategy = build_strategy(strategy_name, capital, reserve, fee)
                else:
                    spec = next(item for item in specs if item["name"] == strategy_name)
                    strategy = make_strategy(spec, capital, reserve, fee)
                setattr(strategy, "TARGET_ALLOC", {pair: 1.0})
                result = run_rebalance_backtest(
                    {pair: backtest_df},
                    strategy,
                    initial_capital=capital,
                    reserve=reserve,
                    fee_rate=fee,
                    execution_mode="next_open",
                )
                full_actions = result.attrs.get("action_log")
                result = result[result["timestamp"] >= start_ts].reset_index(drop=True)
                actions = pd.DataFrame() if full_actions is None else full_actions
                if not actions.empty:
                    actions = actions[actions["timestamp"] >= start_ts].reset_index(drop=True)
                    for row in actions.to_dict("records"):
                        action_rows.append({"strategy": strategy_name, "window_label": window_name, "symbol": pair, **row})
                perf = calculate_portfolio_performance(
                    result,
                    capital,
                    ppy,
                    candle_df=window_df,
                    fee_rate=fee,
                    benchmark_entry_col="open",
                )
                rows.append({
                    "strategy": strategy_name,
                    "window_label": window_name,
                    "symbol": pair,
                    "total_return": float(perf["total_return"]),
                    "max_drawdown": float(perf["max_drawdown"]),
                    "trade_count": int(len(actions)),
                })

    result_frame = pd.DataFrame(rows)
    deltas = []
    summary_rows = []
    for spec in specs:
        delta = compare(result_frame, "v2_28C", spec["name"])
        delta["family"] = spec["family"]
        delta["candidate"] = spec["name"]
        deltas.append(delta)
        changed = (delta["total_return_delta"].abs() > 1e-12) | (delta["trade_count_delta"].abs() > 1e-12)
        summary_rows.append({
            **spec,
            "windows": int(len(delta)),
            "changed_windows": int(changed.sum()),
            "negative_return_deltas": int((delta["total_return_delta"] < -1e-9).sum()),
            "worse_drawdown_deltas": int((delta["max_drawdown_delta"] < -1e-9).sum()),
            "trade_delta_sum": int(delta["trade_count_delta"].sum()),
            "return_delta_sum": float(delta["total_return_delta"].sum()),
            "min_return_delta": float(delta["total_return_delta"].min()),
            "bnb_return_delta_sum": float(delta.loc[delta["symbol"] == "BNB/USDT", "total_return_delta"].sum()),
        })
    return pd.DataFrame(summary_rows), pd.concat(deltas, ignore_index=True), pd.DataFrame(action_rows)


def compare(frame: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    base = frame[frame["strategy"] == baseline]
    cand = frame[frame["strategy"] == candidate]
    merged = cand.merge(base, on=["window_label", "symbol"], suffixes=("_candidate", "_baseline"))
    merged["total_return_delta"] = merged["total_return_candidate"] - merged["total_return_baseline"]
    merged["max_drawdown_delta"] = merged["max_drawdown_candidate"] - merged["max_drawdown_baseline"]
    merged["trade_count_delta"] = merged["trade_count_candidate"] - merged["trade_count_baseline"]
    return merged


def render_report(summary: pd.DataFrame, passed: pd.DataFrame, output_dir: Path) -> str:
    lines = [
        "# Path Counterfactual Rule Screen",
        "",
        f"- Output: `{output_dir}`",
        f"- Candidates screened: `{len(summary)}`",
        f"- Passed hard gates: `{len(passed)}`",
        "",
        "## Top Candidates",
        "",
    ]
    if passed.empty:
        lines.append("No rule passed the hard gates: no negative return deltas, no worse drawdown deltas, positive return delta.")
    else:
        lines.append(passed.head(20).to_markdown(index=False))
    lines.extend(["", "## Best By Return", ""])
    cols = [
        "family",
        "name",
        "changed_windows",
        "negative_return_deltas",
        "worse_drawdown_deltas",
        "trade_delta_sum",
        "return_delta_sum",
        "min_return_delta",
        "bnb_return_delta_sum",
    ]
    lines.append(summary.sort_values("return_delta_sum", ascending=False)[cols].head(30).to_markdown(index=False))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
