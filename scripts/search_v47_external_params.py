#!/usr/bin/env python3
"""Search V4.7 external execution parameters without registering strategies."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.backtest_event_driven import (  # noqa: E402
    calculate_portfolio_performance,
    run_rebalance_backtest,
)
from crypto_spot_v1.benchmark import STRATEGY_CLASSES, V1BenchmarkRunner, build_strategy  # noqa: E402
from crypto_spot_v1.v47.config import V47Config, V47ExecutionConfig, V47OuterConfig  # noqa: E402
from crypto_spot_v1.v47.execution_engine import V47ExecutionEngine  # noqa: E402


DEFAULT_BASE_STRATEGY = "v4_7_clean_event_exec_drift10_intraday_shock_ladder_v7"
WINDOWS = ("train", "validation", "full")
PROFILE_PARAMS = {
    "deep": {
        "deep_only_entry": True,
    },
    "normal": {},
    "relaxed": {
        "entry_rolling365_pos": 0.25,
        "entry_dd365": -0.52,
        "entry_dd180": -0.35,
        "entry_rebound20": 0.06,
        "entry_roc5": -0.06,
        "entry_roc20": -0.22,
    },
}
TREND_PROFILES = [
    (1.50, 1.50),
    (1.75, 1.75),
    (1.90, 1.90),
    (2.05, 2.00),
]
LOW_RECOVERY_PROFILES = [
    (2.00, 1.70),
    (2.30, 1.85),
    (2.60, 2.00),
    (2.90, 2.15),
]
SHOCK_TIER_PROFILES = {
    "none": [],
    "mild": [(-0.25, 0.15, "add_25"), (-0.30, 0.20, "add_30")],
    "v7": [(-0.25, 0.20, "add_25"), (-0.30, 0.30, "add_30")],
}
WORKER_CONFIG: SearchConfig | None = None
WORKER_RUNNER: V1BenchmarkRunner | None = None
WORKER_DATA: dict[str, pd.DataFrame] | None = None
WORKER_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None


@dataclass(frozen=True)
class SearchConfig:
    mode: str
    base_strategy: str
    symbols: list[str]
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    full_start: str
    full_end: str
    output_dir: str
    outer_recommendation_json: str
    max_candidates: int | None
    workers: int
    drawdown_slack: float = 0.03
    trade_count_max_ratio: float = 1.20
    max_gross_max_ratio: float = 1.05
    min_annual_improvement: float = 0.01
    drawdown_penalty: float = 0.70
    trade_penalty: float = 0.10
    financing_penalty: float = 0.05
    max_gross_penalty: float = 0.15
    generalization_penalty: float = 0.50


@dataclass(frozen=True)
class CandidateSpec:
    mode: str
    symbol: str
    candidate: str
    target_pct: float | None = None
    entry_profile: str | None = None
    min_hold_calls: int | None = None
    hard_stop: float = 0.70
    min_raw: float = 0.05
    trend_mult: float | None = None
    trend_cap: float | None = None
    low_recovery_mult: float | None = None
    low_recovery_cap: float | None = None
    shock_reduce_step: float | None = None
    shock_add_step: float | None = None
    shock_extra_tiers: str | None = None
    shock_max_position: float | None = None
    shock_max_gross: float | None = None


@dataclass
class WindowResult:
    mode: str
    symbol: str
    candidate: str
    target_pct: float | None
    entry_profile: str | None
    min_hold_calls: int | None
    trend_mult: float | None
    trend_cap: float | None
    low_recovery_mult: float | None
    low_recovery_cap: float | None
    shock_reduce_step: float | None
    shock_add_step: float | None
    shock_extra_tiers: str | None
    shock_max_position: float | None
    shock_max_gross: float | None
    window: str
    strategy_total_return: float
    strategy_annual_return: float
    strategy_max_drawdown: float
    buyhold_total_return: float
    buyhold_annual_return: float
    buyhold_max_drawdown: float
    trade_count: int
    execution_transform_financing_cost: float
    execution_transform_max_gross_position: float
    outer_buy_count: int
    outer_sell_count: int
    outer_round_trip_count: int
    train_score: float | None = None
    validation_score: float | None = None
    final_score: float | None = None
    passes_constraints: bool | None = None
    reject_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["outer", "leverage", "shock", "combined"], default="outer")
    parser.add_argument("--base-strategy", default=DEFAULT_BASE_STRATEGY)
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,BNB/USDT")
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2026-05-18")
    parser.add_argument("--full-start", default="2020-01-01")
    parser.add_argument("--full-end", default="2026-05-18")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--outer-recommendation-json",
        default="",
        help="combined_recommendation.json from the prior search stage; used as the next-stage starting point.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode not in {"outer", "leverage", "shock", "combined"}:
        raise SystemExit(f"mode {args.mode} is planned but not implemented yet")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "strategy_review" / f"v47_external_param_search_{timestamp}"
    )
    config = SearchConfig(
        mode=args.mode,
        base_strategy=args.base_strategy,
        symbols=parse_symbols(args.symbols),
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        full_start=args.full_start,
        full_end=args.full_end,
        output_dir=str(output_dir),
        outer_recommendation_json=args.outer_recommendation_json,
        max_candidates=args.max_candidates,
        workers=max(1, int(args.workers)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", output_dir)
    runner.config["symbols"] = list(dict.fromkeys(config.symbols + ([] if "BTC/USDT" in config.symbols else ["BTC/USDT"])))
    data = load_data(runner, runner.config["symbols"])

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    combined: dict[str, Any] = {
        "base_strategy": config.base_strategy,
        "mode": config.mode,
        "recommendations": {},
    }

    windows = {
        "train": (as_utc(config.train_start), as_utc(config.train_end)),
        "validation": (as_utc(config.validation_start), as_utc(config.validation_end)),
        "full": (as_utc(config.full_start), as_utc(config.full_end)),
    }
    outer_seeds = load_prior_seeds(config)
    baselines: dict[tuple[str, str], WindowResult] = {}
    for symbol in config.symbols:
        for window_name, (start_ts, end_ts) in windows.items():
            baseline_strategy = None
            baseline_name = config.base_strategy
            if config.mode in {"leverage", "shock"}:
                baseline_spec = make_baseline_spec(symbol, outer_seeds.get(symbol))
                baseline_cls = make_strategy_class(config.base_strategy, baseline_spec)
                baseline_strategy = baseline_cls(
                    initial_capital=float(runner.config["capital"]["initial"]),
                    reserve=float(runner.config["capital"]["reserve"]),
                    fee_rate=float(runner.config["cost"]["fee_rate"]),
                )
                baseline_name = baseline_spec.candidate
            result, events = run_symbol_window(
                strategy_name=baseline_name,
                strategy=baseline_strategy,
                data=data,
                runner=runner,
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                window_name=window_name,
            )
            baselines[(symbol, window_name)] = result
            baseline_rows.append(window_to_row(result))
            event_rows.extend(events)

    candidates = list(generate_candidates(config, outer_seeds))
    if config.max_candidates is not None:
        candidates = candidates[: max(0, config.max_candidates)]

    write_checkpoint(output_dir, config, baseline_rows, candidate_rows, event_rows)
    completed = 0
    if config.workers == 1:
        for candidate in candidates:
            rows, events = evaluate_candidate(candidate, config, runner, data, windows, baselines)
            candidate_rows.extend(rows)
            event_rows.extend(events)
            completed += 1
            write_checkpoint(output_dir, config, baseline_rows, candidate_rows, event_rows)
            print(f"[{completed}/{len(candidates)}] completed {candidate.candidate}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=init_worker,
            initargs=(config, windows),
        ) as executor:
            futures = {
                executor.submit(evaluate_candidate_worker, candidate, baselines): candidate
                for candidate in candidates
            }
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                rows, events = future.result()
                candidate_rows.extend(rows)
                event_rows.extend(events)
                completed += 1
                write_checkpoint(output_dir, config, baseline_rows, candidate_rows, event_rows)
                print(f"[{completed}/{len(candidates)}] completed {candidate.candidate}", flush=True)

    recommendations = build_recommendations(config, candidate_rows, baselines)
    for rec in recommendations:
        recommendation_rows.append(rec)
        symbol = rec["symbol"]
        if rec["decision"] == "use_candidate":
            combined["recommendations"][symbol] = recommendation_payload(config.mode, rec)
        else:
            combined["recommendations"][symbol] = {"decision": "keep_baseline"}

    write_outputs(
        output_dir=output_dir,
        config=config,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        event_rows=event_rows,
        recommendation_rows=recommendation_rows,
        combined=combined,
    )
    print(f"Wrote V4.7 external parameter search outputs to {output_dir}")


def parse_symbols(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def as_utc(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="UTC")


def load_data(runner: V1BenchmarkRunner, symbols: list[str]) -> dict[str, pd.DataFrame]:
    for symbol in symbols:
        try:
            runner.load_data(symbol)
        except ValueError as exc:
            if "No candles loaded" not in str(exc):
                raise
            runner._data_cache[symbol] = load_binance_vision_daily(symbol)
    return runner._inject_btc_regime()


def load_binance_vision_daily(symbol: str) -> pd.DataFrame:
    binance_symbol = symbol.replace("/", "")
    months = pd.period_range("2019-01", pd.Timestamp.utcnow().to_period("M"), freq="M")
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    rows: list[pd.DataFrame] = []
    for month in months:
        url = (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{binance_symbol}/1d/{binance_symbol}-1d-{month}.zip"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            with zf.open(zf.namelist()[0]) as handle:
                rows.append(pd.read_csv(handle, header=None, names=columns))
    if not rows:
        raise ValueError(f"No Binance Vision candles loaded for {symbol} 1d.")
    out = pd.concat(rows, ignore_index=True)
    open_time = pd.to_numeric(out["open_time"], errors="coerce")
    open_time_ms = open_time.where(open_time <= 1e14, open_time / 1000.0)
    out["timestamp"] = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    out = out[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    out.insert(0, "timeframe", "1d")
    out.insert(0, "symbol", symbol)
    out.insert(0, "exchange", "binance")
    return out.sort_values("timestamp").reset_index(drop=True)


def load_prior_seeds(config: SearchConfig) -> dict[str, dict[str, Any]]:
    if config.mode not in {"leverage", "shock", "combined"} or not config.outer_recommendation_json:
        return {}
    path = Path(config.outer_recommendation_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    recommendations = payload.get("recommendations", {})
    seeds: dict[str, dict[str, Any]] = {}
    for symbol, rec in recommendations.items():
        if rec.get("decision") == "use_candidate":
            seeds[symbol] = {
                "target_pct": rec.get("target_pct"),
                "entry_profile": rec.get("entry_profile"),
                "min_hold_calls": rec.get("min_hold_calls"),
                "trend_mult": rec.get("trend_mult"),
                "trend_cap": rec.get("trend_cap"),
                "low_recovery_mult": rec.get("low_recovery_mult"),
                "low_recovery_cap": rec.get("low_recovery_cap"),
                "shock_reduce_step": rec.get("shock_reduce_step"),
                "shock_add_step": rec.get("shock_add_step"),
                "shock_extra_tiers": rec.get("shock_extra_tiers"),
                "shock_max_position": rec.get("shock_max_position"),
                "shock_max_gross": rec.get("shock_max_gross"),
            }
    return seeds


def make_baseline_spec(symbol: str, outer_seed: dict[str, Any] | None) -> CandidateSpec:
    outer_seed = outer_seed or {}
    return CandidateSpec(
        mode="baseline",
        symbol=symbol,
        candidate=f"baseline_{symbol.replace('/', '').lower()}",
        target_pct=outer_seed.get("target_pct"),
        entry_profile=outer_seed.get("entry_profile"),
        min_hold_calls=outer_seed.get("min_hold_calls"),
        trend_mult=outer_seed.get("trend_mult"),
        trend_cap=outer_seed.get("trend_cap"),
        low_recovery_mult=outer_seed.get("low_recovery_mult"),
        low_recovery_cap=outer_seed.get("low_recovery_cap"),
        shock_reduce_step=outer_seed.get("shock_reduce_step"),
        shock_add_step=outer_seed.get("shock_add_step"),
        shock_extra_tiers=outer_seed.get("shock_extra_tiers"),
        shock_max_position=outer_seed.get("shock_max_position"),
        shock_max_gross=outer_seed.get("shock_max_gross"),
    )


def init_worker(config: SearchConfig, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    global WORKER_CONFIG, WORKER_RUNNER, WORKER_DATA, WORKER_WINDOWS
    WORKER_CONFIG = config
    WORKER_WINDOWS = windows
    WORKER_RUNNER = V1BenchmarkRunner(PROJECT_ROOT / "configs" / "backtest_v1.json", Path(config.output_dir))
    WORKER_RUNNER.config["symbols"] = list(dict.fromkeys(config.symbols + ([] if "BTC/USDT" in config.symbols else ["BTC/USDT"])))
    WORKER_DATA = load_data(WORKER_RUNNER, WORKER_RUNNER.config["symbols"])


def evaluate_candidate_worker(
    candidate: CandidateSpec,
    baselines: dict[tuple[str, str], WindowResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if WORKER_CONFIG is None or WORKER_RUNNER is None or WORKER_DATA is None or WORKER_WINDOWS is None:
        raise RuntimeError("worker was not initialized")
    return evaluate_candidate(candidate, WORKER_CONFIG, WORKER_RUNNER, WORKER_DATA, WORKER_WINDOWS, baselines)


def evaluate_candidate(
    candidate: CandidateSpec,
    config: SearchConfig,
    runner: V1BenchmarkRunner,
    data: dict[str, pd.DataFrame],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    baselines: dict[tuple[str, str], WindowResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cls = make_strategy_class(config.base_strategy, candidate)
    window_results: dict[str, WindowResult] = {}
    candidate_events: list[dict[str, Any]] = []
    for window_name, (start_ts, end_ts) in windows.items():
        strategy = cls(
            initial_capital=float(runner.config["capital"]["initial"]),
            reserve=float(runner.config["capital"]["reserve"]),
            fee_rate=float(runner.config["cost"]["fee_rate"]),
        )
        result, events = run_symbol_window(
            strategy_name=candidate.candidate,
            strategy=strategy,
            data=data,
            runner=runner,
            symbol=candidate.symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            window_name=window_name,
        )
        result.target_pct = candidate.target_pct
        result.entry_profile = candidate.entry_profile
        result.min_hold_calls = candidate.min_hold_calls
        result.trend_mult = candidate.trend_mult
        result.trend_cap = candidate.trend_cap
        result.low_recovery_mult = candidate.low_recovery_mult
        result.low_recovery_cap = candidate.low_recovery_cap
        result.shock_reduce_step = candidate.shock_reduce_step
        result.shock_add_step = candidate.shock_add_step
        result.shock_extra_tiers = candidate.shock_extra_tiers
        result.shock_max_position = candidate.shock_max_position
        result.shock_max_gross = candidate.shock_max_gross
        window_results[window_name] = result
        candidate_events.extend(events)

    train_score = score_window(window_results["train"], baselines[(candidate.symbol, "train")], config)
    validation_score = score_window(
        window_results["validation"],
        baselines[(candidate.symbol, "validation")],
        config,
    )
    final_score = validation_score - config.generalization_penalty * max(0.0, train_score - validation_score)
    passes, reason = evaluate_constraints(
        window_results["validation"],
        baselines[(candidate.symbol, "validation")],
        config,
    )
    if candidate.mode == "outer" and window_results["validation"].outer_buy_count < 1:
        passes = False
        reason = append_reason(reason, "inactive_candidate")
    if train_score - validation_score > 0.05:
        reason = append_reason(reason, "likely_overfit")

    rows: list[dict[str, Any]] = []
    for window_name in WINDOWS:
        row_result = window_results[window_name]
        row_result.train_score = train_score
        row_result.validation_score = validation_score
        row_result.final_score = final_score
        row_result.passes_constraints = passes
        row_result.reject_reason = reason
        rows.append(window_to_row(row_result, baselines[(candidate.symbol, window_name)]))
    return rows, candidate_events


def generate_candidates(config: SearchConfig, outer_seeds: dict[str, dict[str, Any]]):
    if config.mode == "outer":
        yield from generate_outer_candidates(config.symbols)
    elif config.mode == "leverage":
        yield from generate_leverage_candidates(config.symbols, outer_seeds)
    elif config.mode == "shock":
        yield from generate_shock_candidates(config.symbols, outer_seeds)
    elif config.mode == "combined":
        yield from generate_combined_candidates(config.symbols, outer_seeds)
    else:
        raise ValueError(f"mode {config.mode} is planned but not implemented yet")


def generate_outer_candidates(symbols: list[str]):
    for symbol in symbols:
        for target_pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for entry_profile in ["deep", "normal", "relaxed"]:
                for min_hold_calls in [90, 120, 180]:
                    yield CandidateSpec(
                        mode="outer",
                        symbol=symbol,
                        candidate=(
                            f"outer_{symbol.replace('/', '').lower()}"
                            f"_target{int(target_pct * 100):02d}_{entry_profile}_hold{min_hold_calls}"
                        ),
                        target_pct=target_pct,
                        entry_profile=entry_profile,
                        min_hold_calls=min_hold_calls,
                    )


def generate_leverage_candidates(symbols: list[str], outer_seeds: dict[str, dict[str, Any]]):
    for symbol in symbols:
        seed = outer_seeds.get(symbol, {})
        for trend_mult, trend_cap in TREND_PROFILES:
            for low_mult, low_cap in LOW_RECOVERY_PROFILES:
                yield CandidateSpec(
                    mode="leverage",
                    symbol=symbol,
                    candidate=(
                        f"leverage_{symbol.replace('/', '').lower()}"
                        f"_tm{pct_code(trend_mult)}_tc{pct_code(trend_cap)}"
                        f"_lm{pct_code(low_mult)}_lc{pct_code(low_cap)}"
                    ),
                    target_pct=seed.get("target_pct"),
                    entry_profile=seed.get("entry_profile"),
                    min_hold_calls=seed.get("min_hold_calls"),
                    trend_mult=trend_mult,
                    trend_cap=trend_cap,
                    low_recovery_mult=low_mult,
                    low_recovery_cap=low_cap,
                )


def pct_code(value: float) -> str:
    return f"{round(float(value) * 100):03d}"


def generate_shock_candidates(symbols: list[str], prior_seeds: dict[str, dict[str, Any]]):
    for symbol in symbols:
        seed = prior_seeds.get(symbol, {})
        for reduce_step in [0.25, 0.35]:
            for add_step in [0.10, 0.20]:
                for tier_profile in ["none", "mild", "v7"]:
                    for max_gross in [2.40, 2.60]:
                        max_position = 2.30 if max_gross <= 2.40 else 2.70
                        yield CandidateSpec(
                            mode="shock",
                            symbol=symbol,
                            candidate=(
                                f"shock_{symbol.replace('/', '').lower()}"
                                f"_rs{pct_code(reduce_step)}_as{pct_code(add_step)}"
                                f"_{tier_profile}_mp{pct_code(max_position)}_mg{pct_code(max_gross)}"
                            ),
                            target_pct=seed.get("target_pct"),
                            entry_profile=seed.get("entry_profile"),
                            min_hold_calls=seed.get("min_hold_calls"),
                            trend_mult=seed.get("trend_mult"),
                            trend_cap=seed.get("trend_cap"),
                            low_recovery_mult=seed.get("low_recovery_mult"),
                            low_recovery_cap=seed.get("low_recovery_cap"),
                            shock_reduce_step=reduce_step,
                            shock_add_step=add_step,
                            shock_extra_tiers=tier_profile,
                            shock_max_position=max_position,
                            shock_max_gross=max_gross,
                        )


def generate_combined_candidates(symbols: list[str], prior_seeds: dict[str, dict[str, Any]]):
    for symbol in symbols:
        seed = prior_seeds.get(symbol, {})
        yield CandidateSpec(
            mode="combined",
            symbol=symbol,
            candidate=f"combined_{symbol.replace('/', '').lower()}",
            target_pct=seed.get("target_pct"),
            entry_profile=seed.get("entry_profile"),
            min_hold_calls=seed.get("min_hold_calls"),
            trend_mult=seed.get("trend_mult"),
            trend_cap=seed.get("trend_cap"),
            low_recovery_mult=seed.get("low_recovery_mult"),
            low_recovery_cap=seed.get("low_recovery_cap"),
            shock_reduce_step=seed.get("shock_reduce_step"),
            shock_add_step=seed.get("shock_add_step"),
            shock_extra_tiers=seed.get("shock_extra_tiers"),
            shock_max_position=seed.get("shock_max_position"),
            shock_max_gross=seed.get("shock_max_gross"),
        )


def make_strategy_class(base_strategy: str, candidate: CandidateSpec):
    base_cls = STRATEGY_CLASSES.get(base_strategy)
    if base_cls is None:
        raise ValueError(f"Unknown base strategy: {base_strategy}")

    class V47ExternalParamSearchStrategy(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            existing_config = getattr(self, "_v47_config", V47Config())
            base_target = dict(V47OuterConfig().target_pct)
            if candidate.target_pct is not None:
                base_target[candidate.symbol] = float(candidate.target_pct)
            profile_params = PROFILE_PARAMS.get(str(candidate.entry_profile), {})
            outer_kwargs = {
                **profile_params,
                "target_pct": base_target,
                "hard_stop": candidate.hard_stop,
                "min_raw": candidate.min_raw,
            }
            if candidate.min_hold_calls is not None:
                outer_kwargs["min_hold_calls"] = int(candidate.min_hold_calls)
            execution_config = existing_config.execution
            if all(
                value is not None
                for value in [
                    candidate.low_recovery_mult,
                    candidate.low_recovery_cap,
                    candidate.trend_mult,
                    candidate.trend_cap,
                ]
            ):
                execution_config = V47ExecutionConfig(
                    borrow_apr=execution_config.borrow_apr,
                    min_target_gap=execution_config.min_target_gap,
                    min_notional=execution_config.min_notional,
                    maintenance_margin=execution_config.maintenance_margin,
                    warning_gross=execution_config.warning_gross,
                    low_recovery_mult=float(candidate.low_recovery_mult),
                    low_recovery_cap=float(candidate.low_recovery_cap),
                    trend_mult=float(candidate.trend_mult),
                    trend_cap=float(candidate.trend_cap),
                )
            self._v47_config = V47Config(
                execution=execution_config,
                outer=V47OuterConfig(**outer_kwargs),
            )
            self._v47_execution_engine = V47ExecutionEngine(self._v47_config)
            if candidate.mode in {"shock", "combined"} and candidate.shock_reduce_step is not None:
                self.EXECUTION_TRANSFORM_INTRADAY_SHOCK_LADDER_V1 = True
                self.EXECUTION_TRANSFORM_INTRADAY_LADDER_REDUCE_STEP = float(candidate.shock_reduce_step)
                self.EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_STEP = float(candidate.shock_add_step)
                self.EXECUTION_TRANSFORM_INTRADAY_LADDER_EXTRA_ADD_TIERS = list(
                    SHOCK_TIER_PROFILES[str(candidate.shock_extra_tiers)]
                )
                self.EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_POSITION = float(candidate.shock_max_position)
                self.EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_GROSS = float(candidate.shock_max_gross)

        @property
        def name(self) -> str:
            return candidate.candidate

    V47ExternalParamSearchStrategy.__name__ = f"V47Search_{candidate.candidate}"
    return V47ExternalParamSearchStrategy


def run_symbol_window(
    *,
    strategy_name: str,
    strategy,
    data: dict[str, pd.DataFrame],
    runner: V1BenchmarkRunner,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    window_name: str,
) -> tuple[WindowResult, list[dict[str, Any]]]:
    df = data[symbol]
    starts = df.index[df["timestamp"] >= start_ts].tolist()
    ends = df.index[df["timestamp"] <= end_ts].tolist()
    if not starts or not ends:
        raise ValueError(f"No data for {symbol} between {start_ts} and {end_ts}")

    eval_start = starts[0]
    eval_end = ends[-1] + 1
    warmup_bars = int(runner.config.get("warmup_bars", 200))
    execution_mode = runner.config.get("execution", {}).get("mode", "next_open")
    backtest_start = max(0, eval_start - warmup_bars)
    backtest_df = df.iloc[backtest_start:eval_end].reset_index(drop=True)
    eval_df = df.iloc[eval_start:eval_end].reset_index(drop=True)

    capital = float(runner.config["capital"]["initial"])
    reserve = float(runner.config["capital"]["reserve"])
    fee = float(runner.config["cost"]["fee_rate"])
    min_notional = runner.config.get("cost", {}).get("min_notional")
    if strategy is None:
        strategy = build_strategy(strategy_name, capital, reserve, fee, min_notional=min_notional)
    elif min_notional is not None and hasattr(strategy, "min_notional"):
        strategy.min_notional = float(min_notional)
    setattr(strategy, "TARGET_ALLOC", {symbol: 1.0})

    result = run_rebalance_backtest(
        {symbol: backtest_df},
        strategy,
        initial_capital=capital,
        reserve=reserve,
        fee_rate=fee,
        execution_mode=execution_mode,
    )
    result = result[(result["timestamp"] >= start_ts) & (result["timestamp"] <= end_ts)].reset_index(drop=True)
    for key, value in list(result.attrs.items()):
        if isinstance(value, pd.DataFrame) and "timestamp" in value.columns:
            result.attrs[key] = value[(value["timestamp"] >= start_ts) & (value["timestamp"] <= end_ts)].reset_index(drop=True)
    metrics = calculate_portfolio_performance(
        result,
        initial_capital=capital,
        periods_per_year=365,
        candle_df=eval_df,
        fee_rate=fee,
        benchmark_entry_col="open" if execution_mode == "next_open" else "close",
    )
    audit = result.attrs.get("execution_transform_audit", pd.DataFrame())
    outer_events = result.attrs.get("outer_overlay_events", pd.DataFrame())
    financing_cost = float(result["cumulative_financing"].iloc[-1]) if not result.empty else 0.0
    max_gross = safe_max(audit, "gross_position")
    buy_count = count_outer_events(outer_events, "buy")
    sell_count = count_outer_events(outer_events, "sell")
    spec = parse_candidate_name(strategy_name)
    window_result = WindowResult(
        mode=spec.get("mode", "baseline"),
        symbol=symbol,
        candidate=strategy_name,
        target_pct=spec.get("target_pct"),
        entry_profile=spec.get("entry_profile"),
        min_hold_calls=spec.get("min_hold_calls"),
        trend_mult=spec.get("trend_mult"),
        trend_cap=spec.get("trend_cap"),
        low_recovery_mult=spec.get("low_recovery_mult"),
        low_recovery_cap=spec.get("low_recovery_cap"),
        shock_reduce_step=spec.get("shock_reduce_step"),
        shock_add_step=spec.get("shock_add_step"),
        shock_extra_tiers=spec.get("shock_extra_tiers"),
        shock_max_position=spec.get("shock_max_position"),
        shock_max_gross=spec.get("shock_max_gross"),
        window=window_name,
        strategy_total_return=float(metrics.get("total_return", float("nan"))),
        strategy_annual_return=float(metrics.get("annual_return", float("nan"))),
        strategy_max_drawdown=float(metrics.get("max_drawdown", float("nan"))),
        buyhold_total_return=float(metrics.get("bh_total_return", float("nan"))),
        buyhold_annual_return=float(metrics.get("bh_annual_return", float("nan"))),
        buyhold_max_drawdown=float(metrics.get("bh_max_drawdown", float("nan"))),
        trade_count=int(metrics.get("trade_count", 0)),
        execution_transform_financing_cost=financing_cost,
        execution_transform_max_gross_position=max_gross,
        outer_buy_count=buy_count,
        outer_sell_count=sell_count,
        outer_round_trip_count=min(buy_count, sell_count),
    )
    events = dataframe_to_records(outer_events)
    for row in events:
        row.setdefault("mode", window_result.mode)
        row.setdefault("candidate", strategy_name)
        row.setdefault("window", window_name)
    return window_result, events


def parse_candidate_name(name: str) -> dict[str, Any]:
    empty = {
        "mode": "baseline",
        "target_pct": None,
        "entry_profile": None,
        "min_hold_calls": None,
        "trend_mult": None,
        "trend_cap": None,
        "low_recovery_mult": None,
        "low_recovery_cap": None,
        "shock_reduce_step": None,
        "shock_add_step": None,
        "shock_extra_tiers": None,
        "shock_max_position": None,
        "shock_max_gross": None,
    }
    if name.startswith("baseline_"):
        return empty
    parts = name.split("_")
    if name.startswith("combined_"):
        return {**empty, "mode": "combined"}
    if name.startswith("leverage_"):
        values = {}
        for prefix, key in [
            ("tm", "trend_mult"),
            ("tc", "trend_cap"),
            ("lm", "low_recovery_mult"),
            ("lc", "low_recovery_cap"),
        ]:
            part = next((item for item in parts if item.startswith(prefix)), "")
            values[key] = int(part.replace(prefix, "")) / 100 if part else None
        return {
            **empty,
            "mode": "leverage",
            **values,
        }
    if name.startswith("shock_"):
        values = {}
        for prefix, key in [
            ("rs", "shock_reduce_step"),
            ("as", "shock_add_step"),
            ("mp", "shock_max_position"),
            ("mg", "shock_max_gross"),
        ]:
            part = next((item for item in parts if item.startswith(prefix)), "")
            values[key] = int(part.replace(prefix, "")) / 100 if part else None
        tier_profile = next((part for part in parts if part in SHOCK_TIER_PROFILES), None)
        return {
            **empty,
            "mode": "shock",
            "shock_extra_tiers": tier_profile,
            **values,
        }
    if not name.startswith("outer_"):
        return empty
    target = next((part for part in parts if part.startswith("target")), "")
    hold = next((part for part in parts if part.startswith("hold")), "")
    profile = next((part for part in parts if part in PROFILE_PARAMS), None)
    return {
        **empty,
        "mode": "outer",
        "target_pct": int(target.replace("target", "")) / 100 if target else None,
        "entry_profile": profile,
        "min_hold_calls": int(hold.replace("hold", "")) if hold else None,
    }


def safe_max(df: pd.DataFrame, column: str) -> float:
    if df is None or df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    return float(values.max()) if not values.empty else 0.0


def count_outer_events(df: pd.DataFrame, event: str) -> int:
    if df is None or df.empty or "event" not in df.columns:
        return 0
    return int((df["event"].astype(str) == event).sum())


def score_window(result: WindowResult, baseline: WindowResult, config: SearchConfig) -> float:
    annual = finite(result.strategy_annual_return)
    dd_penalty = config.drawdown_penalty * max(
        0.0,
        abs(finite(result.strategy_max_drawdown)) - abs(finite(baseline.strategy_max_drawdown)),
    )
    trade_penalty = config.trade_penalty * max(0.0, ratio(result.trade_count, baseline.trade_count) - 1.0)
    financing_penalty = config.financing_penalty * max(
        0.0,
        ratio(result.execution_transform_financing_cost, baseline.execution_transform_financing_cost) - 1.0,
    )
    gross_penalty = config.max_gross_penalty * max(
        0.0,
        ratio(result.execution_transform_max_gross_position, baseline.execution_transform_max_gross_position) - 1.0,
    )
    return annual - dd_penalty - trade_penalty - financing_penalty - gross_penalty


def evaluate_constraints(result: WindowResult, baseline: WindowResult, config: SearchConfig) -> tuple[bool, str]:
    reasons: list[str] = []
    if abs(result.strategy_max_drawdown) > abs(baseline.strategy_max_drawdown) + config.drawdown_slack:
        reasons.append("validation_drawdown_worse")
    if result.trade_count > baseline.trade_count * config.trade_count_max_ratio:
        reasons.append("validation_trade_count_too_high")
    if result.execution_transform_max_gross_position > baseline.execution_transform_max_gross_position * config.max_gross_max_ratio:
        reasons.append("validation_max_gross_too_high")
    return not reasons, ";".join(reasons)


def ratio(value: float, baseline: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    if baseline <= 1e-12:
        return 1.0 if value <= 1e-12 else 999.0
    return float(value) / float(baseline)


def finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def append_reason(current: str, reason: str) -> str:
    if not current:
        return reason
    if reason in current.split(";"):
        return current
    return f"{current};{reason}"


def window_to_row(result: WindowResult, baseline: WindowResult | None = None) -> dict[str, Any]:
    row = asdict(result)
    if baseline is not None:
        for key, value in asdict(baseline).items():
            if key in {
                "mode",
                "symbol",
                "candidate",
                "target_pct",
                "entry_profile",
                "min_hold_calls",
                "trend_mult",
                "trend_cap",
                "low_recovery_mult",
                "low_recovery_cap",
                "shock_reduce_step",
                "shock_add_step",
                "shock_extra_tiers",
                "shock_max_position",
                "shock_max_gross",
                "window",
            }:
                continue
            row[f"baseline_{key}"] = value
    return row


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype(str)
    return out.to_dict(orient="records")


def build_recommendations(
    config: SearchConfig,
    candidate_rows: list[dict[str, Any]],
    baselines: dict[tuple[str, str], WindowResult],
) -> list[dict[str, Any]]:
    df = pd.DataFrame(candidate_rows)
    if df.empty:
        return []
    validation = df[df["window"] == "validation"].copy()
    rows: list[dict[str, Any]] = []
    for symbol in config.symbols:
        symbol_df = validation[validation["symbol"] == symbol].copy()
        passing = symbol_df[symbol_df["passes_constraints"] == True].copy()  # noqa: E712
        if passing.empty:
            rows.append({
                "symbol": symbol,
                "decision": "keep_baseline",
                "candidate": "",
                "target_pct": None,
                "entry_profile": None,
                "min_hold_calls": None,
                "trend_mult": None,
                "trend_cap": None,
                "low_recovery_mult": None,
                "low_recovery_cap": None,
                "shock_reduce_step": None,
                "shock_add_step": None,
                "shock_extra_tiers": None,
                "shock_max_position": None,
                "shock_max_gross": None,
                "validation_annual_improvement": 0.0,
                "validation_drawdown_change": 0.0,
                "likely_overfit": False,
                "reason": "no_candidate_passed_constraints",
            })
            continue
        top = passing.sort_values("final_score", ascending=False).head(10)
        best = top.iloc[0].to_dict()
        baseline = baselines[(symbol, "validation")]
        annual_improvement = float(best["strategy_annual_return"]) - baseline.strategy_annual_return
        drawdown_change = abs(float(best["strategy_max_drawdown"])) - abs(baseline.strategy_max_drawdown)
        decision = "use_candidate" if annual_improvement >= config.min_annual_improvement else "keep_baseline"
        rows.append({
            "symbol": symbol,
            "decision": decision,
            "candidate": best["candidate"] if decision == "use_candidate" else "",
            "target_pct": best["target_pct"] if decision == "use_candidate" else None,
            "entry_profile": best["entry_profile"] if decision == "use_candidate" else None,
            "min_hold_calls": best["min_hold_calls"] if decision == "use_candidate" else None,
            "trend_mult": best.get("trend_mult") if decision == "use_candidate" else None,
            "trend_cap": best.get("trend_cap") if decision == "use_candidate" else None,
            "low_recovery_mult": best.get("low_recovery_mult") if decision == "use_candidate" else None,
            "low_recovery_cap": best.get("low_recovery_cap") if decision == "use_candidate" else None,
            "shock_reduce_step": best.get("shock_reduce_step") if decision == "use_candidate" else None,
            "shock_add_step": best.get("shock_add_step") if decision == "use_candidate" else None,
            "shock_extra_tiers": best.get("shock_extra_tiers") if decision == "use_candidate" else None,
            "shock_max_position": best.get("shock_max_position") if decision == "use_candidate" else None,
            "shock_max_gross": best.get("shock_max_gross") if decision == "use_candidate" else None,
            "validation_annual_improvement": annual_improvement,
            "validation_drawdown_change": drawdown_change,
            "likely_overfit": "likely_overfit" in str(best.get("reject_reason", "")),
            "reason": "annual_improvement_below_1pct" if decision == "keep_baseline" else "best_passing_candidate",
            "top10_candidates": ",".join(top["candidate"].astype(str).tolist()),
        })
    return rows


def recommendation_payload(mode: str, rec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"decision": "use_candidate"}
    if mode == "outer":
        payload.update({
            "target_pct": rec.get("target_pct"),
            "entry_profile": rec.get("entry_profile"),
            "min_hold_calls": rec.get("min_hold_calls"),
        })
    elif mode == "leverage":
        payload.update({
            "target_pct": rec.get("target_pct"),
            "entry_profile": rec.get("entry_profile"),
            "min_hold_calls": rec.get("min_hold_calls"),
            "trend_mult": rec.get("trend_mult"),
            "trend_cap": rec.get("trend_cap"),
            "low_recovery_mult": rec.get("low_recovery_mult"),
            "low_recovery_cap": rec.get("low_recovery_cap"),
        })
    elif mode == "shock":
        payload.update({
            "target_pct": rec.get("target_pct"),
            "entry_profile": rec.get("entry_profile"),
            "min_hold_calls": rec.get("min_hold_calls"),
            "trend_mult": rec.get("trend_mult"),
            "trend_cap": rec.get("trend_cap"),
            "low_recovery_mult": rec.get("low_recovery_mult"),
            "low_recovery_cap": rec.get("low_recovery_cap"),
            "shock_reduce_step": rec.get("shock_reduce_step"),
            "shock_add_step": rec.get("shock_add_step"),
            "shock_extra_tiers": rec.get("shock_extra_tiers"),
            "shock_max_position": rec.get("shock_max_position"),
            "shock_max_gross": rec.get("shock_max_gross"),
        })
    return payload


def write_outputs(
    *,
    output_dir: Path,
    config: SearchConfig,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    recommendation_rows: list[dict[str, Any]],
    combined: dict[str, Any],
) -> None:
    (output_dir / "search_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame([*baseline_rows, *candidate_rows]).to_csv(output_dir / "candidate_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output_dir / "candidate_outer_events.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(output_dir / "symbol_recommendations.csv", index=False)
    (output_dir / "combined_recommendation.json").write_text(
        json.dumps(clean_json(combined), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        build_readme(config, recommendation_rows),
        encoding="utf-8",
    )


def write_checkpoint(
    output_dir: Path,
    config: SearchConfig,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> None:
    (output_dir / "search_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame([*baseline_rows, *candidate_rows]).to_csv(output_dir / "candidate_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output_dir / "candidate_outer_events.csv", index=False)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if pd.isna(value) if value is not None else False:
        return None
    return value


def build_readme(config: SearchConfig, recommendations: list[dict[str, Any]]) -> str:
    lines = [
        "# V4.7 External Parameter Search Report",
        "",
        f"- Base strategy: `{config.base_strategy}`",
        f"- Search mode: `{config.mode}`",
        f"- Train window: {config.train_start} to {config.train_end}",
        f"- Validation window: {config.validation_start} to {config.validation_end}",
        f"- Full review window: {config.full_start} to {config.full_end}",
        "",
        "## Symbol Recommendations",
        "",
    ]
    for rec in recommendations:
        symbol = rec["symbol"]
        if rec["decision"] == "use_candidate":
            lines.extend([
                f"### {symbol}",
                "",
                f"- Decision: use candidate `{rec['candidate']}`",
                f"- Params: {recommendation_param_text(config.mode, rec)}",
                f"- Validation annual return delta vs baseline: {rec['validation_annual_improvement']:.2%}",
                f"- Validation max drawdown absolute delta: {rec['validation_drawdown_change']:.2%}",
                f"- Likely overfit: {'yes' if rec.get('likely_overfit') else 'no'}",
                "",
            ])
        else:
            lines.extend([
                f"### {symbol}",
                "",
                "- Decision: keep baseline",
                f"- Reason: {rec.get('reason', '')}",
                f"- Validation annual return delta vs baseline: {rec.get('validation_annual_improvement', 0.0):.2%}",
                f"- Likely overfit: {'yes' if rec.get('likely_overfit') else 'no'}",
                "",
            ])
    lines.extend([
        "## Next Step",
        "",
        "- Use `symbol_recommendations.csv` for the ranked decision table.",
        "- Use `combined_recommendation.json` as the input seed for the next search stage.",
        "",
    ])
    return "\n".join(lines)


def recommendation_param_text(mode: str, rec: dict[str, Any]) -> str:
    if mode == "shock":
        return (
            f"shock_reduce_step={rec.get('shock_reduce_step')}, "
            f"shock_add_step={rec.get('shock_add_step')}, "
            f"shock_extra_tiers={rec.get('shock_extra_tiers')}, "
            f"shock_max_position={rec.get('shock_max_position')}, "
            f"shock_max_gross={rec.get('shock_max_gross')}"
        )
    if mode == "leverage":
        return (
            f"trend_mult={rec.get('trend_mult')}, trend_cap={rec.get('trend_cap')}, "
            f"low_recovery_mult={rec.get('low_recovery_mult')}, "
            f"low_recovery_cap={rec.get('low_recovery_cap')}"
        )
    return (
        f"target_pct={rec.get('target_pct')}, entry_profile={rec.get('entry_profile')}, "
        f"min_hold_calls={rec.get('min_hold_calls')}"
    )


if __name__ == "__main__":
    main()

