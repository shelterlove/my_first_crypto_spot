# Crypto Spot V1

Long-only crypto spot strategy research and Freqtrade deployment workspace.

The current strategy is `v2_19B`, exposed to Freqtrade as `CryptoSpotV219B`.
The design goal is simple: use a rule-based EMA regime engine to keep long-term
uptrend participation, reduce exposure in weak regimes, and avoid ML-style curve
fitting.

## Current Workflow

Run the Freqtrade-aligned evaluator:

```powershell
python scripts\freqtrade_eval.py --strategy CryptoSpotV219B --timerange 20200101-20260601 --report-window 20251202-20260601 --run-id baseline_v2_19B
```

Run rolling evaluation presets:

```powershell
python scripts\freqtrade_eval.py --strategy CryptoSpotV219B --timerange 20200101-20260601 --rolling-preset standard --run-id rolling_v2_19B
```

The evaluator reports:

- per-pair fixed-allocation results for `BTC/USDT`, `ETH/USDT`, and `BNB/USDT`;
- a synthetic fixed-allocation aggregate portfolio;
- optional rolling windows with both per-pair and aggregate detail.

Shared-wallet portfolio tests are intentionally excluded from this evaluator.
The strategy is reviewed as three independent fixed capital sleeves plus their
equal-weight aggregate.

## Useful Commands

```powershell
python -m py_compile scripts\freqtrade_eval.py scripts\generate_daily_signal.py scripts\check_freqtrade_adapter.py
python scripts\check_freqtrade_adapter.py
python scripts\generate_daily_signal.py --strategy v2_19B
```

Native research backtests are still available for strategy prototyping:

```powershell
python scripts\run_v1_4_remaining.py
```

## Repository Layout

```text
configs/
  backtest_v1.json

freqtrade_user_data/
  config/config.dryrun.example.json
  strategies/CryptoSpotV26.py
  strategies/CryptoSpotV219B.py

scripts/
  freqtrade_eval.py
  check_freqtrade_adapter.py
  generate_daily_signal.py
  run_v1_4_remaining.py
  run_v1_backtest.py

src/crypto_spot_v1/
  strategy.py
  strategy_candidates.py
  strategy_utils.py
  freqtrade_adapter.py
  benchmark.py
  evaluation.py
  metrics.py
```

## Cleanup Policy

Generated outputs are not source of truth and should not be committed:

- `results/`
- `freqtrade_user_data/backtest_results/`
- Python `__pycache__/`

Keep downloaded market data under `freqtrade_user_data/data/` when it is needed
for repeatable local Freqtrade backtests.

## Development Rules

- Keep strategy logic in `src/crypto_spot_v1`.
- Keep Freqtrade strategy files thin; they should adapt lifecycle and execution,
  not duplicate strategy rules.
- Evaluate new ideas first with the lightweight Freqtrade evaluator.
- Use rolling windows only for candidates that pass basic screening.
- Prefer fewer metrics that affect decisions: return, excess return, win rate,
  max drawdown, exposure, trade count, and rolling stability.
