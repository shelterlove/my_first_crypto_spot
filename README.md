# Crypto Spot V1

Native crypto spot strategy research workspace with a thin Freqtrade execution layer.

## Current Recommendation

- Paper-trading candidate: `v3_4I`
- Freqtrade shell: `CryptoSpotV34I`
- Research baseline: `v3`

Recent `v3_5D ~ v3_5H` experiments improved the interpretation of MIXED rebuy
execution, but they did not beat `v3_4I` robustly enough to replace it.

## What This Repo Contains

- native strategy logic under `src/crypto_spot_v1`
- diagnostic and evaluation scripts under `scripts/`
- thin Freqtrade shells under `freqtrade_user_data/strategies/`
- backtest and paper-trading config under `configs/` and `freqtrade_user_data/config/`

## Recommended Workflow

### 1. Native research and diagnostics

```powershell
python scripts\run_window_candidate_test.py --candidate v3_4I --baseline v3 --start 2019-01-01 --end 2024-12-31
python scripts\render_fund_style_report.py --strategy v3_4I --start 2023-05-22 --end 2026-05-22
```

### 2. Freqtrade adapter validation

```powershell
python scripts\check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
```

### 3. Paper trading

```powershell
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --strategy CryptoSpotV34I
```

## Key Files

```text
src/crypto_spot_v1/
  strategy.py
  strategy_candidates.py
  freqtrade_adapter.py
  benchmark.py

scripts/
  run_window_candidate_test.py
  render_fund_style_report.py
  generate_daily_signal.py
  check_freqtrade_adapter.py
  freqtrade_eval.py

freqtrade_user_data/
  config/config.dryrun.example.json
  strategies/CryptoSpotV26.py
  strategies/CryptoSpotV34I.py

docs/
  PAPER_TRADING_READINESS.md
```

## Server Migration and Dry-Run Setup

Use the deployment guide:

- [docs/FREQTRADE_PAPER_TRADING_DEPLOYMENT.md](docs/FREQTRADE_PAPER_TRADING_DEPLOYMENT.md)
- [docs/FREQTRADE_PAPER_TRADING_DEPLOYMENT_CN.md](docs/FREQTRADE_PAPER_TRADING_DEPLOYMENT_CN.md)

## Notes

- Strategy logic should stay in `src/crypto_spot_v1`.
- Freqtrade shells should remain thin adapters.
- `results/` and downloaded market data are generated artifacts, not source of truth.
