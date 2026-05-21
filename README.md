# Crypto Spot V1

Clean V1 project for a long-only crypto spot strategy and its repaired
rolling-window backtest workflow.

V1 scope:

- spot only
- long only
- no leverage
- no shorting
- BTC/USDT, ETH/USDT, BNB/USDT
- 1d timeframe
- repaired backtest methodology: `next_open`, warmup excluded, fee-adjusted Buy & Hold

The accepted strategy is implemented as a single clean `V1SpotStrategy` in
`src/crypto_spot_v1/strategy.py`. The runner registers only `buy_hold` and
`v1`; it does not use any legacy registry or historical version runner.

## Commands

Run a V1 baseline:

```powershell
python scripts\run_v1_backtest.py
```

Run smoke tests:

```powershell
python tests\test_v1_smoke.py
```

Compile-check the package:

```powershell
Get-ChildItem src\crypto_spot_v1\*.py, scripts\run_v1_backtest.py, tests\test_v1_smoke.py | ForEach-Object { python -m py_compile $_.FullName }
```

## Baseline Reference

The repaired V1 reference is:

- latest reproduced run: `results/v1/20260521_224736`
- score: `0.6277`
- mean return: `160.37%`
- mean excess: `9.64%`
- median excess: `0.25%`
- win vs BH: `50.30%`
- mean max drawdown: `-35.71%`
- trades: `37.82`
- exposure: `61.22%`
- turnover: `5.85`

See `docs/V1_MIGRATION_NOTES.md` for migration and verification notes.
