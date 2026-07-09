# Crypto Spot V1

Native crypto spot strategy research workspace. The active deployment line is
V4.8; older V1/V2/V3/Freqtrade material is legacy context only.

## Current Strategy

- Main strategy: `v4_8_eth_bnb`
- Trading symbols: `ETH/USDT`, `BNB/USDT`
- Reference symbols: `BTC/USDT` for BTC regime features only
- Final selection notes: `results\strategy_review\v47_external_final_selection`
- Development guide: [docs/V4_7_DEVELOPMENT_GUIDE.md](docs/V4_7_DEVELOPMENT_GUIDE.md)
- VPS deployment guide: [docs/V48_VPS_DEPLOYMENT.md](docs/V48_VPS_DEPLOYMENT.md)

Current deployment rationale:

- ETH/BNB showed clear risk-adjusted value versus buy & hold.
- BTC was removed from the traded universe because it behaved mainly as a
  defensive substitute and reduced portfolio focus.
- BTC data remains loaded as a market-regime reference.

## Repository Layout

```text
src/crypto_spot_v1/
  strategy_candidates.py       # small active strategy surface
  strategy_legacy.py           # historical inheritance chain still used by v4_7
  benchmark.py                 # small active strategy registry
  backtest_event_driven.py     # event-driven backtest and execution transform
  v42_*.py                     # V4.2 decision/accounting modules
  v45_lifecycle.py             # lifecycle shadow policy used by V4.7

scripts/
  render_strategy_review_chart.py
  analyze_v42_attribution.py
  audit_execution_transform_trend_contribution.py
  generate_daily_signal.py
  binance_testnet_v48_executor.py
  binance_futures_testnet_v48_executor.py
  review_v48_bootstrap_walkforward.py

docs/
  V4_7_DEVELOPMENT_GUIDE.md
```

## Standard Workflow

Run the current V4.8 backtest:

```powershell
$env:DB_PORT='5433'
python scripts\render_strategy_review_chart.py --strategy v4_8_eth_bnb --symbols ETH/USDT,BNB/USDT --start 2020-01-01 --end 2026-05-18 --output-dir results\strategy_review\v4_8_eth_bnb_full_20200101_20260518
```

Run syntax checks:

```powershell
python -m py_compile src\crypto_spot_v1\strategy_candidates.py src\crypto_spot_v1\benchmark.py src\crypto_spot_v1\backtest_event_driven.py scripts\render_strategy_review_chart.py scripts\analyze_v42_attribution.py scripts\audit_execution_transform_trend_contribution.py scripts\generate_daily_signal.py scripts\binance_testnet_v48_executor.py scripts\binance_futures_testnet_v48_executor.py scripts\review_v48_bootstrap_walkforward.py scripts\search_v47_external_params.py
```

Run the native Binance Spot Testnet executor in dry-run mode:

```powershell
$env:DB_PORT='5433'
$env:BINANCE_TESTNET_API_KEY='your_testnet_key'
$env:BINANCE_TESTNET_API_SECRET='your_testnet_secret'
python scripts\binance_testnet_v48_executor.py --config configs\backtest_v1.json --max-order-usdt 25
```

Run the native Binance USD-M Futures Testnet executor in dry-run mode:

```powershell
$env:DB_PORT='5433'
$env:BINANCE_FUTURES_TESTNET_API_KEY='your_futures_testnet_key'
$env:BINANCE_FUTURES_TESTNET_API_SECRET='your_futures_testnet_secret'
python scripts\binance_futures_testnet_v48_executor.py --config configs\backtest_v1.json --exchange-leverage 3 --target-gross-cap 3.00 --max-order-usdt 25
```

Build and serve the local monitor dashboard:

```powershell
python scripts\build_monitor_dashboard_data.py
python -m http.server 8765 --directory web\monitor
```

Create a new experiment by subclassing the current clean strategy, registering
a new explicit strategy name, and writing to a new output directory. Do not
overwrite the V4.8 deployment review outputs.

## Active Strategy Registry

`benchmark.py` intentionally exposes only the active strategy set:

- `buy_hold`
- `v1`
- `v2_6`
- `v4_2_exp_btc_tc_off_base_exit`
- `v4_2_exp_recovery_overlay_outer_qty2x_v1`
- `v4_6`
- `v4_6_outer_qty_v1`
- `v4_6_outer_qty2x_v1`
- `v4_6_trend2_outer_qty2x_v1`
- `v4_7`
- `v4_8_eth_bnb`

Old experiment classes are isolated in `strategy_legacy.py` because `v4_7`
still inherits through several historical layers. They should not be treated as
active entry points unless they are reintroduced deliberately with a new cleanup
plan.

## Notes

- `results/` is generated output and should not be committed.
- Current V4.8 work is native-research first; legacy Freqtrade shells are not
  the active deployment target for this strategy.
- The active deployment path is the native Binance executors. They preserve the
  V4.8 execution-transform layer and explicit spot cash constraints better than
  the legacy Freqtrade shell.
