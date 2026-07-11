# Futures V1

Official ETH/BNB long-only strategy workspace for Binance USD-M Futures
Testnet.

The active strategy is `eth_bnb_futures_v1`. It trades `ETH/USDT` and
`BNB/USDT` on daily candles and uses `BTC/USDT` only as a market-regime
reference.

## Repository Layout

```text
configs/backtest_v1.json                 Strategy universe and backtest config
src/futures_v1/                          Strategy, backtest, data, and metrics code
scripts/sync_binance_klines.py           Daily kline sync
scripts/generate_daily_signal.py         Offline daily signal export
scripts/binance_futures_testnet_executor.py
scripts/run_daemon.py                    Scheduled VPS execution loop
scripts/build_monitor_dashboard_data.py  Static monitor data builder
scripts/serve_monitor.py                 Local/static monitor server
web/monitor/                             Monitor UI
docs/VPS_DEPLOYMENT_CN.md                Chinese VPS deployment guide
tests/                                   Minimal regression and safety tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
copy .env.example .env
```

Edit `.env` with PostgreSQL and Binance USD-M Futures Testnet credentials.
Production Binance keys should not be used in this repository.

## Data

```powershell
python scripts\sync_binance_klines.py --symbols BTC/USDT,ETH/USDT,BNB/USDT --timeframe 1d --start 2020-01-01
```

## Full-Window Review

```powershell
python scripts\render_strategy_review_chart.py
```

The deployable baseline excludes OHLC-only intraday ladder fills that are not
implemented by the testnet executor. Regenerate and review the full-window
metrics before creating a release tag.

The review writes `release_manifest.json` with the Git state, config hash,
package versions, raw OHLCV hashes, execution assumptions, and per-symbol
metrics.

## Risk Research

```powershell
python scripts\research_risk_caps.py
python scripts\research_symbol_weights.py
python scripts\research_btc_regime.py
python scripts\research_funding_stress.py
python scripts\research_execution_leverage.py
python scripts\research_realism_robustness.py
python scripts\research_financing_gate.py
```

These scripts write ignored research tables under `results/research/` and do
not change deployment defaults.

Execution leverage experiment parameters, failed approaches, and decisions are
recorded in `docs/EXECUTION_LEVERAGE_RESEARCH_CN.md`.
Path-dependent cost modeling and robustness attribution are recorded in
`docs/REALISM_ROBUSTNESS_RESEARCH_CN.md`.
Financed-exposure gate experiments and their shadow-only decision are recorded
in `docs/FINANCING_GATE_RESEARCH_CN.md`.

Default review:

- Strategy: `eth_bnb_futures_v1`
- Trading symbols: `ETH/USDT`, `BNB/USDT`
- Reference symbol: `BTC/USDT`
- Window: `2020-01-01` to `2026-05-18`
- Output: `results/strategy_review/official_v1_full_20200101_20260518/`

## Daily Signal

```powershell
python scripts\generate_daily_signal.py --strategy eth_bnb_futures_v1 --config configs\backtest_v1.json --output-dir results\daily_signals
```

## Futures Executor

Use [docs/TESTNET_START_CN.md](docs/TESTNET_START_CN.md) for the first
small-capital testnet cycle and its go/no-go checks.

Dry run is the default:

```powershell
python scripts\binance_futures_testnet_executor.py --config configs\backtest_v1.json --exchange-leverage 2 --target-gross-cap 1.25 --hard-symbol-gross-limit 1.50 --hard-account-gross-limit 1.50
```

Execute only after reviewing the dry-run report:

```powershell
python scripts\binance_futures_testnet_executor.py --config configs\backtest_v1.json --exchange-leverage 2 --target-gross-cap 1.25 --hard-symbol-gross-limit 1.50 --hard-account-gross-limit 1.50 --execute
```

The executor defaults to a `1000` USDT deployment cap, `250` USDT per-order
cap, 25% available-margin reserve, and 30% minimum liquidation buffer. It
writes reports to `results/binance_futures_testnet/` and stores the virtual
sleeve ledger in `runtime/futures_state.json`.

## Daemon

```powershell
python scripts\run_daemon.py --run-at-utc 01:10 --run-on-start --execute --exchange-leverage 2 --target-gross-cap 1.25
```

The daemon syncs candles, runs the futures executor, and rebuilds monitor data.

## Monitor

```powershell
python scripts\build_monitor_dashboard_data.py
python scripts\serve_monitor.py
```

Open `http://127.0.0.1:8765/`.

## Tests

```powershell
python -m compileall -q src scripts tests
python -m pytest -q
```

`results/`, `runtime/`, `logs/`, `.env`, and generated monitor JSON are ignored
runtime data.
