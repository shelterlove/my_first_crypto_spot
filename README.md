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

Dry run is the default:

```powershell
python scripts\binance_futures_testnet_executor.py --config configs\backtest_v1.json --exchange-leverage 3 --target-gross-cap 3.00
```

Execute only after reviewing the dry-run report:

```powershell
python scripts\binance_futures_testnet_executor.py --config configs\backtest_v1.json --exchange-leverage 3 --target-gross-cap 3.00 --execute
```

`--max-order-usdt` defaults to `0`, meaning no per-order cap. The executor
writes reports to `results/binance_futures_testnet/` and stores the virtual
sleeve ledger in `runtime/futures_state.json`.

## Daemon

```powershell
python scripts\run_daemon.py --run-at-utc 01:10 --run-on-start --execute --exchange-leverage 3 --target-gross-cap 3.00
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
