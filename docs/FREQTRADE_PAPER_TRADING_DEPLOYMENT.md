# Freqtrade Paper Trading Deployment

This document is the shortest practical path to move the current recommended
strategy to a server for dry-run paper trading.

## Recommended Target

- Native strategy: `v3_4I`
- Freqtrade shell: `CryptoSpotV34I`
- Exchange model: `binance` spot
- Timeframe: `1d`
- Pairs: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`

## 1. What To Deploy

Copy this repository to the server, including:

- `src/crypto_spot_v1/`
- `freqtrade_user_data/`
- `configs/`
- `scripts/`
- `pyproject.toml`

You do not need to copy:

- `results/`
- local caches
- `__pycache__/`

## 2. Server Prerequisites

- Linux server or VM
- Python `3.10+`
- Freqtrade installed in a venv or container
- exchange API keys suitable for dry-run or read-only operations

If using a Python virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[freqtrade]
```

## 3. Repository Layout On Server

Example:

```text
/opt/crypto_spot_v1/
  src/
  scripts/
  configs/
  freqtrade_user_data/
  .venv/
```

## 4. Environment Variables

Native data loaders use `.env` if present. If you still need DB-backed scripts
on the server, define:

```dotenv
DB_HOST=127.0.0.1
DB_PORT=5433
DB_NAME=quant_db
DB_USER=quant
DB_PASSWORD=quant_password
```

Freqtrade dry-run itself does not require the native DB as long as Freqtrade
market data has already been downloaded into `freqtrade_user_data/data/`.

## 5. Validate The Adapter Before Running

From repo root:

```bash
python scripts/check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade test-pairlist --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json
```

Expected result:

- `CryptoSpotV34I` appears in the strategy list
- pairlist test passes

## 6. Download Data

```bash
freqtrade download-data \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.example.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 1d \
  --timerange 20170101-20260601 \
  --data-format-ohlcv feather \
  --prepend
```

## 7. Dry-Run Config

Base file:

- `freqtrade_user_data/config/config.dryrun.example.json`

Review these fields before use:

- `dry_run`
- `dry_run_wallet`
- `max_open_trades`
- `stake_amount`
- exchange credentials
- pair whitelist

For paper trading, keep:

```json
"dry_run": true
```

## 8. Start The Bot

```bash
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.example.json \
  --strategy CryptoSpotV34I
```

## 9. Operational Checks During The Paper Window

Review regularly:

- generated Freqtrade logs
- action reasons in entry and adjustment tags
- repeated partial buys or sells
- unexpected idle periods
- wallet allocation behavior relative to three-pair expectations

Recommended audit questions:

- Did BTC, ETH, and BNB all receive sensible actions?
- Are partial position adjustments firing as expected?
- Do live signals resemble recent backtest behavior?
- Are any trades repeatedly opened and reduced in a short period?

## 10. Signal Review Helper

For an offline signal snapshot:

```bash
python scripts/generate_daily_signal.py --strategy v3_4I
```

This writes signal artifacts under `results/daily_signals/`.

## 11. Upgrade / Rollback Procedure

Before changing live paper-trading code:

1. stop the bot
2. commit the current repo state or tag it
3. deploy the new code
4. rerun adapter validation
5. restart dry-run

Rollback:

1. stop the bot
2. checkout the last known-good commit
3. rerun validation
4. restart

## 12. What Not To Deploy Yet

Do not paper-trade these as the primary production candidate:

- `v3_5D`
- `v3_5E`
- `v3_5F`
- `v3_5G`
- `v3_5H`

They are useful research variants but did not displace `v3_4I`.

## 13. Recommendation

For the first server run:

- use one bot
- use `CryptoSpotV34I`
- keep dry-run on
- observe for at least 2-4 weeks before discussing real capital
