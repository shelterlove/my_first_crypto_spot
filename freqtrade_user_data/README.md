# Freqtrade Userdir

This directory is the Freqtrade execution layer for `crypto_spot_v1`.

## Current Recommended Shell

- Native strategy: `v3_16A`
- Freqtrade shell: `CryptoSpotV316A`
- Config example: `config/config.dryrun.v316a.example.json`

`v3_4I` / `CryptoSpotV34I` remains available as the older conservative
paper-trading shell.

The Freqtrade shell stays intentionally thin. Indicators, regime state,
cooldowns, and target-position logic remain in `src/crypto_spot_v1`.

## Validation

```bash
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade download-data --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.v316a.example.json --exchange binance --pairs BTC/USDT ETH/USDT BNB/USDT --timeframes 1d --timerange 20170101-20260601 --data-format-ohlcv feather --prepend
python scripts/check_freqtrade_adapter.py
python scripts/freqtrade_eval.py --strategy CryptoSpotV316A --eval-split dev --rolling-preset quick --run-id rolling_v3_16A_dev_quick
```

## Dry Run

```bash
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.v316a.example.json --strategy CryptoSpotV316A
```

## Deployment Note

The research evaluator assumes independent sleeves per pair. A single live
Freqtrade bot uses one wallet. For paper trading this is acceptable, but for
real-capital comparison do not assume one-wallet live sizing is identical to
the research capital model without explicit review.
