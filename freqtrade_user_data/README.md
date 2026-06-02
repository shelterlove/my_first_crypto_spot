# Freqtrade Userdir

This directory is the Freqtrade execution layer for `crypto_spot_v1`.

The strategy files should stay thin: Freqtrade owns the bot lifecycle, while
`src/crypto_spot_v1` remains the source of truth for indicators, market state,
and target-position decisions.

First dry-run target:

```bash
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --strategy CryptoSpotV219B
```

`CryptoSpotV219B` is a thin Freqtrade shell around the native `v2_19B`
strategy. Keep the strategy logic in `src/crypto_spot_v1`; the Freqtrade file
should only adapt lifecycle, staking, and position adjustment calls.

Validation commands:

```bash
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade download-data --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --exchange binance --pairs BTC/USDT ETH/USDT BNB/USDT --timeframes 1d --timerange 20200101-20260601
freqtrade backtesting --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --strategy CryptoSpotV219B --timeframe 1d --timerange 20200101-20260601 --cache none
```

Current status: the Freqtrade shell loads and runs full entry/exit backtests.
It generates signals from a stateful native-strategy simulation over the full
dataframe, preserving confirmation bars, cooldowns, last-trade state, and the
synthetic portfolio path.

Treat Freqtrade performance numbers as execution-layer smoke tests only. The
native strategy uses target-position partial rebalancing and rolling-window
accounting, while Freqtrade reports trade-level entries/exits and may close a
position fully when an exit signal fires.

Signal consistency check:

```bash
python scripts/check_freqtrade_signal_consistency.py
```

This script shows why the first Freqtrade shell differed: stateless per-candle
decisions generated thousands of extra entry/exit signals because they reset the
native strategy state on every bar.
