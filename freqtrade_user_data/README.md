# Freqtrade Userdir

This directory is the Freqtrade execution layer for `crypto_spot_v1`.

`CryptoSpotV219B` is the current shell strategy. It delegates indicators,
regime state, cooldowns, and target-position decisions to the native
`v2_19B` implementation under `src/crypto_spot_v1`.

## Validation

```bash
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade download-data --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --exchange binance --pairs BTC/USDT ETH/USDT BNB/USDT --timeframes 1d --timerange 20200101-20260601
python scripts/freqtrade_eval.py --strategy CryptoSpotV219B --timerange 20200101-20260601 --report-window 20251202-20260601 --run-id baseline_v2_19B
```

Use `scripts/freqtrade_eval.py` for strategy review. It runs one fixed-capital
backtest per pair and then builds the equal-weight aggregate report. This avoids
mixing strategy quality with shared-wallet capital-allocation effects.

## Dry Run

```bash
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --strategy CryptoSpotV219B
```

Before dry-run, verify that order sizing and wallet allocation match the intended
deployment model. The research evaluator assumes independent sleeves per pair;
a single live Freqtrade bot uses one wallet unless deployment is split into
separate bots or equivalent stake controls.
