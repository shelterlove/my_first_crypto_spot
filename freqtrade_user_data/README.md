# Freqtrade Userdir

This directory is the Freqtrade execution layer for `crypto_spot_v1`.

The strategy files should stay thin: Freqtrade owns the bot lifecycle, while
`src/crypto_spot_v1` remains the source of truth for indicators, market state,
and target-position decisions.

First dry-run target:

```bash
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json --strategy CryptoSpotV26
```
