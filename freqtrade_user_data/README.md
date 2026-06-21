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

## Warm Start Alignment

The native backtest path may already hold a residual position when the
Freqtrade bot is started from an empty dry-run/live wallet. The shell therefore
supports a conservative warm-start alignment entry:

- If the latest native action is `hold`, but `native_current_pct` is above
  `min_delta_pct`, the shell emits `enter_long` with tag
  `bootstrap-position-align`.
- `custom_stake_amount()` sizes that entry from the fixed pair sleeve and caps
  the alignment at `bootstrap_position_alignment_max_pct` per pair.
- This is intended for starting a new paper/live bot mid-path. Native strategy
  research logic is unchanged.

Set `bootstrap_position_alignment_enable = False` in the strategy shell if the
bot must wait for the next explicit native buy signal instead.

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
