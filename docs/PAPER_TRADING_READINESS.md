# Paper Trading Readiness

## Current Candidate

- Native strategy: `v3_4I`
- Freqtrade shell: `CryptoSpotV34I`
- Runtime model: one dry-run bot, three fixed pairs (`BTC/USDT`, `ETH/USDT`, `BNB/USDT`)
- Status: suitable for paper trading; not yet approved for real capital

`v3_4I` is the current recommended paper-trading candidate because:

- it beats `buy_hold` on the full `2019-01-01 ~ 2026-05-22` single-pair windows;
- it remains the cleanest promoted version after the later `v3_5x` execution experiments;
- its recent optimization deltas are narrow and auditable.

## Why `v3_4I`

- `v3` is the stable native baseline.
- `v3_4I` adds only the sell-side changes that stayed clean in validation:
  - BTC mature-bull giveback trim
  - inherited narrow trailing-profit-take behavior that remained acceptable
- later `v3_5D ~ v3_5H` MIXED rebuy execution experiments improved behavior
  interpretation but did not beat `v3_4I` robustly enough to replace it.

## Readiness Gates

Before starting dry-run on a server, verify:

- Freqtrade shell loads: `CryptoSpotV34I`
- native adapter sanity check passes
- dry-run config uses the intended pairs and `1d` timeframe
- downloaded data exists for BTC, ETH, and BNB
- logs and signal artifacts are writable on the server
- API keys, if configured, are exchange read-only or dry-run only

## Dry-Run Rules

- Freeze strategy code before the dry-run window starts.
- Run `v3_4I` only. Do not deploy `v3_5D ~ v3_5H`.
- Keep `dry_run: true`.
- Review actions at least weekly:
  - `native_reason`
  - stake sizing
  - repeated partial sells or rebuy loops
  - divergence between expected and produced states
- Stop and investigate if live signal cadence differs materially from backtest expectations.

## Not Ready For Real Capital Until

- 2-4 weeks of dry-run logs are clean
- no unexplained signal churn appears in real-time operation
- order sizing matches the intended per-pair exposure model
- operational recovery is tested: restart, reconnect, missing data, wallet sync
- execution assumptions are acceptable relative to backtest pricing

## Recommended Validation Commands

```powershell
python scripts\check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade test-pairlist --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json
```
