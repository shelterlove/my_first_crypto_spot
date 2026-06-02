# Strategy Review 2026-06-02

## Reference

Current reference:

- Native strategy: `v2_19B`
- Freqtrade strategy: `CryptoSpotV219B`
- Baseline run: `results/freqtrade_eval/baseline_v2_19B_fixed_adj_20260602`
- Quick rolling run: `results/freqtrade_eval/rolling_v2_19B_fixed_adj_quick_20260602`

The older `baseline_v2_19B_full_20260602` and
`rolling_v2_19B_standard_20260602` runs were produced before the Freqtrade
position-adjustment fix. Do not use them for strategy promotion decisions.

## Primary Metrics

Full-period fixed-allocation aggregate:

- strategy return: `2241.39%`
- Buy & Hold return: `1307.29%`
- excess return: `934.11%`
- max drawdown: `-53.85%`
- average exposure: `60.93%`

Quick rolling aggregate:

- windows: `11`
- mean excess: `-2.16%`
- median excess: `-11.14%`
- win rate: `36.36%`
- worst excess: `-104.05%`

The strategy is strong over the full period, but the rolling profile is not yet
stable enough to treat the framework as finished.

## Main Problem

The recurring weakness is not BEAR protection. The main issue is performance in
recovery and choppy uptrend windows, especially 2022-2024.

Repeated pattern:

- the long structure starts improving;
- raw state flickers into `MIXED`;
- `target-reduce` cuts exposure;
- price recovers and the strategy buys back;
- the strategy gives up upside and creates sell-then-rebuy churn.

This is visible in the worst quick-rolling windows:

- `20221216-20231216`: aggregate excess `-65.43%`
- `20230614-20240613`: aggregate excess `-104.05%`
- `20231211-20241210`: aggregate excess `-47.64%`

Target-reduce exits are the most frequent exit type in these windows. Windows
with high target-reduce activity have materially weaker median excess.

## What Not To Change First

Do not weaken these until there is stronger evidence:

- BEAR confirmation and exits;
- trend-break exits;
- risk-reduce exits;
- fast BULL detection;
- asset-specific rules.

These rules are responsible for the strategy's downside protection and for the
full-period outperformance.

## Candidate Plan

Next candidate should be `v2_20B`, not a replacement for `v2_19B` yet.

Design target:

- reduce low-quality repeated `target-reduce` churn;
- keep trend-break and risk-reduce exits unchanged;
- avoid symbol-specific behavior;
- avoid using recent-window performance as the main criterion.

Proposed rule set:

1. Target-reduce hysteresis

   In `MIXED`, after a `target-reduce` sell, block additional `target-reduce`
   sells for the same symbol until either risk worsens or enough bars pass.
   Risk-reduce and trend-break remain allowed.

2. Structural core floor

   When `ema72 > ema168` and `ema168_slope > 0`, a `target-reduce` sell should
   not reduce exposure below a moderate core floor, such as `0.65`. This applies
   only to `target-reduce`.

3. Fast re-entry after target-reduce

   If a recent `target-reduce` was followed by price reclaiming `ema24` with
   positive short-term momentum, reduce buy cooldown so the strategy can restore
   exposure without waiting for full BULL confirmation.

## Validation Order

Use this order for the next iteration:

1. Run targeted windows with `--no-run` only when parsing existing results.
2. Run quick rolling for `v2_20B`.
3. Compare against `v2_19B` using:
   - rolling median excess;
   - rolling win rate;
   - worst rolling excess;
   - max drawdown;
   - per-pair deterioration.
4. Only if quick rolling improves without worsening worst windows, run full
   baseline.
5. Only if full baseline remains competitive, run standard rolling.

## Promotion Requirement

`v2_20B` can replace `v2_19B` only if it improves rolling stability without
materially hurting full-period excess or max drawdown. Recent-window improvement
alone is not sufficient.
