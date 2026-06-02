# Strategy Review 2026-06-02

## Reference

Current reference:

- Native strategy: `v2_21E`
- Freqtrade strategy: `CryptoSpotV221E`
- Baseline run: `results/freqtrade_eval/baseline_v2_21E_20260602`
- Quick rolling run: `results/freqtrade_eval/rolling_v2_21E_quick_20260602`
- Standard rolling run: `results/freqtrade_eval/rolling_v2_21E_standard_20260602`
- Review report: `results/freqtrade_eval/baseline_v2_21E_20260602/review/report.html`

The older `baseline_v2_19B_full_20260602` and
`rolling_v2_19B_standard_20260602` runs were produced before the Freqtrade
position-adjustment fix. Do not use them for strategy promotion decisions.

## Primary Metrics

Full-period fixed-allocation aggregate:

- strategy return: `2563.56%`
- Buy & Hold return: `1307.29%`
- excess return: `1256.27%`
- max drawdown: `-53.85%`
- average exposure: `60.81%`

Quick rolling aggregate:

- windows: `11`
- mean excess: `3.69%`
- median excess: `-2.19%`
- win rate: `45.45%`
- worst excess: `-103.41%`

Standard rolling aggregate:

- windows: `43`
- mean excess: `42.90%`
- median excess: `-1.93%`
- win rate: `48.84%`
- worst excess: `-245.92%`

Decision score:

- score: `80.93`
- grade: `B+`
- decision: `promote_reference`
- CAGR: `75.94%` versus Buy & Hold `57.64%`
- Sharpe: `1.29` versus Buy & Hold `1.01`
- Sortino: `1.52` versus Buy & Hold `1.35`
- Calmar: `1.41` versus Buy & Hold `0.79`

The strategy remains strong over the full period and is more stable than
`v2_19B`, but the rolling profile still shows weakness in 2023-2025 recovery
and slow-uptrend windows.

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

- `20221216-20231216`: aggregate excess `-65.49%`
- `20230614-20240613`: aggregate excess `-103.41%`
- `20231211-20241210`: aggregate excess `-46.97%`

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

## Rejected Candidate: v2_20B

`v2_20B` tested target-reduce hysteresis, a structural MIXED trim cap, and fast
post-trim re-entry. It is rejected and should not be kept in active strategy
code.

Quick rolling comparison:

| strategy | mean excess | median excess | win rate | worst excess |
| --- | ---: | ---: | ---: | ---: |
| `v2_19B` | `-2.16%` | `-11.14%` | `36.36%` | `-104.05%` |
| `v2_20B` | `-26.40%` | `-16.40%` | `36.36%` | `-130.53%` |

The failure is concentrated in strong bull/recovery windows:

- `20200629-20210629`: excess fell from `168.18%` to `25.43%`.
- `20201226-20211226`: excess fell from `-11.14%` to `-130.53%`.

Conclusion: broad target-reduce suppression is too blunt. It slightly improves a
few later choppy windows, but it damages the core job of the strategy: staying
invested through long uptrends.

## Promoted Candidate: v2_20D

`v2_20D` adds a narrow delay-only confirmation for constructive low-risk
`target-reduce` sells. It does not change `trend-break`, `risk-reduce`, BEAR
exits, target tables, buy sizing, or asset-specific behavior.

The delay applies only when all of these are true:

- `sell_setup == "target-reduce"`;
- raw and confirmed state are both `MIXED`;
- `trend_risk <= 1` and `risk_score <= 2`;
- BTC regime is not `BEAR`;
- price is above EMA24;
- ROC5 is positive;
- `ema72 > ema168` and `ema168_slope >= 0`.

The sell is delayed for two strategy calls. If the constructive condition
persists, the sell is allowed. If risk worsens or the constructive condition
breaks, normal sell handling resumes.

Validation versus fixed-adjustment `v2_19B`:

| evaluation | `v2_19B` | `v2_20D` |
| --- | ---: | ---: |
| full excess | `934.11%` | `1169.43%` |
| full max drawdown | `-53.85%` | `-53.85%` |
| quick median excess | `-11.14%` | `-2.19%` |
| quick win rate | `36.36%` | `45.45%` |
| quick worst excess | `-104.05%` | `-104.05%` |
| standard median excess | `-8.12%` | `-1.89%` |
| standard win rate | `44.19%` | `48.84%` |
| standard worst excess | `-246.76%` | `-246.76%` |
| decision score | `78.04` | `80.82` |
| Calmar | `1.34` | `1.39` |

The improvement is mostly from BNB windows, while BTC and ETH show only small
mean-excess declines and no worse median or worst rolling excess. The rule is
asset-general and logically tied to market structure, so this is acceptable.

## Promoted Candidate: v2_21E

`v2_21E` is a narrow follow-up to `v2_20D`. It does not change target tables,
BEAR exits, `risk-reduce`, or `trend-break`. It only addresses one observed
failure mode: structural recovery buys that are immediately unwound by routine
`target-reduce` sells.

The added rules are:

- in constructive `MIXED`, use the existing recovery-override buy path when
  price is above EMA24 and EMA168, EMA72 remains above EMA168, EMA168 slope is
  positive, ROC5 is positive, and BTC regime is not BEAR;
- after a `safe-recovery` buy, delay routine `target-reduce` for 2 calls if
  structure remains constructive and risk is not above 3;
- `risk-reduce`, `trend-break`, and BEAR exits are never blocked by this grace
  period.

Validation versus `v2_20D`:

| evaluation | `v2_20D` | `v2_21E` |
| --- | ---: | ---: |
| full excess | `1169.43 pp` | `1256.27 pp` |
| full max drawdown | `-53.85%` | `-53.85%` |
| score | `80.82` | `80.93` |
| quick mean excess | `3.03 pp` | `3.69 pp` |
| quick worst excess | `-104.05 pp` | `-103.41 pp` |
| standard mean excess | `41.44 pp` | `42.90 pp` |
| standard median excess | `-1.89 pp` | `-1.93 pp` |
| standard win rate | `48.84%` | `48.84%` |
| standard worst excess | `-246.76 pp` | `-245.92 pp` |

This is a small improvement, not a new regime. The main bottleneck remains
slow-uptrend participation in 2023-2025 windows. `v2_21E` is acceptable as the
new reference because it improves full-period return and worst rolling excess
without weakening the defensive exits.

## Candidate Plan

The next candidate should focus on the remaining 2023-2025 recovery weakness,
not on broad sell suppression.

Design target:

- reduce low-quality repeated `target-reduce` churn;
- keep trend-break and risk-reduce exits unchanged;
- avoid symbol-specific behavior;
- avoid using recent-window performance as the main criterion.

Possible rule set:

1. Fast re-entry after target-reduce

   If a recent `target-reduce` was followed by price reclaiming `ema24` with
   positive short-term momentum, reduce buy cooldown so the strategy can restore
   exposure without waiting for full BULL confirmation.

2. Delay-only confirmation

   Test a short delay before low-risk `target-reduce` in constructive `MIXED`,
   but do not reduce sell size or block later sells. Risk-reduce and trend-break
   remain unchanged.

## Validation Order

Use this order for the next iteration:

1. Run targeted windows with `--no-run` only when parsing existing results.
2. Run quick rolling for the candidate.
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

Any candidate can replace `v2_19B` only if it improves rolling stability without
materially hurting full-period excess or max drawdown. Recent-window improvement
alone is not sufficient.
