# Strategy Iteration Notes

## Current Reference

The current reference candidate is `v2_21E`, deployed through the thin
Freqtrade shell `CryptoSpotV221E`.

## Core Diagnosis

The strategy is not mainly limited by prediction complexity. Its main risk is
participation timing:

- it should stay invested enough during durable uptrends;
- it should avoid defensive rules that repeatedly sell normal bull-market
  pullbacks;
- it should not fix underperformance with coin-specific exceptions;
- it should keep BEAR protection simple and explicit.

BNB weakness in recent Freqtrade tests is not a reason to add BNB-only rules.
The correct next step is to inspect whether the same rule creates late entries,
early exits, or long cash periods across multiple assets and windows.

## Iteration Rules

- Do not optimize for a single symbol.
- Prefer rules that are defensible before seeing the backtest.
- Test new ideas with fixed-allocation per-pair results first.
- Promote only after rolling windows show stable behavior under
  `docs/EVALUATION_METRIC_POLICY.md`.
- Treat recent-window results as diagnostics, not promotion evidence.
- Keep rejected experiments out of active strategy code.

## 2026-06-02 Target-Reduce Iteration

Added `scripts/analyze_target_reduce_regret.py` to inspect every
`target-reduce` exit against 3/5/10/20 day forward returns and the market
structure at the exit.

Findings:

- broad `target-reduce` suppression is harmful and remains rejected;
- fast re-entry cooldown alone (`v2_20C`) had no behavior difference versus
  `v2_19B`, so it was removed;
- narrow delay-only confirmation for constructive low-risk `target-reduce`
  sells (`v2_20D`) improved full-period and rolling stability without worsening
  worst rolling excess or max drawdown.

## 2026-06-03 Structural Recovery Iteration

`v2_21E` keeps `v2_20D` sell defense intact and adds two narrow recovery rules:

- structural MIXED recovery can use the existing recovery-override buy path
  when price is above EMA24 and EMA168, EMA72 remains above EMA168, EMA168 slope
  is positive, ROC5 is positive, and BTC regime is not BEAR;
- after such a `safe-recovery` buy, routine `target-reduce` sells get a 2-bar
  grace period. `risk-reduce`, `trend-break`, and BEAR exits are unchanged.

Validation versus `v2_20D`:

| metric | `v2_20D` | `v2_21E` |
| --- | ---: | ---: |
| full excess | `1169.43 pp` | `1256.27 pp` |
| score | `80.82` | `80.93` |
| standard mean excess | `41.44 pp` | `42.90 pp` |
| standard median excess | `-1.89 pp` | `-1.93 pp` |
| standard win rate | `48.84%` | `48.84%` |
| standard worst excess | `-246.76 pp` | `-245.92 pp` |
| max drawdown | `-53.85%` | `-53.85%` |

This is a small improvement, not a structural breakthrough. `v2_21E` is the
reference because it improves full-period and mean rolling performance without
materially weakening drawdown or worst-window behavior. The remaining bottleneck
is still 2023-2025 recovery and slow-uptrend participation, not BEAR defense.
