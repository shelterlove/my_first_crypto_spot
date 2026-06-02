# Strategy Iteration Notes

## Current Reference

The current reference candidate is `v2_20D`, deployed through the thin
Freqtrade shell `CryptoSpotV220D`.

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

`v2_20D` is now the reference. The remaining bottleneck is still 2023-2025
recovery and slow-uptrend participation, not BEAR defense.
