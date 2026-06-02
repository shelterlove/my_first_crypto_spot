# Strategy Iteration Notes

## Current Reference

The current reference candidate is `v2_19B`, deployed through the thin
Freqtrade shell `CryptoSpotV219B`.

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
- Promote only after rolling windows show stable behavior.
- Keep rejected experiments out of active strategy code.
