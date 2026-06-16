# Strategy Architecture Notes: v3_15E

This document records the current strategy-research architecture after the
v3_15A-v3_15E cleanup pass. It is scoped to native strategy research and does
not cover deployment, Freqtrade dry-run operation, or server work.

## Current Research Base

`v3_15E` is the current architecture base.

It is behavior-equivalent to `v3_15D`, `v3_15C`, `v3_15B`, and `v3_15A` on the
checked windows, while keeping the effective `v3_14D` trading behavior.

Checked windows:

- `2019-01-01` to `2024-12-31`
- `2021-01-01` to `2024-12-31`
- `2025-01-01` to `2026-05-22`

In all three `v3_15E` vs `v3_15D` checks:

- return delta: `0`
- max drawdown delta: `0`
- trade count delta: `0`
- changed trades: none

Relevant result directories:

- `results/diagnostics/window_v3_15E_vs_v3_15D_20190101_20241231_r3`
- `results/diagnostics/window_v3_15E_vs_v3_15D_20210101_20241231_secondary_r1`
- `results/diagnostics/window_v3_15E_vs_v3_15D_20250101_20260522_validation_r1`

## Effective Inheritance Path

The active research path is:

```text
V315EStrategy
-> V315DStrategy
-> V315CStrategy
-> V315BStrategy
-> V315AStrategy
-> V314CStrategy
-> V314BStrategy
-> V314AStrategy
-> V312CStrategy
-> V312BStrategy
-> V312AStrategy
-> V311AStrategy
-> V310CStrategy
-> V310BStrategy
-> V39FStrategy
-> ...
```

Do not evaluate future candidates by reading only the newest class. Several
important behaviors still come from earlier classes in this chain.

## Four-Layer Design

The intended architecture has four layers.

### 1. Signal Layer

Question: is this a potentially good buy or sell location?

Current examples:

- `peak_memory_light_trim`
- `profit_stage2_failed_rebound_trim`
- `mature_bull_giveback_trim`
- `trailing-profit-take`
- `safe-recovery`
- `trend-cont`
- `pullback`
- pending buy release: mean reversion or breakout

Important distinction:

- `target-gap` and `target-reduce` are not high-quality signals by themselves.
- They are position needs created by the target-position system.
- Future versions should treat them as low-priority candidates that require
  execution permission.

### 2. State Layer

Question: does current market state support the candidate?

Current inputs include:

- `raw_state`
- `confirmed_state`
- `trend_risk`
- `drawdown_risk`
- `risk_score`
- `btc_regime`
- EMA distance and slope
- `rolling_365d_pos`
- `donchian_pos`
- `roc_20`
- peak memory and days from peak

Current issue:

These inputs are still repeated across many mechanisms. That is acceptable for
now because v3_15E is an architecture bridge, but future cleanup should extract
common labels such as:

- high-profit retreat
- failed rebound
- weak recovery
- confirmed breakout
- deep risk
- external BTC risk

### 3. Execution Arbiter Layer

Question: should this candidate execute now, be deferred, be capped, or be
cancelled?

This is the most important layer for reducing internal conflict.

Current v3_15E progress:

- pending buy fills use `candidate -> decision -> Action`;
- regular buys use `candidate -> decision -> Action`;
- sell actions are wrapped into sell candidates without changing existing sell
  behavior;
- post-profit-trim rebuy protection is centralized through
  `_evaluate_buy_permission`;
- `profit_cycle` centralizes path memory for peak trim and stage2 trim.

Still incomplete:

- regular sells are not fully generated from candidates yet; v3_15E wraps the
  existing sell chain to avoid losing older active sell mechanisms;
- late sell intent and other sell-side special cases still live in older chain
  logic;
- there is not yet one unified priority table for all buy and sell candidates.

Target priority order:

1. trend break / core override defense
2. risk reduce
3. profit trim and stage2 trim
4. late sell intent fill
5. pending buy intent fill
6. safe recovery / trend continuation / pullback
7. target-gap / target-reduce position needs

### 4. Sizing Layer

Question: if execution is allowed, how much should trade?

Current sizing sources:

- target-position gap
- state config `max_buy`
- `max_sell`
- peak trim size
- stage2 trim size
- intent budget
- recovery override sizing
- bull guard sizing

Design rule:

Sizing should not decide whether a trade is good. It should only size an
already allowed candidate.

## Profit Cycle State

`v3_15B` introduced `_profit_cycle` as the current path-memory container for
profit-taking.

It tracks:

- `stage1_call`
- `stage1_sell_price`
- `stage1_anchor_peak`
- `stage2_call`
- `stage2_sell_price`
- `stage2_anchor_peak`
- `last_trim_call`
- `last_trim_sell_price`

Current behavior intentionally preserves old semantics:

- `peak_memory_light_trim` opens the stage2 profit cycle;
- `mature_bull_giveback_trim` does not open a stage2 cycle;
- after stage2 trim, the cycle stays in stage 2 and does not repeatedly fire
  additional stage2 trims from the same stage1 anchor.

This is a structure cleanup, not a new trading rule.

## Candidate Flow Status

Current status by path:

| Path | Status |
| --- | --- |
| pending buy fill | candidate/decision/action |
| regular buy | candidate/decision/action |
| regular sell | existing action wrapped as candidate |
| profit trim sell | still generated by existing sell chain |
| trailing profit take | still generated by existing sell chain |
| late sell intent | still generated by older intent logic |

The sell side is intentionally more conservative. A direct rewrite of
`_maybe_sell` lost active sell mechanisms such as `trailing-profit-take` and
`mature_bull_giveback_trim`, so v3_15E keeps the existing sell chain and wraps
its output.

## Keep / Do Not Prioritize

Keep as active architecture concepts:

- native event-driven evaluation;
- target-position system;
- explicit signal/state/execution/sizing separation;
- post-sell buy permission;
- profit cycle memory;
- pending intent with current-state revalidation;
- risk-reduce and trend-break defense.

Do not prioritize now:

- Freqtrade summary metrics as strategy-quality evidence;
- broad high-price selling rules;
- broad MIXED buy vetoes;
- coin-specific fitting;
- more parameter tuning before architecture responsibilities are clear.

## Known Remaining Problems

1. `target-gap` still behaves too much like a buy signal.

It should become `position_buy_need`: a request to move toward target, not a
standalone reason to buy.

2. `target-reduce` still behaves too much like a sell signal.

It should become `position_sell_need`: a request to reduce exposure, subject to
sell quality and risk priority.

3. The sell side is not fully candidate-native.

v3_15E wraps sell actions after the old chain generates them. A future version
can migrate sell generation mechanism by mechanism, but only with strict
trade-level equivalence checks.

4. State labels are duplicated.

Many mechanisms inspect the same EMA, Donchian, ROC, BTC regime, and risk
fields. Future cleanup should extract reusable labels instead of repeating
condition groups.

5. Historical candidates remain in `strategy_candidates.py`.

They are useful for reproduction while the research path is moving quickly, but
the file is now too large. Do not delete historical candidates until the active
path and research conclusions are documented enough to reproduce important
comparisons.

## Next Steps

Recommended next candidate:

```text
v3_15F
```

Goal:

- keep behavior close to v3_15E at first;
- explicitly rename internal regular target-gap candidate semantics to
  `position_buy_need`;
- explicitly rename regular target-reduce semantics to `position_sell_need`;
- keep risk-reduce, trend-break, profit trim, and trailing profit take as
  higher-priority sell signals.

After that, a behavior-changing candidate can test whether low-priority
position needs should require stronger buy/sell quality confirmation in MIXED.

## Deletion Guidance

Safe to delete:

- `__pycache__`
- `.pytest_cache`
- empty temporary probe directories
- failed intermediate diagnostic runs when a final passing run is retained

Do not delete yet:

- historical strategy classes in `strategy_candidates.py`;
- research scripts under `scripts/`;
- Freqtrade shells and docs;
- diagnostics used to support current conclusions.

Reason:

These files are not all active in the current path, but many still support
reproduction, review, or documentation references. Prefer archiving and indexing
before physical deletion.
