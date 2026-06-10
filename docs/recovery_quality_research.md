# Recovery Quality Research Notes

Last updated: 2026-06-07

## Current Baseline

- Stable baseline: `v2_21E`
- Active conservative candidate: `v2_28C`
- Rejected research candidates in this round: `v2_29A`, `v2_29B`, `v2_29C`, `v2_30A`, `v2_31A`, `v2_32A`, `v2_32B`, `v2_33A`, `v2_33B`
- No-edge research candidates in this round: `v2_31B`, `v2_31C`

## Design Goal

The recovery quality layer is an external market scoring layer, not the core position strategy itself.

The intended split is:

- Core position layer decides the strategy target and state range.
- External risk/control layer decides whether approaching that target is acceptable, how quickly to approach it, and when post-buy target reductions should be allowed to fire.

This means the scoring layer should be conservative. It should block or slow low-quality recovery exposure before it causes churn, but it should not rewrite normal MIXED/BULL behavior unless the evidence is very strong.

## What Worked

`v2_28C` remains the only clean candidate so far.

Complete rolling workflow versus `v2_21E`:

- Windows: 165
- Changed windows: 34
- Negative return deltas: 0
- Worse drawdown deltas: 0
- Trade delta sum: -42
- Return delta sum: +0.132345

By symbol:

- BTC/USDT: +0.105747, 21 positive windows, 0 negative windows
- ETH/USDT: +0.026598, 13 positive windows, 0 negative windows
- BNB/USDT: unchanged

Interpretation: a narrow post-safe-recovery target-reduce deadband can remove short-term churn without changing the strategy's broader risk profile.

## What Failed

### Static Recovery Quality Bands

`v2_29A` and `v2_29B` tried to use static recovery quality bands to cap or boost MIXED buy targets.

Smoke results were immediately weak:

- `v2_29A`: 4 negative return windows, 4 worse drawdown windows
- `v2_29B`: 5 negative return windows, 4 worse drawdown windows

Main issue: static indicators did not cleanly separate high-quality and low-quality recovery.

Offline refined labels also showed the problem:

- HIGH group still had weak 60d forward behavior.
- LOW group still contained strong rebound samples.
- BNB failure cases could be partially identified, but not cleanly enough to justify target caps as a general rule.

### High-Quality Trim Smoothing Only

`v2_29C` removed buy-target changes and kept only high-quality target-reduce smoothing.

Smoke result:

- 1 negative return window
- 0 worse drawdown windows
- One useful ETH improvement, but BNB worsened slightly

Interpretation: safer than target caps, but still inferior to `v2_28C`.

### Stateful Recovery Path Score

`v2_30A` widened the post-safe-recovery deadband and required a stronger path score after the buy.

Smoke looked clean:

- Negative return deltas: 0
- Worse drawdown deltas: 0
- Return gain appeared better than `v2_28C`

Complete rolling workflow failed:

- Windows: 165
- Changed windows: 44
- Negative return deltas: 23
- Worse drawdown deltas: 12
- Trade delta sum: -88
- Return delta sum: -0.514624

By symbol:

- BTC/USDT: +0.105747, 21 positive windows, 0 negative windows
- ETH/USDT: unchanged
- BNB/USDT: -0.620371, 23 negative windows, 12 worse drawdown windows

Interpretation: the extra stateful score reproduced the BTC benefit already captured by `v2_28C`, lost the ETH benefit, and introduced systemic BNB harm. The failure mode is not a single bad window; it is a recurring alt-specific over-tolerance after recovery buys.

### Low-Quality Recovery Veto

`v2_31A` tested a conservative-looking external recovery slowdown layer over `v2_28C`.

Design:

- Do not boost target allocation.
- Do not relax tiny buys.
- Slow or veto low-quality `safe-recovery` and low-position `target-gap` buys.
- Use multiple weak conditions: BTC weakness, EMA168 structure, Donchian/range weakness, drawdown, ROC/volume, ATR rank, and 365d price position.

Feather smoke result:

- `v2_28C` vs `v2_21E`: +0.011896, 0 negative windows, 0 worse drawdown windows.
- `v2_31A` vs `v2_21E`: -0.536901, 3 negative windows, 1 worse drawdown window.

Failure mode:

- `v2_31A` misclassified early strong-bull MIXED target-gap buys, especially 2019-03.
- The 4% slowdown created repeated small buys instead of clean risk reduction.
- This confirms that buy-size slowdown can reintroduce the exact repeated-probe behavior the project is trying to avoid.

`v2_31B` tightened the idea to a severe low-quality `target-gap` veto only.

Feather complete result:

- `v2_31B` matched `v2_28C` exactly.
- `v2_31B` vs `v2_28C`: 0 changed windows, 0 return delta, 0 trade delta.

`v2_31C` relaxed the severe veto slightly while adding a bull-repair deny gate.

Feather smoke result:

- `v2_31C` matched `v2_28C` exactly.
- Guard hits: 0.

Interpretation: the useful range is currently empty. If the veto is broad enough to trigger, it misclassifies strong recovery; if it is strict enough to avoid strong recovery, the core strategy has already avoided those buys.

### Data-Guided Veto/Probe Follow-Up

After mining the existing diagnostics, three more focused hypotheses were tested.

`v2_32A` tested a BNB-only safe-recovery veto for high historical rank plus weakening path.

Rationale from diagnostics:

- Actual BNB `safe-recovery` buys had weak 60d behavior.
- `rolling_365d_pos >= 0.62` plus weakening ROC/volume captured several bad BNB recovery buys.

Feather smoke result:

- `v2_32A` vs `v2_28C`: -1.050294 return delta.
- Negative windows: 3.
- Worse drawdown windows: 3.

Failure mode:

- The veto removed BNB recovery buys around 2019-08 and 2024-07.
- Although some of these buys had weak short forward returns, they were still needed for the later portfolio path.
- Short-horizon bad labels did not translate into better rolling-window behavior.

`v2_32B` tested a high-ATR BULL target-gap chase veto.

Rationale from diagnostics:

- Actual `atr_high & bull_state & target_gap` buys had negative 60d average behavior.

Feather smoke result:

- `v2_32B` vs `v2_28C`: -0.773474 return delta.
- Negative windows: 5.
- Worse drawdown windows: 5.

Failure mode:

- The veto delayed BTC/ETH/BNB BULL replenishment into different later buys.
- BTC sometimes improved, but ETH/BNB degraded enough to fail smoke.
- High ATR alone is not a reliable chase veto because it often appears in valid continuation phases.

`v2_33A` tested a one-shot BNB bear-market reclaim probe below the 8% tiny-buy floor.

Rationale from diagnostics:

- Missed BNB target-gap candidates in BTC BEAR regimes had very large forward returns in 2019.
- The probe was restricted to low current exposure, high volume, positive ROC20, and local reclaim conditions.

Feather smoke result:

- `v2_33A` vs `v2_28C`: -2.970261 return delta.
- Negative windows: 1.
- Worse drawdown windows: 2.
- Probe hits: 2.

Failure mode:

- The 2019 BNB probe looked attractive in forward-event analysis but badly polluted the full path.
- Adding even a small buy changed later position state and trade timing enough to reduce rolling-window performance.
- Missed-opportunity diagnostics cannot be used as direct counterfactual PnL without simulating the altered path.

`v2_33B` tightened the BNB bear reclaim probe to require `raw_state == MIXED` and `confirmed_state == MIXED`.

Path replay smoke result:

- `v2_33B` vs `v2_28C`: -3.252485 return delta.
- Negative windows: 1.
- Worse drawdown windows: 2.

Failure mode:

- The earlier `confirmed_state == BEAR` probe was removed, but a later 2019-02 probe still polluted the path.
- The changed path skipped later baseline safe-recovery and target-gap actions, then created a different sequence of BULL buys.
- This confirms that a single small probe can materially alter subsequent execution timing.

## Path Counterfactual Tool

Added `scripts/analyze_path_counterfactual_replay.py`.

Purpose:

- Load local feather candles.
- Replay registered strategies over the same engine.
- Compare candidates against a baseline at path level.
- Save smoke/complete deltas, by-symbol summaries, and smoke action diffs.

Important runs:

- `results/diagnostics/path_counterfactual_replay_smoke_20260608`
- `results/diagnostics/path_counterfactual_replay_complete_20260608`
- `results/diagnostics/path_counterfactual_replay_v2_33B_smoke_20260608`

The complete replay showed an important nuance:

- `v2_33A` complete return delta was +0.880886 versus `v2_28C`.
- But it had 11 worse drawdown windows and failed smoke/path-pollution badly.

Conclusion: do not promote candidates based on complete return alone. Smoke/path-pollution and drawdown deltas remain hard gates.

## Current Bottleneck

The bottleneck is not lack of indicators. The problem is that the available spot-market technical indicators are not stable enough to rank recovery quality across BTC, ETH, and BNB with one static rule.

Useful signals exist, but they are conditional:

- BTC responds well to narrow post-recovery churn control.
- ETH also benefits from the narrow `v2_28C` version.
- BNB is vulnerable to delayed target-reduce exits when the deadband is widened.
- Historical price percentile, EMA position, Donchian position, ROC, drawdown-from-high, and volume strength help describe context, but they do not form a universal high/low recovery classifier.
- Severe low-quality target-gap conditions mostly do not fire under the existing core strategy; broadening them quickly hits valid early-bull recovery.
- Short forward-return labels for actual buys are not sufficient to justify vetoes; the rolling path can still need those buys.
- Missed-opportunity events are especially dangerous: the best forward rows can become bad full-path interventions once they alter later position state.

## Updated Strategy Direction

Do not continue lowering tiny-buy thresholds or widening post-buy deadbands globally.

Recommended next direction:

1. Keep `v2_28C` as the only active candidate.
2. Treat scoring as a veto/slowdown layer, not as a target-boost layer.
3. Require symbol-aware validation for any recovery-quality rule.
4. For any new rule, first prove it does not change BNB negatively.
5. Prefer rules that reduce exposure after weak recovery evidence instead of rules that increase exposure after strong-looking evidence.

Possible future candidate shape:

- Start from `v2_28C`.
- Add a low-quality veto only when multiple independent risk conditions agree:
  - BTC regime is BEAR or deteriorating.
  - Asset is below EMA168 or EMA168 slope is negative.
  - Donchian position is weak.
  - 120d drawdown remains large.
  - 365d price percentile is high enough to imply rebound exhaustion, or low enough to imply unresolved breakdown.
  - Recovery buy is followed by weakening ROC/volume within a short path window.
- Do not raise target allocation based on this layer.
- Do not globally increase post-recovery sell deadband.

The next real breakthrough probably requires path-aware simulation before implementation, not another static pre-buy score. Any future candidate should first be tested as a counterfactual path replay, because event-level forward returns have repeatedly failed to predict strategy-level deltas.

## Promotion Rule

A future candidate should be rejected unless it passes all of these:

- Smoke windows: no negative return deltas and no worse drawdown deltas.
- Complete rolling workflow: no systematic symbol-specific damage.
- BNB must remain unchanged or positive.
- Return improvement should exceed `v2_28C`'s +0.132345 complete-window total, otherwise the added complexity is not justified.

## Rule Screen 2026-06-08

Added `scripts/screen_path_counterfactual_rules.py` to screen unregistered path-level rule overlays before turning them into formal strategy candidates.

Run:

- `results/diagnostics/path_counterfactual_rule_screen_20260608_rerun`

Rules screened:

- BNB safe-recovery veto using 365d historical price percentile, Donchian position, ROC, and volume weakness.
- BULL target-gap high-ATR chase veto, with and without overheat confirmation.
- BNB bear-market reclaim probe using low current exposure, BTC BEAR, volume strength, positive ROC20, Donchian reclaim, low 365d percentile, and EMA168 reclaim.

Result:

- Candidates screened: 21.
- Passed hard gates: 0.
- Best return group was BNB bear probe variants: `+0.317437` total return delta, but only changed one window and worsened drawdown once.
- Safe-recovery veto variants either did not fire or produced negative return and worse drawdown.
- High-ATR target-gap veto variants were consistently negative, with return deltas from `-1.221647` to `-1.337026`.

Conclusion:

- None of these static pre-buy overlays is worth promoting.
- Historical price percentile is useful as context, but not sufficient as a standalone screen when combined with Donchian/ROC/volume.
- The current bottleneck remains path dependence: a rule that looks sensible at event level can still damage later target-gap, safe-recovery, and target-reduce sequencing.
- Next screening should move from single-day pre-buy veto/probe rules toward short path-state rules, for example requiring deterioration after a weak recovery attempt before reducing the external market score.

## Indicator Screen 2026-06-08

Added `scripts/screen_recovery_indicator_candidates.py`.

Purpose:

- Screen indicators before changing strategy code.
- Separate pre-event structural indicators from post-event path confirmation indicators.
- Score rules as conservative low-quality recovery controls: blocking bad samples helps, blocking good samples is heavily penalized.

Run:

- `results/diagnostics/recovery_indicator_screen_20260608_v2`

Best indicator families:

- BTC macro weakness:
  - `btc_price_rank_180d <= 0.255` hit 97 bad samples, 0 good samples.
  - `btc_price_vs_ema72 <= 0.000422752 AND btc_ema24_slope <= -0.00195981` hit 98 bad samples, 0 good samples.
  - These mostly describe 2022-style weak macro conditions, so they are useful external score context but not enough for a recovery-specific candidate.
- Post-path historical-position deterioration:
  - `post_10d_rolling_365d_pos_delta <= -0.00739441` hit 149 bad samples and 10 good samples.
  - Combining post-5d/post-10d 365d-rank deterioration improved precision and kept coverage across BTC/ETH/BNB.
  - This is closer to the intended external market-score layer because it waits for recovery failure evidence after a buy.

Path replay follow-up:

- Added temporary `post_recovery_fade_cap` overlays to `scripts/screen_path_counterfactual_rules.py`.
- These overlays are not registered strategy candidates.
- They only activate after a recovery/target-gap buy, then cap exposure if 365d historical-position rank keeps falling and momentum/structure are weak.

Important path replay runs:

- `results/diagnostics/path_counterfactual_postfade_cap35_d02_20260608`
- `results/diagnostics/path_counterfactual_postfade_bnb_cap35_d02_20260608`
- `results/diagnostics/path_counterfactual_postfade_bnb_slope_grid_cap35_d02_20260608`
- `results/diagnostics/path_counterfactual_postfade_bnb_cap_grid_btcup003_d02_20260608`
- `results/diagnostics/path_counterfactual_postfade_bnb_drop_grid_btcup003_cap30_20260608`

Findings:

- Broad all-symbol post-fade cap had large positive total return (`+8.275053`) but failed hard gates with 9 negative return windows and 7 worse drawdown windows.
- BNB-only post-fade cap cleaned up most damage, but still failed with 3 negative return windows.
- Adding BTC long-term uptrend context was the key filter.
- Best screened rule:
  - Symbol: `BNB/USDT` only.
  - External cap: `30%`.
  - Activate only after recovery/target-gap buy age is 5 to 30 bars.
  - Require `rolling_365d_pos` to fall at least `0.02` from the buy snapshot.
  - Require weak momentum/structure: `roc_10 <= -0.03` or `roc_20 <= -0.05`, plus `price_vs_ema72 < 0` or `donchian_pos < 0.45`.
  - Require BTC long-term trend context: `btc_ema168_slope >= 0.03`.
- Best smoke result: `post_fade_bnb_btcup0.03_cap0.30_d0.02`
  - Return delta: `+0.968546`.
  - Negative return windows: `0`.
  - Worse drawdown windows: `0`.
  - Changed windows: `3`.
  - Positive windows: `strong_bull BNB`, `path_pollution BNB`, `full_dev_tail BNB`.

Interpretation:

- The first promising indicator is not a pre-buy high/low recovery classifier.
- It is a symbol-aware, post-buy failure confirmation signal.
- The useful pattern is: BTC long-term trend still supports risk, but BNB's own recovery attempt starts fading quickly. In that case, the external layer should force BNB back to light exposure instead of continuing to follow the internal target.
- This matches the original two-layer design: the core strategy still decides target/state, while the external score layer limits how close execution may move toward that target.

Next validation:

- Do not promote yet.
- Register a formal candidate only if we want to test this rule through the standard workflow.
- Before promotion, run a complete rolling replay equivalent and verify the effect is not just the three smoke windows.

## V2.34 Formal Candidate Test

Registered formal candidates:

- `v2_34A`: BNB post-recovery fade cap from the best screen rule.
- `v2_34B`: `v2_34A` plus `btc_price_vs_ema72 <= 0`.
- `v2_34C`: `v2_34A` plus `btc_price_vs_ema72 <= -0.02`.
- `v2_34D`: `v2_34A` plus BNB `rolling_365d_pos <= 0.35`.

Implementation:

- Base remains `v2_28C`.
- The new layer does not raise target allocation.
- It only adds a BNB sell cap after a recovery/target-gap buy has started fading.
- Trigger core:
  - buy age 5 to 30 bars;
  - BNB `rolling_365d_pos` down at least `0.02` from the buy snapshot;
  - weak momentum: `roc_10 <= -0.03` or `roc_20 <= -0.05`;
  - weak structure: `price_vs_ema72 < 0` or `donchian_pos < 0.45`;
  - BTC `ema168_slope >= 0.03`;
  - cap BNB exposure to `30%`.

Runs:

- `results/diagnostics/path_counterfactual_replay_v2_34A_smoke_20260608`
- `results/diagnostics/path_counterfactual_replay_v2_34A_complete_20260608`
- `results/diagnostics/path_counterfactual_replay_v2_34BC_smoke_20260608`
- `results/diagnostics/path_counterfactual_replay_v2_34D_smoke_20260608`

Results:

- `v2_34A` smoke:
  - return delta `+0.911464`;
  - negative return windows `0`;
  - worse drawdown windows `0`.
- `v2_34A` complete:
  - return delta `+2.660757`;
  - changed windows `59`;
  - negative return windows `21`;
  - worse drawdown windows `12`.
- `v2_34B` smoke:
  - return delta `-1.403272`;
  - negative return windows `2`;
  - worse drawdown windows `2`.
- `v2_34C` smoke:
  - return delta `-0.074895`;
  - negative return windows `1`;
  - worse drawdown windows `2`.
- `v2_34D` smoke:
  - return delta `-0.936316`;
  - negative return windows `2`;
  - worse drawdown windows `2`.

Conclusion:

- `v2_34A` is not promotable despite positive complete return, because rolling-window damage is too frequent.
- `v2_34B/C/D` should be rejected immediately because they fail smoke.
- The screened indicator was real, but the formally replayed strategy remains too path-dependent.
- The main failure mode is BNB warmup/path-state sensitivity: external cap actions in 2019 can improve some full windows but hurt rolling windows whose measurement starts after the cap changed the inherited position state.

Next direction:

- Do not keep tightening this cap rule unless a new path replay idea directly addresses inherited-position damage.
- If continuing this family, the next test should be an execution-only buy throttle after fade confirmation, not a forced sell cap.
- Forced post-buy sells are too path-sensitive for BNB unless the signal is tied to a true trend-break/risk-reduce condition.

## MIXED Historical Label Research, Train Older Years / Test Recent 2 Years

Run:

- `python scripts\analyze_mixed_historical_profiles.py --test-years 2 --run-id mixed_historical_profiles_oos2y_db_20260608`
- Output: `results/diagnostics/mixed_historical_profiles_oos2y_db_20260608`

Data split:

- DB daily data covers BTC/ETH/BNB from `2019-12-31 16:00 UTC` to `2026-05-22 16:00 UTC`.
- MIXED sample range: `2019-12-31 16:00 UTC` to `2026-02-01 16:00 UTC`.
- Train: before `2024-02-01 16:00 UTC`, 838 MIXED rows.
- Test: from `2024-02-01 16:00 UTC`, 600 MIXED rows.

Main finding:

- MIXED is not one neutral regime. It contains at least one useful recovery-reclaim structure and several risk structures.
- The goal should not be to replace the core market-state layer. These labels are better suited for the external market-score layer that controls how much of the internal target position may be executed.

Positive labels:

- `RECLAIM_EMA72_LONG_DOWN` is the strongest reusable MIXED profile.
  - Meaning: price has reclaimed EMA72 while the long EMA structure is still not fully repaired.
  - Train 60d median return: `+11.21%`, positive rate `64.93%`.
  - Test 60d median return: `+44.40%`, positive rate `94.12%`.
- `supportive_reclaim` is the best candidate composite.
  - Train 60d median return: `+19.69%`, positive rate `82.83%`.
  - Test 60d median return: `+44.40%`, positive rate `94.12%`.
  - Interpretation: a MIXED reclaim is materially better when BTC context is supportive. This should reduce suppression, but should not automatically boost target allocation.
- `high_volume_reclaim` is positive but weaker out of sample.
  - Train 60d median return: `+22.58%`, positive rate `93.20%`.
  - Test 60d median return: `+6.44%`, positive rate `79.17%`.
  - Interpretation: volume can confirm reclaim, but should be a secondary factor rather than a standalone positive rule.

Negative labels:

- `BTC_RANGE_WEAK` is a stable negative BTC context.
  - Train 60d median return: `-6.21%`, positive rate `41.54%`.
  - Test 60d median return: `-4.83%`, positive rate `35.44%`.
  - Interpretation: weak BTC range is suitable for an external risk-score penalty.
- `high_volume_pullback` is a stable negative MIXED pattern.
  - Train 60d median return: `-8.84%`, positive rate `10.00%`.
  - Test 60d median return: `-19.56%`, positive rate `19.23%`.
  - Interpretation: high-volume pullback should reduce allowed execution toward target.

Unstable or caution-only labels:

- `PULLBACK_ABOVE_EMA168` is not stable as a hard rule.
  - Train 60d median return: `+10.90%`.
  - Test 60d median return: `-4.25%`.
  - Use only as a light penalty or observation.
- `TOP_CHASE_RISK` is recently negative but not train-stable.
  - Train 60d median return: `+34.76%`.
  - Test 60d median return: `-6.53%`.
  - Use only as a recent-cycle caution until path tests confirm value.
- `bear_reclaim_trap` is historically very negative, but lacks clean recent-2y confirmation in this DB split.
  - Full sample 60d median return: `-16.47%`, positive rate `15.87%`.
  - Use as a caution label, not as a promoted rule yet.
- `BTC_BEAR` alone is not a clean bearish filter in MIXED samples.
  - Train and test 60d median returns are both positive.
  - A BTC bear context can still contain valid alt recovery entries. It must be combined with asset-level failed reclaim, no-reclaim, or weak-range patterns.

Proposed MIXED label set for future score research:

- `MIXED_SUPPORTIVE_RECLAIM`: price reclaims EMA72 while long structure is still repairing, with supportive BTC context. This should avoid unnecessary buy suppression.
- `MIXED_VOLUME_CONFIRMED_RECLAIM`: reclaim plus volume confirmation. Positive but only secondary.
- `MIXED_WEAK_RANGE_RISK`: BTC weak range or high-volume pullback. This should reduce allowed execution toward target.
- `MIXED_UNCLEAR`: all remaining MIXED rows. Keep original strategy behavior.

Design implication:

- Do not use MIXED sublabels to directly rewrite target position yet.
- First use them as a conservative external risk-score layer:
  - positive labels reduce penalties;
  - negative labels cap execution or slow target approach;
  - no label leaves the core strategy unchanged.
