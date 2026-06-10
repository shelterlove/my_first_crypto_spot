# Risk Signal Design Notes

## Drawdown Risk Anchor

Current `v2_21E` uses a position-cycle `peak_price` to help calculate drawdown risk. This answers a position-management question:

- How much has price fallen from the high reached during the current holding cycle?

This is useful for profit giveback and current-position management, but it should not be treated as the full market-risk signal.

For future design, split risk into separate concepts:

- Market drawdown risk: use market data peaks such as `dd_from_120d_high`, `dd_from_180d_high`, `rolling_365d_pos`, and `price_rank_365d`.
- Position PnL risk: use average entry cost fields such as `avg_cost`, `price_vs_avg_entry_pct`, and `unrealized_pnl_pct`.
- Trend structure risk: use EMA alignment, EMA slopes, BTC regime, Donchian position, and ROC.

Do not let position-cycle `peak_price` alone decide market quality. A better rule should require agreement between market drawdown, position PnL, and trend structure before increasing defensive pressure.

## External Score Layer

The intended architecture is two-layer:

- Core position layer: keeps the existing strategy logic for market state, target allocation, and defensive priority.
- External market-score layer: decides how much of the core target may be executed under current market quality.

The external layer should be conservative. It should usually throttle or delay execution, not force large sells or rewrite the strategy's target position.

## MIXED Label Inputs

Recent train/test research used older years for discovery and the latest two years for out-of-sample checks. The useful MIXED labels should be treated as score inputs, not as direct strategy states.

Positive or penalty-reducing inputs:

- `MIXED_SUPPORTIVE_RECLAIM`: price reclaims EMA72 while long EMA structure is still repairing, with supportive BTC context.
  - This is the strongest positive MIXED structure found so far.
  - Use it to avoid unnecessary suppression of recovery buys.
  - Do not use it by itself to raise the core target.
- `MIXED_VOLUME_CONFIRMED_RECLAIM`: reclaim with volume confirmation.
  - Positive but weaker out of sample.
  - Use as a secondary confirmation only.

Negative or penalty-adding inputs:

- `MIXED_WEAK_RANGE_RISK`: BTC is in weak range, or the asset shows high-volume pullback behavior.
  - BTC weak range was stable negative in both older-year research and recent two-year testing.
  - High-volume pullback was also stable negative.
  - Use this to reduce allowed execution toward target.

Caution-only inputs:

- Pullback above EMA168, top-chase risk, and bear-reclaim-trap patterns may be useful, but they are not clean enough for hard rules yet.
- `BTC_BEAR` alone is not a reliable bearish filter inside MIXED. It needs asset-level failed-reclaim or weak-range confirmation.

## Risk Score Direction

For the next design pass, the score should answer:

- Is the core strategy allowed to move close to its target?
- Should execution be slowed or capped?
- Is the evidence strong enough to require defensive behavior, or only enough to avoid adding risk?

Initial conservative interpretation:

- High score: allow normal execution toward the core target.
- Medium score: allow partial movement toward the target.
- Low score: cap execution and avoid repeated small recovery buys.
- Critical score: only if existing trend-break or risk-reduce conditions agree; do not invent a new forced-sell path from weak score alone.

## Initial Risk-Cap Proposal

The preferred implementation form is an upper bound:

- `final_allowed_position = min(core_target_position, external_risk_cap)`

This is easier to reason about than multiplying the core target by a score, and it reduces path-dependent behavior.

Use these first-pass caps for research:

- `100%`: high-quality MIXED reclaim.
- `75%`: positive reclaim, but not enough evidence for full release.
- `60%`: neutral/default MIXED repair zone.
- `40%`: weak range, pullback, or top-chase caution.
- `25%`: critical weak structure; allow only probe-sized exposure unless the core strategy is already defensive.

Suggested MIXED cap rules:

- `100%` cap:
  - `profile == RECLAIM_EMA72_LONG_DOWN`
  - and `btc_signal == BTC_SUPPORTIVE`
  - Rationale: train/test split was strongly positive and stable.
- `75%` cap:
  - reclaim with weaker confirmation, such as `RECLAIM_EMA72_LONG_DOWN` with `BTC_RANGE_IMPROVING`, or reclaim with `volume_strength >= 1.15`.
  - Rationale: positive but weaker out of sample; use as partial release only.
- `60%` cap:
  - default MIXED.
  - `LOST_EMA168_LONG_UP` also belongs here unless additional weak-range or failed-structure evidence appears.
  - Rationale: lost EMA168 was not reliably bearish in train/test; it is a volatile repair zone rather than an automatic danger state.
- `40%` cap:
  - `BTC_RANGE_WEAK`.
  - `PULLBACK_ABOVE_EMA168`.
  - `ABOVE_EMA72_LONG_UP` with `rolling_365d_pos >= 0.75`.
  - Rationale: these are caution labels. They should slow or cap execution, not force sells.
- `25%` cap:
  - high-volume pullback: `PULLBACK_ABOVE_EMA168` and `volume_strength >= 1.15`.
  - below-EMA168 long-down structure with weak BTC context.
  - weak BTC range plus failed asset structure: `price_vs_ema72 < 0`, `roc_20 < 0`, and `ema24_slope < 0`.
  - Rationale: these are the clearest low-quality recovery or weak-range patterns.

Offline MIXED train/test check:

- In the recent two-year test split, the proposed hierarchy produced:
  - `25%` cap: 67 rows, 60d median return `-6.00%`, positive rate `32.84%`.
  - `40%` cap: 98 rows, 60d median return `-3.32%`, positive rate `39.80%`.
  - `60%` cap: 350 rows, 60d median return `+2.39%`, positive rate `56.00%`.
  - `100%` cap: 85 rows, 60d median return `+44.40%`, positive rate `94.12%`.
- This is not a backtest result. It only says the cap labels have useful out-of-sample ordering on MIXED bars.

Important limitation:

- The cap layer will miss some very early bottoms, especially when price is still below long EMAs.
- That is acceptable for this layer because its purpose is risk control, not bottom-fishing.
- If early-bottom capture is desired later, it should be a separate small-probe rule, not a reason to raise the general risk cap.

## Optimized MIXED Cap Research, 2026-06-09

Run:

- `python scripts\analyze_external_risk_cap_design.py --run-id external_risk_cap_design_mixed_oos2y_20260609`
- Output: `results/diagnostics/external_risk_cap_design_mixed_oos2y_20260609`

Research split:

- Train: `2019-12-31 16:00 UTC` to `2023-12-24 16:00 UTC`, 838 MIXED rows.
- Test: `2024-05-01 16:00 UTC` to `2026-02-01 16:00 UTC`, 600 MIXED rows.

Key optimization:

- Do not reward generic trend strength inside MIXED.
- `ABOVE_EMA72_LONG_UP` and `LOST_EMA168_LONG_UP + BTC_SUPPORTIVE` looked good in older rows but failed badly in the recent two-year test.
- Therefore, high caps should be reserved for strict reclaim repair, not for all apparently strong MIXED structures.

Optimized cap rules:

- `100%` full release:
  - `profile == RECLAIM_EMA72_LONG_DOWN`
  - and `btc_signal == BTC_SUPPORTIVE`
  - This is the only full-release condition.
- `75%` partial release:
  - `RECLAIM_EMA72_LONG_DOWN` with `BTC_RANGE_IMPROVING` and no sharp negative `roc_20`.
  - or reclaim with `volume_strength >= 1.15`, unless BTC is bear/weak range.
  - Treat this as provisional because the latest two-year split had no 75% samples.
- `60%` default:
  - default MIXED repair zone.
  - most `LOST_EMA168_LONG_UP` cases remain here, not lower, because lost EMA168 was not reliably bearish by itself.
- `40%` caution:
  - `BTC_RANGE_WEAK`.
  - `PULLBACK_ABOVE_EMA168`.
  - `ABOVE_EMA72_LONG_UP` with high historical rank, low Donchian position, or BTC bear context.
  - `LOST_EMA168_LONG_UP + BTC_SUPPORTIVE` only when price is still below EMA72, volume is high, or ATR rank is high.
- `25%` critical:
  - pullback with high volume, sharp negative `roc_20`, or weak BTC range.
  - below-EMA168 long-down structure with BTC bear or weak range.
  - weak BTC range plus failed asset structure: below EMA72, negative `roc_20`, and negative EMA24 slope.
  - BTC bear rally trap: BTC bear, asset at or below mid historical rank, and asset price above EMA72.

Optimized train/test check:

- Train:
  - `25%`: 209 rows, 60d median return `-8.28%`, positive rate `40.19%`, down60 median `-19.95%`.
  - `40%`: 95 rows, 60d median return `+17.82%`, positive rate `66.32%`, down60 median `-12.04%`.
  - `60%`: 336 rows, 60d median return `+31.31%`, positive rate `75.00%`, down60 median `-7.50%`.
  - `75%`: 57 rows, 60d median return `+7.39%`, positive rate `63.16%`, down60 median `-11.10%`.
  - `100%`: 141 rows, 60d median return `+23.13%`, positive rate `90.78%`, down60 median `-8.92%`.
- Test:
  - `25%`: 91 rows, 60d median return `-6.00%`, positive rate `31.87%`, down60 median `-14.74%`.
  - `40%`: 121 rows, 60d median return `-5.89%`, positive rate `35.54%`, down60 median `-17.46%`.
  - `60%`: 303 rows, 60d median return `+3.90%`, positive rate `61.06%`, down60 median `-9.18%`.
  - `100%`: 85 rows, 60d median return `+44.40%`, positive rate `94.12%`, down60 median `-7.24%`.

Interpretation:

- The optimized design gives useful out-of-sample ordering:
  - `25%` and `40%` are weak/risky in the recent two-year test.
  - `60%` is neutral-to-positive.
  - `100%` is clearly positive.
- The `40%` bucket was positive in train but negative in test. This is acceptable for a caution bucket because the rule is not meant to identify shorts; it is meant to prevent full release when evidence is unstable.
- The `75%` bucket needs more validation because it had train samples but no recent-test samples.

Current best research version:

- Use `100%`, `60%`, `40%`, and `25%` as the primary tested caps.
- Keep `75%` in the design, but do not rely on it for promotion until more windows confirm it.
- If this becomes a candidate strategy, start with buy-throttle behavior only. Do not force sells from the cap layer.

75% cap validation note:

- The latest two-year test split had no independent `75%` cap samples.
- Test split reclaim rows:
  - `RECLAIM_EMA72_LONG_DOWN`: 85 rows.
  - `RECLAIM_EMA72_LONG_DOWN + BTC_SUPPORTIVE`: 85 rows.
  - `RECLAIM_EMA72_LONG_DOWN + BTC_RANGE_IMPROVING`: 0 rows.
  - `RECLAIM_EMA72_LONG_DOWN + BTC_BEAR`: 0 rows.
  - `RECLAIM_EMA72_LONG_DOWN + BTC_RANGE_WEAK`: 0 rows.
- There were 24 reclaim rows with `volume_strength >= 1.15`, but all were also `BTC_SUPPORTIVE`, so they were correctly upgraded to the `100%` cap by priority.
- Therefore, `75%` should remain a conceptual/provisional bucket only.
- For the first strategy candidate, prefer a four-bucket implementation:
  - `100%`: strict supportive reclaim.
  - `60%`: default MIXED.
  - `40%`: caution.
  - `25%`: critical.

## First-Step Drawdown Anchor Test, 2026-06-09

Candidate:

- `v2_35B`
- Baseline: `v2_21E`
- Scope: first-step risk anchor only.
- No MIXED sublabel cap, no external position cap, no sell-forcing overlay.

Change:

- Replace position-cycle `peak_price` drawdown risk with market drawdown anchors.
- Keep `avg_cost` as the position PnL reference.
- Drawdown risk now requires:
  - active position and valid average cost;
  - unrealized profit above threshold;
  - market drawdown from 120d/180d high above threshold;
  - price below EMA24/EMA72.

Rules tested:

- Risk 2:
  - `profit_pct > 30%`
  - and `max(dd_from_120d_high, dd_from_180d_high) > 25%`
  - and `price < ema72`
- Risk 1:
  - `profit_pct > 20%`
  - and `max(dd_from_120d_high, dd_from_180d_high) > 15%`
  - and `price < ema24`

Lightweight test:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_21E --candidate v2_35B --years 3 --run-id recent3y_v2_35B_vs_v2_21E_drawdown_anchor_20260609`
- Output: `results/diagnostics/recent3y_v2_35B_vs_v2_21E_drawdown_anchor_20260609`
- Window: `2023-05-23 16:00 UTC` to `2026-05-22 16:00 UTC`.

Result:

- BTC:
  - return delta `+0.073683`
  - max drawdown delta `+0.007979`
  - trade count delta `-3`
- ETH:
  - return delta `-0.084120`
  - max drawdown delta `0.000000`
  - trade count delta `+2`
- BNB:
  - return delta `+0.009703`
  - max drawdown delta approximately `0`
  - trade count delta `0`
- Total return delta: `-0.000735`.
- Worse drawdown windows: `0`.

Interpretation:

- This is materially better than the external-cap test because it does not damage all symbols.
- The direction is not promotable yet because ETH degradation is too large.
- The promising part is BTC: replacing position peak with market drawdown avoided some late-risk behavior and improved return/drawdown.
- The failure mode is ETH 2025-11 path churn: changed drawdown risk shifted target-reduce/safe-recovery timing and reduced final return without improving drawdown.

Next design implication:

- First-step risk anchor is worth further analysis, but not with a blanket replacement.
- A safer next candidate should only replace `peak_price` drawdown risk when the position is not in a fresh recovery path, or should require stronger market confirmation before increasing risk.
- Do not combine this with external cap until the single-step variant is stable.

## Layered Giveback Cap Refactor Test, 2026-06-09

Candidate:

- `v2_36A`
- Baseline: `v2_21E`
- Goal: test structural separation of profit-giveback drawdown from core `risk_score`.

Implementation:

- Core `drawdown_risk` returns `0`.
- Original 21E profit-giveback conditions are moved into an independent target cap layer.
- Cap form:
  - `target = min(core_target, lookup_target(raw_state, trend_risk + giveback_risk))`
- The cap still uses the original position-cycle `peak_price` logic to isolate the effect of layering from the effect of changing the risk anchor.

Recent 3Y smoke:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_21E --candidate v2_36A --years 3 --run-id recent3y_v2_36A_vs_v2_21E_layered_giveback_cap_20260609`
- Output: `results/diagnostics/recent3y_v2_36A_vs_v2_21E_layered_giveback_cap_20260609`
- Result:
  - BTC return delta `-0.034641`, drawdown delta `-0.002698`.
  - ETH return delta `+0.014132`, drawdown delta `-0.029081`.
  - BNB return delta `-0.050795`, drawdown delta `-0.017254`.
  - Return delta sum `-0.071304`; all three symbols had worse drawdown.

Complete rolling:

- Command: `python scripts\run_v1_candidate_workflow.py --baseline v2_21E --candidate v2_36A --stage complete`
- Output: `results/v1_candidate_workflow/20260609_112119_v2_36A_vs_v2_21E_complete`
- Windows: `165`.
- Changed windows: `105`.
- Negative return deltas: `62`.
- Worse drawdown deltas: `56`.
- Trade delta sum: `-61`.
- Return delta sum: `-13.034812`.

By symbol:

- BNB:
  - return delta sum `-7.081945`.
  - negative return windows `20`.
  - worse drawdown windows `9`.
- BTC:
  - return delta sum `-3.976280`.
  - negative return windows `17`.
  - worse drawdown windows `24`.
- ETH:
  - return delta sum `-1.976586`.
  - negative return windows `25`.
  - worse drawdown windows `23`.

Interpretation:

- This refactor is not behavior-preserving.
- Removing giveback risk from core `risk_score` changes recovery, cooldown, and target-reduce sequencing even though the same giveback condition is reintroduced as a cap.
- The damage is broad and concentrated in important trend windows, especially 2020-2022 BNB/BTC.
- The experiment confirms the architectural problem, but `v2_36A` is not a usable candidate.

Decision:

- Reject `v2_36A`.
- Do not move 21E's existing profit-giveback drawdown out of core `risk_score` as a blanket change.
- Any future refactor must first prove behavior parity with `v2_21E` before changing semantics.
- A safer path is to add new external caps alongside the existing core risk, not to remove existing drawdown risk from the core.

## Behavior-Equivalent Layer Scaffold, 2026-06-09

Candidate:

- `v2_36B`
- Baseline: `v2_21E`
- Goal: prove a target-layering scaffold can be introduced without changing strategy behavior.

Implementation:

- Keep core 21E behavior unchanged.
- Keep `drawdown_risk` inside core `risk_score`.
- Wrap `_compose_target` into:
  - `_core_layer_target(...)`
  - `_risk_cap_layer(...)`
- `_risk_cap_layer(...)` currently returns `_target_cap()`, so it should not constrain the core target.
- Final target remains:
  - `min(core_layer_target, risk_cap_layer)`

Recent 3Y equivalence check:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_21E --candidate v2_36B --years 3 --run-id recent3y_v2_36B_vs_v2_21E_equivalent_layer_scaffold_20260609`
- Output: `results/diagnostics/recent3y_v2_36B_vs_v2_21E_equivalent_layer_scaffold_20260609`
- BTC/ETH/BNB return deltas: all `0.000000`.
- BTC/ETH/BNB max drawdown deltas: all `0.000000`.
- Trade count deltas: all `0`.

Complete rolling:

- Command: `python scripts\run_v1_candidate_workflow.py --baseline v2_21E --candidate v2_36B --stage complete`
- Output: `results/v1_candidate_workflow/20260609_123019_v2_36B_vs_v2_21E_complete`
- Windows: `165`.
- Changed windows: `0`.
- Negative return deltas: `0`.
- Worse drawdown deltas: `0`.
- Trade delta sum: `0`.
- Return delta sum: `0.000000`.

Decision:

- `v2_36B` is a valid behavior-equivalent scaffold.
- Use this as the base for future layered experiments.
- Future changes should modify `_risk_cap_layer(...)` only, while keeping core `risk_score` untouched unless a separate experiment explicitly targets core behavior.

## Behavior-Equivalent Target-Band Pipeline, 2026-06-09

Candidate:

- `v2_36C`
- Baseline: `v2_21E`
- Goal: refactor the full 21E action flow into explicit target-band and execution steps without changing behavior.

Implementation:

- Inherits the `v2_36B` target-layer scaffold.
- Keeps all 21E semantics unchanged.
- Splits the main action flow into:
  - `_prepare_position_context(...)`
  - `_build_market_context(...)`
  - `_build_signal_context(...)`
  - `_build_target_band(...)`
  - `_maybe_sell(...)`
  - `_maybe_buy(...)`
- Makes the implicit 21E target interval explicit:
  - `sell_boundary`
  - `buy_boundary`
- Keeps sell-before-buy ordering.
- Preserves V24/V221E path state side effects:
  - `_latest_bar`
  - `_current_price`
  - `_last_sell_price`
  - `_last_recovery_buy_call`
- Keeps `drawdown_risk` inside core `risk_score`; no semantic migration.

Recent 3Y equivalence check:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_21E --candidate v2_36C --years 3 --run-id recent3y_v2_36C_vs_v2_21E_target_band_refactor_20260609`
- Output: `results/diagnostics/recent3y_v2_36C_vs_v2_21E_target_band_refactor_20260609`
- BTC/ETH/BNB return deltas: all `0.000000`.
- BTC/ETH/BNB max drawdown deltas: all `0.000000`.
- Trade count deltas: all `0`.

Complete rolling:

- Command: `python scripts\run_v1_candidate_workflow.py --baseline v2_21E --candidate v2_36C --stage complete`
- Output: `results/v1_candidate_workflow/20260609_134550_v2_36C_vs_v2_21E_complete`
- Windows: `165`.
- Changed windows: `0`.
- Negative return deltas: `0`.
- Worse drawdown deltas: `0`.
- Trade delta sum: `0`.
- Return delta sum: `0.000000`.

Decision:

- `v2_36C` is the preferred behavior-equivalent refactor base.
- Future work should start from `v2_36C`, not `v2_36A`.
- Next experiments should only add constraints to `_risk_cap_layer(...)` or execution-specific helpers while keeping the target-band pipeline intact.

## Post Safe-Recovery Target-Reduce Block, 2026-06-09

Goal:

- Test whether the frequent `safe-recovery buy -> short-term target-reduce sell` churn can be reduced without weakening true risk exits.
- Baseline: `v2_36C`.

Candidates:

- `v2_36D`: block ordinary `target-reduce` for 3 calls after a `safe-recovery` buy, but never block BEAR, BTC BEAR, `drawdown_risk > 0`, `risk_score >= 3`, or `trend_risk > 1`.
- `v2_36E`: same as `v2_36D`, plus `volume_strength >= 0.75`.

Historical 5Y excluding recent 2Y:

- Window: `2019-05-23` to `2024-05-22`.
- `v2_36D` output: `results/diagnostics/hist5y_ex_recent2y_v2_36D_vs_v2_36C_recovery_trim_block_20260609`.
- `v2_36D` deltas vs `v2_36C`:
  - BTC: `+0.021450`, drawdown unchanged, trades `-2`.
  - ETH: `+0.045162`, drawdown unchanged, trades unchanged.
  - BNB: `-0.410426`, drawdown unchanged, trades `-2`.
  - Return delta sum: `-0.343815`.
- `v2_36D` failed because it protected BNB 2023-05-29 target-reduce in a weak-volume recovery, delaying the sell from `311.7` to `306.8` and removing a later 2023-06-10 target-gap buy.

- `v2_36E` output: `results/diagnostics/hist5y_ex_recent2y_v2_36E_vs_v2_36C_recovery_trim_block_volume_20260609`.
- `v2_36E` deltas vs `v2_36C`:
  - BTC: `+0.021450`, drawdown unchanged, trades `-2`.
  - ETH: `+0.045162`, drawdown unchanged, trades unchanged.
  - BNB: `0.000000`, drawdown unchanged, trades unchanged.
  - Return delta sum: `+0.066611`.
- `v2_36E` changed only:
  - Removed BTC 2023-06-18 `target-reduce` and 2023-06-19 `safe-recovery` buy-back churn.
  - Delayed ETH 2024-05-14 `target-reduce` to 2024-05-16 at a better price.

Decision:

- `v2_36D` is rejected; it proves a plain recent-recovery block is too broad.
- `v2_36E` is a small positive smoke result, not enough for promotion.
- The useful signal is not "recent safe-recovery" alone; it needs at least a basic recovery-quality gate such as volume strength.
- Next test should keep the protection narrow and compare against recent holdout / rolling windows before considering any broader target-reduce smoothing.

## Target-Reduce Sell-Fly Diagnostics, 2026-06-09

Goal:

- Focus the MIXED score research on the main objective: avoiding `target-reduce` exits that sell too early before a recovery rally.
- Use action-level events from `v2_36C`, not generic MIXED bars.

Historical 5Y excluding recent 2Y:

- Source actions: `results/diagnostics/hist5y_ex_recent2y_v2_36C_vs_v2_21E_20260609/actions.csv`.
- Diagnostic output: `results/diagnostics/target_reduce_mixed_score_hist5y_ex_recent2y_20260609`.
- `target-reduce` events: `42`.
- Sell-fly labels: `17` bad, `13` severe.

Important finding:

- Existing MIXED external-cap labels are useful for buy-side risk, but they do not directly solve sell-fly.
- Most sell-fly events landed in the default `60%` cap bucket:
  - `DEFAULT_MIXED_CAP60`: 25 events, 14 bad, 11 severe, 60d median return `+46.4%`.
  - `SUPPORTIVE_RECLAIM_CAP100`: 1 event, 1 bad.
  - `CAUTION_CAP40` and `CRITICAL_CAP25` mostly contained correct defensive sells.
- Therefore, reusing the buy-side cap labels as a sell-protection rule is not enough.

Best current-bar sell-fly feature:

- `ema168_slope < 0` during `target-reduce`:
  - Historical 5Y ex recent 2Y: 13 hits, 11 bad, 9 severe, 2 good, 60d median return `+58.6%`.
  - This usually overlaps `LOST_EMA168_LONG_UP`, price below EMA72/EMA168, and `risk_score == 2`, `drawdown_risk == 0`.
- Interpretation: many sell-fly cases are not clean bullish reclaims. They are early recovery / capitulation-rebound zones where the long trend is still technically down, so the core target stays low and `target-reduce` trims too early.

Recent 2Y check:

- Source actions: `results/diagnostics/recent2y_v2_36C_action_log_20260609/actions.csv`.
- Diagnostic output: `results/diagnostics/target_reduce_mixed_score_recent2y_20260609`.
- `target-reduce` events: 27.
- Sell-fly labels: 1 bad, 0 severe.
- `ema168_slope < 0`: 3 hits, 1 bad, 0 severe.
- Conclusion: `ema168_slope < 0` is not stable enough for a hard target-reduce block.

Design implication:

- Do not hard-disable `target-reduce` from a MIXED score alone.
- The next candidate should test softening only:
  - apply only to ordinary `target-reduce`;
  - never affect `risk-reduce`, `trend-break`, or BEAR;
  - prefer raising the sell threshold or reducing `max_sell`, not blocking the sell entirely;
  - require `drawdown_risk == 0` so profit-giveback defense remains active.
- Candidate shape to test later:
  - `raw_state == MIXED and confirmed_state == MIXED`;
  - `sell_setup == target-reduce`;
  - `ema168_slope < 0`;
  - `drawdown_risk == 0`;
  - soften sell size/threshold instead of forbidding the action.

## V2.36F Mixed Early-Repair Target-Reduce Softening, 2026-06-09

Candidate:

- `v2_36F`
- Baseline: `v2_36C`
- Implements the action-level sell-fly idea from the previous diagnostic.

Rule:

- Applies only inside `_adjust_sell_execution(...)`.
- Conditions:
  - `sell_setup == target-reduce`;
  - `raw_state == MIXED`;
  - `confirmed_state == MIXED`;
  - `drawdown_risk == 0`;
  - `ema168_slope < 0`.
- Effect:
  - `sell_threshold >= 12%`;
  - `max_sell <= 8%`;
  - does not affect `risk-reduce`, `trend-break`, BEAR, or profit-giveback drawdown defense.

Historical 5Y excluding recent 2Y:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_36C --candidate v2_36F --years 5 --end 2024-05-22 --run-id hist5y_ex_recent2y_v2_36F_vs_v2_36C_mixed_early_repair_trim_20260609`
- Output: `results/diagnostics/hist5y_ex_recent2y_v2_36F_vs_v2_36C_mixed_early_repair_trim_20260609`
- Deltas:
  - BTC: `+0.145761`, drawdown unchanged, trades `+4`.
  - ETH: `+1.097706`, drawdown unchanged, trades unchanged.
  - BNB: `+1.145075`, drawdown unchanged, trades `+3`.
  - Return delta sum: `+2.388542`.
- Main changed behavior:
  - 2020-03 crash-rebound target-reduce sells are cut roughly in half rather than executed as larger immediate trims.
  - BTC 2023-09-28 target-reduce of about `102` notional becomes three smaller sells around `33` each before the safe-recovery buy.
  - BNB 2020-07 target-reduce cluster is split into smaller daily trims, reducing low-area sell pressure before the rally.
- This matches the intended behavior: soften ordinary trims in early-repair MIXED rather than blocking exits.

Recent 2Y smoke:

- Command: `python scripts\run_recent3y_candidate_test.py --baseline v2_36C --candidate v2_36F --years 2 --run-id recent2y_v2_36F_vs_v2_36C_mixed_early_repair_trim_20260609`
- Output: `results/diagnostics/recent2y_v2_36F_vs_v2_36C_mixed_early_repair_trim_20260609`
- Deltas:
  - BTC: `-0.001792`, drawdown slightly better by `+0.000006`, trades `+3`.
  - ETH: `0.000000`, drawdown unchanged, trades unchanged.
  - BNB: `+0.020843`, drawdown unchanged, trades `-1`.
  - Return delta sum: `+0.019052`.
- Main recent behavior:
  - BTC 2025-04 softening slightly hurts.
  - BNB 2024-06 and 2025-02 softening helps.
  - No drawdown degradation.

Decision:

- `v2_36F` is the first sell-fly candidate with a meaningful positive historical smoke result and a non-negative recent-2Y smoke.
- It should not be promoted yet.
- Next validation should be rolling quick/standard, with special attention to:
  - whether extra small trims create unacceptable churn;
  - whether BTC 2025-04-like cases repeat;
  - whether the improvement is overly concentrated in 2020 crash-rebound paths.

## Sell-Fly Score Iteration, 2026-06-09

Purpose:

- Refine the broad `v2_36F` sell-fly rule before paying the cost of rolling tests.
- Validate action-level scores on `target-reduce` events:
  - Research set: 5Y excluding recent 2Y.
  - Test set: recent 2Y.

Diagnostic output:

- `results/diagnostics/sellfly_score_design_20260609`
- `sellfly_score_train_events_iter2.csv`
- `sellfly_score_test_events_iter2.csv`

Key findings:

- A simple additive score overfits.
  - The first additive score put 26/42 historical events into the high bucket, including many correct sells.
  - In recent 2Y it put 11 events in the high bucket but captured no sell-fly events.
- The useful structure is categorical, not linear:
  - panic/capitulation repair;
  - strict early repair;
  - supportive recovery.
- The only clean and stable category is panic/capitulation repair.

Best diagnostic rules:

- `panic`:
  - `target-reduce`;
  - `MIXED/MIXED`;
  - `drawdown_risk == 0`;
  - `ema168_slope < 0`;
  - `volume_strength >= 1.15`;
  - `roc_20 <= -0.20`;
  - `donchian_pos >= 0.45`.
  - Train: 8 hits, 7 bad, 5 severe, 1 good.
  - Recent 2Y: 0 hits.
- `strict early repair`:
  - `ema168_slope <= -0.0019` with breakdown vetoes.
  - Train quality is good, but real-path tests still caused small post-2020 negatives.
- `supportive recovery`:
  - High precision in the static event table, but no additional real-path effect in the tested strategy candidate.

Candidate results:

- `v2_36G`: categorical score with panic strong, strict early-repair medium, supportive light.
  - 5Y ex recent 2Y: return delta sum `+2.319773`, drawdown unchanged.
  - 4Y ending 2024-05-22: return delta sum `-0.024195`.
  - 3Y ending 2024-05-22: return delta sum `-0.013487`.
  - 2Y ending 2024-05-22: return delta sum `-0.004685`.
  - Recent 2Y: return delta sum `-0.000428`.
  - Decision: better than `v2_36F` but still not clean enough; medium score triggered unstable BTC 2023/2025 trims.
- `v2_36H`: panic-only sell-fly softening.
  - 5Y ex recent 2Y:
    - BTC `+0.147105`;
    - ETH `-0.006637`;
    - BNB `+0.979333`;
    - sum `+1.119801`;
    - drawdown unchanged.
  - 4Y ending 2024-05-22: all deltas `0`.
  - Recent 2Y: all deltas `0`.
  - Decision: clean conservative module, but narrow and entirely driven by early-2020 panic repair.
- `v2_36I`: panic plus supportive light softening.
  - Same result as `v2_36H`; supportive light did not create additional real-path changes.

Current conclusion:

- The acceptable score boundary is panic-only.
- This is not a broad sell-fly solution; it is a conservative crash-repair protection.
- Do not run rolling for `v2_36F` or `v2_36G`.
- `v2_36H` is the only candidate worth optional rolling, but only if the goal is to add a narrow panic-repair guard.
- To build a broader sell-fly score, more evidence is needed for non-panic cases such as BTC 2023-09/10 and BNB 2023-11; current rules are not robust enough.
