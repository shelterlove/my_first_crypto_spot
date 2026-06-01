# Strategy Iteration Notes

## 2026-06-02

Current reference candidate: `v2_12A`.

Main diagnosis from the complete `v2_12A` run:

- BULL underperformance is concentrated in windows that are underexposed early.
- In BULL loser windows, average exposure is lower in the first 30/60/120 days, but higher late in the window.
- This means the main drag is not a simple lack of long-term exposure. It is delayed participation near the start of major uptrends.
- The delay is often caused by BEAR-to-MIXED confirmation and BEAR `max_buy=0.05`, where the intended small buy can fall below the minimum trade notional.

Tested and rejected:

- `v2_16A/B/C`: limited starter targets during early reversal. No material effect because the existing MIXED target is already high once MIXED is confirmed.
- `v2_17A/B/C`: globally faster MIXED confirmation. Hurt screening score, indicating the existing MIXED confirmation filters useful false recoveries.
- `v2_18A/B/C`: conditional fast BEAR-to-MIXED confirmation on strong reversal. Did not improve screening score, so the earlier entry cost outweighed the captured upside.

Conclusion:

- Do not promote earlier BEAR-to-MIXED entry rules without stronger evidence.
- The next useful work should focus on diagnostics, not new rules: identify whether early underexposure happens mostly after true cycle lows, after mid-cycle corrections, or because of the minimum notional / BEAR buy-size interaction.
