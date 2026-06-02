# Evaluation Metric Policy

The evaluation layer should be small enough to run often and rich enough to
explain failures.

## Primary Metrics

Keep these in the main review:

- total return;
- Buy & Hold return;
- excess return;
- win rate versus Buy & Hold;
- max drawdown;
- average exposure;
- trade count.

These metrics directly answer whether the strategy earns enough, controls risk,
uses capital as intended, and trades at a realistic frequency.

## Required Scenarios

Every serious candidate should be reviewed in two shapes:

- per-pair fixed allocation for BTC, ETH, and BNB;
- equal-weight aggregate built from those independent sleeves.

Promising candidates should also run rolling windows with multiple start dates
and lengths. Rolling windows are for stability review, not first-pass screening.

## Excluded From Primary Review

Do not keep duplicate or low-actionability metrics in the main report:

- score component dumps;
- skewness, kurtosis, and VaR-style statistics;
- large per-bar diagnostics;
- full action logs for every screening run;
- shared-wallet portfolio comparisons.

Detailed trade logs and diagnostics are useful only when a candidate is worth
investigating or when a specific losing window needs explanation.
