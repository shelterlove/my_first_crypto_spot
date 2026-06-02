# Evaluation Metric Policy

The evaluation layer is split into two tiers.

## Research Mode

Research mode is for screening new candidates. It should stay small and fast.

Keep:

- `model_review.md/json`: primary decision summary.
- `summary_metrics.csv`: score, return, excess return, win rate, drawdown, exposure, trades, turnover.
- `benchmark_metrics.csv`: Buy & Hold, exposure-matched Buy & Hold, simple EMA168 filter.
- `regime_performance_report.csv`: BULL / sideways / BEAR behavior.
- `raw_backtest_results.csv`: window-level audit rows.
- `strategy_manifest.json`, `config_snapshot.json`, `experiment_metadata.json`: reproducibility.

Do not write in research mode:

- Full action logs or equity curves.
- Final score component CSV. It overlaps with `model_review` and `summary_metrics`.
- Risk distribution statistics such as VaR, skewness, and kurtosis.
- Signal attribution, blocked-buy, sell-too-early, or per-bar diagnostics.

## Complete Mode

Complete mode is for candidates that survive research screening.

Keep additional complete-only files:

- `action_logs.csv.gz` and `equity_curves.csv.gz`.
- `early_exposure_report.csv`: first 30/60/120 day exposure and return capture.
- Signal attribution and state transition reports.
- Buy-blocked and sell-too-early diagnostics.
- Timestamp/accounting audits.
- Cost and warmup sensitivity reports.
- HTML report.

## Metric Selection Rules

A metric should remain in the primary review only if it affects a decision:

- Does the strategy beat Buy & Hold often enough? Use median excess and win rate.
- Does it reduce drawdown? Use mean max drawdown and drawdown reduction.
- Is the return retained while reducing risk? Use retention ratio and mean/median return.
- Is it deployable? Use trade count, turnover, and explicit execution assumptions.
- Does it fail in a specific market state? Use regime breakdown.
- Does it miss early trend participation? Use `early_exposure_report.csv` in complete mode.

Metrics that mostly duplicate these should stay out of research output.

## Execution Assumptions

`cost.min_notional` is explicit. The backtest uses normalized capital, so a fixed exchange minimum can materially change strategy behavior. For percentage-style research, keep it low or zero; for deployment rehearsal, set it to the exchange/order-size constraint.
