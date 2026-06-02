# Evaluation Metric Policy

The evaluation standard is fixed around long-term robustness, not recent
performance. A short recent window can explain current behavior, but it must not
promote or reject a strategy by itself.

## Evaluation Tiers

Use three tiers for every serious candidate.

### 1. Primary Evaluation

Primary evaluation decides whether a strategy is worth keeping.

Required scenarios:

- per-pair fixed allocation for `BTC/USDT`, `ETH/USDT`, and `BNB/USDT`;
- equal-weight fixed-allocation aggregate built from those three independent
  sleeves;
- rolling windows with multiple start dates for candidates that pass the first
  full-period screen.

Primary metrics:

- total return;
- Buy & Hold return;
- excess return;
- rolling median excess;
- rolling win rate versus Buy & Hold;
- worst rolling excess;
- max drawdown;
- average exposure;
- trade count.

### 2. Risk Evaluation

Risk evaluation checks that improved return is not coming from unacceptable
damage elsewhere.

Keep:

- aggregate max drawdown;
- per-pair max drawdown;
- worst rolling window;
- downside-window behavior;
- exposure during clear downtrends;
- abnormal trade-count increases.

Do not promote a strategy that improves mean return while materially worsening
the worst windows or drawdown profile.

### 3. Behavior Evaluation

Behavior evaluation checks whether the trading logic is defensible before
looking at the result.

Review trade logs for:

- delayed entry after durable trend recovery;
- repeated sell-then-rebuy churn;
- target-reduce sells during normal bull-market pullbacks;
- risk-reduce and trend-break sells during genuine structure damage;
- long cash periods while the long-term structure is improving.

Rules must stay asset-general. Do not add BTC-, ETH-, or BNB-specific logic just
because one symbol is dragging a backtest.

## Promotion Rules

Compare every candidate against the current reference strategy, not against an
absolute score.

A candidate can replace the reference only if most of these are true:

- full-period aggregate excess is not worse than the reference;
- rolling median excess improves;
- rolling win rate does not decline materially;
- worst rolling excess does not worsen materially;
- aggregate max drawdown does not worsen by more than about 2-3 percentage
  points;
- no single pair shows a large unexplained deterioration;
- trade count stays in the same broad regime;
- behavior review confirms the rule is logically consistent with the long-term
  strategy philosophy.

If a candidate improves recent performance but fails these checks, keep it as an
experiment, not as the new reference.

## Rejection Rules

Reject or keep as research-only if any of these happen:

- improvement is concentrated in one symbol only;
- results depend mainly on one short recent period;
- worst rolling windows worsen substantially;
- max drawdown worsens without a clear compensating stability gain;
- the rule causes repeated sell-then-rebuy churn;
- the rule weakens BEAR exits or trend-break protection without strong evidence;
- the explanation would not make sense before seeing the backtest result.

## Recent Window Policy

Recent-window performance is a diagnostic only.

Use it to answer:

- whether the strategy behaves sensibly in the current market regime;
- whether dry-run monitoring should expect more cash, more exposure, or more
  trades;
- whether recent losses are caused by reasonable defense or by avoidable churn.

Do not use it as a promotion criterion. A strategy should not be promoted
because the latest 3-6 months look good, and it should not be rejected only
because the latest 3-6 months look bad.

## Output Discipline

Keep primary reports compact. The main report should include only metrics that
affect a decision.

Keep in primary reports:

- `summary.csv` / `summary.json`;
- `report.md`;
- `trades.csv` for full-period runs;
- `rolling_summary.csv` and `rolling_detail.csv` for rolling runs;
- aggregate equity curve.

Exclude from primary reports:

- score component dumps;
- skewness, kurtosis, and VaR-style statistics;
- large per-bar diagnostics;
- full action logs for every screening run;
- shared-wallet portfolio comparisons.

Detailed diagnostics are useful only after a candidate is worth investigating or
when a specific losing window needs explanation.
