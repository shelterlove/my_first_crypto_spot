# Evaluation Metric Policy

The evaluation standard is fixed around long-term robustness, not recent
performance. A short recent window can explain current behavior, but it must not
promote or reject a strategy by itself.

## Evaluation Tiers

Use three tiers for every serious candidate.

### Data Split Discipline

Strategy iteration must use fixed time splits to reduce researcher overfitting.

| Split | Timerange | Purpose |
| --- | --- | --- |
| `dev` | `20200101-20231231` | Strategy design and trade-level diagnostics |
| `validation` | `20240101-20250531` | Candidate selection without trade-level rule tuning |
| `holdout` | `20250601-20260601` | Final review only; do not use for strategy design |

Use `scripts/freqtrade_eval.py --eval-split <split>` to apply these ranges.
Do not inspect holdout trades to design new rules. Holdout is only for a
candidate that already passed development and validation checks.

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

Primary reports must show both the equal-weight aggregate and the individual
BTC/ETH/BNB sleeves. Do not judge a candidate from the aggregate alone.

### Decision Score

Use `scripts/review_freqtrade_eval.py` after a serious candidate has baseline
and rolling results.

The score is a numeric summary, not a promotion shortcut. Promotion still
requires the hard checks below to pass. Every sub-score is clamped to `[0, 100]`;
values better than the full-score target are capped at `100`, and values worse
than the fail target are floored at `0`.

| Component | Weight | Purpose |
| --- | ---: | --- |
| `return_score` | 25 | Full-period excess, relative Buy & Hold return, pair coverage, weakest pair, CAGR |
| `robustness_score` | 25 | Rolling median excess, rolling win rate, worst rolling excess, pair medians |
| `risk_score` | 20 | Absolute drawdown, per-pair drawdown, rolling drawdown, underwater days, relative risk |
| `risk_adjusted_score` | 20 | Sharpe, Sortino, Calmar versus absolute targets and Buy & Hold |
| `behavior_score` | 10 | Trade frequency, average exposure, per-pair trade balance |

Manual logic review is no longer part of the numeric score. It is handled as two
hard checks:

- rule generality: the rule must not contain symbol-specific thresholds or fit
  only one historical segment;
- defense integrity: BEAR exits, trend-break exits, and risk-reduce exits must
  not be weakened without explicit evidence.

Component formulas in `scripts/review_freqtrade_eval.py`:

- `return_score`
  - `excess_score = linear_score(aggregate_excess_pp, fail=-50, full=200)`
  - if Buy & Hold return is positive:
    `relative_score = linear_score(strategy_return / buyhold_return, fail=0.80, full=1.20)`
  - if Buy & Hold return is not positive:
    `relative_score = linear_score(aggregate_excess_pp, fail=0, full=100)`
  - `pair_positive = percent of BTC/ETH/BNB with positive full-period excess`
  - `weakest_pair_score = linear_score(min_pair_excess_pp, fail=-50, full=100)`
  - if Buy & Hold CAGR is positive:
    `cagr_score = linear_score(strategy_cagr / buyhold_cagr, fail=0.80, full=1.20)`
  - if Buy & Hold CAGR is not positive:
    `cagr_score = linear_score(strategy_cagr - buyhold_cagr, fail=0, full=30pp)`
  - final: `0.30 * excess_score + 0.25 * relative_score + 0.20 * pair_positive + 0.15 * weakest_pair_score + 0.10 * cagr_score`
- `robustness_score`
  - `median_score = linear_score(rolling_median_excess_pp, fail=-40, full=10)`
  - `win_score = linear_score(rolling_win_rate_pct, fail=35, full=55)`
  - `worst_score = linear_score(worst_rolling_excess_pp, fail=-300, full=-100)`
  - `pair_score = mean(linear_score(pair_median_excess_pp, fail=-40, full=10))`
  - final: `0.35 * median_score + 0.25 * win_score + 0.25 * worst_score + 0.15 * pair_score`
- `risk_score`
  - aggregate drawdown full target: `<= 40%`; fail target: `>= 70%`
  - worst single-pair drawdown full target: `<= 45%`; fail target: `>= 75%`
  - worst rolling drawdown full target: `<= 45%`; fail target: `>= 75%`
  - underwater days full target: `<= 500`; fail target: `>= 1200`
  - relative drawdown full target: `strategy_dd / buyhold_dd <= 0.60`; fail target: `>= 1.00`
  - relative underwater full target: `strategy_underwater / buyhold_underwater <= 0.70`; fail target: `>= 1.20`
  - final: `0.30 * aggregate_dd + 0.20 * pair_dd + 0.20 * rolling_dd + 0.15 * underwater + 0.15 * relative_risk`
- `risk_adjusted_score`
  - absolute targets for this long-only spot strategy:
    - Calmar: fail `0.0`, full `1.20`
    - Sortino: fail `0.0`, full `1.80`
    - Sharpe: fail `0.0`, full `1.20`
  - relative targets versus Buy & Hold:
    - fail if the ratio is `0.80`
    - full score if the ratio is `1.20`
  - final: `0.60 * absolute_risk_adjusted + 0.40 * relative_to_buyhold`
- `behavior_score`
  - trade frequency full score: `1` to `12` trades per pair per year
  - if below `1`, linearly score from `0` to `1`
  - if above `12`, linearly decay to zero at `20`
  - exposure full score: `50%` to `70%`, linearly decays to zero at `35%` and `85%`
  - per-pair trade balance full score: max/min trade count `<= 2`; fail at `>= 4`
  - final: `0.45 * trade_frequency + 0.35 * exposure + 0.20 * trade_balance`

`linear_score(value, fail, full)` means a linear score from `0` at `fail` to
`100` at `full`, clamped to `[0, 100]`. For lower-is-better metrics, the
direction is reversed with the same clamp.

`excess_return_pct` is measured in percentage points:

```text
strategy_return_pct - buy_hold_return_pct
```

Therefore a worst excess below `-100` means the strategy lagged Buy & Hold by
more than 100 percentage points. It does not mean the strategy lost more than
100%.

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
- `review/score.json`, `review/score_components.csv`,
  `review/promotion_checks.csv`, and `review/report.html` for promoted
  candidates or serious contenders.

Exclude from primary reports:

- score component dumps;
- skewness, kurtosis, and VaR-style statistics;
- large per-bar diagnostics;
- full action logs for every screening run;
- shared-wallet portfolio comparisons.

Detailed diagnostics are useful only after a candidate is worth investigating or
when a specific losing window needs explanation.
