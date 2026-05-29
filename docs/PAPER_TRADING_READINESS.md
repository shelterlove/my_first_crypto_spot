# Paper Trading Readiness

## Candidate

- Strategy: `v2_6`
- Class: `V26Strategy`
- Evaluation run: `results/v1_eval_upgrade/20260529_123908_v2_6`
- Status: ready for paper trading, not ready for live capital

## Why This Version

`v2_6` keeps the stable `v2_4` framework and adds one conservative long-core rule:
when confirmed `MIXED` still has constructive EMA168 structure, sell targets use a
BULL target with a risk penalty. This keeps more core exposure when the long-term
trend is intact, without changing BEAR targets, buy sizing, or trend-break exits.

## Gate Results

| Gate | Requirement | v2_6 |
|---|---:|---:|
| Score | `>= v2_4` | `0.6500` |
| Median excess vs BH | `>= 8%` | `8.50%` |
| Win rate vs BH | `>= 54%` | `54.55%` |
| BTC median excess | `>= 0` | `2.09%` |
| BULL win rate | `>= 35%` | `35.05%` |
| BEAR win rate | `>= 90%` | `94.44%` |
| Mean max drawdown | not worse than `v2_4` by more than 1% | `-37.68%` |
| Turnover | not more than 10% above `v2_4` | `4.51` |
| Simple EMA168 median excess | `>= 0` | `5.54%` |
| Path exposure-matched median excess | `>= 0` | `1.15%` |
| Diagnostic quality | all pass | pass |

Cost stress remained positive:

| Scenario | Mean excess vs BH | Win rate vs BH |
|---|---:|---:|
| base | `27.53%` | `54.55%` |
| realistic | `26.98%` | `53.94%` |
| conservative | `25.32%` | `53.33%` |
| stress | `23.11%` | `52.73%` |

## Known Weakness

- BULL median excess remains negative at `-17.89%`.
- The strategy is still a rule overlay on the target-table framework, not a fully
  separated core/tactical portfolio engine.
- Paper trading should validate signal behavior, execution assumptions, and
  operational logging. It should not be treated as proof of live robustness.

## Paper Trading Rules

- Freeze strategy code and parameters at `v2_6`.
- Generate and archive `strategy_manifest.json` for every evaluation or signal run.
- Use `scripts/generate_daily_signal.py --strategy v2_6` for signal-only review.
- Run paper trading for at least 3 months.
- Do not modify rules during the paper-trading window.
- Review every generated action reason before enabling any live execution.
- Track divergence between expected next-open execution and paper exchange fills.
- Stop the paper test if action frequency materially exceeds backtest behavior.

## Promotion Beyond Paper Trading

Do not use real capital until paper trading confirms:

- signal cadence matches backtest expectations,
- no unexplained state transitions or action reasons occur,
- execution cost and slippage stay within stress assumptions,
- BTC behavior remains acceptable,
- logs are complete enough to audit every action.
