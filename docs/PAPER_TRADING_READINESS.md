# Paper Trading Readiness

## Current Candidate

- Native strategy: `v2_20D`
- Freqtrade strategy: `CryptoSpotV220D`
- Evaluation model: fixed allocation per pair plus equal-weight aggregate
- Status: reference candidate passed fixed-allocation baseline plus quick and
  standard rolling review; paper trading still requires live signal audit.

## Readiness Gates

A candidate is suitable for paper trading when it passes these checks:

- Freqtrade adapter compiles and loads.
- Fixed-allocation per-pair reports are available for BTC, ETH, and BNB.
- Aggregate return, excess return, win rate, drawdown, exposure, and trade count
  are reviewed over full-period and rolling-window runs.
- `review/report.html`, `score.json`, and promotion checks are generated and
  archived for the candidate.
- No unexplained divergence exists between native strategy state and Freqtrade
  entry/exit behavior.
- Recent-window behavior is explainable from trade logs, especially losing
  windows and prolonged cash periods.

## Paper Trading Rules

- Freeze strategy code and parameters before starting the paper window.
- Keep the bot in dry-run for at least 3 months.
- Review every generated action reason before enabling real orders.
- Track fill price, fees, skipped orders, position size, and state transitions.
- Stop the paper test if action frequency or exposure differs materially from
  backtest expectations.

## Not Ready For Real Capital Until

- signal cadence matches backtest expectations;
- execution cost and slippage are within stress assumptions;
- BTC, ETH, and BNB each have understandable behavior;
- logs are complete enough to audit every action;
- no emergency manual intervention is needed during the paper window.
