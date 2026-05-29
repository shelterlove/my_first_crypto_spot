# V1 Migration Notes

V1 preserves the accepted V6.19 behavior under the repaired backtest
methodology, but the runtime strategy code is now flattened into a clean
`V1SpotStrategy` implementation.

## Migration Completed In This Workspace

- Expose the accepted behavior as `v1`.
- Flatten the accepted strategy into `src/crypto_spot_v1/strategy.py`.
- Keep the repaired event-driven backtest, rolling-window, metrics, evaluation,
  diagnostics, and HTML report runtime inside `src/crypto_spot_v1`.
- Use the repaired standard methodology: `next_open`, warmup excluded,
  fee-adjusted Buy & Hold.
- Write new V1 results under `results`.
- Do not register historical legacy strategy versions in the V1 runner; only
  local V1 baseline/candidate strategies are allowed.
- Do not optimize strategy rules during cleanup.
- Keep database credentials sourced from this project's `.env` or process
  environment variables.

## Verification

- Compile check passed for package modules, runner, and smoke test.
- Smoke test passed for execution timing, warmup exclusion, accounting
  reconstruction, action-count reconciliation, and Buy & Hold fee-adjusted
  initial buy.
- Full V1 baseline reproduced at `results/v1/20260521_232254`.
- V1 metrics match repaired legacy V6.19 run `results/v6_19/20260521_163754`
  exactly on score, return, excess, drawdown, trades, exposure, and turnover.
- Result artifact checks:
  - `actions_before_window = 0`
  - `equity_before_window = 0`
  - `bad_signal_timestamp = 0`
  - `execution_modes = ["next_open"]`
  - `action_count_delta = 0`

## Repository Status

This directory is now the V1 project root. Do not change V1 strategy rules
before optimization work is explicitly started.
