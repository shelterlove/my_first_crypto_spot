# Backtest Workflow

This repo has two evaluation stacks. Use the native V1 stack for strategy
research and candidate screening. Use the Freqtrade stack only after a candidate
has passed V1 development checks and needs deployment-aligned validation.

## Current Strategy State

| Strategy | Status | Notes |
| --- | --- | --- |
| `v2_21E` | baseline | Stable native V1 research baseline. |
| `v2_25F` | rejected | Post safe-recovery sell deadband hurt BNB 2019-08. |
| `v2_26A-F` | rejected / diagnostic only | External cap and deny variants did not produce stable improvements. |
| `v2_27A-D` | rejected | Tiny-buy relaxation weakened the existing guard. |
| `v2_28A` | inactive | Structure gate was too strict and did not trigger. |
| `v2_28B` | inactive | Correct condition family, but the buy-after window was too narrow. |
| `v2_28C` | active candidate | Conservative execution-layer target-reduce deadband candidate. |

## Evaluation Layers

1. Diagnostics
   - Goal: explain a behavior or identify possible score features.
   - Typical outputs: event CSVs, feature summaries, rule candidates, short reports.
   - Do not promote or reject a strategy from diagnostics alone.

2. Native V1 smoke
   - Goal: quickly reject candidates with obvious path damage.
   - Fixed windows:
     - `strong_bull`: `2019-02-25` to `2021-02-24`
     - `post_covid`: `2020-03-21` to `2021-03-21`
     - `path_pollution`: `2018-06-30` to `2021-06-29`
     - `bear_rally`: `2022-08-01` to `2022-12-31`
     - `bear_defence`: `2021-12-11` to `2022-12-11`
   - A candidate fails smoke if strong-bull recovery does not improve or if 2022
     defense windows materially worsen.

3. Native V1 rolling / complete
   - Goal: check whether smoke improvements survive the full V1 research grid.
   - Use `research` first for fast screening, then workflow `complete` for a
     full-grid candidate-vs-baseline delta check.
   - Use direct `run_v1_backtest.py --mode complete` only after workflow
     complete passes and a full artifact/HTML audit is needed.
   - Compare every serious candidate against `v2_21E`.

4. Freqtrade validation
   - Goal: deployment-aligned validation after V1 complete passes.
   - Do not tune rules on validation or holdout trades.
   - `2025-01-01` and later data is validation / holdout only unless the split
     policy is explicitly changed.

## Standard Commands

Native V1 workflow helper:

```powershell
python scripts\run_v1_candidate_workflow.py --candidate v2_28C --baseline v2_21E --stage smoke
python scripts\run_v1_candidate_workflow.py --candidate v2_28C --baseline v2_21E --stage research
python scripts\run_v1_candidate_workflow.py --candidate v2_28C --baseline v2_21E --stage complete
```

The workflow helper keeps `complete` lightweight enough for iteration: it uses
the complete rolling grid but skips heavy action/equity/per-bar artifacts. Run
the direct evaluator for the final native V1 audit.

Direct native V1 evaluator:

```powershell
python scripts\run_v1_backtest.py --candidate v2_28C --mode research --step-multiplier 1
python scripts\run_v1_backtest.py --candidate v2_28C --mode complete
```

Freqtrade validation remains separate:

```powershell
python scripts\freqtrade_eval.py --strategy CryptoSpotV219B --eval-split dev --rolling-preset quick --run-id rolling_v2_19B_dev_quick
```

## Output Locations

- Native V1 workflow helper: `results/v1_candidate_workflow/`
- Native V1 evaluator: `results/v1_eval_upgrade/`
- Research diagnostics: `results/diagnostics/`
- Freqtrade evaluation: `results/freqtrade_eval/`
- Freqtrade raw backtests: `freqtrade_user_data/backtest_results/`

Generated outputs are not source of truth and should not be committed.
