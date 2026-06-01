#!/usr/bin/env python3
"""Run the clean V1 baseline or a registered V1 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.benchmark import V1BenchmarkRunner
from crypto_spot_v1.evaluation import RESEARCH_MODE, VALID_MODES, normalize_mode, save_evaluation_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="v1",
        help="Registered strategy candidate to compare against buy_hold.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=RESEARCH_MODE,
        help="research for fast candidate screening; complete for full diagnostics/audit.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostics when mode resolves to complete.",
    )
    parser.add_argument(
        "--step-multiplier",
        type=int,
        default=None,
        help="Rolling-window step multiplier. Defaults to 2 in research and 1 in complete.",
    )
    return parser.parse_args()


def main(
    candidate_name: str,
    mode: str,
    diagnostics: bool = False,
    step_multiplier: int | None = None,
) -> None:
    config_path = PROJECT_ROOT / "configs" / "backtest_v1.json"
    output_dir = PROJECT_ROOT / "results"
    runner = V1BenchmarkRunner(str(config_path), output_dir=str(output_dir))

    effective_mode = normalize_mode(mode)
    if step_multiplier is None:
        step_multiplier = 2 if effective_mode == RESEARCH_MODE else 1
    if step_multiplier < 1:
        raise ValueError("--step-multiplier must be >= 1")
    runner.config.setdefault("evaluation", {})["window_step_multiplier"] = step_multiplier
    results = runner.run_all(
        candidate_name,
        collect_artifacts=(effective_mode != RESEARCH_MODE),
        window_step_multiplier=step_multiplier,
    )
    scores = runner.score_all(results)
    verdict = runner.check_promotion(results, candidate_name)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_evaluation_run(
        runner=runner,
        results=results,
        scores=scores,
        verdict=verdict,
        candidate_name=candidate_name,
        mode=effective_mode,
        timestamp=ts,
        config_path=config_path,
        output_root=output_dir,
        diagnostics_enabled=diagnostics or effective_mode != RESEARCH_MODE,
    )

    print(f"{candidate_name} {effective_mode} evaluation complete")
    print(f"window_step_multiplier={step_multiplier}")
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: score={score:.4f}")
    print(f"run_dir={run_dir}")
    if effective_mode != RESEARCH_MODE:
        print(f"report_path={run_dir / 'html_report.html'}")


if __name__ == "__main__":
    args = parse_args()
    main(args.candidate, args.mode, args.diagnostics, args.step_multiplier)
